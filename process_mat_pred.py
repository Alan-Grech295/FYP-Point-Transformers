"""
Author: Benny
Date: Nov 2019
"""
import importlib
import json
import logging
import os
import platform
import shutil
import struct
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path
from typing import Tuple, List, Optional, Callable, BinaryIO

import hydra
import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaterialDataset
from trial_manager import TrialManager

type_to_size = {
    "f": 4,
    "f2": 8,
    "f3": 12,
    "f4": 16,
    "i": 4
}

type_to_num_elements = {
    "f": 1,
    "f2": 2,
    "f3": 3,
    "f4": 4,
    "i": 1
}

type_to_dtype = {
    "f": "f4",
    "f2": "f4",
    "f3": "f4",
    "f4": "f4",
    "i": "i4"
}


def __save_element(el_type: str, row: np.ndarray, start_index: int, little_endian: bool) -> Tuple[bytes, int]:
    b = bytearray()
    if el_type.startswith('f'):
        save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}f', v)
    elif el_type.startswith('i'):
        save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}I', int(v))
    else:
        assert False, f"Invalid element type '{el_type}' provided"

    for i in range(type_to_num_elements[el_type]):
        b.extend(save_func(row[start_index]))
        start_index += 1

    return b, start_index


def row_to_bytes(meta, row):
    row_bytes = bytearray()
    el_index = 0
    for v in meta["header"].values():
        b, el_index = __save_element(v, row, el_index, meta["isLittleEndian"])
        row_bytes.extend(b)
    return row_bytes


def get_header_columns(meta: dict, columns: List[str],
                       process_columns: Optional[Callable[[str, str, List[int]], List[int]]] = None):
    col_index = 0
    cols = []
    for k, v in meta["header"].items():
        col_size = type_to_num_elements[v]
        if any([fnmatch(k, c) for c in columns]):
            cur_cols = list(range(col_index, col_index + col_size))
            if process_columns:
                cur_cols = process_columns(k, v, cur_cols)
            cols.extend(cur_cols)
        col_index += col_size
    return cols


def get_header_dtype(meta: dict) -> np.dtype:
    fields = []
    for k, v in meta["header"].items():
        np_type = ("<" if meta["isLittleEndian"] else ">") + type_to_dtype[v]
        count = type_to_num_elements[v]
        if count == 1:
            fields.append((k, np_type))
        else:
            for i in range(count):
                fields.append((f"{k}_{i}", np_type))

    return np.dtype(fields)


def read_file(meta: dict, file: BinaryIO, num_rows=10_000):
    dtype = get_header_dtype(meta)
    contents = np.fromfile(file, dtype=dtype, count=num_rows)
    return np.column_stack([contents[field].astype(np.float32) for field in contents.dtype.names])


@hydra.main(config_path='config', config_name='mat_pred', version_base=None)
def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    omegaconf.OmegaConf.set_struct(args, False)
    TrialManager.supress_checks()
    TrialManager.set_trial("default")

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('prediction/')

    process_dataset = MaterialDataset(root=root, npoints=args.num_point, num_samples_per_ds=1, randomized=False,
                                      dataset_type="clean")
    trainDataLoader = DataLoader(process_dataset, batch_size=1, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
        args)
    try:
        classifier = torch.compile(classifier, backend="eager")
    except:
        pass

    classifier = classifier.cuda()

    try:
        checkpoint = torch.load(f'best_model_mat_pred_{TrialManager().trial_name}.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except Exception as e:
        logger.warning(f'Could not find pretrained model {e}')

    classifier = classifier.eval()

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                      smoothing=0.9):
            data: torch.Tensor = torch.Tensor(data).float().cuda()
            target_albedo, target_metallic, target_norm, target_occ = (torch.Tensor(target[..., :3]).float().cuda(),
                                                                       torch.Tensor(
                                                                           target[..., 4:6]).float().cuda(),
                                                                       torch.Tensor(
                                                                           target[..., 6:9]).float().cuda(),
                                                                       torch.Tensor(
                                                                           target[..., 10:11]).float().cuda())

            pred_albedo, pred_metallic, pred_occ = classifier(data, target_norm)

            path = process_dataset.data_paths[i]
            with open(path.with_suffix('.meta.json'), 'r') as f:
                meta = json.load(f, object_pairs_hook=OrderedDict)

            with open(path, 'rb') as f:
                predicted_data = read_file(meta, f, args.num_point)

            insert_col = get_header_columns(meta, ["Metallic"], lambda _1, _2, cols: [cols[0], cols[-1]])[-1] + 1

            header: List[Tuple[str, str]] = list(meta["header"].items())
            metallic_index = header.index(("Metallic", "f4")) + 1
            header.insert(metallic_index, ("Pred Occlusion", "f4"))
            header.insert(metallic_index, ("Pred Metallic", "f4"))
            header.insert(metallic_index, ("Pred Albedo", "f4"))

            meta["header"] = OrderedDict(header)

            # print(pred_albedo[0])

            albedo = pred_albedo[0].cpu().numpy()
            albedo = np.append(albedo, np.ones((albedo.shape[0], 1)), axis=1)
            metallic = pred_metallic[0].cpu().numpy()
            metallic = np.insert(metallic, 1, np.zeros((metallic.shape[0], 2)).T, axis=1)
            occ = pred_occ[0].cpu().numpy()
            occ = np.repeat(occ, 4, axis=1)

            # dir_radiance = rotated_dirs[0].cpu().numpy()
            # dir_0 = np.append(dir_radiance[:, :3], np.ones((dir_radiance.shape[0], 1)), axis=1)
            # rad_0 = np.append(dir_radiance[:, 3:6], np.ones((dir_radiance.shape[0], 1)), axis=1)
            #
            # received_normal = target[0, :, :4]
            # received_normal = np.append(received_normal, np.ones((received_normal.shape[0], 1)), axis=1)

            # predicted_data[..., albedo_header_cols] = pred_albedo[0].cpu().numpy()

            predicted = np.concatenate((albedo, metallic, occ), axis=1)

            predicted_data = np.hstack((predicted_data[:, :insert_col], predicted, predicted_data[:, insert_col:]))
            # predicted_data = np.insert(predicted_data, obj=insert_col, values=metallic.T, axis=1)
            # predicted_data = np.insert(predicted_data, obj=insert_col, values=albedo.T, axis=1)
            # predicted_data[..., metallic_header_cols] = pred_metallic[0].cpu().numpy()

            num_rows = args.num_point
            meta["numExtraData"] += 3

            with open(path.with_name(path.stem + '_predicted.meta.json'), 'w') as f:
                json.dump(meta, f)

            with open(path.with_name(path.stem + '_predicted.data'), 'wb') as f:
                for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
                    f.write(row_to_bytes(meta, predicted_data[i]))


if __name__ == '__main__':
    main()
