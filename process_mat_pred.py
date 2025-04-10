"""
Author: Benny
Date: Nov 2019
"""
import importlib
import json
import logging
import math
import os
import platform
import shutil
import struct
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path
from typing import Tuple, List, Optional, Callable, BinaryIO
import matplotlib.pyplot as plt
from scipy import stats

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


def map_hdr(x, bias):
    return torch.log(1 + bias * x) / math.log(1 + bias)


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

    root = hydra.utils.to_absolute_path('prediction/Trial')

    process_dataset = MaterialDataset(root=root, npoints=args.num_point, num_samples_per_ds=1, randomized=False,
                                      dataset_type="clean")
    trainDataLoader = DataLoader(process_dataset, batch_size=1, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
        args)

    try:
        checkpoint = torch.load(f'best_model_mat_pred_{TrialManager().trial_name}.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except Exception as e:
        logger.warning(f'Could not find pretrained model {e}')

    try:
        classifier = torch.compile(classifier, backend="eager")
    except:
        pass

    classifier = classifier.cuda()

    classifier = classifier.eval()

    n_bins = 60
    bins = np.zeros(n_bins, dtype=np.longlong)
    bin_edges = None

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            data: torch.Tensor = torch.Tensor(data).float().cuda()
            target_albedo, target_metallic, target_norm, target_occ, target_hdr = (
                torch.Tensor(target[..., :3]).float().cuda(),
                torch.Tensor(
                    target[..., 4:6]).float().cuda(),
                torch.Tensor(
                    target[..., 6:9]).float().cuda(),
                torch.Tensor(
                    target[..., 10:11]).float().cuda(),
                torch.Tensor(
                    target[..., 11:]).float().cuda(),
            )

            pred_albedo, pred_metallic, pred_occ, pred_hdr, radiance_indices = classifier(data, target_norm)

            # HDR Statistics calcs
            # hdr = target[..., 11:].flatten()
            # transformed = np.log(hdr + 0.00001)
            # mean = torch.mean(transformed)
            # std = torch.std(transformed)
            # norm_dist = torch.distributions.Normal(mean, std)
            # transformed = norm_dist.cdf(transformed)
            # # transformed = stats.norm.cdf(transformed, loc=mean, scale=std)
            # cur_bins, bin_edges = np.histogram(transformed, bins=n_bins, range=(0, 1))
            # bins += cur_bins
            # # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # # plt.bar(bin_centers, cur_bins, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
            # # plt.show()
            # continue

            # target_metallic = target_metallic.cpu()
            # plt.scatter(x=target_metallic[..., 0], y=target_metallic[..., 1], s=1)
            # plt.xlabel("Metallic")
            # plt.ylabel("Smoothness")
            # plt.show()

            # b = (0 < target_metallic[..., 1]) & (target_metallic[..., 1] < 0.1)
            # pred_hdr = pred_hdr.reshape(pred_hdr.shape[0], pred_hdr.shape[1], -1, 3)
            # hdr_lum = torch.sqrt(0.299 * pred_hdr[..., 0] ** 2 + 0.587 * pred_hdr[..., 1] ** 2 + 0.114 * pred_hdr[..., 2] ** 2)
            # dirs = data[..., 3:].reshape(data.shape[0], data.shape[1], -1, 3)[..., range(0, 60, 2), :]
            #
            # hdr_lum = hdr_lum[b, :]
            # dirs = dirs[b, :, :]
            #
            # scaled_dirs = dirs * hdr_lum.unsqueeze(-1)
            #
            # min_lum = torch.min(hdr_lum, dim=-1, keepdim=True)[0].unsqueeze(-1)
            #
            # dirs = (dirs * min_lum).cpu()
            # scaled_dirs = scaled_dirs.cpu()

            # for j in range(dirs.shape[0]):
            #     fig = plt.figure()
            #     ax = fig.add_subplot(projection='3d')
            #
            #     ax.scatter(scaled_dirs[j, :, 0], scaled_dirs[j, :, 1], scaled_dirs[j, :, 2])
            #     ax.scatter(dirs[j, :, 0], dirs[j, :, 1], dirs[j, :, 2])
            #
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #
            #     plt.show(block=True)

            path = process_dataset.data_paths[i]
            with open(path.with_suffix('.meta.json'), 'r') as f:
                meta = json.load(f, object_pairs_hook=OrderedDict)

            with open(path, 'rb') as f:
                predicted_data = read_file(meta, f, args.num_point)

            insert_col = get_header_columns(meta, ["Metallic"], lambda _1, _2, cols: [cols[0], cols[-1]])[-1] + 1

            header: List[Tuple[str, str]] = list(meta["header"].items())

            metallic_index = header.index(("Metallic", "f4")) + 1

            num_view_dirs = radiance_indices.shape[-1]
            for i in reversed(range(num_view_dirs)):
                header.insert(metallic_index, (f"View Radiance {i + 1}", "f4"))

            for i in reversed(range(num_view_dirs)):
                header.insert(metallic_index, (f"Pred HDR Radiance {i + 1}", "f4"))

            header.insert(metallic_index, ("Pred Occlusion", "f4"))
            header.insert(metallic_index, ("Pred Metallic", "f4"))
            header.insert(metallic_index, ("Pred Albedo", "f4"))

            meta["header"] = OrderedDict(header)

            albedo = pred_albedo[0].cpu().numpy()
            albedo = np.append(albedo, np.ones((albedo.shape[0], 1)), axis=1)
            metallic = pred_metallic[0].cpu().numpy()
            metallic = np.insert(metallic, 1, np.zeros((metallic.shape[0], 2)).T, axis=1)
            occ = pred_occ[0].cpu().numpy()
            occ = np.repeat(occ, 4, axis=1)

            radiances = data[0, :, [i for i in range(3, data.shape[-1]) if ((i // 3) % 2) == 0]]
            radiance_indices = radiance_indices[0].unsqueeze(-1) * 3

            # Create indices for the 3 values (radiance) per viewpoint
            offsets = torch.arange(3, device=radiance_indices.device).view(1, 3)  # (1, 6)
            radiance_indices = (radiance_indices + offsets).view(radiance_indices.shape[0], -1)

            view_dir_radiances = torch.gather(radiances, -1, radiance_indices)
            fixed_view_dir_radiances = torch.zeros(view_dir_radiances.shape[0], num_view_dirs * 4, device=view_dir_radiances.device)
            view_dir_indices = [i for i in range(num_view_dirs * 4) if (i % 4) != 3]
            fixed_view_dir_radiances[:, view_dir_indices] = view_dir_radiances
            view_dir_radiances = fixed_view_dir_radiances.cpu().numpy()

            hdr_radiances = torch.zeros(view_dir_radiances.shape[0], num_view_dirs * 4, device=pred_hdr.device)
            hdr_radiances[:, view_dir_indices] = pred_hdr[0]
            hdr_radiances = hdr_radiances.cpu().numpy()

            predicted = np.concatenate((albedo, metallic, occ, hdr_radiances, view_dir_radiances), axis=1)

            predicted_data = np.hstack((predicted_data[:, :insert_col], predicted, predicted_data[:, insert_col:]))

            num_rows = args.num_point
            meta["numExtraData"] += 3 + num_view_dirs * 2
            meta["numPoints"] = num_rows

            with open(path.with_name(path.stem + '_predicted.meta.json'), 'w') as f:
                json.dump(meta, f)

            with open(path.with_name(path.stem + '_predicted.data'), 'wb') as f:
                for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
                    f.write(row_to_bytes(meta, predicted_data[i]))

            # light_points_per_row = round(pred_light.shape[1] ** (1. / 3))
            #
            # light_data = {
            #     "Intensities": pred_light[0].expand(-1, 3).reshape(light_points_per_row, light_points_per_row,
            #                                                        light_points_per_row, -1).tolist(),
            #     "Min": torch.min(data[0, :, :3], dim=0).values.tolist(),
            #     "Max": torch.max(data[0, :, :3], dim=0).values.tolist(),
            # }
            #
            # with open(path.with_name(path.stem + '_predicted_light_data.json'), 'w') as f:
            #     json.dump(light_data, f)

        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.bar(bin_centers, bins, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
        # plt.show()


if __name__ == '__main__':
    main()
