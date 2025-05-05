"""
Author: Benny
Date: Nov 2019
"""
import importlib
import json
import logging
import math
import os
import shutil
import struct
from collections import OrderedDict
from fnmatch import fnmatch
from typing import Tuple, List, Optional, Callable, BinaryIO

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaterialDataset

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
        save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}i', int(v))
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
    return np.column_stack([contents[field].astype(np.float64) for field in contents.dtype.names])


def map_hdr(x, bias):
    return torch.log(1 + bias * x) / math.log(1 + bias)


def generate_fibonnaci_sphere(num_directions: int, device) -> torch.Tensor:
    """
    Generates approximately uniformly distributed points on a unit sphere
    using the Fibonacci lattice method.

    Args:
        num_directions: The desired number of direction vectors (points on the sphere).
        device: The torch device ('cpu', 'cuda', etc.) to create the tensor on.

    Returns:
        A tensor of shape (num_directions, 3) containing normalized 3D direction vectors.
    """

    indices = torch.arange(num_directions, dtype=torch.float32, device=device) + 0.5
    phi = math.pi * (3.0 - math.sqrt(5.0))

    y = 1.0 - (2.0 * indices) / num_directions

    # Radius of the circle at height y
    radius = torch.sqrt(1.0 - y * y)

    # Angle increment (longitude)
    theta = phi * indices

    # Calculate x and z coordinates
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    # Stack coordinates into vectors
    # The vectors should theoretically already be normalized because x^2 + y^2 + z^2 = 1
    directions = torch.stack([x, y, z], dim=-1)

    directions = F.normalize(directions, p=2, dim=-1)

    return directions


@hydra.main(config_path='config', config_name='mat_pred', version_base=None)
def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('prediction/Trial')

    process_dataset = MaterialDataset(root=root, npoints=args.num_point, num_samples_per_ds=5, randomized=False,
                                      dataset_type="rendered")
    trainDataLoader = DataLoader(process_dataset, batch_size=1, num_workers=10, shuffle=False)

    '''MODEL LOADING'''
    args.input_dim = 3 + (3 + 3) * 64
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerMat')(
        args)

    try:
        checkpoint = torch.load(f'best_model_mat_pred.pth')
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

    # with torch.cuda.amp.autocast(enabled=True):
    #     for i, (data, target, env_map) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
    #         xyz = torch.Tensor(data[..., :3]).float().cuda()
    #         normals = torch.Tensor(data[..., 3:6]).float().cuda()
    #         albedo = torch.Tensor(target[..., :3]).float().cuda()
    #         metallic = torch.Tensor(target[..., 4:5]).float().cuda()
    #         smoothness = torch.Tensor(target[..., 5:6]).float().cuda()
    #         env_visibility = torch.Tensor(target[..., 6:]).float().cuda()
    #
    #         env_map = torch.Tensor(env_map).float().cuda()
    #         max = torch.max(env_map)
    #
    #         B, N, _ = xyz.shape
    #
    #         view_dirs = generate_fibonnaci_sphere(24, xyz.device)
    #         view_dirs = view_dirs.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    #
    #         radiance = renderer(normals, albedo, metallic, smoothness, view_dirs, env_visibility, env_map, exposure=0.0001)
    #
    #         # Saving radiance
    #         path = process_dataset.data_paths[i]
    #         with open(path.with_suffix('.meta.json'), 'r') as f:
    #             meta = json.load(f, object_pairs_hook=OrderedDict)
    #
    #         with open(path, 'rb') as f:
    #             predicted_data = read_file(meta, f, args.num_point)
    #
    #         header: List[Tuple[str, str]] = list(meta["header"].items())
    #         for j in range(radiance.shape[2]):
    #             header.append((f"View Direction {j + 1}", "f3"))
    #             header.append((f"View Radiance {j + 1}", "f3"))
    #
    #         meta["header"] = OrderedDict(header)
    #         meta["numViewpoints"] = radiance.shape[2]
    #
    #         np_radiance = radiance[0].cpu().numpy()
    #         np_dirs = view_dirs[0].cpu().numpy()
    #         view_radiance = np.empty((N, radiance.shape[2] * 2, 3))
    #         view_radiance[:, range(0, view_radiance.shape[1], 2), :] = np_dirs
    #         view_radiance[:, range(1, view_radiance.shape[1], 2), :] = np_radiance
    #         view_radiance = view_radiance.reshape(N, -1)
    #
    #         predicted_data = np.hstack((predicted_data, view_radiance))
    #
    #         num_rows = args.num_point
    #         meta["numPoints"] = num_rows
    #
    #         with open(path.with_name(path.stem + '_predicted.meta.json'), 'w') as f:
    #             json.dump(meta, f)
    #
    #         with open(path.with_name(path.stem + '_predicted.data'), 'wb') as f:
    #             for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
    #                 f.write(row_to_bytes(meta, predicted_data[i]))
    #
    # return

    results = [np.empty((args.num_point * process_dataset.num_samples_per_ds, 8), dtype=np.float32) for _ in
               range(len(process_dataset.data_paths))]

    with torch.no_grad():
        for i, (data, target, view_radiances, env_map, exposure) in tqdm(enumerate(trainDataLoader),
                                                                         total=len(trainDataLoader),
                                                                         smoothing=0.9):
            xyz = torch.Tensor(data[..., :3]).float().cuda()
            normals = torch.Tensor(data[..., 3:6]).float().cuda()
            target_albedo = torch.Tensor(target[..., :3]).float()
            # env_visibility = torch.Tensor(target[..., 6:]).float().cuda()

            B, N = data.shape[0], data.shape[1]
            view_radiances = view_radiances.reshape(B, N, 3, -1, 3)
            target_view_dirs = torch.Tensor(view_radiances[..., 0, :, :]).float().cuda()
            target_view_rads = torch.Tensor(view_radiances[..., 1, :, :]).float().cuda()
            # target_view_hdr_rads = torch.Tensor(view_radiances[..., 2, :, :]).float().cuda()

            # target_env_map = torch.Tensor(env_map).float().cuda()
            # min_env_map, max_env_map = torch.min(target_env_map), torch.max(target_env_map)

            pred_albedo, pred_metallic, pred_smoothness = classifier(xyz, normals, target_view_dirs,
                                                                     target_view_rads)

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

            albedo = pred_albedo[0].cpu().numpy()
            albedo = np.append(albedo, np.ones((albedo.shape[0], 1)), axis=1)
            metallic = torch.cat((pred_metallic, pred_smoothness), dim=-1)[0].cpu().numpy()
            metallic = np.insert(metallic, 1, np.zeros((metallic.shape[0], 2)).T, axis=1)

            predicted = np.concatenate((albedo, metallic), axis=1)

            ds_index = i % len(process_dataset.data_paths)
            offset_index = i // len(process_dataset.data_paths) * process_dataset.npoints

            results[ds_index][offset_index:offset_index + process_dataset.npoints] = predicted

            continue

            path = process_dataset.data_paths[i]
            with open(path.with_suffix('.meta.json'), 'r') as f:
                meta = json.load(f, object_pairs_hook=OrderedDict)

            with open(path, 'rb') as f:
                predicted_data = read_file(meta, f, args.num_point)

            insert_col = get_header_columns(meta, ["Metallic"], lambda _1, _2, cols: [cols[0], cols[-1]])[-1] + 1

            header: List[Tuple[str, str]] = list(meta["header"].items())

            metallic_index = header.index(("Metallic", "f4")) + 1

            header.insert(metallic_index, ("Pred Metallic", "f4"))
            header.insert(metallic_index, ("Pred Albedo", "f4"))

            meta["header"] = OrderedDict(header)

            albedo = pred_albedo[0].cpu().numpy()
            albedo = np.append(albedo, np.ones((albedo.shape[0], 1)), axis=1)
            metallic = torch.cat((pred_metallic, pred_smoothness), dim=-1)[0].cpu().numpy()
            metallic = np.insert(metallic, 1, np.zeros((metallic.shape[0], 2)).T, axis=1)

            predicted = np.concatenate((albedo, metallic), axis=1)

            predicted_data = np.hstack((predicted_data[:, :insert_col], predicted, predicted_data[:, insert_col:]))

            num_rows = args.num_point
            meta["numExtraData"] += 2
            meta["numPoints"] = num_rows

            with open(path.with_name(path.stem + '_predicted.meta.json'), 'w') as f:
                json.dump(meta, f)

            with open(path.with_name(path.stem + '_predicted.data'), 'wb') as f:
                for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
                    f.write(row_to_bytes(meta, predicted_data[i]))

        num_rows = process_dataset.num_samples_per_ds * process_dataset.npoints

        for i, path in enumerate(process_dataset.data_paths):
            with open(path.with_suffix('.meta.json'), 'r') as f:
                meta = json.load(f, object_pairs_hook=OrderedDict)

            with open(path, 'rb') as f:
                predicted_data = read_file(meta, f, num_rows)

            insert_col = get_header_columns(meta, ["Metallic"], lambda _1, _2, cols: [cols[0], cols[-1]])[-1] + 1

            header: List[Tuple[str, str]] = list(meta["header"].items())

            metallic_index = header.index(("Metallic", "f4")) + 1

            header.insert(metallic_index, ("Pred Metallic", "f4"))
            header.insert(metallic_index, ("Pred Albedo", "f4"))

            meta["header"] = OrderedDict(header)

            predicted = results[i]

            pred_albedo = predicted[:, :3]
            pred_metallic = predicted[:, 4:5]
            pred_smoothness = predicted[:, 7:8]

            target_data = predicted_data[:, get_header_columns(meta, ["Albedo", "Metallic"])]

            target_albedo = target_data[:, :3]
            target_metallic = target_data[:, 4:5]
            target_smoothness = target_data[:, 7:8]

            def mse_loss(a, b):
                return ((a - b) ** 2).mean()

            def mae_loss(a, b):
                return (np.absolute(a - b)).mean()

            albedo_loss = mae_loss(target_albedo, pred_albedo)
            metallic_loss = mae_loss(target_metallic, pred_metallic)
            smoothness_loss = mae_loss(target_smoothness, pred_smoothness)

            print(f"Loss for {path}: Albedo - {albedo_loss}, Metallic - {metallic_loss}, Smoothness - {smoothness_loss}")

            predicted_data = np.hstack((predicted_data[:, :insert_col], predicted, predicted_data[:, insert_col:]))

            meta["numPoints"] = num_rows
            meta["numExtraData"] += 2

            with open(path.with_name(path.stem + '_predicted.meta.json'), 'w') as f:
                json.dump(meta, f)

            with open(path.with_name(path.stem + '_predicted.data'), 'wb') as f:
                for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
                    f.write(row_to_bytes(meta, predicted_data[i]))

        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.bar(bin_centers, bins, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
        # plt.show()


if __name__ == '__main__':
    main()
