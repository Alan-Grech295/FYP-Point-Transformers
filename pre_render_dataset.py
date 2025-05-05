import json
import math
import multiprocessing
import os.path
import shutil
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import BinaryIO, Tuple, List, Callable, Optional
import torch.nn.functional as F

import numpy as np
import torch
from tqdm import tqdm

from dataset import MaterialDataset
from renderer import Renderer


def generate_fibonnaci_sphere(num_directions: int, angle_deg: float, device) -> torch.Tensor:
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

    y_min = math.sin(math.radians(angle_deg))
    y = 1.0 - ((1.0 - y_min) * indices) / num_directions

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


def rotation_matrix_from_normals(normals, up=(0, 1, 0)):
    def normalize(v):
        """ Normalizes a tensor along the last dimension, avoiding division by zero. """
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

    up = torch.tensor(up, dtype=torch.float32, device=normals.device).expand_as(normals)

    # Compute rotation axis (cross product)
    axis = torch.cross(up, normals, dim=-1)
    axis = normalize(axis)  # Normalize axis

    # Compute rotation angle (dot product and arccos)
    dot = torch.sum(normals * up, dim=-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(dot)  # Angle in radians

    # Rodrigues' formula components
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Skew-symmetric matrix of axis
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    K = torch.stack([
        torch.zeros_like(x), -z, y,
        z, torch.zeros_like(x), -x,
        -y, x, torch.zeros_like(x)
    ], dim=-1).view(*axis.shape[:-1], 3, 3)

    # Compute rotation matrix: R = I + sin(angle) * K + (1 - cos(angle)) * K^2
    I = torch.eye(3, device=normals.device).expand(*axis.shape[:-1], 3, 3)
    R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1)) * (K @ K)

    return R  # Shape: (B, N, 3, 3)


def rotate_directions(dirs, normals):
    """ Rotates only the view directions in an interleaved (dir, rad) tensor. """
    # Compute rotation matrix
    R = rotation_matrix_from_normals(normals, up=(0, 1, 0))  # Shape (B, N, 3, 3)

    # Rotate view directions: R @ view_dirs15
    # rotated_dirs = torch.matmul(dirs.unsqueeze(-2), R_T.unsqueeze(2)).squeeze(-2)

    rotated_dirs = torch.einsum("b n i j, b n k j -> b n k i", R, dirs)

    return rotated_dirs


class DatasetRenderer:
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

    def __init__(self, root='./data/material_pred'):
        self.root = root
        self.dataset_loader = MaterialDataset(root, 1e9, 1, False, "raw", cache_size_gb=0)
        self.renderer = Renderer(chunk_size=4)

    @staticmethod
    def __to_type(b: bytes, type: str, little_endian: bool):
        if type.startswith('f'):
            return struct.unpack(f'{"<" if little_endian else ">"}f', b)[0]
        elif type == 'i':
            return int.from_bytes(b, 'little' if little_endian else 'big')

        assert False, f"Invalid type provided '{type}'"

    def __read_file(self, file: BinaryIO, dtype):
        contents = np.fromfile(file, dtype=dtype)
        return np.column_stack([contents[field].astype(np.float64) for field in contents.dtype.names])

    def __get_header_dtype(self, meta: dict) -> np.dtype:
        fields = []
        for k, v in meta["header"].items():
            np_type = ("<" if meta["isLittleEndian"] else ">") + self.type_to_dtype[v]
            count = self.type_to_num_elements[v]
            if count == 1:
                fields.append((k, np_type))
            else:
                for i in range(count):
                    fields.append((f"{k}_{i}", np_type))

        return np.dtype(fields)

    def __to_structured_array(self, arr: np.ndarray, dtype: np.dtype):
        num_rows = arr.shape[0]
        structured_array = np.empty(num_rows, dtype=dtype)

        current_col_idx = 0
        for field_name, (field_dtype, field_shape) in dtype.fields.items():
            # Slice the relevant columns from the regular array that correspond to this field
            data_slice_from_regular = arr[:, current_col_idx: current_col_idx + 1]

            structured_array[field_name] = data_slice_from_regular[:, 0].astype(
                field_dtype.base)  # Cast and assign the column

            current_col_idx += 1  # Move to the columns for the next field

        return structured_array

    def clean_row(self, meta, row: np.ndarray):
        element_index = 0
        valid_view_dirs = []
        valid_view_radiances = []
        valid_hdr_view_radiances = []
        views_to_clean = []
        header_items = list(meta["header"].items())
        has_hdr_radiances = "hasHdrRadiances" in meta and meta["hasHdrRadiances"]
        for i in range(len(header_items)):
            k, v = header_items[i]
            num_elements = self.type_to_num_elements[v]
            if k.startswith('View Direction'):
                dir_index = k[len('View Direction'):].strip()
                r_k, r_v = header_items[i + 1]
                if has_hdr_radiances:
                    hdr_k, hdr_v = header_items[i + 2]
                    hdr_num_elements = self.type_to_num_elements[hdr_v]
                    assert hdr_k.endswith(dir_index), "View direction and HDR radiance must be in order"

                assert r_k.endswith(dir_index), "View direction and radiance must be in order"
                r_num_elements = self.type_to_num_elements[r_v]
                if all(np.isfinite(row[element_index + num_elements:element_index + num_elements + r_num_elements])):
                    valid_view_dirs.append(row[element_index:element_index + num_elements])
                    valid_view_radiances.append(row[
                                                element_index + num_elements:element_index + num_elements + r_num_elements])
                    if has_hdr_radiances:
                        start_radiance_index = element_index + num_elements + r_num_elements
                        valid_hdr_view_radiances.append(row[
                                                        start_radiance_index:start_radiance_index + hdr_num_elements])

                else:
                    if has_hdr_radiances:
                        views_to_clean.append((element_index, num_elements, r_num_elements, hdr_num_elements))
                    else:
                        views_to_clean.append((element_index, num_elements, r_num_elements))

            element_index += num_elements

        valid_view_dirs = np.array(valid_view_dirs)
        valid_view_radiances = np.array(valid_view_radiances)
        valid_hdr_view_radiances = np.array(valid_hdr_view_radiances)

        if len(valid_view_dirs) == 0:
            return

        for vals in views_to_clean:
            if has_hdr_radiances:
                invalid_view_index, dir_size, r_size, hdr_size = vals
            else:
                invalid_view_index, dir_size, r_size = vals

            dir = row[invalid_view_index:invalid_view_index + dir_size]
            dots = np.dot(valid_view_dirs, dir)
            closest_dir_indices = np.argpartition(dots, -(min(4, len(dots))))
            if len(closest_dir_indices) > 4:
                closest_dir_indices = closest_dir_indices[-4:]
            weighted_avg = np.average(
                valid_view_radiances[closest_dir_indices],
                weights=abs(dots[closest_dir_indices]),
                axis=0,
                keepdims=True).squeeze()
            row[invalid_view_index + dir_size:invalid_view_index + dir_size + r_size] = weighted_avg

            if has_hdr_radiances:
                hdr_weighted_avg = np.average(
                    valid_hdr_view_radiances[closest_dir_indices],
                    weights=abs(dots[closest_dir_indices]),
                    axis=0,
                    keepdims=True).squeeze()
                hdr_start_index = invalid_view_index + dir_size + r_size
                row[hdr_start_index:hdr_start_index + hdr_size] = hdr_weighted_avg

        clean_percent = len(valid_view_radiances) / (len(valid_view_radiances) + len(views_to_clean))
        assert 0 <= clean_percent <= 1, "Clean percentage range is not between 0 and 1"
        assert (len(valid_view_radiances) + len(views_to_clean)) == meta[
            "numViewpoints"], "Invalid number of viewpoints"
        row[-1] = clean_percent

    def compute_occlusion(self, meta: OrderedDict, data: np.ndarray) -> np.ndarray:
        occ_index = 0
        for k, v in meta["header"].items():
            if k == "Occlusion":
                break

            occ_index += self.type_to_num_elements[v]

        data = np.insert(data, occ_index + 2, np.zeros(data.shape[0]), axis=1)

        max_occ = np.max(data[:, occ_index])
        if np.isnan(max_occ) or max_occ == 0:
            data[:, occ_index] = 0
            data[:, occ_index + 1] = 0
            max_occ = 1
        data[:, occ_index + 2] = data[:, occ_index] / max_occ
        meta["header"]["Occlusion"] = "f3"

        return data

    def __save_element(self, el_type: str, row: np.ndarray, start_index: int, little_endian: bool) -> Tuple[bytes, int]:
        b = bytearray()
        if el_type.startswith('f'):
            save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}f', v)
        elif el_type.startswith('i'):
            save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}i', int(v))
        else:
            assert False, f"Invalid element type '{el_type}' provided"

        for i in range(self.type_to_num_elements[el_type]):
            b.extend(save_func(row[start_index]))
            start_index += 1

        return b, start_index

    def row_to_bytes(self, meta, row):
        row_bytes = bytearray()
        el_index = 0
        for v in meta["header"].values():
            b, el_index = self.__save_element(v, row, el_index, meta["isLittleEndian"])
            row_bytes.extend(b)
        return row_bytes

    def __get_header_columns(self, meta: dict, columns: List[str],
                             process_columns: Optional[Callable[[str, str, List[int]], List[int]]] = None):
        col_index = 0
        cols = []
        for k, v in meta["header"].items():
            col_size = self.type_to_num_elements[v]
            if any([fnmatch(k, c) for c in columns]):
                cur_cols = list(range(col_index, col_index + col_size))
                if process_columns:
                    cur_cols = process_columns(k, v, cur_cols)
                cols.extend(cur_cols)
            col_index += col_size
        return cols

    def process_file(self, index):
        path = self.dataset_loader.data_paths[index]
        print(f"Started rendering {path}")
        with open(path.with_suffix('.meta.json'), 'r') as f:
            meta = json.load(f, object_pairs_hook=OrderedDict)

        dtype = self.__get_header_dtype(meta)

        num_rows = meta["numPoints"]
        with open(path, 'rb') as f:
            data = self.__read_file(f, dtype)

        # Render dataset
        ds_data, ds_target, env_map = self.dataset_loader[index]
        normals: torch.Tensor = torch.Tensor(ds_data[..., 3:6]).unsqueeze(0).float().cuda()
        albedo: torch.Tensor = torch.Tensor(ds_target[..., :3]).unsqueeze(0).float().cuda()
        metallic: torch.Tensor = torch.Tensor(ds_target[..., 4:5]).unsqueeze(0).float().cuda()
        smoothness: torch.Tensor = torch.Tensor(ds_target[..., 5:6]).unsqueeze(0).float().cuda()
        env_visibility: torch.Tensor = torch.Tensor(ds_target[..., 6:]).unsqueeze(0).float().cuda()

        env_map = torch.Tensor(env_map).unsqueeze(0).float().cuda()

        # exposure = 0.0001
        num_view_dirs = 32

        view_dirs = generate_fibonnaci_sphere(num_view_dirs, 10, normals.device).unsqueeze(0).unsqueeze(0).expand(-1,
                                                                                                                  num_rows,
                                                                                                                  -1,
                                                                                                                  -1)
        view_dirs = rotate_directions(view_dirs, normals)

        radiance, hdr_radiance, exposure = self.renderer(normals, albedo, metallic, smoothness, view_dirs,
                                                         env_visibility, env_map,
                                                         exposure=None, return_hdr=True, return_exposure=True)

        if not torch.isfinite(radiance).all() or not torch.isfinite(hdr_radiance).all():
            print("WARNING: Invalid values detected")

        meta["exposure"] = exposure.item()
        meta["numViewpoints"] = num_view_dirs

        # Add view direction, radiance and hdr radiance to header
        header = list(meta["header"].items())
        header += [(f"View Direction {i + 1}", "f3") for i in range(num_view_dirs)]
        header += [(f"Radiance {i + 1}", "f3") for i in range(num_view_dirs)]
        header += [(f"HDR Radiance {i + 1}", "f3") for i in range(num_view_dirs)]
        meta["header"] = OrderedDict(header)

        np_view_dirs = view_dirs[0].reshape(num_rows, -1).cpu().numpy()
        np_radiance = radiance[0].reshape(num_rows, -1).cpu().numpy()
        np_hdr_radiance = hdr_radiance[0].reshape(num_rows, -1).cpu().numpy()
        data = np.hstack((data, np_view_dirs, np_radiance, np_hdr_radiance))

        # excl_occ = np.hstack((data[:, :16], data[:, 18:]))
        #
        # assert np.isfinite(excl_occ).all()

        with open(path.with_name(path.stem + '_rendered.meta.json'), 'w') as f:
            json.dump(meta, f)

        # assert np.isfinite(data).all(), f"Found nan in dataset {path} {np.count_nonzero(np.isfinite(data))}"
        new_dtype = self.__get_header_dtype(meta)
        structured_array = self.__to_structured_array(data, new_dtype)

        data_bytes = structured_array.tobytes()

        with open(path.with_name(path.stem + '_rendered.data'), 'wb') as f:
            f.write(data_bytes)

        print(f"Finished rendering {path}")

    def render(self):
        num_processes = 12  # multiprocessing.cpu_count()

        # for i in range(len(self.dataset_loader)):
        #     self.process_file(i)

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(self.process_file, range(len(self.dataset_loader)))


if __name__ == "__main__":
    # dataset_renderer = DatasetRenderer(root="/mnt/e/FYP Dataset/32768_64")
    # dataset_renderer = DatasetRenderer(root="/mnt/d/Dev/Point-Transformers/prediction/Trial")
    dataset_renderer = DatasetRenderer(root="D:\\Dev\\Point-Transformers\\prediction\\Trial")
    # dataset_renderer = DatasetRenderer(root="E:/FYP Dataset/32768_64/train/Outdoor/DirLight")
    # dataset_cleaner = DatasetRenderer(root="E:\\FYP Dataset\\32768_64\\train\\Outdoor\\Skybox")
    dataset_renderer.render()
    # dataset_cleaner.copy_light_data()
