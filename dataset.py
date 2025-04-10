import math
import struct
import time
from fnmatch import fnmatch
from io import BytesIO
from pathlib import Path

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from typing import List, TextIO, BinaryIO, Tuple, Literal, Optional, Callable

from pointnet_util import farthest_point_sample, pc_normalize
import json

from timer import Timer


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500,
                 split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


class MaterialDataset(Dataset):
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

    def __init__(self, root='./data/material_pred', npoints=100000, num_samples_per_ds=1, randomized=True,
                 dataset_type: Literal["all", "raw", "clean"] = "clean", with_light_data=False, cache_size_gb=1):
        self.npoints = npoints
        self.num_samples_per_ds = num_samples_per_ds
        self.with_light_data = with_light_data
        if '*' in str(root):
            self.root, self.wildcard_prepend = str(root).split('*', maxsplit=1)
        else:
            self.root = root
            self.wildcard_prepend = ""

        if len(self.wildcard_prepend) > 0:
            self.wildcard_prepend = "*" + self.wildcard_prepend

        if dataset_type == "all" or dataset_type == "raw":
            self.data_paths = list(Path(self.root).rglob(os.path.join(self.wildcard_prepend, "*.data")))
            if dataset_type == "raw":
                self.data_paths = [p for p in self.data_paths if not p.stem.endswith("_cleaned")]
        else:
            self.data_paths = list(Path(self.root).rglob(os.path.join(self.wildcard_prepend, "*_cleaned.data")))

        self.cache = {}  # from index to (point_set, ) tuple
        self.cache_size_bytes = 1_073_741_824 * cache_size_gb
        self.cur_size_bytes = 0
        self.randomized = randomized

    @staticmethod
    def __to_type(b: bytes, type: str, little_endian: bool):
        if type.startswith('f'):
            return struct.unpack(f'{"<" if little_endian else ">"}f', b)[0]
        elif type == 'i':
            return int.from_bytes(b, 'little' if little_endian else 'big')

        assert False, f"Invalid type provided '{type}'"

    def __read_file(self, meta: dict, file: BinaryIO, offset_rows=0, num_rows=-1):
        dtype = self.__get_header_dtype(meta)
        offset = offset_rows * dtype.itemsize
        contents = np.fromfile(file, dtype=dtype, offset=offset, count=num_rows)
        return np.column_stack([contents[field].astype(np.float32) for field in contents.dtype.names])

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

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        ds_index = index % len(self.data_paths)

        if ds_index in self.cache:
            data, target = self.cache[ds_index]

            num_rows = min(self.npoints, data.shape[0])
            np.random.seed(index)
            if self.randomized:
                indices = np.random.choice(data.shape[0], num_rows, replace=False)
            else:
                indices = np.arange(num_rows)
            data = data[indices, :]
            target = target[indices, :]

            return data, target
        else:
            path = self.data_paths[ds_index]
            with open(path.with_suffix('.meta.json'), 'r') as f:
                meta = json.load(f)

            num_rows = min(self.npoints, meta["numPoints"])
            # offset = (index // len(self.data_paths)) * num_rows

            with open(path, 'rb') as f:
                rows = self.__read_file(meta, f)

            np.random.seed(index)
            if self.randomized:
                indices = np.random.choice(rows.shape[0], num_rows, replace=False)
            else:
                indices = np.arange(num_rows)

            # rows = rows[indices, :]

            def process_cols(key: str, data_type: str, cols: List[int]) -> List[int]:
                if key == "Metallic":
                    # Only Red and Alpha channels are used in metallic map https://docs.unity3d.com/6000.0/Documentation/Manual/StandardShaderMaterialParameterMetallic.html
                    return [cols[0], cols[-1]]
                elif key == "Occlusion":
                    return [cols[2]]

                return cols

            data_col_indices = self.__get_header_columns(meta, ["Position", "View Direction *", "Radiance *"])
            data_cols = rows[:, data_col_indices]
            target_col_indices = self.__get_header_columns(meta, ["Albedo", "Metallic", "Normal", "Occlusion"],
                                                           process_cols)
            if "hasHdrRadiances" in meta and meta["hasHdrRadiances"]:
                target_col_indices.extend(self.__get_header_columns(meta, ["HDR Radiance *"]))
            target_cols = rows[:, target_col_indices]

            # if self.with_light_data:
            #     light_data_path = path.with_name(path.stem + "_light_data.json")
            #     with open(light_data_path, 'r') as f:
            #         light_data = json.load(f)
            #
            #     intensities = np.array(light_data["Intensities"], dtype=np.float32)
            #     min_intensity = np.min(intensities)
            #     max_intensity = np.max(intensities)
            #
            #     if np.isclose(min_intensity, max_intensity):
            #         intensities = np.ones_like(intensities)
            #     else:
            #         intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
            #
            #     light_bounds_min = np.array(light_data["Min"], dtype=np.float32)
            #     light_bounds_max = np.array(light_data["Max"], dtype=np.float32)
            #
            # out_tuple = [data_cols, target_cols]
            # if self.with_light_data:
            #     out_tuple.extend([intensities, light_bounds_min, light_bounds_max])
            #
            # out_tuple = tuple(out_tuple)

            new_size_bytes = self.cur_size_bytes + (rows.shape[0] * rows.shape[1] * 4)
            if ds_index not in self.cache and new_size_bytes <= self.cache_size_bytes:
                self.cache[ds_index] = (data_cols, target_cols)
                self.cur_size_bytes = new_size_bytes
            return data_cols[indices, :], target_cols[indices, :]

    def __len__(self):
        return len(self.data_paths) * self.num_samples_per_ds


if __name__ == '__main__':
    # data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    # for point,label in DataLoader:
    #     print(point.shape)
    #     print(label.shape)

    data = MaterialDataset(root="E:\\FYP Dataset\\32768_64\\train\\Outdoor\\DirLight", npoints=10000, dataset_type="raw")
    train, target = data[1]
    print(train.shape, target.shape)
    print(train, target)
