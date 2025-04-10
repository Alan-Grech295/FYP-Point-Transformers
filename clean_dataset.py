import json
import multiprocessing
import os.path
import shutil
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import BinaryIO, Tuple, List, Callable, Optional

import numpy as np
from tqdm import tqdm


class DatasetCleaner:
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

    def __init__(self, root='./data/material_pred', skip_cleaned: bool = True):
        self.root = root
        self.data_paths = sorted(
            [path for path in Path(root).rglob("*.data") if
             not path.stem.endswith("_cleaned") and (not skip_cleaned or not os.path.exists(
                 path.with_name(path.stem + "_cleaned.data")))],
            key=lambda path: os.path.getsize(path))

    @staticmethod
    def __to_type(b: bytes, type: str, little_endian: bool):
        if type.startswith('f'):
            return struct.unpack(f'{"<" if little_endian else ">"}f', b)[0]
        elif type == 'i':
            return int.from_bytes(b, 'little' if little_endian else 'big')

        assert False, f"Invalid type provided '{type}'"

    def __read_file(self, file: BinaryIO, dtype):
        contents = np.fromfile(file, dtype=dtype)
        return np.column_stack([contents[field].astype(np.float32) for field in contents.dtype.names])
        # num_row_elements = sum([self.type_to_num_elements[v] for v in meta["header"].values()])
        # rows = np.empty((num_rows, num_row_elements))
        # for r in tqdm(range(num_rows), desc=f"Processing {file.name}", total=num_rows, smoothing=0.9):
        #     element_index = 0
        #     for k, v in meta["header"].items():
        #         num_elements = self.type_to_num_elements[v]
        #         for i in range(num_elements):
        #             # Assuming all elements are aligned to 4 bytes
        #             rows[r, element_index] = self.__to_type(file.read(4), v, meta["isLittleEndian"])
        #             element_index += 1
        #
        # return rows

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
            save_func = lambda v: struct.pack(f'{"<" if little_endian else ">"}I', int(v))
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

    def clean_file(self, path):
        print(f"Started cleaning {path}")
        with open(path.with_suffix('.meta.json'), 'r') as f:
            meta = json.load(f, object_pairs_hook=OrderedDict)

        dtype = self.__get_header_dtype(meta)

        num_rows = meta["numPoints"]
        with open(path, 'rb') as f:
            data = self.__read_file(f, dtype)

        if np.all(np.isfinite(data)) and meta["header"]["Occlusion"] == "f3":
            print("Dataset already clean")
            return

        # Clean invalid views
        meta["header"]["Correct Viewpoint Ratio"] = "f"  # Add Correct Viewpoint Ratio header
        data = np.append(data, np.zeros((data.shape[0], 1)), axis=1)  # Add last column for clean rows ratio
        for i in tqdm(range(num_rows), total=num_rows, desc="Cleaning dataset", smoothing=0.9):
            self.clean_row(meta, data[i])

        # Normalize HDR radiance values
        if "hasHdrRadiances" in meta and meta["hasHdrRadiances"]:
            hdr_cols = self.__get_header_columns(meta, ["HDR Radiance *"])
            hdr_radiances = data[:, hdr_cols]
            min = np.min(hdr_radiances)
            max = np.max(hdr_radiances)

            hdr_radiances = (hdr_radiances - min) / (max - min)
            data[:, hdr_cols] = hdr_radiances

        # Add better computed occlusion
        data = self.compute_occlusion(meta, data)

        assert np.all(np.isfinite(data)), f"Dataset '{path}' was not properly cleaned"

        with open(path.with_name(path.stem + '_cleaned.meta.json'), 'w') as f:
            json.dump(meta, f)

        with open(path.with_name(path.stem + '_cleaned.data'), 'wb') as f:
            for i in tqdm(range(num_rows), total=num_rows, desc="Saving dataset", smoothing=0.9):
                f.write(self.row_to_bytes(meta, data[i]))

        light_data_path = path.with_name(path.stem + '_light_data.json')
        cleaned_light_data_path = path.with_name(path.stem + '_cleaned_light_data.json')
        if os.path.exists(light_data_path):
            shutil.copyfile(light_data_path, cleaned_light_data_path)
        print(f"Finished cleaning {path}")

    def clean(self):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(self.clean_file, self.data_paths)

    def copy_light_data(self):
        for path in self.data_paths:
            light_data_path = path.with_name(path.stem + '_light_data.json')
            cleaned_light_data_path = path.with_name(path.stem + '_cleaned_light_data.json')
            if os.path.exists(light_data_path):
                shutil.copyfile(light_data_path, cleaned_light_data_path)


if __name__ == "__main__":
    dataset_cleaner = DatasetCleaner(root="./test_sample", skip_cleaned=True)
    dataset_cleaner.clean()
    # dataset_cleaner.copy_light_data()
