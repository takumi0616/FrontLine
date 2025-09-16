import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
from collections import deque
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
from main_v3_config import ORIG_H, ORIG_W, print_memory_usage


class FrontalDatasetStage1(Dataset):
    """
    Stage1: GSM(93ch = 31vars*3times) -> 6-class segmentation target from front GT (5ch -> class map)
    """
    def __init__(self, months, nc_gsm_dir, nc_0p5_dir, cache_size=50):
        self.months = months
        self.nc_gsm_dir = nc_gsm_dir
        self.nc_0p5_dir = nc_0p5_dir
        self.data_index = []
        self.lat = None
        self.lon = None
        self.cache = {}
        self.cache_size = cache_size
        self.file_cache = {}
        self.file_cache_size = 10
        self.prepare_index()

    def prepare_index(self):
        print_memory_usage("Before Stage1 prepare_index")
        for month in self.months:
            gsm_file = os.path.join(self.nc_gsm_dir, f"gsm{month}.nc")
            front_file = os.path.join(self.nc_0p5_dir, f"{month}.nc")

            if not os.path.exists(gsm_file):
                print(f"{gsm_file} が見つかりません。スキップします。")
                continue
            if not os.path.exists(front_file):
                print(f"{front_file} が見つかりません。スキップします。")
                continue

            ds_gsm = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)

            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]

            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)

            for t in times:
                t_now = pd.to_datetime(t)
                t_prev = t_now - pd.Timedelta(hours=6)
                t_next = t_now + pd.Timedelta(hours=6)
                t_prev_np = np.datetime64(t_prev)
                t_next_np = np.datetime64(t_next)
                if t_prev_np not in ds_gsm["time"] or t_next_np not in ds_gsm["time"]:
                    continue

                self.data_index.append(
                    {
                        "gsm_file": gsm_file,
                        "front_file": front_file,
                        "t_now": t_now,
                        "t_prev_np": t_prev_np,
                        "t_next_np": t_next_np,
                        "t": t,
                    }
                )

            ds_gsm.close()
            ds_front.close()
            del ds_gsm, ds_front
            gc.collect()

        print_memory_usage("After Stage1 prepare_index")
        print(f"Stage1データインデックス構築完了：{len(self.data_index)}サンプル")
        gc.collect()

    def load_single_item(self, idx):
        item = self.data_index[idx]
        gsm_file = item["gsm_file"]
        front_file = item["front_file"]
        t = item["t"]
        t_prev_np = item["t_prev_np"]
        t_next_np = item["t_next_np"]
        t_now = item["t_now"]

        if gsm_file in self.file_cache:
            ds_gsm = self.file_cache[gsm_file]
        else:
            ds_gsm = xr.open_dataset(gsm_file)
            if len(self.file_cache) >= self.file_cache_size:
                oldest_key = next(iter(self.file_cache.keys()))
                self.file_cache[oldest_key].close()
                del self.file_cache[oldest_key]
            self.file_cache[gsm_file] = ds_gsm

        if front_file in self.file_cache:
            ds_front = self.file_cache[front_file]
        else:
            ds_front = xr.open_dataset(front_file)
            if len(self.file_cache) >= self.file_cache_size:
                oldest_key = next(iter(self.file_cache.keys()))
                self.file_cache[oldest_key].close()
                del self.file_cache[oldest_key]
            self.file_cache[front_file] = ds_front

        gsm_data_prev = ds_gsm.sel(time=t_prev_np).to_array().load().values
        gsm_data_now = ds_gsm.sel(time=t).to_array().load().values
        gsm_data_next = ds_gsm.sel(time=t_next_np).to_array().load().values
        front_data = ds_front.sel(time=t).to_array().load().values
        gsm_data = np.concatenate([gsm_data_prev, gsm_data_now, gsm_data_next], axis=0)

        return gsm_data.astype(np.float32), front_data.astype(np.float32), t_now

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        if idx in self.cache:
            gsm_data, front_data, time_dt = self.cache[idx]
        else:
            gsm_data, front_data, time_dt = self.load_single_item(idx)
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache.keys()))
                del self.cache[oldest_key]
            self.cache[idx] = (gsm_data, front_data, time_dt)

        time_str = str(time_dt)
        gsm_tensor = torch.from_numpy(gsm_data)
        target_cls = torch.zeros(ORIG_H, ORIG_W, dtype=torch.long)
        for c in range(5):
            mask = front_data[c, :, :] == 1
            target_cls[mask] = c + 1

        return gsm_tensor, target_cls, time_str


class FrontalRefinementDataset(Dataset):
    """
    Stage2: refinement dataset
    - train/val mode: augment GT class map to create degraded input and target class map
    - test mode: read Stage1 probabilities and convert to class map as input; target is zeros
    """
    def __init__(
        self,
        months,
        nc_0p5_dir,
        mode="train",
        stage1_out_dir=None,
        n_augment=8,
        prob_dilation=0.8,
        prob_create_gaps=0.8,
        prob_random_pixel_change=0.8,
        prob_add_fake_front=0.8,
        dilation_kernel_range=(2, 3),
        num_gaps_range=(2, 4),
        gap_size_range=(3, 5),
        num_pix_to_change_range=(20, 100),
        num_fake_front_range=(2, 10),
        cache_size=50,
    ):
        import random  # local to avoid leaking global RNG for workers

        self.months = months
        self.nc_0p5_dir = nc_0p5_dir
        self.mode = mode
        self.stage1_out_dir = stage1_out_dir
        self.n_augment = n_augment
        self.data_index = []
        self.lat = None
        self.lon = None
        self.cache = {}
        self.cache_size = cache_size

        self.prob_dilation = prob_dilation
        self.prob_create_gaps = prob_create_gaps
        self.prob_random_pixel_change = prob_random_pixel_change
        self.prob_add_fake_front = prob_add_fake_front

        self.dilation_kernel_range = dilation_kernel_range
        self.num_gaps_range = num_gaps_range
        self.gap_size_range = gap_size_range
        self.num_pix_to_change_range = num_pix_to_change_range
        self.num_fake_front_range = num_fake_front_range

        if self.mode in ["train", "val"]:
            self.prepare_index_trainval()
        else:
            self.prepare_index_test()

    def prepare_index_trainval(self):
        print_memory_usage("Before Stage2 prepare_index_trainval")
        for month in self.months:
            front_file = os.path.join(self.nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(front_file):
                print(f"{front_file} が見つかりません。スキップします。(Stage2)")
                continue

            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_front["lat"].values[:ORIG_H]
                self.lon = ds_front["lon"].values[:ORIG_W]

            times = ds_front["time"].values
            for t in times:
                time_dt = pd.to_datetime(t)
                for aug_idx in range(self.n_augment):
                    self.data_index.append(
                        {
                            "front_file": front_file,
                            "t": t,
                            "time_dt": time_dt,
                            "aug_idx": aug_idx,
                            "is_augmented": True,
                        }
                    )

            ds_front.close()
            del ds_front
            gc.collect()

        print_memory_usage("After Stage2 prepare_index_trainval")
        print(f"Stage2データインデックス構築完了（train/val）：{len(self.data_index)}サンプル")
        gc.collect()

    def prepare_index_test(self):
        print_memory_usage("Before Stage2 prepare_index_test")
        if not os.path.exists(self.stage1_out_dir):
            print(f"Stage1 出力ディレクトリ {self.stage1_out_dir} がありません。(Stage2)")
            return

        files = sorted([f for f in os.listdir(self.stage1_out_dir) if f.endswith(".nc")])
        for f in files:
            nc_path = os.path.join(self.stage1_out_dir, f)
            ds = xr.open_dataset(nc_path)

            if self.lat is None or self.lon is None:
                self.lat = ds["lat"].values
                self.lon = ds["lon"].values

            time_val = ds["time"].values[0] if "time" in ds else None
            time_dt = pd.to_datetime(time_val) if time_val is not None else None

            self.data_index.append(
                {
                    "nc_path": nc_path,
                    "time_dt": time_dt,
                    "is_augmented": False,
                }
            )

            ds.close()
            del ds
            gc.collect()

        print_memory_usage("After Stage2 prepare_index_test")
        print(f"Stage2データインデックス構築完了（test）：{len(self.data_index)}サンプル")
        gc.collect()

    def load_single_item(self, idx):
        import numpy as np
        import cv2
        import random

        item = self.data_index[idx]
        if item.get("is_augmented", False) and "front_file" in item:
            front_file = item["front_file"]
            t = item["t"]
            time_dt = item["time_dt"]
            aug_idx = item["aug_idx"]
            ds_front = xr.open_dataset(front_file)
            front_data = ds_front.sel(time=t).to_array().values
            target_cls = np.zeros((ORIG_H, ORIG_W), dtype=np.int64)
            for c in range(5):
                mask = front_data[c, :, :] == 1
                target_cls[mask] = c + 1

            np.random.seed((hash(str(time_dt)) + aug_idx) & 0x7FFFFFFF)
            degraded = self.degrade_front_data(target_cls)

            ds_front.close()

            return degraded.astype(np.int64), target_cls.astype(np.int64), time_dt

        elif "nc_path" in item:
            nc_path = item["nc_path"]
            time_dt = item["time_dt"]
            ds = xr.open_dataset(nc_path)
            probs_np = ds["probabilities"].isel(time=0).values
            degraded_cls = np.argmax(probs_np, axis=-1)
            target_cls = np.zeros_like(degraded_cls, dtype=np.int64)

            ds.close()

            return degraded_cls, target_cls, time_dt

        else:
            raise ValueError(f"Invalid data index item: {item}")

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        import torch

        if idx in self.cache:
            in_cls, tgt_cls, time_dt = self.cache[idx]
        else:
            in_cls, tgt_cls, time_dt = self.load_single_item(idx)
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache.keys()))
                del self.cache[oldest_key]
            self.cache[idx] = (in_cls, tgt_cls, time_dt)

        time_str = str(time_dt) if time_dt is not None else "None"
        x_tensor = torch.from_numpy(in_cls).float().unsqueeze(0)
        y_tensor = torch.from_numpy(tgt_cls).long()
        return x_tensor, y_tensor, time_str

    def create_irregular_shape(self, h, w, cy, cx, max_shape_size=16):
        import random
        visited = set()
        visited.add((cy, cx))
        queue = deque()
        queue.append((cy, cx))

        target_size = random.randint(1, max_shape_size)
        shape_points = []

        while queue and len(shape_points) < target_size:
            y, x = queue.popleft()
            shape_points.append((y, x))

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    if np.random.rand() < 0.8:
                        queue.append((ny, nx))
        return shape_points

    def degrade_front_data(self, cls_map):
        import random
        import cv2
        degraded = cls_map.copy()
        for c in [1, 2, 3, 4, 5]:
            mask = (degraded == c).astype(np.uint8)
            if np.random.rand() < self.prob_dilation:
                ksize = np.random.randint(self.dilation_kernel_range[0], self.dilation_kernel_range[1] + 1)
                kernel = np.ones((ksize, ksize), np.uint8)
                mask_dil = cv2.dilate(mask, kernel, iterations=1)
                degraded[mask_dil == 1] = c

            if np.random.rand() < self.prob_create_gaps:
                num_gaps = np.random.randint(self.num_gaps_range[0], self.num_gaps_range[1] + 1)
                indices = np.transpose(np.nonzero(mask))
                if len(indices) > 0:
                    for _ in range(num_gaps):
                        idx = indices[np.random.randint(0, len(indices))]
                        yy, xx = idx
                        shape_points = self.create_irregular_shape(
                            degraded.shape[0], degraded.shape[1], yy, xx, max_shape_size=np.random.randint(3, 6)
                        )
                        for (sy, sx) in shape_points:
                            if degraded[sy, sx] == c:
                                degraded[sy, sx] = 0
        h, w = degraded.shape
        if np.random.rand() < self.prob_random_pixel_change:
            num_pix_to_change = np.random.randint(self.num_pix_to_change_range[0], self.num_pix_to_change_range[1] + 1)
            num_shapes = np.random.randint(1, 6)
            for _ in range(num_shapes):
                cy = np.random.randint(0, h)
                cx = np.random.randint(0, w)
                old_c = degraded[cy, cx]
                if old_c in [1, 2, 3, 4, 5]:
                    shape_points = self.create_irregular_shape(h, w, cy, cx, 25)
                    possible_classes = [0, 1, 2, 3, 4, 5]
                    possible_classes.remove(old_c)
                    new_c = possible_classes[np.random.randint(0, len(possible_classes))]

                    count_changed = 0
                    for (sy, sx) in shape_points:
                        if degraded[sy, sx] == old_c:
                            degraded[sy, sx] = new_c
                            count_changed += 1
                            if count_changed >= num_pix_to_change:
                                break
        if np.random.rand() < self.prob_add_fake_front:
            num_fake_front = np.random.randint(self.num_fake_front_range[0], self.num_fake_front_range[1] + 1)
            for _ in range(num_fake_front):
                cy = np.random.randint(0, h)
                cx = np.random.randint(0, w)
                if degraded[cy, cx] == 0:
                    shape_points = self.create_irregular_shape(h, w, cy, cx, 25)
                    new_c = [1, 2, 3, 4, 5][np.random.randint(0, 5)]
                    for (sy, sx) in shape_points:
                        if degraded[sy, sx] == 0:
                            degraded[sy, sx] = new_c

        return degraded


class Stage2DiffusionTrainDataset(Dataset):
    """
    前線GT (nc_0p5_dirの 5ch: warm,cold,stationary,occluded,warm_cold) を
    6ch(one-hot: none, warm, cold, stationary, occluded, warm_cold) 確率として返す
    """
    def __init__(self, months, nc_0p5_dir):
        self.months = months
        self.nc_0p5_dir = nc_0p5_dir
        self.index = []
        self.lat = None
        self.lon = None
        for month in self.months:
            f = os.path.join(self.nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(f):
                print(f"[Stage2-DiffTrain] GT not found: {f}")
                continue
            ds = xr.open_dataset(f)
            if self.lat is None or self.lon is None:
                self.lat = ds["lat"].values[:ORIG_H]
                self.lon = ds["lon"].values[:ORIG_W]
            for t in ds["time"].values:
                self.index.append((f, t))
            ds.close()
        print(f"[Stage2-DiffTrain] samples: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, t = self.index[idx]
        ds = xr.open_dataset(f)
        arr5 = ds.sel(time=t).to_array().values  # (5,H,W) 0/1
        ds.close()
        h, w = arr5.shape[1], arr5.shape[2]
        any_front = (arr5.sum(axis=0) > 0).astype(np.float32)
        none_ch = (1.0 - any_front).astype(np.float32)
        # channel order: 0:none, 1:warm,2:cold,3:stationary,4:occluded,5:warm_cold
        prob6 = np.zeros((6, h, w), dtype=np.float32)
        prob6[0] = none_ch
        for c in range(5):
            prob6[c + 1] = arr5[c].astype(np.float32)
        # normalize
        s = prob6.sum(axis=0, keepdims=True) + 1e-8
        prob6 = prob6 / s
        x = torch.from_numpy(prob6)  # (6,H,W)
        return x, str(pd.to_datetime(t))


class Stage2DiffusionTestDataset(Dataset):
    """
    Stage1のprobファイル(nc)を読み込み、(B,6,H,W)の確率を返す
    """
    def __init__(self, stage1_out_dir):
        self.files = sorted(
            [os.path.join(stage1_out_dir, f) for f in os.listdir(stage1_out_dir) if f.endswith(".nc")]
        )
        self.lat = None
        self.lon = None
        if len(self.files) > 0:
            ds0 = xr.open_dataset(self.files[0])
            self.lat = ds0["lat"].values
            self.lon = ds0["lon"].values
            ds0.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        ds = xr.open_dataset(f)
        probs = ds["probabilities"].isel(time=0).values  # (H,W,C=6)
        tval = ds["time"].values[0] if "time" in ds else None
        ds.close()
        # (6,H,W)
        probs = np.transpose(probs, (2, 0, 1)).astype(np.float32)
        # normalize for safety
        s = probs.sum(axis=0, keepdims=True) + 1e-8
        probs = probs / s
        return torch.from_numpy(probs), os.path.basename(f).replace("prob_", "").replace(".nc", "")


__all__ = [
    "FrontalDatasetStage1",
    "FrontalRefinementDataset",
    "Stage2DiffusionTrainDataset",
    "Stage2DiffusionTestDataset",
]
