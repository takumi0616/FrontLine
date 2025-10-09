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
    概要:
        Stage1（Swin-UNet）の学習/評価用データセット。
        GSM 物理量の 3時刻スタック（93ch = 31変数 × 3時刻）を入力とし、
        GT 前線（5chバイナリ: warm/cold/stationary/occluded/warm_cold）から
        6クラス（none=0, warm=1, cold=2, stationary=3, occluded=4, warm_cold=5）の
        クラスマップ（H×W）をターゲットとして返す。

    入力（__init__ 引数）:
        - months (Iterable[int|str]): 対象の年月キー（例: 201401, 201402, ...）
        - nc_gsm_dir (str): GSM の NetCDF（31変数×時刻）ディレクトリ
        - nc_0p5_dir (str): 前線 GT の NetCDF（5chバイナリ×時刻）ディレクトリ
        - cache_size (int): サンプルキャッシュの保持上限（LRU 的に古いものから破棄）
        - file_cache_size (int): 開いた NetCDF ファイルハンドルの保持上限（LRU 的に破棄）

    処理:
        - prepare_index() で、3時刻 (t-6h, t, t+6h) がそろう時刻 t を列挙して data_index を構築
        - __getitem__() で GSM(93ch) とターゲットクラスマップ（6クラス）と時刻文字列を返す

    出力（__getitem__ 戻り値）:
        - gsm_tensor (Tensor): 形状 (93, ORIG_H, ORIG_W) の float32 テンソル
        - target_cls (Tensor): 形状 (ORIG_H, ORIG_W) の long テンソル（値域 0..5）
        - time_str (str): サンプルの時刻文字列
    """
    def __init__(self, months, nc_gsm_dir, nc_0p5_dir, cache_size=50, file_cache_size=10):
        self.months = months
        self.nc_gsm_dir = nc_gsm_dir
        self.nc_0p5_dir = nc_0p5_dir
        self.data_index = []  # 各サンプルのメタ情報リスト
        self.lat = None
        self.lon = None
        self.cache = {}  # サンプルキャッシュ: idx -> (gsm, front, time_dt)
        self.cache_size = cache_size
        self.file_cache = {}  # ファイルキャッシュ: path -> xarray.Dataset
        self.file_cache_size = file_cache_size
        self.prepare_index()

    def prepare_index(self):
        """
        概要:
            データのインデックス（どのファイルのどの時刻を1サンプルにするか）を構築する。

        入力:
            なし（インスタンス変数 months, nc_gsm_dir, nc_0p5_dir を参照）

        処理:
            - 各 month について GSM ファイルと前線ファイルの存在をチェック
            - 共通に存在する時刻 t を抽出し、さらに t-6h と t+6h が GSM にあるもののみ採用
            - 採用した時刻に対して data_index にメタ情報を積む
            - lat/lon は最初に見つけたファイルから ORIG_H/ORIG_W の範囲で確定

        出力:
            なし（self.data_index を更新）
        """
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

            # lat/lon の初期化（固定のグリッド定義を使用）
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]

            # 共通時刻をソートして取得
            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)

            # 3時刻がそろう時だけサンプル化
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
        """
        概要:
            data_index[idx] が指す1サンプル分の GSM 3時刻スタックと GT 前線 5ch を読み出す。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - file_cache を用いて対象の NetCDF を開く（上限超過時は古いものから close して削除）
            - GSM: t-6h, t, t+6h をそれぞれ (31, H, W) として読み、チャネル方向に連結して (93, H, W)
            - 前線: t の (5, H, W) を読み出す

        出力:
            - gsm_data (np.ndarray): (93, ORIG_H, ORIG_W), float32
            - front_data (np.ndarray): (5, ORIG_H, ORIG_W), float32
            - t_now (pd.Timestamp): 現在時刻
        """
        item = self.data_index[idx]
        gsm_file = item["gsm_file"]
        front_file = item["front_file"]
        t = item["t"]
        t_prev_np = item["t_prev_np"]
        t_next_np = item["t_next_np"]
        t_now = item["t_now"]

        # GSM ファイルハンドル（キャッシュ）
        if gsm_file in self.file_cache:
            ds_gsm = self.file_cache[gsm_file]
        else:
            ds_gsm = xr.open_dataset(gsm_file)
            if len(self.file_cache) >= self.file_cache_size:
                oldest_key = next(iter(self.file_cache.keys()))
                self.file_cache[oldest_key].close()
                del self.file_cache[oldest_key]
            self.file_cache[gsm_file] = ds_gsm

        # 前線ファイルハンドル（キャッシュ）
        if front_file in self.file_cache:
            ds_front = self.file_cache[front_file]
        else:
            ds_front = xr.open_dataset(front_file)
            if len(self.file_cache) >= self.file_cache_size:
                oldest_key = next(iter(self.file_cache.keys()))
                self.file_cache[oldest_key].close()
                del self.file_cache[oldest_key]
            self.file_cache[front_file] = ds_front

        # 3時刻分の GSM を取得しチャネル結合: (31, H, W) × 3 -> (93, H, W)
        gsm_data_prev = ds_gsm.sel(time=t_prev_np).to_array().load().values
        gsm_data_now = ds_gsm.sel(time=t).to_array().load().values
        gsm_data_next = ds_gsm.sel(time=t_next_np).to_array().load().values
        front_data = ds_front.sel(time=t).to_array().load().values  # (5, H, W)
        gsm_data = np.concatenate([gsm_data_prev, gsm_data_now, gsm_data_next], axis=0)

        return gsm_data.astype(np.float32), front_data.astype(np.float32), t_now

    def __len__(self):
        """
        概要:
            サンプル数（インデックス長）を返す。
        出力:
            - n (int): サンプル総数
        """
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        概要:
            インデックス idx のサンプルをテンソル形式にして返す。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - サンプルキャッシュにあれば再利用、なければ load_single_item() でロードし、LRU で格納
            - 前線 5ch バイナリ -> 6クラスのクラスマップ（0: none, 1..5: 前線）を生成
            - GSM は torch.Tensor(float32)、ターゲットは torch.LongTensor に変換

        出力:
            - gsm_tensor (Tensor): (93, ORIG_H, ORIG_W)
            - target_cls (Tensor): (ORIG_H, ORIG_W), 値域 0..5
            - time_str (str): 時刻文字列
        """
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
            target_cls[mask] = c + 1  # 1..5 を前線クラスに対応づけ

        return gsm_tensor, target_cls, time_str


class FrontalRefinementDataset(Dataset):
    """
    概要:
        Stage2（前線補正）用データセット。
        - train/val: GT クラスマップから擬似劣化（dilation, gaps, random change, fake front）を生成し、
                     (入力=劣化クラスマップ, 目標=GTクラスマップ) のペアを返す。
        - test: Stage1 出力の確率 nc を読み、argmax によるクラスマップを入力として返す（目標はゼロ配列）。

    入力（__init__ 引数）:
        - months (Iterable[int|str]): train/val で使う年月キー（test では未使用）
        - nc_0p5_dir (str): GT 前線 nc ディレクトリ（train/val で使用）
        - mode (str): "train" | "val" | "test"
        - stage1_out_dir (str|None): test モードで読む Stage1 の .nc 出力フォルダ
        - n_augment (int): 1つの時刻につき何通りの劣化サンプルを作るか
        - prob_* 系, *_range 系: 劣化操作の確率やパラメータ範囲
        - cache_size (int): サンプルキャッシュ上限

    出力（__getitem__ 戻り値）:
        - x_tensor (Tensor): (1, H, W) の float32（入力クラスマップを 1ch で表現）
        - y_tensor (Tensor): (H, W) の long（目標クラス。test では全0）
        - time_str (str): 時刻文字列（test ではファイル名由来、存在しなければ "None"）
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
        import random  # DataLoader ワーカ間での RNG 干渉を避けるためローカル import

        self.months = months
        self.nc_0p5_dir = nc_0p5_dir
        self.mode = mode
        self.stage1_out_dir = stage1_out_dir
        self.n_augment = n_augment
        self.data_index = []  # サンプルインデックス
        self.lat = None
        self.lon = None
        self.cache = {}
        self.cache_size = cache_size

        # 劣化操作の確率
        self.prob_dilation = prob_dilation
        self.prob_create_gaps = prob_create_gaps
        self.prob_random_pixel_change = prob_random_pixel_change
        self.prob_add_fake_front = prob_add_fake_front

        # 劣化操作のパラメータ
        self.dilation_kernel_range = dilation_kernel_range
        self.num_gaps_range = num_gaps_range
        self.gap_size_range = gap_size_range
        self.num_pix_to_change_range = num_pix_to_change_range
        self.num_fake_front_range = num_fake_front_range

        # インデックス構築
        if self.mode in ["train", "val"]:
            self.prepare_index_trainval()
        else:
            self.prepare_index_test()

    def prepare_index_trainval(self):
        """
        概要:
            train/val 用に、各時刻に対して n_augment 個の劣化サンプルを作るためのインデックスを構築。

        入力:
            なし（self.months, self.nc_0p5_dir, self.n_augment を参照）

        処理:
            - 各 month の前線 GT nc を開き、全時刻 t を列挙
            - 各 t について aug_idx=0..n_augment-1 の項目を data_index に追加
            - lat/lon を初回ファイルから取得
        """
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
        """
        概要:
            test 用に、Stage1 の確率出力 nc ファイル群を列挙して data_index を構築。

        入力:
            なし（self.stage1_out_dir を参照）

        処理:
            - stage1_out_dir の .nc を列挙
            - 各ファイルについて lat/lon（先頭のみ）と time を取り出して data_index に積む
        """
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
        """
        概要:
            data_index[idx] をもとに 1 サンプルを作成して返す（劣化または Stage1 出力の読込み）。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - train/val: 前線 GT の 5ch を読んでクラスマップにし、degrade_front_data() で劣化版を生成
            - test: Stage1 の probabilities (H, W, C=6) を読んで argmax のクラスマップを入力とする

        出力:
            - degraded_cls (np.ndarray): 入力クラスマップ (H, W), int64
            - target_cls (np.ndarray): 目標クラスマップ (H, W), int64（test では全0）
            - time_dt (pd.Timestamp|None): 時刻
        """
        import numpy as np
        import cv2
        import random

        item = self.data_index[idx]
        if item.get("is_augmented", False) and "front_file" in item:
            # train/val サンプル
            front_file = item["front_file"]
            t = item["t"]
            time_dt = item["time_dt"]
            aug_idx = item["aug_idx"]
            ds_front = xr.open_dataset(front_file)
            front_data = ds_front.sel(time=t).to_array().values  # (5, H, W) 0/1
            target_cls = np.zeros((ORIG_H, ORIG_W), dtype=np.int64)
            for c in range(5):
                mask = front_data[c, :, :] == 1
                target_cls[mask] = c + 1  # 1..5 にエンコード

            # 擬似乱数シードを時刻×拡張番号から決めて擬似再現性を担保
            np.random.seed((hash(str(time_dt)) + aug_idx) & 0x7FFFFFFF)
            degraded = self.degrade_front_data(target_cls)

            ds_front.close()

            return degraded.astype(np.int64), target_cls.astype(np.int64), time_dt

        elif "nc_path" in item:
            # test サンプル（Stage1 出力）
            nc_path = item["nc_path"]
            time_dt = item["time_dt"]
            ds = xr.open_dataset(nc_path)
            probs_np = ds["probabilities"].isel(time=0).values  # (H, W, C=6)
            degraded_cls = np.argmax(probs_np, axis=-1)  # (H, W)
            target_cls = np.zeros_like(degraded_cls, dtype=np.int64)  # ダミー目標（未使用）

            ds.close()

            return degraded_cls, target_cls, time_dt

        else:
            raise ValueError(f"Invalid data index item: {item}")

    def __len__(self):
        """
        概要:
            サンプル数（インデックス長）を返す。
        出力:
            - n (int): サンプル総数
        """
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        概要:
            インデックス idx のサンプル（入力1ch、目標クラス）をテンソルで返す。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - キャッシュ参照→なければ load_single_item()
            - x は (1, H, W) float32、y は (H, W) long に整形して返す

        出力:
            - x_tensor (Tensor): (1, H, W) float32
            - y_tensor (Tensor): (H, W) long
            - time_str (str): 時刻文字列（None の場合は "None"）
        """
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
        # 入力は 1ch に詰めて float 化
        x_tensor = torch.from_numpy(in_cls).float().unsqueeze(0)
        y_tensor = torch.from_numpy(tgt_cls).long()
        return x_tensor, y_tensor, time_str

    def create_irregular_shape(self, h, w, cy, cx, max_shape_size=16):
        """
        概要:
            起点 (cy, cx) から上下左右にランダムに広がる不規則な塊（領域）を生成する。

        入力:
            - h, w (int): 画像の高さ・幅
            - cy, cx (int): 生成開始位置（中心）
            - max_shape_size (int): 最大ピクセル数

        処理:
            - BFS 的に隣接4近傍へ 0.8 の確率で増殖しながら shape_points を収集

        出力:
            - shape_points (List[Tuple[int,int]]): 領域を構成する画素座標のリスト
        """
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
        """
        概要:
            クラスマップ cls_map（0..5）に様々な劣化操作を適用して擬似的に品質を下げた入力を生成する。

        入力:
            - cls_map (np.ndarray): (H, W) int64、0..5 のクラスラベル

        処理:
            - 膨張: クラス領域を膨らませる（確率 self.prob_dilation）
            - ギャップ生成: 領域の一部を 0（none）で塗りつぶし欠損を作る（確率 self.prob_create_gaps）
            - ランダム画素置換: ランダムに領域内の画素を別クラスへ置換（確率 self.prob_random_pixel_change）
            - 偽前線追加: 背景から新しい小領域を別クラスで作る（確率 self.prob_add_fake_front）

        出力:
            - degraded (np.ndarray): 劣化後のクラスマップ (H, W) int64
        """
        import random
        import cv2
        degraded = cls_map.copy()
        # 各前線クラスごとに処理
        for c in [1, 2, 3, 4, 5]:
            mask = (degraded == c).astype(np.uint8)
            # 膨張
            if np.random.rand() < self.prob_dilation:
                ksize = np.random.randint(self.dilation_kernel_range[0], self.dilation_kernel_range[1] + 1)
                kernel = np.ones((ksize, ksize), np.uint8)
                mask_dil = cv2.dilate(mask, kernel, iterations=1)
                degraded[mask_dil == 1] = c

            # ギャップ生成
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
                                degraded[sy, sx] = 0  # none に落とす
        h, w = degraded.shape

        # ランダム画素置換（不規則領域単位で）
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

        # 偽前線追加（背景から）
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
    概要:
        拡散モデル（Stage2 DiffusionCorrector）の学習用データセット。
        前線 GT の 5ch バイナリ（warm, cold, stationary, occluded, warm_cold）から
        6ch の確率マップ（none + 5 前線）を生成して返す。

    入力（__init__ 引数）:
        - months (Iterable[int|str]): 使用する年月キー
        - nc_0p5_dir (str): 前線 GT nc ディレクトリ

    出力（__getitem__ 戻り値）:
        - x (Tensor): (6, H, W) float32、各画素でチャネル和=1 になるよう正規化済み
        - time_str (str): 時刻文字列
    """
    def __init__(self, months, nc_0p5_dir):
        self.months = months
        self.nc_0p5_dir = nc_0p5_dir
        self.index = []  # (file, time) のタプル配列
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
        """
        概要:
            サンプル数を返す。
        出力:
            - n (int): サンプル総数
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        概要:
            idx 番目の (file, time) から 6ch 確率マップを生成して返す。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - 5ch バイナリ (5, H, W) を読み、none チャネル = 1 - any(front) を計算
            - 6ch を stack してチャネル和=1 になるよう正規化
            - torch.Tensor に変換

        出力:
            - x (Tensor): (6, H, W) float32
            - time_str (str): 時刻文字列
        """
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
        # normalize（数値安定のため微小値を加算）
        s = prob6.sum(axis=0, keepdims=True) + 1e-8
        prob6 = prob6 / s
        x = torch.from_numpy(prob6)  # (6,H,W)
        return x, str(pd.to_datetime(t))


class Stage2DiffusionTestDataset(Dataset):
    """
    概要:
        拡散モデルの推論用（test）データセット。
        Stage1 の .nc ファイルから (H, W, C=6) の確率を読み、(6, H, W) テンソルで返す。

    入力（__init__ 引数）:
        - stage1_out_dir (str): Stage1 の確率出力 .nc フォルダ

    出力（__getitem__ 戻り値）:
        - probs (Tensor): (6, H, W) float32、チャネル和=1 にスケール済み
        - name (str): ファイル名起源の識別子（"prob_*.nc" -> "*"）
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
        """
        概要:
            ファイル数を返す。
        出力:
            - n (int): サンプル総数
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        概要:
            idx 番目の Stage1 .nc から確率を読み、(6, H, W) に整形して返す。

        入力:
            - idx (int): サンプルインデックス

        処理:
            - probabilities[time=0] を (H, W, C) として取得
            - 転置で (C, H, W) に並べ替え
            - 数値安定のためチャネル和で割って正規化

        出力:
            - probs (Tensor): (6, H, W) float32
            - name (str): ファイル名から生成した識別子
        """
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
