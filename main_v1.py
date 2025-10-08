import torch
import os
import gc
import psutil  
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  
import pandas as pd
from datetime import datetime
import multiprocessing
import cv2 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from collections import deque 
from skimage.morphology import skeletonize 
from matplotlib import gridspec
import re
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import torch.backends.cudnn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
import math
import pandas as pd
from skan import Skeleton 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import matplotlib.pyplot as plt
import glob
import subprocess
import shutil
import tempfile
from collections import defaultdict
import xml.etree.ElementTree as ET
import shap 
from typing import Optional
import os, gc, random, glob, math, psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from typing import Optional
import shap                            
import torch
import torch.nn as nn

# Swin-UNet core implementation is moved out to keep this file lean
# Ensure this file's directory is on sys.path so local module import works when run from repo root
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
from swin_unet import SwinTransformerSys

# ======================================
# Global configuration (centralized params)
# ======================================
CFG = {
    "SEED": 0,
    "THREADS": 4,
    "PATHS": {
        "nc_gsm_dir": "./128_128/nc_gsm9",
        "nc_0p5_dir": "./128_128/nc_0p5_bulge_v2",
        "stage1_out_dir": "./v1_result/stage1_nc",
        "stage2_out_dir": "./v1_result/stage2_nc",
        "stage3_out_dir": "./v1_result/stage3_nc",
        "model_s1_save_dir": "./v1_result/stage1_model",
        "model_s2_save_dir": "./v1_result/stage2_model",
        "output_visual_dir": "./v1_result/visualizations",
        "stage4_svg_dir": "./v1_result/stage4_svg",
    },
    "IMAGE": {
        "ORIG_H": 128,
        "ORIG_W": 128,
    },
    "STAGE1": {
        "num_classes": 6,
        "in_chans": 93,
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "test_months":  (2023, 1, 2023, 12),
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05
        },
        "model": {
            "img_size": 128,
            "patch_size": 2,
            "embed_dim": 192,
            "depths": [2, 2, 2, 2],
            "depths_decoder": [1, 2, 2, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 16,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "norm_layer": nn.LayerNorm,
            "ape": True,
            "patch_norm": True,
            "use_checkpoint": False,
            "final_upsample": "expand_first"
        }
    },
    "STAGE2": {
        "num_classes": 6,
        "in_chans": 1,
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_val": 1,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05
        },
        "model": {
            "img_size": 128,
            "patch_size": 2,
            "embed_dim": 96,
            "depths": [2, 2, 2, 2],
            "depths_decoder": [1, 2, 2, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 16,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "norm_layer": nn.LayerNorm,
            "ape": False,
            "patch_norm": True,
            "use_checkpoint": False,
            "final_upsample": "expand_first"
        },
        "augment": {
            "n_augment": 10,
            "prob_dilation": 0.8,
            "prob_create_gaps": 0.8,
            "prob_random_pixel_change": 0.8,
            "prob_add_fake_front": 0.8,
            "dilation_kernel_range": (2, 3),
            "num_gaps_range": (2, 4),
            "gap_size_range": (3, 5),
            "num_pix_to_change_range": (20, 100),
            "num_fake_front_range": (2, 10)
        }
    },
    "VISUALIZATION": {
        "class_colors": {
            0: "#FFFFFF",
            1: "#FF0000",
            2: "#0000FF",
            3: "#008015",
            4: "#800080",
            5: "#FFA500"
        },
        "pressure_vmin": -40,
        "pressure_vmax": 40,
        "pressure_levels": 21,
        "parallel_factor": 4
    },
    "VIDEO": {
        "image_folder": "./v1_result/visualizations/",
        "output_folder": "./v1_result/",
        "frame_rate": 4,
        "low_res_scale": 4,
        "low_res_frame_rate": 2
    },
    "SHAP": {
        "use_gpu": True,
        "max_samples_per_class": 500,
        "out_root": "./v1_result/shap_stage1",
        "free_mem_threshold_gb": 4.0
    }
}

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")

def create_comparison_videos(image_folder="./v1_result/visualizations/",
                             output_folder="./v1_result/",
                             frame_rate=4,
                             low_res_scale=4,  
                             low_res_frame_rate=2):  
    """
    概要:
        可視化で生成された比較画像（comparison_*.png）から、月別および通年の動画(mp4)を作成する。

    入力:
        - image_folder (str): 入力画像ディレクトリ（comparison_*.png が格納されていること）
        - output_folder (str): 出力動画の保存ディレクトリ
        - frame_rate (int|float): 通常解像度の動画のフレームレート
        - low_res_scale (int): 低解像度動画の縮小係数（width/height を 1/low_res_scale に縮小）
        - low_res_frame_rate (int|float): 低解像度動画のフレームレート

    処理:
        - ファイル名から年月(YYYYMM)を抽出してグルーピング
        - 月別に OpenCV の VideoWriter で mp4 を生成
        - 通年の結合動画を作成
        - さらに ffmpeg を用いて低解像度版（libx264 yuv420p）を生成

    出力:
        - output_folder に mp4 を保存（例: comparison_202301.mp4, comparison_2023_full_year.mp4, comparison_2023_full_year_low.mp4）
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(image_folder, 'comparison_*.png'))
    image_files.sort(key=lambda x: os.path.basename(x))
    monthly_images = defaultdict(list)

    for img_file in image_files:
        timestamp = os.path.basename(img_file).replace('comparison_', '').replace('.png', '')
        month_str = timestamp[:6]
        monthly_images[month_str].append(img_file)

    all_image_files = []
    for month in sorted(monthly_images.keys()):
        images = monthly_images[month]
        images.sort(key=lambda x: os.path.basename(x))
        all_image_files.extend(images)

        if images:
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape

            output_video = os.path.join(output_folder, f'comparison_{month}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

            for image in images:
                img = cv2.imread(image)
                video.write(img)

            video.release()
            print(f"[動画作成] {month} の動画を保存しました。")
        else:
            print(f"[動画作成] {month} の画像がありません。")

    if all_image_files:
        all_image_files.sort(key=lambda x: os.path.basename(x))
        frame = cv2.imread(all_image_files[0])
        height, width, layers = frame.shape

        output_video_all = os.path.join(output_folder, 'comparison_2023_full_year.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_all, fourcc, frame_rate, (width, height))
        
        for image in all_image_files:
            img = cv2.imread(image)
            video.write(img)

        video.release()
        print("[動画作成] 1月から12月までの統合動画を保存しました。")
        temp_dir = tempfile.mkdtemp()

        low_width = (width // low_res_scale) // 2 * 2
        low_height = (height // low_res_scale) // 2 * 2

        for idx, image in enumerate(all_image_files):
            img = cv2.imread(image)
            img_small = cv2.resize(img, (low_width, low_height))
            temp_image_path = os.path.join(temp_dir, f'frame_{idx:06d}.png')
            cv2.imwrite(temp_image_path, img_small)

        output_video_low = os.path.join(output_folder, 'comparison_2023_full_year_low.mp4')

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-r', str(low_res_frame_rate),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-vcodec', 'libx264',
            '-crf', '30',
            '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p',
            output_video_low
        ]

        subprocess.run(ffmpeg_cmd)
        shutil.rmtree(temp_dir)

        print("[動画作成] 1月から12月までの統合動画（低画質版）を保存しました。")
    else:
        print("[動画作成] 年間動画用の画像ファイルが見つかりませんでした。")

def print_memory_usage(msg=""):
    """
    概要:
        現在プロセスのRSS(実使用メモリ)をMB単位で出力する簡易ユーティリティ。

    入力:
        - msg (str): 付加メッセージ（どのタイミングかを示すラベル）

    処理:
        - psutil で自プロセスの RSS を取得し、MB単位に換算して print

    出力:
        - なし（標準出力へログ出力）
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024 
    print(f"[Memory] {msg} memory usage: {mem_info:.2f} MB")

seed = CFG["SEED"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'Random seed set as {seed}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
nc_gsm_dir = CFG["PATHS"]["nc_gsm_dir"]
nc_0p5_dir = CFG["PATHS"]["nc_0p5_dir"]
stage1_out_dir = CFG["PATHS"]["stage1_out_dir"]
stage2_out_dir = CFG["PATHS"]["stage2_out_dir"]
stage3_out_dir = CFG["PATHS"]["stage3_out_dir"]
model_s1_save_dir = CFG["PATHS"]["model_s1_save_dir"]
model_s2_save_dir = CFG["PATHS"]["model_s2_save_dir"]
output_visual_dir = CFG["PATHS"]["output_visual_dir"]

ORIG_H = CFG["IMAGE"]["ORIG_H"]
ORIG_W = CFG["IMAGE"]["ORIG_W"]

def get_available_months(start_year, start_month, end_year, end_month):
    """
    概要:
        開始年月から終了年月までを含む YYYYMM の文字列リストを生成する。

    入力:
        - start_year (int), start_month (int): 開始年・月
        - end_year (int), end_month (int): 終了年・月（含む）

    処理:
        - datetime を用いて1ヶ月ずつ進めながら YYYYMM 形式の文字列にして蓄積

    出力:
        - months (List[str]): 例 ["201401", ..., "202212"]
    """
    months = []
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    while current <= end:
        months.append(current.strftime('%Y%m'))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return months

class FrontalDatasetStage1(Dataset):
    """
    概要:
        Stage1（Swin-UNet）学習/評価用データセット。
        GSM の多変量データ（t-6h, t0, t+6h を結合）を入力テンソル(93ch)として返し、
        0/1の前線ラスタ（5クラス）をクラスID(0..5)の2次元マップに変換して教師として返す。

    入力:
        - months (List[str]): 対象年月のリスト（YYYYMM）
        - nc_gsm_dir (str): GSM NetCDF のディレクトリ
        - nc_0p5_dir (str): 前線ラスタ NetCDF のディレクトリ
        - cache_size (int): サンプルキャッシュの最大保持数

    処理:
        - prepare_index() で各月のファイル存在/時刻整合を確認し、データインデックスを構築
        - __getitem__ でインデックスから入力(93ch)と教師(クラスID HxW)を生成
        - メモリ節約のため、最近アクセス分をキャッシュして再利用

    出力:
        - __getitem__ -> (gsm_tensor: FloatTensor[93,H,W], target_cls: LongTensor[H,W], time_str: str)
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
        """
        概要:
            GSM/前線NetCDFを走査し、共通時刻かつ t±6h が存在するサンプルのみを data_index に登録する。

        入力:
            なし（コンストラクタで与えられたパス/設定を使用）

        処理:
            - 月ごとに gsm{YYYYMM}.nc と {YYYYMM}.nc を開き、共通 time を抽出
            - 各 time について t-6h, t+6h の両方があるときのみ採用し、必要メタ情報を data_index に保存

        出力:
            なし（内部状態 data_index, lat, lon を更新）
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

            ds_gsm   = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)

            if self.lat is None or self.lon is None:
                self.lat = ds_gsm['lat'].values[:ORIG_H]
                self.lon = ds_gsm['lon'].values[:ORIG_W]

            times = np.intersect1d(ds_gsm['time'], ds_front['time'])
            times = np.sort(times)

            for t in times:
                t_now = pd.to_datetime(t)
                t_prev = t_now - pd.Timedelta(hours=6)
                t_next = t_now + pd.Timedelta(hours=6)
                t_prev_np = np.datetime64(t_prev)
                t_next_np = np.datetime64(t_next)
                if t_prev_np not in ds_gsm['time'] or t_next_np not in ds_gsm['time']:
                    continue

                self.data_index.append({
                    'gsm_file': gsm_file,
                    'front_file': front_file,
                    't_now': t_now,
                    't_prev_np': t_prev_np,
                    't_next_np': t_next_np,
                    't': t
                })

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
            data_index[idx] で指定される1サンプルをディスクから読み出し、学習用の入力/教師を組み立てる。

        入力:
            - idx (int): サンプルのインデックス

        処理:
            - ファイルキャッシュ（self.file_cache）を用いて NetCDF を再利用
            - GSM(t-6h, t0, t+6h) を連結し 93ch の入力配列を作成
            - 前線ラスタ(5ch)をクラスID(1..5)へマッピングし、背景を0として target_cls を作成

        出力:
            - gsm_data (np.ndarray[float32]): (93,H,W)
            - front_data (np.ndarray[float32]): (5,H,W) 元の one-hot ラスタ（返却は内部のみ）
            - t_now (pd.Timestamp): サンプルの時刻
        """
        item = self.data_index[idx]
        gsm_file = item['gsm_file']
        front_file = item['front_file']
        t = item['t']
        t_prev_np = item['t_prev_np']
        t_next_np = item['t_next_np']
        t_now = item['t_now']

        if gsm_file in self.file_cache:
            ds_gsm = self.file_cache[gsm_file]
        else:
            ds_gsm   = xr.open_dataset(gsm_file)
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
        gsm_data_now  = ds_gsm.sel(time=t).to_array().load().values
        gsm_data_next = ds_gsm.sel(time=t_next_np).to_array().load().values
        front_data    = ds_front.sel(time=t).to_array().load().values
        gsm_data = np.concatenate([gsm_data_prev, gsm_data_now, gsm_data_next], axis=0)

        return gsm_data.astype(np.float32), front_data.astype(np.float32), t_now

    def __len__(self):
        """
        概要:
            登録済みサンプル数を返す。

        入力:
            なし

        処理:
            - data_index の長さを返却

        出力:
            - n (int)
        """
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        概要:
            PyTorch DataLoader から呼ばれ、idx 番目のテンソル/教師/時刻文字列を返す。

        入力:
            - idx (int): サンプル番号

        処理:
            - サンプルキャッシュ（self.cache）を優先的に利用
            - load_single_item で取得した (gsm, front, t) から
              gsm を Tensor[93,H,W] へ、front をクラスID Tensor[H,W] へ変換

        出力:
            - gsm_tensor (torch.FloatTensor): (93,H,W)
            - target_cls (torch.LongTensor): (H,W), 値は 0..5
            - time_str (str): ISO 形式の時刻文字列
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
            mask = (front_data[c, :, :] == 1)
            target_cls[mask] = c+1

        return gsm_tensor, target_cls, time_str

class SwinUnetModel(nn.Module):
    """
    概要:
        Swin-UNet 本体の薄いラッパ。CFG 由来のハイパーパラメータで SwinTransformerSys を構築する。

    入力:
        - num_classes (int): 出力クラス数（例 6）
        - in_chans (int): 入力チャネル数（例 93）
        - model_cfg (dict|None): 明示指定があればそれを使用、None の場合は CFG["STAGE1"]["model"]

    処理:
        - SwinTransformerSys を初期化して self.swin_unet に保持

    出力:
        - forward(x): ロジット (B,C,H,W)
    """
    def __init__(self, num_classes=6, in_chans=93, model_cfg=None):
        super(SwinUnetModel, self).__init__()
        cfg = model_cfg if model_cfg is not None else CFG["STAGE1"]["model"]
        self.swin_unet = SwinTransformerSys(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            depths_decoder=cfg["depths_decoder"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            norm_layer=cfg["norm_layer"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            final_upsample=cfg["final_upsample"]
        )

    def forward(self, x):
        """
        概要:
            入力テンソルからクラスごとのロジットを推論する。

        入力:
            - x (Tensor): (B, in_chans, H, W)

        処理:
            - SwinTransformerSys にそのまま渡してロジットを得る

        出力:
            - logits (Tensor): (B, num_classes, H, W)
        """
        logits = self.swin_unet(x)
        return logits

class DiceLoss(nn.Module):
    """
    概要:
        マルチクラス Dice 損失（クラスごとに Dice を計算し平均）を返す。

    入力:
        - classes (int): クラス数
        - forward(inputs, targets)
            - inputs: (B, C, H, W) ロジット（内部で softmax）
            - targets: (B, H, W) クラスID

    処理:
        - softmax で確率化
        - 各クラスについて flatten して Dice を計算
        - クラス平均を返す

    出力:
        - loss (Tensor): スカラー
    """
    def __init__(self, classes):
        super(DiceLoss, self).__init__()
        self.classes = classes
    def forward(self, inputs, targets):
        smooth = 1e-5
        total_loss = 0
        inputs = torch.softmax(inputs, dim=1)

        for i in range(self.classes):
            inp_flat  = inputs[:, i].contiguous().view(-1)
            tgt_flat = (targets == i).float().view(-1)
            intersection = (inp_flat * tgt_flat).sum()
            dice_score = (2.0*intersection + smooth) / (inp_flat.sum() + tgt_flat.sum() + smooth)
            dice_loss = 1 - dice_score
            total_loss += dice_loss
        return total_loss / self.classes

num_classes_stage1 = CFG["STAGE1"]["num_classes"]
ce_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss(classes=num_classes_stage1)

def combined_loss(inputs, targets):
    """
    概要:
        クロスエントロピー + Dice の合算損失。

    入力:
        - inputs (Tensor): (B, C, H, W) ロジット
        - targets (Tensor): (B, H, W) クラスID

    処理:
        - CE と Dice を計算し加算

    出力:
        - loss (Tensor): スカラー
    """
    loss_ce = ce_loss(inputs, targets)
    loss_dc = dice_loss(inputs, targets)
    return loss_ce + loss_dc

def train_stage1_one_epoch(model, dataloader, optimizer, epoch, num_classes):
    """
    概要:
        Stage1 モデルの1エポック学習を行い、平均損失と各クラス精度をログ出力する。

    入力:
        - model (nn.Module): 学習対象モデル
        - dataloader (DataLoader): 学習データローダ
        - optimizer (Optimizer): 最適化手法
        - epoch (int): 現在のエポック番号（0始まり）
        - num_classes (int): クラス数

    処理:
        - バッチ毎に forward/backward/step
        - ロス/精度を集計

    出力:
        - avg_epoch_loss (float): エポック平均損失
    """
    print_memory_usage(f"Before Stage1 train epoch={epoch+1}")
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = [0]*num_classes
    total = [0]*num_classes
    batch_count = 0

    pbar = tqdm(dataloader, desc=f"[Stage1][Train Epoch {epoch+1}]")
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        total_loss += loss_val
        batch_count += 1
        
        _, predicted = torch.max(outputs, dim=1)
        for i in range(num_classes):
            correct[i] += ((predicted == i) & (targets == i)).sum().item()
            total[i]   += (targets == i).sum().item()

        if (batch_idx+1) % 10 == 0:
            avg_loss_10 = running_loss / 10
            pbar.set_postfix({'Loss': f'{avg_loss_10:.4f}'})
            running_loss = 0.0

    avg_epoch_loss = total_loss / max(1, batch_count)
    print_memory_usage(f"After Stage1 train epoch={epoch+1}")
    gc.collect()

    print(f"\n[Stage1][Train Epoch {epoch+1}] Loss: {avg_epoch_loss:.4f}")
    print(f"[Stage1][Train Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i]/total[i]*100) if total[i]>0 else 0
        print(f"  Class {i}: {acc:.2f} %")
        
    return avg_epoch_loss

def test_stage1_one_epoch(model, dataloader, epoch, num_classes):
    """
    概要:
        Stage1 の検証（勾配停止）を1エポック分実施し、平均損失と各クラス精度を出力。

    入力:
        - model, dataloader, epoch, num_classes: train_stage1_one_epoch と同様

    処理:
        - no_grad で推論しロス/精度を集計

    出力:
        - avg_loss (float): 平均損失
    """
    print_memory_usage(f"Before Stage1 test epoch={epoch+1}")
    model.eval()
    test_loss = 0.0
    correct = [0]*num_classes
    total = [0]*num_classes
    batch_count = 0

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            test_loss += loss.item()
            batch_count += 1

            _, predicted = torch.max(outputs, dim=1)
            for i in range(num_classes):
                correct[i] += ((predicted == i) & (targets == i)).sum().item()
                total[i]   += (targets == i).sum().item()

    print_memory_usage(f"After Stage1 test epoch={epoch+1}")
    gc.collect()

    avg_loss = test_loss / max(1, batch_count)
    print(f"\n[Stage1][Test Epoch {epoch+1}] Loss: {avg_loss:.4f}")
    print(f"[Stage1][Test Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i]/total[i]*100) if total[i]>0 else 0
        print(f"  Class {i}: {acc:.2f} %")
        
    return avg_loss

def evaluate_stage1(model, dataloader, save_nc_dir=None):
    """
    概要:
        Stage1 モデルの推論を実行し、メトリクス計算と NetCDF 保存（任意）を行う。

    入力:
        - model (nn.Module)
        - dataloader (DataLoader)
        - save_nc_dir (str|None): 出力先ディレクトリ。None の場合は保存しない。

    処理:
        - 全サンプルで softmax により確率と予測クラスを集計
        - precision/recall/f1/accuracy を算出
        - save_nc_dir 指定時は (lat, lon, class) の DataArray を NetCDF 出力

    出力:
        - なし（ログ/ファイル出力）
    """
    print_memory_usage("Before evaluate_stage1")
    evaluate_start = time.time()
    
    model.eval()
    all_probs = []
    all_outputs = []
    all_targets = []
    all_times = []

    inference_start = time.time()
    with torch.no_grad():
        for inputs, targets, times in tqdm(dataloader, desc="[Stage1] Evaluate"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            pred_cls = torch.argmax(prob, dim=1)

            all_probs.append(prob.cpu())
            all_outputs.append(pred_cls.cpu())
            all_targets.append(targets.cpu())
            all_times.extend(times)
    inference_end = time.time()
    print(f"[Stage1] 推論処理時間: {format_time(inference_end - inference_start)}")
    metrics_start = time.time()
    all_probs = torch.cat(all_probs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    pred_flat = all_outputs.view(-1).numpy()
    targ_flat = all_targets.view(-1).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        targ_flat, pred_flat, labels=range(num_classes_stage1), average=None, zero_division=0
    )
    accuracy = (pred_flat == targ_flat).sum() / len(targ_flat)*100
    print(f"\n[Stage1] Pixel Accuracy (all classes): {accuracy:.2f}%")
    for i in range(num_classes_stage1):
        print(f"  Class{i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    metrics_end = time.time()
    print(f"[Stage1] 評価指標計算時間: {format_time(metrics_end - metrics_start)}")

    if save_nc_dir is not None:
        save_start = time.time()
        os.makedirs(save_nc_dir, exist_ok=True)
        for i in range(all_probs.shape[0]):
            time_str = all_times[i]
            probs_np = all_probs[i].numpy()
            probs_np = np.transpose(probs_np, (1,2,0)) 
            lat = dataloader.dataset.lat
            lon = dataloader.dataset.lon
            da = xr.DataArray(probs_np, dims=["lat","lon","class"],
                              coords={"lat": lat, "lon": lon, "class": np.arange(num_classes_stage1)})
            ds = xr.Dataset({"probabilities": da})
            ds = ds.expand_dims("time")
            ds["time"] = [pd.to_datetime(time_str)]
            date_str = pd.to_datetime(time_str).strftime('%Y%m%d%H%M')
            ds.to_netcdf(os.path.join(save_nc_dir, f"prob_{date_str}.nc"), engine='netcdf4') 
            del ds, da, probs_np
            gc.collect()

        print(f"[Stage1] Probabilities saved to {save_nc_dir}")
        save_end = time.time()
        print(f"[Stage1] 結果保存時間: {format_time(save_end - save_start)}")

    cleanup_start = time.time()
    del all_probs, all_outputs, all_targets
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage1] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")
    
    evaluate_end = time.time()
    print(f"[Stage1] 評価全体の実行時間: {format_time(evaluate_end - evaluate_start)}")
    print_memory_usage("After evaluate_stage1")

def run_stage1():
    """
    概要:
        Stage1（Swin-UNet, 93入力→6クラス）の学習・検証・推論・保存を一括実行する。

    入力:
        なし（CFG から設定を参照）

    処理:
        - データセット/データローダ構築
        - モデル/オプティマイザ初期化
        - チェックポイント再開（あれば）
        - 学習ループ（各エポックでtrain/test・チェックポイント保存）
        - 最良モデル保存、Loss曲線/CSV出力、評価推論の NetCDF 保存

    出力:
        - なし（成果物は v1_result 下に保存）
    """
    print_memory_usage("Start Stage 1")
    stage1_start = time.time()
    ds_start = time.time()
    y1, m1, y2, m2 = CFG["STAGE1"]["train_months"]
    train_months = get_available_months(y1, m1, y2, m2)
    y1, m1, y2, m2 = CFG["STAGE1"]["test_months"]
    test_months  = get_available_months(y1, m1, y2, m2)

    train_dataset_s1 = FrontalDatasetStage1(train_months, nc_gsm_dir, nc_0p5_dir)
    test_dataset_s1  = FrontalDatasetStage1(test_months,  nc_gsm_dir, nc_0p5_dir)
    train_loader_s1  = DataLoader(train_dataset_s1, batch_size=CFG["STAGE1"]["dataloader"]["batch_size_train"], shuffle=True, num_workers=CFG["STAGE1"]["dataloader"]["num_workers"])
    test_loader_s1   = DataLoader(test_dataset_s1,  batch_size=CFG["STAGE1"]["dataloader"]["batch_size_test"],  shuffle=False, num_workers=CFG["STAGE1"]["dataloader"]["num_workers"])
    ds_end = time.time()
    print(f"[Stage1] データセット準備時間: {format_time(ds_end - ds_start)}")

    print(f"[Stage1] Train dataset size: {len(train_dataset_s1)}")
    print(f"[Stage1] Test  dataset size: {len(test_dataset_s1)}")
    model_init_start = time.time()
    model_s1 = SwinUnetModel(num_classes=CFG["STAGE1"]["num_classes"], in_chans=CFG["STAGE1"]["in_chans"], model_cfg=CFG["STAGE1"]["model"]).to(device)
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=CFG["STAGE1"]["optimizer"]["lr"], weight_decay=CFG["STAGE1"]["optimizer"]["weight_decay"])
    model_init_end = time.time()
    print(f"[Stage1] モデル初期化時間: {format_time(model_init_end - model_init_start)}")

    num_epochs_stage1 = CFG["STAGE1"]["epochs"]
    ckpt_start = time.time()
    start_epoch = 0
    os.makedirs(model_s1_save_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(model_s1_save_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(model_s1_save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_s1.load_state_dict(state_dict)
        optimizer_s1.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Stage1] 既存のチェックポイント {latest_checkpoint} から学習を再開します（エポック {start_epoch} から）")
    else:
        print("[Stage1] 新規に学習を開始します")
    ckpt_end = time.time()
    print(f"[Stage1] チェックポイント読み込み時間: {format_time(ckpt_end - ckpt_start)}")

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs_stage1):
        epoch_start = time.time()
        train_loss = train_stage1_one_epoch(model_s1, train_loader_s1, optimizer_s1, epoch, CFG["STAGE1"]["num_classes"])
        test_loss = test_stage1_one_epoch(model_s1, test_loader_s1, epoch, CFG["STAGE1"]["num_classes"])
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model_s1.state_dict()
            best_epoch = epoch
            print(f"[Stage1] 新しい最良モデルを見つけました（エポック {epoch+1}）: テスト損失 = {test_loss:.4f}")
        
        epoch_end = time.time()
        print(f"[Stage1] エポック {epoch+1} 実行時間: {format_time(epoch_end - epoch_start)}")
        save_start = time.time()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_s1.state_dict(),
            'optimizer_state_dict': optimizer_s1.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }

        checkpoint_path = os.path.join(model_s1_save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"[Stage1] チェックポイントを保存しました: checkpoint_epoch_{epoch}.pth")
        if epoch > 0:
            previous_checkpoint_path = os.path.join(model_s1_save_dir, f'checkpoint_epoch_{epoch - 1}.pth')
            if os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
                print(f"[Stage1] 前回のチェックポイントを削除しました: checkpoint_epoch_{epoch - 1}.pth")
        save_end = time.time()
        print(f"[Stage1] チェックポイント保存時間: {format_time(save_end - save_start)}")
    
    training_end = time.time()
    print(f"[Stage1] 学習ループ全体の実行時間: {format_time(training_end - training_start)}")
    final_save_start = time.time()
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(model_s1_save_dir, 'model_final.pth'))
        print(f"[Stage1] 最良モデル（エポック {best_epoch+1}）を model_final.pth として保存しました")
    else:
        torch.save(model_s1.state_dict(), os.path.join(model_s1_save_dir, 'model_final.pth'))
        print(f"[Stage1] 最終的なモデルを保存しました: model_final.pth")
    
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs = list(range(start_epoch + 1, start_epoch + len(train_losses) + 1))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, test_losses, 'r-', label='Test Loss')
        if best_epoch >= 0:
            plt.axvline(x=best_epoch+1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage1 Training and Test Loss')
        plt.legend()
        plt.grid(True)
        min_loss = min(min(train_losses), min(test_losses))
        max_loss = max(max(train_losses), max(test_losses))
        margin = (max_loss - min_loss) * 0.1
        plt.ylim([max(0, min_loss - margin), max_loss + margin])
        
        loss_curve_path = os.path.join(model_s1_save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[Stage1] Loss曲線を保存しました: {loss_curve_path}")
        # Save loss history as CSV for logging
        try:
            epochs_col = epochs
        except NameError:
            epochs_col = list(range(1, len(train_losses) + 1))
        df_loss_s1 = pd.DataFrame({
            "epoch": epochs_col,
            "train_loss": train_losses,
            "test_loss": test_losses
        })
        csv_path_s1 = os.path.join(model_s1_save_dir, "loss_history.csv")
        df_loss_s1.to_csv(csv_path_s1, index=False)
        # Also place a copy in v1_result root for easy access
        try:
            root_csv_s1 = os.path.join(os.path.dirname(model_s1_save_dir), "loss_history_stage1.csv")
            df_loss_s1.to_csv(root_csv_s1, index=False)
        except Exception as e:
            print(f"[Stage1] Loss history CSV copy skipped: {e}")
    
    final_save_end = time.time()
    print(f"[Stage1] 最終モデル保存時間: {format_time(final_save_end - final_save_start)}")
    eval_start = time.time()
    if best_model_state is not None:
        model_s1.load_state_dict(best_model_state)
        print(f"[Stage1] 評価のために最良モデル（エポック {best_epoch+1}）をロードしました")
    
    evaluate_stage1(model_s1, test_loader_s1, save_nc_dir=stage1_out_dir)
    eval_end = time.time()
    print(f"[Stage1] 評価時間: {format_time(eval_end - eval_start)}")
    cleanup_start = time.time()
    del train_dataset_s1, test_dataset_s1, train_loader_s1, test_loader_s1
    del model_s1, optimizer_s1
    torch.cuda.empty_cache()
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage1] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    stage1_end = time.time()
    print(f"[Stage1] 全体の実行時間: {format_time(stage1_end - stage1_start)}")
    print_memory_usage("After Stage 1")

class FrontalRefinementDataset(Dataset):
    """
    概要:
        Stage2（補正学習）用データセット。
        - train/val: GT から擬似的な劣化をかけたクラスマップを入力、GT を教師として返す
        - test: Stage1 の確率から得た最頻クラスを入力、教師はダミー（0）を返す

    入力:
        - months (List[str]|None): train/val 対象の年月リスト（test では None）
        - nc_0p5_dir (str): GT 前線 NetCDF のディレクトリ
        - mode (str): 'train'|'val'|'test'
        - stage1_out_dir (str|None): test モード時に参照する Stage1 出力（prob_*.nc）ディレクトリ
        - 各種劣化オプション（確率・範囲など）
        - cache_size (int): サンプルキャッシュ数

    処理:
        - prepare_index_trainval/prepare_index_test で data_index を構築
        - __getitem__ で入力(1ch, H, W) と教師(クラスID HxW) を返す

    出力:
        - __getitem__ -> (x_tensor: FloatTensor[1,H,W], y_tensor: LongTensor[H,W], time_str: str)
    """
    def __init__(self, months, nc_0p5_dir, mode='train', stage1_out_dir=None,
                 n_augment=8,
                 prob_dilation=0.8,
                 prob_create_gaps=0.8,
                 prob_random_pixel_change=0.8,
                 prob_add_fake_front=0.8,
                 dilation_kernel_range=(2,3),
                 num_gaps_range=(2,4),
                 gap_size_range=(3,5),
                 num_pix_to_change_range=(20,100),
                 num_fake_front_range=(2,10),
                 cache_size=50):
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

        if self.mode in ['train','val']:
            self.prepare_index_trainval()
        else:
            self.prepare_index_test()

    def prepare_index_trainval(self):
        """
        概要:
            train/val 用に GT 前線ファイルを走査し、時刻ごとに n_augment 回の劣化サンプルを作成するための
            data_index を構築する。

        入力:
            なし

        処理:
            - 各時刻 t について aug_idx=0..n_augment-1 のエントリを作成

        出力:
            なし（self.data_index を更新）
        """
        print_memory_usage("Before Stage2 prepare_index_trainval")
        for month in self.months:
            front_file = os.path.join(self.nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(front_file):
                print(f"{front_file} が見つかりません。スキップします。(Stage2)")
                continue

            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_front['lat'].values[:ORIG_H]
                self.lon = ds_front['lon'].values[:ORIG_W]

            times = ds_front['time'].values
            for t in times:
                time_dt = pd.to_datetime(t)
                for aug_idx in range(self.n_augment):
                    self.data_index.append({
                        'front_file': front_file,
                        't': t,
                        'time_dt': time_dt,
                        'aug_idx': aug_idx,
                        'is_augmented': True
                    })

            ds_front.close()
            del ds_front
            gc.collect()

        print_memory_usage("After Stage2 prepare_index_trainval")
        print(f"Stage2データインデックス構築完了（train/val）：{len(self.data_index)}サンプル")
        gc.collect()

    def prepare_index_test(self):
        """
        概要:
            test モードでは Stage1 出力の確率 NetCDF 群を走査し、各ファイルを1サンプルとして data_index を作成する。

        入力:
            なし

        処理:
            - stage1_out_dir 内の .nc を列挙し、time を取得して data_index に格納

        出力:
            なし（self.data_index を更新）
        """
        print_memory_usage("Before Stage2 prepare_index_test")
        if not os.path.exists(self.stage1_out_dir):
            print(f"Stage1 出力ディレクトリ {self.stage1_out_dir} がありません。(Stage2)")
            return

        files = sorted([f for f in os.listdir(self.stage1_out_dir) if f.endswith('.nc')])
        for f in files:
            nc_path = os.path.join(self.stage1_out_dir, f)
            ds = xr.open_dataset(nc_path)

            if self.lat is None or self.lon is None:
                self.lat = ds['lat'].values
                self.lon = ds['lon'].values

            time_val = ds['time'].values[0] if 'time' in ds else None
            time_dt = pd.to_datetime(time_val) if time_val is not None else None

            self.data_index.append({
                'nc_path': nc_path,
                'time_dt': time_dt,
                'is_augmented': False
            })

            ds.close()
            del ds
            gc.collect()

        print_memory_usage("After Stage2 prepare_index_test")
        print(f"Stage2データインデックス構築完了（test）：{len(self.data_index)}サンプル")
        gc.collect()

    def load_single_item(self, idx):
        """
        概要:
            data_index[idx] をもとに、train/val は GT を劣化して入力を作成、test は Stage1 の argmax を入力として返す。

        入力:
            - idx (int): サンプル番号

        処理:
            - train/val: GT one-hot をクラスIDへ、確率的な劣化処理(degrade_front_data)を適用
            - test: Stage1 の probabilities から argmax を取り入力クラスマップを作成、教師は0で埋める

        出力:
            - in_cls (np.ndarray[int64]): 入力クラスマップ (H,W)
            - tgt_cls (np.ndarray[int64]): 教師クラスマップ (H,W)
            - time_dt (pd.Timestamp|None): 時刻
        """
        item = self.data_index[idx]
        if item.get('is_augmented', False) and 'front_file' in item:
            front_file = item['front_file']
            t = item['t']
            time_dt = item['time_dt']
            aug_idx = item['aug_idx']
            ds_front = xr.open_dataset(front_file)
            front_data = ds_front.sel(time=t).to_array().values
            target_cls = np.zeros((ORIG_H, ORIG_W), dtype=np.int64)
            for c in range(5):
                mask = (front_data[c, :, :] == 1)
                target_cls[mask] = c+1

            np.random.seed((hash(str(time_dt)) + aug_idx) & 0x7FFFFFFF)
            degraded = self.degrade_front_data(target_cls)
            
            ds_front.close()
            
            return degraded.astype(np.int64), target_cls.astype(np.int64), time_dt
        
        elif 'nc_path' in item:
            nc_path = item['nc_path']
            time_dt = item['time_dt']
            ds = xr.open_dataset(nc_path)
            probs_np = ds['probabilities'].isel(time=0).values
            degraded_cls = np.argmax(probs_np, axis=-1)
            target_cls = np.zeros_like(degraded_cls, dtype=np.int64)
            
            ds.close()
            
            return degraded_cls, target_cls, time_dt
        
        else:
            raise ValueError(f"Invalid data index item: {item}")

    def __len__(self):
        """
        概要:
            サンプル数を返す。

        入力:
            なし

        処理:
            - data_index の長さを返却

        出力:
            - n (int)
        """
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        概要:
            idx 番目の (入力1ch, 教師, 時刻文字列) を返す。

        入力:
            - idx (int)

        処理:
            - キャッシュを優先、無い場合は load_single_item
            - 入力は float の (1,H,W)、教師は long の (H,W) に変換

        出力:
            - x_tensor (FloatTensor[1,H,W]), y_tensor (LongTensor[H,W]), time_str (str)
        """
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
        """
        概要:
            中心座標から BFS 風に拡張して不規則な形状の点集合を作る（劣化用の穴やフェイク前線に利用）。

        入力:
            - h, w (int): 画像サイズ
            - cy, cx (int): 初期中心座標
            - max_shape_size (int): 形状の最大ピクセル数

        処理:
            - 4近傍で確率的に拡張し、座標集合を返す

        出力:
            - shape_points (List[Tuple[int,int]]): (y,x) の座標リスト
        """
        visited = set()
        visited.add((cy, cx))
        queue = deque()
        queue.append((cy, cx))

        target_size = random.randint(1, max_shape_size)
        shape_points = []

        while queue and len(shape_points)<target_size:
            y,x = queue.popleft()
            shape_points.append((y,x))

            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0<=ny<h and 0<=nx<w and (ny,nx) not in visited:
                    visited.add((ny,nx))
                    if random.random()<0.8:
                        queue.append((ny,nx))
        return shape_points

    def degrade_front_data(self, cls_map):
        """
        概要:
            クラスマップに対し、膨張やギャップ作成、ランダム置換、偽前線追加等の劣化を施す。

        入力:
            - cls_map (np.ndarray[int]): (H,W) 0..5 のクラスID

        処理:
            - dilation（確率的）
            - create_gaps（穴あけ）
            - random pixel change（形状ごとクラス変更）
            - add_fake_front（背景に擬似前線追加）

        出力:
            - degraded (np.ndarray[int]): 劣化後クラスマップ
        """
        degraded = cls_map.copy()
        for c in [1,2,3,4,5]:
            mask = (degraded == c).astype(np.uint8)
            if random.random()<self.prob_dilation:
                ksize = random.randint(*self.dilation_kernel_range)
                kernel = np.ones((ksize,ksize), np.uint8)
                mask_dil = cv2.dilate(mask, kernel, iterations=1)
                degraded[mask_dil==1] = c

            if random.random()<self.prob_create_gaps:
                num_gaps = random.randint(*self.num_gaps_range)
                indices = np.transpose(np.nonzero(mask))
                if len(indices)>0:
                    for _ in range(num_gaps):
                        idx = random.choice(indices)
                        yy,xx = idx
                        shape_points = self.create_irregular_shape(degraded.shape[0], degraded.shape[1], yy,xx, max_shape_size=random.randint(3,5))
                        for (sy,sx) in shape_points:
                            if degraded[sy,sx] == c:
                                degraded[sy,sx] = 0
        h,w = degraded.shape
        if random.random()<self.prob_random_pixel_change:
            num_pix_to_change = random.randint(*self.num_pix_to_change_range)
            num_shapes = random.randint(1,5)
            for _ in range(num_shapes):
                cy = random.randint(0,h-1)
                cx = random.randint(0,w-1)
                old_c = degraded[cy,cx]
                if old_c in [1,2,3,4,5]:
                    shape_points = self.create_irregular_shape(h,w,cy,cx,25)
                    possible_classes = [0,1,2,3,4,5]
                    possible_classes.remove(old_c)
                    new_c = random.choice(possible_classes)

                    count_changed=0
                    for (sy,sx) in shape_points:
                        if degraded[sy,sx]==old_c:
                            degraded[sy,sx] = new_c
                            count_changed+=1
                            if count_changed>=num_pix_to_change:
                                break
        if random.random()<self.prob_add_fake_front:
            num_fake_front = random.randint(*self.num_fake_front_range)
            for _ in range(num_fake_front):
                cy = random.randint(0,h-1)
                cx = random.randint(0,w-1)
                if degraded[cy,cx]==0:
                    shape_points = self.create_irregular_shape(h,w,cy,cx,25)
                    new_c = random.choice([1,2,3,4,5])
                    for (sy,sx) in shape_points:
                        if degraded[sy,sx]==0:
                            degraded[sy,sx] = new_c

        return degraded

class SwinUnetModelStage2(nn.Module):
    """
    概要:
        Stage2 用の Swin-UNet ラッパ（入力 1ch → 出力 6クラス）。

    入力:
        - num_classes (int), in_chans (int), model_cfg (dict|None)

    処理:
        - SwinTransformerSys を初期化

    出力:
        - forward(x): (B,C,H,W)
    """
    def __init__(self, num_classes=6, in_chans=1, model_cfg=None):
        super(SwinUnetModelStage2,self).__init__()
        cfg = model_cfg if model_cfg is not None else CFG["STAGE2"]["model"]
        self.swin_unet = SwinTransformerSys(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            depths_decoder=cfg["depths_decoder"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            norm_layer=cfg["norm_layer"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            final_upsample=cfg["final_upsample"]
        )
    def forward(self,x):
        """
        概要:
            Stage2 入力（1ch クラスマップ）からロジットを出力する。

        入力:
            - x (Tensor): (B,1,H,W)

        処理:
            - Swin-UNet へ前伝播

        出力:
            - logits (Tensor): (B,6,H,W)
        """
        return self.swin_unet(x)

def train_stage2_one_epoch(model, dataloader, optimizer, epoch, num_classes):
    """
    概要:
        Stage2 モデルの1エポック学習を実施。

    入力:
        - model, dataloader, optimizer, epoch, num_classes: Stage1 と同様

    処理:
        - forward/backward/step、精度集計

    出力:
        - avg_epoch_loss (float)
    """
    print_memory_usage(f"Before Stage2 train epoch={epoch+1}")
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = [0]*num_classes
    total = [0]*num_classes
    batch_count = 0

    pbar = tqdm(dataloader, desc=f"[Stage2][Train Epoch {epoch+1}]")
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        total_loss += loss_val
        batch_count += 1
        
        _, predicted = torch.max(outputs, dim=1)
        for i in range(num_classes):
            correct[i] += ((predicted==i)&(targets==i)).sum().item()
            total[i] += (targets==i).sum().item()

        if (batch_idx+1)%10==0:
            avg_loss_10 = running_loss/10
            pbar.set_postfix({'Loss':f'{avg_loss_10:.4f}'})
            running_loss=0.0

    avg_epoch_loss = total_loss / max(1, batch_count)
    print_memory_usage(f"After Stage2 train epoch={epoch+1}")
    gc.collect()

    print(f"\n[Stage2][Train Epoch {epoch+1}] Loss: {avg_epoch_loss:.4f}")
    print(f"[Stage2][Train Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i]/total[i]*100) if total[i]>0 else 0
        print(f"  Class {i}: {acc:.2f} %")
        
    return avg_epoch_loss

def test_stage2_one_epoch(model, dataloader, epoch, num_classes):
    """
    概要:
        Stage2 の検証を1エポック分実施。

    入力:
        - model, dataloader, epoch, num_classes: Stage1 と同様

    処理:
        - no_grad で推論しロス/精度を集計

    出力:
        - avg_loss (float)
    """
    print_memory_usage(f"Before Stage2 test epoch={epoch+1}")
    model.eval()
    test_loss=0.0
    correct=[0]*num_classes
    total=[0]*num_classes
    batch_count = 0

    with torch.no_grad():
        for inputs,targets,_ in dataloader:
            inputs,targets=inputs.to(device), targets.to(device)
            outputs=model(inputs)
            loss=combined_loss(outputs,targets)
            test_loss+=loss.item()
            batch_count += 1

            _,predicted=torch.max(outputs,dim=1)
            for i in range(num_classes):
                correct[i]+=((predicted==i)&(targets==i)).sum().item()
                total[i]+=(targets==i).sum().item()

    print_memory_usage(f"After Stage2 test epoch={epoch+1}")
    gc.collect()

    avg_loss=test_loss / max(1, batch_count)
    print(f"\n[Stage2][Test Epoch {epoch+1}] Loss: {avg_loss:.4f}")
    print(f"[Stage2][Test Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc=(correct[i]/total[i]*100) if total[i]>0 else 0
        print(f"  Class {i}: {acc:.2f} %")
        
    return avg_loss

def evaluate_stage2(model, dataloader, save_nc_dir=None):
    """
    概要:
        Stage2 モデルの推論・メトリクス計算・NetCDF保存（任意）を行う。

    入力:
        - model (nn.Module)
        - dataloader (DataLoader)
        - save_nc_dir (str|None): 保存先

    処理:
        - 確率/予測クラスを集約し、精度指標と混同行列を計算
        - save_nc_dir 指定時は probabilities を NetCDF 保存

    出力:
        - なし（ログ/ファイル出力）
    """
    print_memory_usage("Before evaluate_stage2")
    evaluate_start = time.time()
    model.eval()
    all_probs=[]
    all_outputs=[]
    all_targets=[]
    all_times=[]
    inference_start = time.time()
    with torch.no_grad():
        for inputs,targets,times in tqdm(dataloader,desc="[Stage2] Evaluate"):
            inputs,targets=inputs.to(device), targets.to(device)
            outputs=model(inputs)
            prob=torch.softmax(outputs,dim=1)
            pred_cls=torch.argmax(prob,dim=1)
            all_probs.append(prob.cpu())
            all_outputs.append(pred_cls.cpu())
            all_targets.append(targets.cpu())
            all_times.extend(times)
    inference_end = time.time()
    print(f"[Stage2] 推論処理時間: {format_time(inference_end - inference_start)}")
    metrics_start = time.time()
    all_probs=torch.cat(all_probs,dim=0)
    all_outputs=torch.cat(all_outputs,dim=0)
    all_targets=torch.cat(all_targets,dim=0)
    if (all_targets.numpy().sum()>0):
        pred_flat=all_outputs.view(-1).numpy()
        targ_flat=all_targets.view(-1).numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            targ_flat, pred_flat, labels=list(range(CFG["STAGE2"]["num_classes"])), average=None, zero_division=0)
        accuracy = (pred_flat==targ_flat).sum()/len(targ_flat)*100
        print(f"\n[Stage2] Pixel Accuracy (all classes): {accuracy:.2f}%")
        for i, cls_id in enumerate(list(range(CFG["STAGE2"]["num_classes"]))):
            print(f"  Class{cls_id}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

        cm=confusion_matrix(targ_flat,pred_flat,labels=list(range(CFG["STAGE2"]["num_classes"])))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                    xticklabels=list(range(CFG["STAGE2"]["num_classes"])),
                    yticklabels=list(range(CFG["STAGE2"]["num_classes"])))
        plt.title('[Stage2] Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('stage2_confusion_matrix.png')
        plt.close()
    metrics_end = time.time()
    print(f"[Stage2] 評価指標計算時間: {format_time(metrics_end - metrics_start)}")
    if save_nc_dir is not None:
        save_start = time.time()
        os.makedirs(save_nc_dir,exist_ok=True)
        lat=dataloader.dataset.lat
        lon=dataloader.dataset.lon
        for i in range(all_probs.shape[0]):
            time_str=all_times[i]
            probs_np=all_probs[i].numpy()
            probs_np=np.transpose(probs_np,(1,2,0))
            da=xr.DataArray(probs_np,dims=["lat","lon","class"],
                            coords={"lat":lat,"lon":lon,"class":np.arange(CFG["STAGE2"]["num_classes"])})
            ds=xr.Dataset({"probabilities":da})
            ds=ds.expand_dims('time')
            ds['time']=[pd.to_datetime(time_str)]
            date_str=pd.to_datetime(time_str).strftime('%Y%m%d%H%M')
            ds.to_netcdf(os.path.join(save_nc_dir, f"refined_{date_str}.nc"), engine='netcdf4')
            del ds, da, probs_np
            gc.collect()
        print(f"[Stage2] Refined probabilities saved to {save_nc_dir}")
        save_end = time.time()
        print(f"[Stage2] 結果保存時間: {format_time(save_end - save_start)}")

    cleanup_start = time.time()
    del all_probs, all_outputs, all_targets
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage2] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")
    
    evaluate_end = time.time()
    print(f"[Stage2] 評価全体の実行時間: {format_time(evaluate_end - evaluate_start)}")
    print_memory_usage("After evaluate_stage2")

def run_stage2():
    """
    概要:
        Stage2（補正モデル, 入力1ch→6クラス）の学習・検証・推論・保存を実行。

    入力:
        なし（CFG 参照）

    処理:
        - train/val データの構築（GT 劣化ベース）
        - 学習ループとチェックポイント保存
        - 最良モデルで test データ（Stage1出力）を推論し NetCDF 保存

    出力:
        - なし（成果物は v1_result 下に保存）
    """
    print_memory_usage("Start Stage 2")
    stage2_start = time.time()
    ds_start = time.time()
    y1, m1, y2, m2 = CFG["STAGE2"]["train_months"]
    train_months = get_available_months(y1, m1, y2, m2)

    aug = CFG["STAGE2"]["augment"]
    train_dataset_s2=FrontalRefinementDataset(
        months=train_months,
        nc_0p5_dir=nc_0p5_dir,
        mode='train',
        stage1_out_dir=None,
        n_augment=aug["n_augment"],
        prob_dilation=aug["prob_dilation"],
        prob_create_gaps=aug["prob_create_gaps"],
        prob_random_pixel_change=aug["prob_random_pixel_change"],
        prob_add_fake_front=aug["prob_add_fake_front"],
        dilation_kernel_range=aug["dilation_kernel_range"],
        num_gaps_range=aug["num_gaps_range"],
        gap_size_range=aug["gap_size_range"],
        num_pix_to_change_range=aug["num_pix_to_change_range"],
        num_fake_front_range=aug["num_fake_front_range"]
    )
    val_dataset_s2=FrontalRefinementDataset(
        months=train_months,
        nc_0p5_dir=nc_0p5_dir,
        mode='val',
        stage1_out_dir=None,
        n_augment=aug["n_augment"],
        prob_dilation=aug["prob_dilation"],
        prob_create_gaps=aug["prob_create_gaps"],
        prob_random_pixel_change=aug["prob_random_pixel_change"],
        prob_add_fake_front=aug["prob_add_fake_front"],
        dilation_kernel_range=aug["dilation_kernel_range"],
        num_gaps_range=aug["num_gaps_range"],
        gap_size_range=aug["gap_size_range"],
        num_pix_to_change_range=aug["num_pix_to_change_range"],
        num_fake_front_range=aug["num_fake_front_range"]
    )
    train_loader_s2=DataLoader(train_dataset_s2,batch_size=CFG["STAGE2"]["dataloader"]["batch_size_train"],shuffle=True,num_workers=CFG["STAGE2"]["dataloader"]["num_workers"])
    val_loader_s2=DataLoader(val_dataset_s2,batch_size=CFG["STAGE2"]["dataloader"]["batch_size_val"],shuffle=False,num_workers=CFG["STAGE2"]["dataloader"]["num_workers"])
    ds_end = time.time()
    print(f"[Stage2] データセット準備時間: {format_time(ds_end - ds_start)}")
    print(f"[Stage2] Train dataset size: {len(train_dataset_s2)}")
    print(f"[Stage2] Val   dataset size: {len(val_dataset_s2)}")
    model_init_start = time.time()
    model_s2=SwinUnetModelStage2(num_classes=CFG["STAGE2"]["num_classes"],in_chans=CFG["STAGE2"]["in_chans"], model_cfg=CFG["STAGE2"]["model"]).to(device)
    optimizer_s2=optim.AdamW(model_s2.parameters(),lr=CFG["STAGE2"]["optimizer"]["lr"],weight_decay=CFG["STAGE2"]["optimizer"]["weight_decay"])
    model_init_end = time.time()
    print(f"[Stage2] モデル初期化時間: {format_time(model_init_end - model_init_start)}")
    num_epochs_stage2 = CFG["STAGE2"]["epochs"]
    ckpt_start = time.time()
    start_epoch=0
    os.makedirs(model_s2_save_dir,exist_ok=True)
    checkpoint_files=[f for f in os.listdir(model_s2_save_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        latest_checkpoint=checkpoint_files[-1]
        checkpoint_path=os.path.join(model_s2_save_dir,latest_checkpoint)
        checkpoint=torch.load(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_s2.load_state_dict(state_dict)
        optimizer_s2.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']+1
        print(f"[Stage2] 既存のチェックポイント {latest_checkpoint} から学習を再開します（エポック {start_epoch} から）")
    else:
        print("[Stage2] 新規に学習を開始します")
    ckpt_end = time.time()
    print(f"[Stage2] チェックポイント読み込み時間: {format_time(ckpt_end - ckpt_start)}")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs_stage2):
        epoch_start = time.time()
        train_loss = train_stage2_one_epoch(model_s2, train_loader_s2, optimizer_s2, epoch, CFG["STAGE2"]["num_classes"])
        val_loss = test_stage2_one_epoch(model_s2, val_loader_s2, epoch, CFG["STAGE2"]["num_classes"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_s2.state_dict()
            best_epoch = epoch
            print(f"[Stage2] 新しい最良モデルを見つけました（エポック {epoch+1}）: 検証損失 = {val_loss:.4f}")
        
        epoch_end = time.time()
        print(f"[Stage2] エポック {epoch+1} 実行時間: {format_time(epoch_end - epoch_start)}")
        save_start = time.time()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_s2.state_dict(),
            'optimizer_state_dict': optimizer_s2.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        checkpoint_path = os.path.join(model_s2_save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"[Stage2] チェックポイントを保存しました: checkpoint_epoch_{epoch}.pth")
        if epoch > 0:
            previous_checkpoint_path = os.path.join(model_s2_save_dir, f'checkpoint_epoch_{epoch - 1}.pth')
            if os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
                print(f"[Stage2] 前回のチェックポイントを削除しました: checkpoint_epoch_{epoch - 1}.pth")
        save_end = time.time()
        print(f"[Stage2] チェックポイント保存時間: {format_time(save_end - save_start)}")
    
    training_end = time.time()
    print(f"[Stage2] 学習ループ全体の実行時間: {format_time(training_end - training_start)}")
    final_save_start = time.time()
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(model_s2_save_dir, 'model_final.pth'))
        print(f"[Stage2] 最良モデル（エポック {best_epoch+1}）を model_final.pth として保存しました")
    else:
        torch.save(model_s2.state_dict(), os.path.join(model_s2_save_dir, 'model_final.pth'))
        print(f"[Stage2] 最終的なモデルを保存しました: model_final.pth")
    
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs = list(range(start_epoch + 1, start_epoch + len(train_losses) + 1))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        if best_epoch >= 0:
            plt.axvline(x=best_epoch+1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Stage2 Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        min_loss = min(min(train_losses), min(val_losses))
        max_loss = max(max(train_losses), max(val_losses))
        margin = (max_loss - min_loss) * 0.1
        plt.ylim([max(0, min_loss - margin), max_loss + margin])
        
        loss_curve_path = os.path.join(model_s2_save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[Stage2] Loss曲線を保存しました: {loss_curve_path}")
        # Save loss history as CSV for logging
        try:
            epochs_col = epochs
        except NameError:
            epochs_col = list(range(1, len(train_losses) + 1))
        df_loss_s2 = pd.DataFrame({
            "epoch": epochs_col,
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        csv_path_s2 = os.path.join(model_s2_save_dir, "loss_history.csv")
        df_loss_s2.to_csv(csv_path_s2, index=False)
        # Also place a copy in v1_result root for easy access
        try:
            root_csv_s2 = os.path.join(os.path.dirname(model_s2_save_dir), "loss_history_stage2.csv")
            df_loss_s2.to_csv(root_csv_s2, index=False)
        except Exception as e:
            print(f"[Stage2] Loss history CSV copy skipped: {e}")
    
    final_save_end = time.time()
    print(f"[Stage2] 最終モデル保存時間: {format_time(final_save_end - final_save_start)}")
    eval_start = time.time()
    test_dataset_s2=FrontalRefinementDataset(
        months=None,
        nc_0p5_dir=nc_0p5_dir,
        mode='test',
        stage1_out_dir=stage1_out_dir
    )
    test_loader_s2=DataLoader(test_dataset_s2,batch_size=CFG["STAGE2"]["dataloader"]["batch_size_test"],shuffle=False,num_workers=CFG["STAGE2"]["dataloader"]["num_workers"])
    print(f"[Stage2] Test dataset size (Stage1結果): {len(test_dataset_s2)}")
    if best_model_state is not None:
        model_s2.load_state_dict(best_model_state)
        print(f"[Stage2] 評価のために最良モデル（エポック {best_epoch+1}）をロードしました")
    
    evaluate_stage2(model_s2, test_loader_s2, save_nc_dir=stage2_out_dir)
    eval_end = time.time()
    print(f"[Stage2] 評価時間: {format_time(eval_end - eval_start)}")
    cleanup_start = time.time()
    del train_dataset_s2, val_dataset_s2, train_loader_s2, val_loader_s2
    del test_dataset_s2, test_loader_s2
    del model_s2, optimizer_s2
    torch.cuda.empty_cache()
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage2] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    stage2_end = time.time()
    print(f"[Stage2] 全体の実行時間: {format_time(stage2_end - stage2_start)}")
    print_memory_usage("After Stage 2")

def evaluate_stage3(stage2_nc_dir, save_nc_dir):
    """
    概要:
        Stage2 出力の確率から argmax でクラスマップを作り、スケルトン化した結果を NetCDF に保存する。

    入力:
        - stage2_nc_dir (str): Stage2 NetCDF ディレクトリ
        - save_nc_dir (str): 出力先ディレクトリ

    処理:
        - refined_*.nc を順次読み、class_map を生成
        - skimage.morphology.skeletonize で 1ピクセル幅の骨格抽出
        - 結果を class_map として保存

    出力:
        - なし（ファイル出力/ログ）
    """
    print_memory_usage("Before evaluate_stage3")
    evaluate_start = time.time()
    setup_start = time.time()
    os.makedirs(save_nc_dir, exist_ok=True)
    stage2_files = sorted([f for f in os.listdir(stage2_nc_dir) if f.endswith('.nc')])
    setup_end = time.time()
    print(f"[Stage3] 設定準備時間: {format_time(setup_end - setup_start)}")

    processing_start = time.time()
    for f in tqdm(stage2_files, desc="[Stage3] Evaluate"):
        file_start = time.time()
        stage2_path = os.path.join(stage2_nc_dir, f)
        ds = xr.open_dataset(stage2_path)
        time_str = ds['time'].values[0]
        time_dt = pd.to_datetime(time_str)
        date_str = time_dt.strftime('%Y%m%d%H%M')

        probs_np = ds['probabilities'].isel(time=0).values
        lat      = ds['lat'].values
        lon      = ds['lon'].values
        ds.close()
        class_map = np.argmax(probs_np, axis=-1)
        binary_mask = (class_map >= 1).astype(np.uint8)
        skeleton = skeletonize(binary_mask).astype(np.uint8)

        skeletonized_map = np.zeros_like(class_map)
        h, w = class_map.shape
        for y in range(h):
            for x in range(w):
                if skeleton[y, x]:
                    if class_map[y, x] >= 1:
                        skeletonized_map[y, x] = class_map[y, x]
                    else:
                        neighbors = class_map[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)]
                        neighbor_classes = neighbors[neighbors >= 1]
                        if len(neighbor_classes) > 0:
                            counts = np.bincount(neighbor_classes)
                            skeletonized_map[y, x] = np.argmax(counts)
                        else:
                            skeletonized_map[y, x] = 0

        lat = ds['lat'].values
        lon = ds['lon'].values
        da = xr.DataArray(skeletonized_map, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        ds_out = xr.Dataset({"class_map": da})
        ds_out = ds_out.expand_dims('time')
        ds_out['time'] = [time_dt]
        ds_out.to_netcdf(os.path.join(save_nc_dir, f"skeleton_{date_str}.nc"), engine='netcdf4')
        del ds, probs_np, class_map, skeleton, skeletonized_map
        gc.collect()
        
        file_end = time.time()
        if (stage2_files.index(f) + 1) % 10 == 0:
            print(f"[Stage3] {stage2_files.index(f) + 1}/{len(stage2_files)}ファイル処理時間: {format_time(file_end - file_start)}")
    
    processing_end = time.time()
    print(f"[Stage3] 全ファイル処理時間: {format_time(processing_end - processing_start)}")

    print(f"[Stage3] Skeletonized results saved to {save_nc_dir}")
    
    evaluate_end = time.time()
    print(f"[Stage3] 評価全体の実行時間: {format_time(evaluate_end - evaluate_start)}")
    print_memory_usage("After evaluate_stage3")

def run_stage3():
    """
    概要:
        Stage3（スケルトン化）を実行。Stage2 の確率出力から前線骨格を抽出する。

    入力:
        なし

    処理:
        - evaluate_stage3 を呼び出し

    出力:
        - なし
    """
    print_memory_usage("Start Stage 3")
    evaluate_stage3(stage2_nc_dir=stage2_out_dir, save_nc_dir=stage3_out_dir)
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("After Stage 3")

# --------------------------------------------------
# 可視化
# --------------------------------------------------
def visualize_results(stage1_nc_dir,stage2_nc_dir,stage3_nc_dir,original_nc_dir,output_dir):
    """
    概要:
        Stage1/2/3 と元データを並べて地図上に描画し、比較PNGを出力する。

    入力:
        - stage1_nc_dir, stage2_nc_dir, stage3_nc_dir (str): 各段の NetCDF 置き場
        - original_nc_dir (str): 元の前線データ NetCDF 置き場
        - output_dir (str): 画像出力先

    処理:
        - 共通時刻を抽出し、multiprocessing で並列描画
        - 気圧偏差のコンター/塗りつぶし + 各クラスマップを pcolormesh でオーバレイ
        - 低気圧中心(任意)をプロット

    出力:
        - output_dir に comparison_YYYYMMDDHHMM.png を保存
    """
    print("可視化処理を開始します。")
    os.makedirs(output_dir,exist_ok=True)

    stage1_files=sorted([f for f in os.listdir(stage1_nc_dir) if f.endswith('.nc')])
    stage2_files=sorted([f for f in os.listdir(stage2_nc_dir) if f.endswith('.nc')])
    stage3_files=sorted([f for f in os.listdir(stage3_nc_dir) if f.endswith('.nc')])

    stage1_dict={ f.split('_')[1].split('.')[0]: os.path.join(stage1_nc_dir,f) for f in stage1_files}
    stage2_dict={ f.split('_')[1].split('.')[0]: os.path.join(stage2_nc_dir,f) for f in stage2_files}
    stage3_dict={ f.split('_')[1].split('.')[0]: os.path.join(stage3_nc_dir,f) for f in stage3_files}

    common_times= sorted(set(stage1_dict.keys()) & set(stage2_dict.keys()) & set(stage3_dict.keys()))
    print(f"共通の時間数: {len(common_times)}")

    class_colors = CFG["VISUALIZATION"]["class_colors"]
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds=np.arange(len(class_colors)+1)-0.5
    norm=mcolors.BoundaryNorm(bounds,cmap.N)

    pressure_vmin=CFG["VISUALIZATION"]["pressure_vmin"]
    pressure_vmax=CFG["VISUALIZATION"]["pressure_vmax"]
    pressure_levels=np.linspace(pressure_vmin,pressure_vmax,CFG["VISUALIZATION"]["pressure_levels"])
    pressure_norm=mcolors.Normalize(vmin=pressure_vmin,vmax=pressure_vmax)
    cmap_pressure=plt.get_cmap('RdBu_r')

    nc_gsm_alt=nc_gsm_dir

    inputs=[]
    for t_str in common_times:
        inputs.append((
            t_str,
            stage1_dict[t_str],
            stage2_dict[t_str],
            stage3_dict[t_str],
            original_nc_dir,
            nc_gsm_alt,
            output_dir,
            class_colors,
            cmap,
            norm,
            pressure_levels,
            pressure_norm,
            cmap_pressure
        ))
    print("キャッシュ・地図データ生成のため、最初の1件のみシリアル処理します。")
    process_single_time(inputs[0])
    num_processes=max(1, multiprocessing.cpu_count()//CFG["VISUALIZATION"]["parallel_factor"])
    print(f"{num_processes}個のプロセスで並列処理を開始します。")
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_single_time,inputs),total=len(inputs),desc='可視化処理中'))
    print("可視化処理が完了しました。")

def process_single_time(args):
    """
    概要:
        1つの時刻に対し、Stage1/2/3/GT を地図上で可視化し、比較PNGを保存する。

    入力:
        - args (tuple): 内部で展開される描画に必要なパラメータ一式

    処理:
        - NetCDF 読み取り、クラスマップ作成、GSM から気圧偏差を計算
        - cartopy で背景地図、contour/contourf、pcolormesh を重ね描き

    出力:
        - PNG ファイル（存在済ならスキップ）
    """
    import gc
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import numpy as np
    import pandas as pd
    import matplotlib.colors as mcolors

    (time_str,
     stage1_nc_path,
     stage2_nc_path,
     stage3_nc_path,
     original_nc_dir,
     nc_gsm_alt,
     output_dir,
     class_colors,
     cmap,
     norm,
     pressure_levels,
     pressure_norm,
     cmap_pressure) = args

    output_filename=f"comparison_{time_str}.png"
    output_path=os.path.join(output_dir,output_filename)

    if os.path.exists(output_path):
        print(f"{output_filename} は既に存在します。スキップします。")
        return

    ds_s1 = xr.open_dataset(stage1_nc_path)
    probs_s1=ds_s1['probabilities'].isel(time=0).values
    class_map_s1=np.argmax(probs_s1,axis=-1)
    lat=ds_s1['lat'].values
    lon=ds_s1['lon'].values
    ds_s1.close()
    del ds_s1, probs_s1
    gc.collect()

    ds_s2 = xr.open_dataset(stage2_nc_path)
    probs_s2=ds_s2['probabilities'].isel(time=0).values
    class_map_s2=np.argmax(probs_s2,axis=-1)
    ds_s2.close()
    del ds_s2, probs_s2
    gc.collect()

    ds_s3= xr.open_dataset(stage3_nc_path)
    class_map_s3= ds_s3['class_map'].isel(time=0).values
    ds_s3.close()
    del ds_s3
    gc.collect()

    month_str=time_str[:6]
    original_file=os.path.join(original_nc_dir,f"{month_str}.nc")
    if not os.path.exists(original_file):
        print(f"元の前線データが見つかりません: {original_file}")
        return
    ds_orig=xr.open_dataset(original_file)

    time_dt=pd.to_datetime(time_str,format='%Y%m%d%H%M')
    if time_dt in ds_orig['time']:
        orig_data=ds_orig.sel(time=time_dt)
    else:
        time_diff=np.abs(ds_orig['time']-time_dt)
        min_time_diff=time_diff.min()
        if min_time_diff<=np.timedelta64(3,'h'):
            nearest_time= ds_orig['time'].values[ time_diff.argmin() ]
            orig_data= ds_orig.sel(time=nearest_time)
            print(f"時間が一致しないため、最も近い時間 {nearest_time} を使用します。")
        else:
            print(f"時間 {time_str} が元のデータに存在しません: {original_file}")
            ds_orig.close()
            return

    class_map_orig = np.zeros((len(lat),len(lon)),dtype=np.int64)
    var_names = {
        1:'warm',
        2:'cold',
        3:'stationary',
        4:'occluded',
        5:'warm_cold'
    }
    for cid,varn in var_names.items():
        if varn in orig_data.data_vars:
            mask=orig_data[varn].values
            class_map_orig[mask==1]=cid
    ds_orig.close()
    del ds_orig, orig_data
    gc.collect()
    lowcenter_nc7_dir = nc_gsm_alt
    nc7_file = os.path.join(lowcenter_nc7_dir, f"gsm{month_str}.nc")
    low_mask = None
    low_center_exists = False
    if os.path.exists(nc7_file):
        ds_nc7 = xr.open_dataset(nc7_file)
        lowcenter_time_idx = None
        try:
            ds_times = pd.to_datetime(ds_nc7['time'].values)
            if time_dt in ds_times:
                lowcenter_time_idx = int(np.where(ds_times==time_dt)[0][0])
            else:
                timediffs = np.abs(ds_times - time_dt)
                minidx = timediffs.argmin()
                if timediffs[minidx] <= pd.Timedelta(hours=3):
                    lowcenter_time_idx = minidx
                    print(f"[低気圧中心] 時間が一致せず,最も近い {ds_nc7['time'].values[minidx]} を使用")
                else:
                    print(f"[低気圧中心] 時間不一致、一致するデータなし: {time_str}")
            if lowcenter_time_idx is not None:
                lowcenter_arr = ds_nc7['surface_low_center'].isel(time=lowcenter_time_idx).values
                low_mask = (lowcenter_arr == 1)
                low_center_exists = True
            ds_nc7.close()
        except Exception as e:
            print(f"[低気圧中心] 読み取り失敗 ({nc7_file}, {time_str}): {e}")
            if 'ds_nc7' in locals():
                ds_nc7.close()
    else:
        print(f"[低気圧中心] ファイルが見つかりません: {nc7_file} ({time_str})")

    gsm_file=os.path.join(nc_gsm_alt,f"gsm{month_str}.nc")
    if not os.path.exists(gsm_file):
        print(f"GSMデータが見つかりません: {gsm_file}")
        return
    ds_gsm=xr.open_dataset(gsm_file)
    if time_dt in ds_gsm['time']:
        gsm_dat=ds_gsm.sel(time=time_dt)
    else:
        time_diff=np.abs(ds_gsm['time']-time_dt)
        min_time_diff=time_diff.min()
        if min_time_diff<=np.timedelta64(3,'h'):
            nearest_time=ds_gsm['time'].values[time_diff.argmin()]
            gsm_dat=ds_gsm.sel(time=nearest_time)
            print(f"GSM時間が一致しないため、最も近い時間 {nearest_time} を使用します。")
        else:
            print(f"時間 {time_str} がGSMデータに存在しません: {gsm_file}")
            ds_gsm.close()
            return
    if 'surface_prmsl' in gsm_dat:
        surface_prmsl = gsm_dat['surface_prmsl'].values
    else:
        print(f"'surface_prmsl' 変数が存在しません: {gsm_file}")
        ds_gsm.close()
        return
    ds_gsm.close()
    del ds_gsm, gsm_dat
    gc.collect()

    area_mean=np.nanmean(surface_prmsl)
    pressure_dev=surface_prmsl-area_mean

    lon_grid,lat_grid=np.meshgrid(lon,lat)

    fig=plt.figure(figsize=(24,6))
    from matplotlib import gridspec
    gs=gridspec.GridSpec(1,5,width_ratios=[1,1,1,1,0.05],wspace=0.1)

    ax0=plt.subplot(gs[0],projection=ccrs.PlateCarree())
    ax1=plt.subplot(gs[1],projection=ccrs.PlateCarree())
    ax2=plt.subplot(gs[2],projection=ccrs.PlateCarree())
    ax3=plt.subplot(gs[3],projection=ccrs.PlateCarree())
    axes=[ax0,ax1,ax2,ax3]

    extent=[lon.min(),lon.max(),lat.min(),lat.max()]

    for ax in axes:
        ax.set_extent(extent,crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),edgecolor='black')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'),linestyle=':')
        ax.add_feature(cfeature.LAKES.with_scale('10m'),alpha=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale('10m'))
        gl=ax.gridlines(draw_labels=True, linewidth=0.5,color='gray',linestyle='--')
        gl.top_labels=False
        gl.right_labels=False
        ax.tick_params(labelsize=8)

    for ax in axes:
        cf=ax.contourf(
            lon_grid,lat_grid,pressure_dev,
            levels=pressure_levels,cmap=cmap_pressure,extend='both',
            norm=pressure_norm,transform=ccrs.PlateCarree(),zorder=0
        )
        cs=ax.contour(
            lon_grid,lat_grid,pressure_dev,
            levels=pressure_levels,colors='black',linestyles='--',linewidths=1.5,
            transform=ccrs.PlateCarree(),zorder=1
        )

    im1=ax0.pcolormesh(
        lon_grid,lat_grid,class_map_s1,
        cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),
        alpha=0.6,zorder=2
    )
    ax0.set_title(f'Stage1 結果\n{time_str}')

    im2=ax1.pcolormesh(
        lon_grid,lat_grid,class_map_s2,
        cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),
        alpha=0.6,zorder=2
    )
    ax1.set_title(f'Stage2 結果\n{time_str}')

    im3=ax2.pcolormesh(
        lon_grid,lat_grid,class_map_s3,
        cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),
        alpha=0.6,zorder=2
    )
    ax2.set_title(f'Stage3 結果（スケルトン化）\n{time_str}')

    im4=ax3.pcolormesh(
        lon_grid,lat_grid,class_map_orig,
        cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),
        alpha=0.6,zorder=2
    )
    ax3.set_title(f'元の前線データ\n{time_str}')

    if low_center_exists and (low_mask is not None) and (low_mask.shape == (lat.size, lon.size)):
        y_idx, x_idx = np.where(low_mask)
        low_lats = lat[y_idx]
        low_lons = lon[x_idx]

        for ax in axes:
            ax.plot(low_lons, low_lats, 'rx', markersize=8, markeredgewidth=2, zorder=6, label="低気圧中心")
    else:
        print(f"低気圧中心データ取得不可: {nc7_file}, {time_str}")

    cax=plt.subplot(gs[4])
    sm=plt.cm.ScalarMappable(cmap=cmap_pressure,norm=pressure_norm)
    sm.set_array([])
    cbar=plt.colorbar(sm,cax=cax,orientation='vertical')
    cbar.set_label('海面更正気圧の偏差 (hPa)')

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(output_path,dpi=300,bbox_inches='tight')
    plt.close()
    del class_map_s1,class_map_s2,class_map_s3,class_map_orig, lon_grid,lat_grid,pressure_dev
    gc.collect()

def run_visualization():
    """
    概要:
        Stage1/2/3 の結果をまとめて可視化し、PNG を出力する。

    入力:
        なし

    処理:
        - visualize_results を呼び出し
        - GPUメモリ/キャッシュのクリーンアップ

    出力:
        - なし
    """
    print_memory_usage("Start Visualization")
    vis_start = time.time()

    visualize_start = time.time()
    visualize_results(
        stage1_nc_dir=stage1_out_dir,
        stage2_nc_dir=stage2_out_dir,
        stage3_nc_dir=stage3_out_dir,
        original_nc_dir=nc_0p5_dir,
        output_dir=output_visual_dir
    )
    visualize_end = time.time()
    print(f"[Visualization] 結果の可視化処理時間: {format_time(visualize_end - visualize_start)}")

    cleanup_start = time.time()
    torch.cuda.empty_cache()
    gc.collect()
    cleanup_end = time.time()
    print(f"[Visualization] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    vis_end = time.time()
    print(f"[Visualization] 全体の実行時間: {format_time(vis_end - vis_start)}")
    print_memory_usage("After Visualization")

def compute_metrics(y_true, y_pred, labels):
    """
    概要:
        予測と正解から Accuracy, Macro Precision/Recall/F1, Cohen's Kappa を計算する。

    入力:
        - y_true (np.ndarray): 1次元の正解ラベル
        - y_pred (np.ndarray): 1次元の予測ラベル
        - labels (List[int]): 評価に含めるラベル集合

    処理:
        - precision_recall_fscore_support を macro 平均で利用
        - accuracy と kappa を算出

    出力:
        - (acc, macro_prec, macro_rec, macro_f1, kappa): いずれも数値（%はすでに換算済）
    """
    acc = np.mean(y_true == y_pred) * 100
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_prec *= 100
    macro_rec *= 100
    macro_f1  *= 100
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, macro_prec, macro_rec, macro_f1, kappa

def run_evaluation():
    """
    概要:
        2023年の Stage1/2/3 出力を対象に、共通時刻での総合評価図/ログを生成する。

    入力:
        なし（CFG のパスを参照）

    処理:
        - Stage1/2/3 の NetCDF を読み、argmax/クラスマップを取得
        - 元データから GT を作成（±3h 許容で最近傍許容）
        - 混同行列/各種メトリクス/比率やRMSEなどを計算
        - 図 (evaluation_summary.png) と詳細ログ (evaluation_summary.log) を書き出し

    出力:
        - なし（ファイル出力）
    """
    print("[Evaluation] Start evaluation for 2023 data (6 classes).")

    ratio_buf_s1 = {c:[] for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_buf_s2 = {c:[] for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_buf_s3 = {c:[] for c in range(1, CFG["STAGE1"]["num_classes"])}
    stage1_files = sorted([f for f in os.listdir(stage1_out_dir) if f.startswith("prob_2023") and f.endswith('.nc')])
    stage2_files = sorted([f for f in os.listdir(stage2_out_dir) if f.startswith("refined_2023") and f.endswith('.nc')])
    stage3_files = sorted([f for f in os.listdir(stage3_out_dir) if f.startswith("skeleton_2023") and f.endswith('.nc')])
    
    stage1_dict = {}
    stage2_dict = {}
    stage3_dict = {}
    
    def extract_time(fname, prefix):
        return fname.replace(prefix, "").replace(".nc", "")
    
    for f in stage1_files:
        token = extract_time(f, "prob_")
        stage1_dict[token] = os.path.join(stage1_out_dir, f)
    for f in stage2_files:
        token = extract_time(f, "refined_")
        stage2_dict[token] = os.path.join(stage2_out_dir, f)
    for f in stage3_files:
        token = extract_time(f, "skeleton_")
        stage3_dict[token] = os.path.join(stage3_out_dir, f)
    
    common_keys = sorted(set(stage1_dict.keys()) & set(stage2_dict.keys()) & set(stage3_dict.keys()))
    if len(common_keys) == 0:
        print("[Evaluation] Not found any 2023 common times among stage1/2/3.")
        return
    
    stage1_pred_list = []
    stage2_pred_list = []
    stage3_pred_list = []
    gt_list = []
    
    for key in common_keys:
        ds1 = xr.open_dataset(stage1_dict[key])
        probs_s1 = ds1['probabilities'].isel(time=0).values
        ds1.close()
        pred_s1 = np.argmax(probs_s1, axis=-1)
        ds2 = xr.open_dataset(stage2_dict[key])
        probs_s2 = ds2['probabilities'].isel(time=0).values
        ds2.close()
        pred_s2 = np.argmax(probs_s2, axis=-1)
        ds3 = xr.open_dataset(stage3_dict[key])
        pred_s3 = ds3['class_map'].isel(time=0).values
        ds3.close()
        stage1_pred_list.append(pred_s1)
        stage2_pred_list.append(pred_s2)
        stage3_pred_list.append(pred_s3)
        month_str = key[:6]
        gtf = os.path.join(nc_0p5_dir, f"{month_str}.nc")
        if not os.path.exists(gtf):
            print(f"GroundTruth file not found: {gtf}")
            gt_list.append(np.zeros_like(pred_s1))
            continue
        ds_gt = xr.open_dataset(gtf)
        t_dt = pd.to_datetime(key, format='%Y%m%d%H%M')
        if t_dt in ds_gt['time']:
            front_data = ds_gt.sel(time=t_dt).to_array().values
        else:
            diff_ = np.abs(ds_gt['time'].values - np.datetime64(t_dt))
            idx_ = diff_.argmin()
            if diff_[idx_] <= np.timedelta64(3, 'h'):
                front_data = ds_gt.sel(time=ds_gt['time'][idx_]).to_array().values
            else:
                print(f"No GT time close for {key} in {gtf}")
                ds_gt.close()
                gt_list.append(np.zeros_like(pred_s1))
                continue
        ds_gt.close()
        gt_map = np.zeros_like(pred_s1)
        for c in range(5):
            mask = (front_data[c, :, :] == 1)
            gt_map[mask] = c+1
        gt_list.append(gt_map)
        # Update pixel-count ratio buffers per class (pred/GT), avoiding division by zero
        for c in range(1, CFG["STAGE1"]["num_classes"]):
            gt_cnt = int((gt_map == c).sum())
            if gt_cnt > 0:
                ratio_buf_s1[c].append(float((pred_s1 == c).sum() / gt_cnt))
                ratio_buf_s2[c].append(float((pred_s2 == c).sum() / gt_cnt))
                ratio_buf_s3[c].append(float((pred_s3 == c).sum() / gt_cnt))
        del ds_gt, front_data
    
    stage1_all = np.concatenate([arr.flatten() for arr in stage1_pred_list], axis=0)
    stage2_all = np.concatenate([arr.flatten() for arr in stage2_pred_list], axis=0)
    stage3_all = np.concatenate([arr.flatten() for arr in stage3_pred_list], axis=0)
    gt_all = np.concatenate([arr.flatten() for arr in gt_list], axis=0)
    
    label_all = list(range(CFG["STAGE1"]["num_classes"]))
    cm_s1 = confusion_matrix(gt_all, stage1_all, labels=label_all)
    cm_s2 = confusion_matrix(gt_all, stage2_all, labels=label_all)
    cm_s3 = confusion_matrix(gt_all, stage3_all, labels=label_all)
    
    total_cnt = len(gt_list)
    count_s1 = sum(1 for i in range(total_cnt) if set(np.unique(stage1_pred_list[i])) == set(np.unique(gt_list[i])))
    count_s2 = sum(1 for i in range(total_cnt) if set(np.unique(stage2_pred_list[i])) == set(np.unique(gt_list[i])))
    count_s3 = sum(1 for i in range(total_cnt) if set(np.unique(stage3_pred_list[i])) == set(np.unique(gt_list[i])))
    ratio_s1 = (count_s1 / total_cnt * 100) if total_cnt > 0 else 0
    ratio_s2 = (count_s2 / total_cnt * 100) if total_cnt > 0 else 0
    ratio_s3 = (count_s3 / total_cnt * 100) if total_cnt > 0 else 0
    
    def calc_stage_metrics(y_true, y_pred, labels):
        return compute_metrics(y_true, y_pred, labels)

    acc1, mp1, mr1, mf1, kappa1 = calc_stage_metrics(gt_all, stage1_all, label_all)
    acc2, mp2, mr2, mf2, kappa2 = calc_stage_metrics(gt_all, stage2_all, label_all)
    acc3, mp3, mr3, mf3, kappa3 = calc_stage_metrics(gt_all, stage3_all, label_all)
    
    df_metrics_full = pd.DataFrame({
        "Accuracy (%)": [acc1, acc2, acc3],
        "Macro Precision (%)": [mp1, mp2, mp3],
        "Macro Recall (%)": [mr1, mr2, mr3],
        "Macro F1 (%)": [mf1, mf2, mf3],
        "Cohen Kappa": [kappa1, kappa2, kappa3]
    }, index=["Stage1", "Stage2", "Stage3"])
    
    filter_mask = (gt_all != 0)
    stage1_all_f = stage1_all[filter_mask]
    stage2_all_f = stage2_all[filter_mask]
    stage3_all_f = stage3_all[filter_mask]
    gt_all_f = gt_all[filter_mask]
    labels_5 = list(range(1, CFG["STAGE1"]["num_classes"]))
    
    if len(gt_all_f) > 0:
        acc1_f, mp1_f, mr1_f, mf1_f, kappa1_f = calc_stage_metrics(gt_all_f, stage1_all_f, labels_5)
        acc2_f, mp2_f, mr2_f, mf2_f, kappa2_f = calc_stage_metrics(gt_all_f, stage2_all_f, labels_5)
        acc3_f, mp3_f, mr3_f, mf3_f, kappa3_f = calc_stage_metrics(gt_all_f, stage3_all_f, labels_5)
    else:
        acc1_f = mp1_f = mr1_f = mf1_f = kappa1_f = 0
        acc2_f = mp2_f = mr2_f = mf2_f = kappa2_f = 0
        acc3_f = mp3_f = mr3_f = mf3_f = kappa3_f = 0
    
    df_metrics_filtered = pd.DataFrame({
        "Accuracy (%)": [acc1_f, acc2_f, acc3_f],
        "Macro Precision (%)": [mp1_f, mp2_f, mp3_f],
        "Macro Recall (%)": [mr1_f, mr2_f, mr3_f],
        "Macro F1 (%)": [mf1_f, mf2_f, mf3_f],
        "Cohen Kappa": [kappa1_f, kappa2_f, kappa3_f]
    }, index=["Stage1", "Stage2", "Stage3"])
    
    ratio_s1_cls = {c: (float(np.mean(ratio_buf_s1[c])) if len(ratio_buf_s1[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_s2_cls = {c: (float(np.mean(ratio_buf_s2[c])) if len(ratio_buf_s2[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_s3_cls = {c: (float(np.mean(ratio_buf_s3[c])) if len(ratio_buf_s3[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}

    rmse_s1_cls, rmse_s2_cls, rmse_s3_cls = {}, {}, {}
    for c in range(1, CFG["STAGE1"]["num_classes"]):
        gt_bin        = (gt_all       == c).astype(np.float32)
        pred_s1_bin   = (stage1_all   == c).astype(np.float32)
        pred_s2_bin   = (stage2_all   == c).astype(np.float32)
        pred_s3_bin   = (stage3_all   == c).astype(np.float32)
        rmse_s1_cls[c]= np.sqrt(np.mean((pred_s1_bin - gt_bin) ** 2))
        rmse_s2_cls[c]= np.sqrt(np.mean((pred_s2_bin - gt_bin) ** 2))
        rmse_s3_cls[c]= np.sqrt(np.mean((pred_s3_bin - gt_bin) ** 2))

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.4)
    
    def normalize_cm(cm):
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        return cm_norm
        
    cm_s1_norm = normalize_cm(cm_s1)
    cm_s2_norm = normalize_cm(cm_s2)
    cm_s3_norm = normalize_cm(cm_s3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm_s1_norm, annot=cm_s1, fmt='d', cmap='Blues',
                xticklabels=label_all, yticklabels=label_all, ax=ax1,
                vmin=0, vmax=1.0)
    ax1.set_title("Stage1 Confusion Matrix (All Classes)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(cm_s2_norm, annot=cm_s2, fmt='d', cmap='Blues',
                xticklabels=label_all, yticklabels=label_all, ax=ax2,
                vmin=0, vmax=1.0)
    ax2.set_title("Stage2 Confusion Matrix (All Classes)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(cm_s3_norm, annot=cm_s3, fmt='d', cmap='Blues',
                xticklabels=label_all, yticklabels=label_all, ax=ax3,
                vmin=0, vmax=1.0)
    ax3.set_title("Stage3 Confusion Matrix (All Classes)")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax_table_full = fig.add_subplot(gs[1, :])
    ax_table_full.axis('off')
    table_data_full = df_metrics_full.round(2).values
    row_labels_full = df_metrics_full.index.tolist()
    col_labels_full = df_metrics_full.columns.tolist()
    table_full = ax_table_full.table(cellText=table_data_full,
                                     rowLabels=row_labels_full,
                                     colLabels=col_labels_full,
                                     cellLoc='center',
                                     loc='center')
    table_full.auto_set_font_size(False)
    table_full.set_fontsize(10)
    ax_table_full.set_title("Evaluation Metrics (All Classes)", fontweight="bold", pad=20)
    ax_table_filtered = fig.add_subplot(gs[2, :])
    ax_table_filtered.axis('off')
    table_data_filtered = df_metrics_filtered.round(2).values
    row_labels_filtered = df_metrics_filtered.index.tolist()
    col_labels_filtered = df_metrics_filtered.columns.tolist()
    table_filtered = ax_table_filtered.table(cellText=table_data_filtered,
                                             rowLabels=row_labels_filtered,
                                             colLabels=col_labels_filtered,
                                             cellLoc='center',
                                             loc='center')
    table_filtered.auto_set_font_size(False)
    table_filtered.set_fontsize(10)
    ax_table_filtered.set_title("Evaluation Metrics (Front Only: Classes 1-5)", fontweight="bold", pad=20)
    ratio_text  = "Pixel-count ratio (pred/GT) 〈mean, cls1-〉\n" + \
                  "  S1: " + ", ".join([f"C{c}:{ratio_s1_cls[c]:.2f}" for c in range(1, CFG['STAGE1']['num_classes'])]) + "\n" + \
                  "  S2: " + ", ".join([f"C{c}:{ratio_s2_cls[c]:.2f}" for c in range(1, CFG['STAGE1']['num_classes'])]) + "\n" + \
                  "  S3: " + ", ".join([f"C{c}:{ratio_s3_cls[c]:.2f}" for c in range(1, CFG['STAGE1']['num_classes'])])
    rmse_text   = "RMSE (cls1-)\n" + \
                  "  S1: " + ", ".join([f"C{c}:{rmse_s1_cls[c]:.3f}" for c in range(1, CFG['STAGE1']['num_classes'])]) + "\n" + \
                  "  S2: " + ", ".join([f"C{c}:{rmse_s2_cls[c]:.3f}" for c in range(1, CFG['STAGE1']['num_classes'])]) + "\n" + \
                  "  S3: " + ", ".join([f"C{c}:{rmse_s3_cls[c]:.3f}" for c in range(1, CFG['STAGE1']['num_classes'])])

    summary_text = (f"Time-wise Presence-set match  (%): "
                    f"S1 {ratio_s1:.2f}, S2 {ratio_s2:.2f}, S3 {ratio_s3:.2f}\n"
                    f"{ratio_text}\n{rmse_text}")
    fig.text(0.5, 0.005, summary_text, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_fig = "v1_result/evaluation_summary.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    # Write detailed text-based summary log
    try:
        v1_root = os.path.dirname(CFG["PATHS"]["output_visual_dir"])
        os.makedirs(v1_root, exist_ok=True)
        log_path = os.path.join(v1_root, "evaluation_summary.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== Evaluation Summary ===\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write(f"Common timestamps evaluated: {len(common_keys)}\n\n")

            f.write("Time-wise Presence-set match (%)\n")
            f.write(f"  Stage1: {ratio_s1:.2f}\n  Stage2: {ratio_s2:.2f}\n  Stage3: {ratio_s3:.2f}\n\n")

            f.write("Pixel-count ratio (pred/GT) per class (1-5)\n")
            for c in range(1, CFG["STAGE1"]["num_classes"]):
                f.write(f"  Class {c}: S1={ratio_s1_cls[c]:.4f}, S2={ratio_s2_cls[c]:.4f}, S3={ratio_s3_cls[c]:.4f}\n")
            f.write("\n")

            f.write("RMSE per class (1-5)\n")
            for c in range(1, CFG["STAGE1"]["num_classes"]):
                f.write(f"  Class {c}: S1={rmse_s1_cls[c]:.4f}, S2={rmse_s2_cls[c]:.4f}, S3={rmse_s3_cls[c]:.4f}\n")
            f.write("\n")

            f.write("Confusion Matrix (All classes, raw counts)\n")
            f.write("[Stage1]\n"); f.write(pd.DataFrame(cm_s1).to_string(index=True, header=True)); f.write("\n")
            f.write("[Stage2]\n"); f.write(pd.DataFrame(cm_s2).to_string(index=True, header=True)); f.write("\n")
            f.write("[Stage3]\n"); f.write(pd.DataFrame(cm_s3).to_string(index=True, header=True)); f.write("\n\n")

            f.write("Evaluation Metrics (All Classes)\n")
            f.write(df_metrics_full.round(4).to_string()); f.write("\n\n")

            f.write("Evaluation Metrics (Front Only: Classes 1-5)\n")
            f.write(df_metrics_filtered.round(4).to_string()); f.write("\n\n")

            # Append loss history if available
            f.write("Loss History (from training)\n")
            try:
                s1_csv = os.path.join(model_s1_save_dir, "loss_history.csv")
                if os.path.exists(s1_csv):
                    df1 = pd.read_csv(s1_csv)
                    f.write("[Stage1]\n")
                    f.write(df1.to_string(index=False)); f.write("\n")
                else:
                    f.write("[Stage1] loss_history.csv not found.\n")
            except Exception as e:
                f.write(f"[Stage1] Error reading loss history: {e}\n")
            try:
                s2_csv = os.path.join(model_s2_save_dir, "loss_history.csv")
                if os.path.exists(s2_csv):
                    df2 = pd.read_csv(s2_csv)
                    f.write("[Stage2]\n")
                    f.write(df2.to_string(index=False)); f.write("\n")
                else:
                    f.write("[Stage2] loss_history.csv not found.\n")
            except Exception as e:
                f.write(f"[Stage2] Error reading loss history: {e}\n")

            f.write("\nArtifacts\n")
            f.write(f"  Figure: {out_fig}\n")
            f.write(f"  Stage1 loss curve: {os.path.join(model_s1_save_dir, 'loss_curve.png')}\n")
            f.write(f"  Stage2 loss curve: {os.path.join(model_s2_save_dir, 'loss_curve.png')}\n")
        print(f"[Evaluation] Summary log -> {log_path}")
    except Exception as e:
        print(f"[Evaluation] Failed to write evaluation_summary.log: {e}")

    print(f"[Evaluation] Done. Figure -> {out_fig}")

def smooth_polyline(points, window_size=3):
    """
    概要:
        ポリラインの座標列を移動平均で平滑化する。

    入力:
        - points (List[Tuple[float,float]]): (x,y) の点列
        - window_size (int): 平滑化窓幅（奇数推奨）

    処理:
        - 各点を中心に前後 half 幅で平均を取りスムージング

    出力:
        - smoothed (List[Tuple[float,float]]): 平滑化後の点列
    """
    if len(points) <= window_size:
        return points
    smoothed = []
    half = window_size // 2
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        avg_x = sum(p[0] for p in points[start:end]) / (end - start)
        avg_y = sum(p[1] for p in points[start:end]) / (end - start)
        smoothed.append((avg_x, avg_y))
    return smoothed

def extract_polylines_using_skan(class_map, lat, lon):
    """
    概要:
        スケルトン抽出ライブラリ(skan)を用いて、各クラス(1..5)の細線化ポリラインを抽出する。

    入力:
        - class_map (np.ndarray[int]): (H,W) クラスIDマップ（0=背景, 1..5=前線）
        - lat (np.ndarray): 緯度配列（H）
        - lon (np.ndarray): 経度配列（W）

    処理:
        - クラスごとに2値マスクを作り、Skeleton からパス座標を取得
        - 小領域でスケルトンが失敗する場合は、重心点を2点で疑似ポリライン化

    出力:
        - polylines (List[Tuple[int, List[Tuple[float,float]]]]): (クラスID, [(lon,lat), ...]) の配列
    """
    polylines = []
    for c in range(1, 6):
        mask = (class_map == c).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        try:
            skel = Skeleton(mask)
            for i in range(skel.n_paths):
                coords = skel.path_coordinates(i)
                coords_int = np.rint(coords).astype(int)
                points_geo = [(lon[::-1][col], lat[::-1][row]) for (row, col) in coords_int]
                polylines.append((c, points_geo))
        except ValueError as e:
            if "index pointer size" in str(e):
                ys, xs = np.nonzero(mask)
                if len(ys) > 0:
                    centroid_row = int(np.round(np.mean(ys)))
                    centroid_col = int(np.round(np.mean(xs)))
                    pt = (lon[::-1][centroid_col], lat[::-1][centroid_row])
                    polylines.append((c, [pt, pt]))
                    print(f"[Info] 小領域のクラス {c} に対して、重心 {pt} を用いポリラインを作成しました。")
                else:
                    print(f"[Warning] クラス {c} のmaskが空です。")
            else:
                print("skan.Skeleton エラー（クラス {}）: {}".format(c, e))
    return polylines

def save_polylines_as_svg(polylines, viewBox, output_path, smoothing_window=3):
    """
    概要:
        ポリライン（経緯度座標）を SVG の polyline として可視化し保存する。

    入力:
        - polylines (List[Tuple[int, List[Tuple[float,float]]]]): (クラスID, 点列)
        - viewBox (Tuple[float,float,float,float]): (min_lon, min_lat, width, height)
        - output_path (str): 保存先 SVG パス
        - smoothing_window (int): 出力前に適用する平滑化窓幅

    処理:
        - 背景グリッド/目盛りを描画
        - クラス色で polyline を描画（必要なら平滑化）
        - SVG をファイル保存

    出力:
        - なし（ファイル出力）
    """
    class_colors = {
        1: "#FF0000",   # 温暖前線（赤）
        2: "#0000FF",   # 寒冷前線（青）
        3: "#008015",   # 停滞前線（緑）
        4: "#800080",   # 閉塞前線（紫）
        5: "#FFA500",   # 前線の繋ぎ目（橙）
    }
    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    min_lon, min_lat, width, height = viewBox
    svg_lines.append('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="{0:.4f} {1:.4f} {2:.4f} {3:.4f}">'.format(min_lon, min_lat, width, height))
    lon_interval = width / 10.0 if width != 0 else 1
    lat_interval = height / 10.0 if height != 0 else 1
    for lx in np.arange(min_lon, min_lon + width + lon_interval, lon_interval):
        svg_lines.append('<line x1="{0:.4f}" y1="{1:.4f}" x2="{0:.4f}" y2="{2:.4f}" stroke="#CCCCCC" stroke-width="0.1" />'.format(lx, min_lat, min_lat+height))
        svg_lines.append('<text x="{0:.4f}" y="{1:.4f}" font-size="0.5" fill="#000000" text-anchor="middle" dy="0.5">{0:.2f}</text>'.format(lx, min_lat+height))
    for ly in np.arange(min_lat, min_lat + height + lat_interval, lat_interval):
        svg_lines.append('<line x1="{0:.4f}" y1="{1:.4f}" x2="{2:.4f}" y2="{1:.4f}" stroke="#CCCCCC" stroke-width="0.1" />'.format(min_lon, ly, min_lon+width))
        svg_lines.append('<text x="{0:.4f}" y="{1:.4f}" font-size="0.5" fill="#000000" text-anchor="start" dx="0.2" dy="0.3">{1:.2f}</text>'.format(min_lon, ly))
    
    for poly in polylines:
        cls, points = poly
        if smoothing_window > 1:
            points = smooth_polyline(points, window_size=smoothing_window)
        points_str = " ".join("{0:.4f},{1:.4f}".format(pt[0], pt[1]) for pt in points)
        color = class_colors.get(cls, "#000000")
        svg_lines.append('<polyline fill="none" stroke="{0}" stroke-width="0.5" points="{1}" />'.format(color, points_str))
    
    svg_lines.append('</svg>')
    with open(output_path, "w", encoding="utf-8") as f:
        for line in svg_lines:
            f.write(line + "\n")
    print("SVG saved:", output_path)

def evaluate_stage4(stage3_nc_dir, output_svg_dir):
    """
    概要:
        Stage3 のスケルトン化クラスマップをポリラインに変換し、SVG として出力する。

    入力:
        - stage3_nc_dir (str): skeleton_*.nc のディレクトリ
        - output_svg_dir (str): SVG 出力先

    処理:
        - 各ファイルを読み、extract_polylines_using_skan でポリライン抽出
        - viewBox を経緯度の最小/最大から算出し、SVG を保存

    出力:
        - なし（SVG ファイル）
    """
    os.makedirs(output_svg_dir, exist_ok=True)
    skeleton_files = sorted([f for f in os.listdir(stage3_nc_dir) if f.startswith("skeleton_") and f.endswith(".nc")])
    
    for f in skeleton_files:
        nc_path = os.path.join(stage3_nc_dir, f)
        ds = xr.open_dataset(nc_path)
        time_val = ds["time"].values[0]
        time_dt = pd.to_datetime(time_val)
        date_str = time_dt.strftime("%Y%m%d%H%M")
        class_map = ds["class_map"].isel(time=0).values
        lat = ds["lat"].values
        lon = ds["lon"].values
        ds.close()
        gc.collect()
        polylines = extract_polylines_using_skan(class_map, lat, lon)
        lon_fixed = lon[::-1]
        lat_fixed = lat[::-1]
        min_lon_val = float(np.min(lon_fixed))
        max_lon_val = float(np.max(lon_fixed))
        min_lat_val = float(np.min(lat_fixed))
        max_lat_val = float(np.max(lat_fixed))
        viewBox = (min_lon_val, min_lat_val, max_lon_val - min_lon_val, max_lat_val - min_lat_val)
        
        output_path = os.path.join(output_svg_dir, "skeleton_{}.svg".format(date_str))
        save_polylines_as_svg(polylines, viewBox, output_path, smoothing_window=3)
        
        del ds, class_map, lat, lon, polylines
        gc.collect()

def run_stage4():
    """
    概要:
        Stage4（SVG 出力）を実行。Stage3 の骨格結果をベクタ形式に変換する。

    入力:
        なし

    処理:
        - evaluate_stage4 を呼び出し

    出力:
        - なし
    """
    stage3_nc_dir = stage3_out_dir
    output_svg_dir = CFG["PATHS"]["stage4_svg_dir"]
    evaluate_stage4(stage3_nc_dir, output_svg_dir)
    print("【Stage4 Improved】 SVG 出力処理が完了しました。")

for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = str(CFG["THREADS"])
torch.set_num_threads(CFG["THREADS"])
def _rss_mb():  return psutil.Process(os.getpid()).memory_info().rss/1024/1024
def _gpu_mb():  return torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
def _mem(tag):  print(f"[Mem] {tag:18s}  CPU:{_rss_mb():7.1f}MB  GPU:{_gpu_mb():7.1f}MB")
def _load_model(ckpt:str, device):
    """
    概要:
        学習済み Stage1 モデルをチェックポイントからロードし、推論モードで返す。

    入力:
        - ckpt (str): チェックポイントパス（model_final.pth など）
        - device (torch.device): 配置デバイス

    処理:
        - SwinUnetModel を構築し state_dict を読み込み
        - DataParallel 由来の "module." プレフィックスを除去
        - eval()/no-grad 用に requires_grad=False を設定

    出力:
        - net (nn.Module): ロード済みモデル（eval）
    """
    # Avoid self-import; directly construct the wrapper model
    net = SwinUnetModel(num_classes=CFG["STAGE1"]["num_classes"], in_chans=CFG["STAGE1"]["in_chans"], model_cfg=CFG["STAGE1"]["model"])
    obj = torch.load(ckpt, map_location="cpu")
    sd  = obj if isinstance(obj, OrderedDict) else obj["model_state_dict"]
    sd  = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k,v in sd.items())
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net
_gsm_base=None
def _gsm_vars():
    """
    概要:
        GSM NetCDF の最初のファイルから変数名リストを抽出（キャッシュ）する。

    入力:
        なし

    処理:
        - nc_gsm_dir/gsm*.nc の先頭ファイルを開き data_vars を列挙

    出力:
        - 変数名リスト (List[str])
    """
    global _gsm_base
    if _gsm_base is None:
        import xarray as xr, glob
        f=sorted(glob.glob(os.path.join(nc_gsm_dir,"gsm*.nc")))[0]
        _gsm_base=list(xr.open_dataset(f).data_vars)
    return _gsm_base
VAR_NAMES_93=[f"{v}_{t}" for t in("t-6h","t0","t+6h") for v in _gsm_vars()]
class OnlineMoments:
    """
    概要:
        特徴量ごとの |SHAP| 平均、符号付き平均、分散（からの標準偏差）を逐次更新で推定するユーティリティ。

    入力:
        - n_feat (int): 特徴量数（93）

    処理:
        - update(arr): (C,H,W) の SHAP を受け取り、チャネル平均を逐次統計に反映
        - ave_abs()/ave()/std(): 推定値を返す

    出力:
        - プロパティ様の各メソッドで数値配列を返却
    """
    def __init__(self, n_feat=93):
        self.n=0
        self.mu  = np.zeros(n_feat, dtype=np.float64)
        self.M2  = np.zeros(n_feat, dtype=np.float64)
        self.raw = np.zeros(n_feat, dtype=np.float64)

    def _to_CHW(self, arr:np.ndarray)->np.ndarray:
        """
        概要:
            入力配列を (C,H,W) に整形する（末尾がチャネルなら転置）。

        入力:
            - arr (np.ndarray or Tensor): 3次元の SHAP 配列

        処理:
            - Tensor の場合は numpy に変換
            - 形状を検査し (C,H,W) に転置/検証

        出力:
            - arr_chw (np.ndarray): (C,H,W)
        """
        arr = np.squeeze(arr) 
        if arr.ndim != 3:
            raise ValueError(f"Unexpected ndim: {arr.ndim}, shape={arr.shape}")
        if arr.shape[ -1 ] == self.mu.size and arr.shape[0] != self.mu.size:
            arr = np.transpose(arr, (2,0,1)) 
        if arr.shape[0] != self.mu.size:
            raise ValueError(f"Channel数が合いません: {arr.shape}")
        return arr

    def update(self, arr):
        """
        概要:
            SHAP 配列を受け取り、逐次統計（平均|SHAP|、平均、分散）を更新する。

        入力:
            - arr (np.ndarray|Tensor): (C,H,W) or (H,W,C) 形式

        処理:
            - チャネルごとに |.|平均、符号付き平均を算出
            - Welford 法に基づき分散用の M2 を更新

        出力:
            - なし（内部状態更新）
        """
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = self._to_CHW(arr)

        abs_m = np.abs(arr).mean(axis=(1,2))
        sig_m = arr.mean(axis=(1,2))        

        self.n += 1
        delta   = abs_m - self.mu
        self.mu += delta / self.n
        self.M2 += delta * (abs_m - self.mu)
        self.raw += sig_m
    def ave_abs(self): return self.mu
    def ave(self):     return self.raw / max(self.n,1)
    def std(self):     return np.sqrt(self.M2 / max(self.n-1,1))
def _save_summary(cls_tag, stats, X, S, out_dir):
    """
    概要:
        クラス別の SHAP 集計結果をCSV/PNGで保存する（beeswarm/bar/waterfall）。

    入力:
        - cls_tag (str): クラス名の表示用
        - stats (OnlineMoments): 逐次統計
        - X (np.ndarray): 入力のチャネル平均 (N,93)
        - S (np.ndarray): SHAP のチャネル平均 (N,93)
        - out_dir (str): 出力ディレクトリ

    処理:
        - CSV: AveAbs_SHAP / Ave_SHAP / Std_SHAP を変数名付きで出力
        - 3種類の SHAP 可視化を保存

    出力:
        - なし（ファイル出力）
    """
    os.makedirs(out_dir, exist_ok=True)
    df=pd.DataFrame({"variable":VAR_NAMES_93,
                     "AveAbs_SHAP":stats.ave_abs(),
                     "Ave_SHAP":stats.ave(),
                     "Std_SHAP":stats.std()})\
          .sort_values("AveAbs_SHAP",ascending=False)
    csv=os.path.join(out_dir,f"class{cls_tag}_summary.csv")
    df.to_csv(csv,index=False)

    plt.figure(); shap.summary_plot(S, X,
                                    feature_names=VAR_NAMES_93,
                                    show=False, plot_size=(9,4))
    plt.title(f"{cls_tag}  summary (beeswarm)"); plt.tight_layout()
    plt.savefig(csv.replace(".csv","_beeswarm.png"),dpi=200); plt.close()

    plt.figure(); shap.summary_plot(S, X,
                                    feature_names=VAR_NAMES_93,
                                    plot_type="bar", show=False)
    plt.title(f"{cls_tag}  mean(|SHAP|)"); plt.tight_layout()
    plt.savefig(csv.replace(".csv","_bar.png"),dpi=200); plt.close()

    expl=shap.Explanation(values=S[0],base_values=0,
                          data=X[0],feature_names=VAR_NAMES_93)
    plt.figure(); shap.plots.waterfall(expl,max_display=20,show=False)
    plt.title(f"{cls_tag}  waterfall (sample0)"); plt.tight_layout()
    plt.savefig(csv.replace(".csv","_waterfall.png"),dpi=200); plt.close()
def _pick_gpu(th=CFG["SHAP"]["free_mem_threshold_gb"]):
    """
    概要:
        空きメモリ量がしきい値以上の GPU を選ぶ。満たさなければ None。

    入力:
        - th (float): 必要な空きメモリ(GB)

    処理:
        - torch.cuda.mem_get_info で各GPUの空き容量を比較

    出力:
        - device_id (int|None)
    """
    if not torch.cuda.is_available(): return None
    best=-1; bid=None
    for i in range(torch.cuda.device_count()):
        free,_=torch.cuda.mem_get_info(i); free/=1024**3
        if free>best: best, bid = free, i
    return bid if best>=th else None
def _safe_shap(expl,x,ns=16):
    """
    概要:
        GPU OOM を避けるため、nsamples を半減しながら SHAP を再試行する。

    入力:
        - expl: shap.GradientExplainer
        - x (Tensor): 入力 (1,B,C,H,W) の想定に準じる
        - ns (int): 初期サンプル数

    処理:
        - OOM を検知したら ns を半分にして再試行、1未満で例外

    出力:
        - shap_values（Explainer に準じた戻り値）
    """
    while True:
        try:  return expl.shap_values(x,nsamples=ns)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            ns//=2
            if ns<1: raise
            print(f"[SHAP] OOM → nsamples={ns} で再試行"); torch.cuda.empty_cache()
def run_stage1_shap_evaluation_cpu(use_gpu=True,
                                   max_samples_per_class=500,
                                   out_root="./v1_result/shap_stage1"):
    """
    概要:
        Stage1 モデルに対する SHAP 解析を行い、クラス別に重要度統計と可視化を保存する。

    入力:
        - use_gpu (bool): GPU を使うか（空き容量が足りなければCPU）
        - max_samples_per_class (int): クラスごとの最大サンプル数
        - out_root (str): 出力ルートディレクトリ

    処理:
        - 2023年データからランダムにサンプルし、各クラスに出現するピクセルがあるサンプルのみ対象
        - shap.GradientExplainer によりチャネル毎の重要度を推定
        - OnlineMoments で逐次統計、_save_summary で保存

    出力:
        - なし（CSV/PNG 保存）
    """
    print("\n========== Stage-1 SHAP 解析 ==========")
    months=get_available_months(2023,1,2023,12)
    ds = FrontalDatasetStage1(months, nc_gsm_dir, nc_0p5_dir)
    idxs=list(range(len(ds))); random.shuffle(idxs)
    gid=_pick_gpu() if use_gpu else None
    device=torch.device(f"cuda:{gid}") if gid is not None else torch.device("cpu")
    print(f"使用デバイス : {device}")
    model=_load_model(os.path.join(model_s1_save_dir,"model_final.pth"),device)
    class Wrap(nn.Module):
        def __init__(self,net,cid): super().__init__(); self.net,self.cid=net,cid
        def forward(self,x): return self.net(x)[:,self.cid].mean((1,2),keepdim=True)
    bg,_ ,_=ds[idxs[0]]; bg=bg.unsqueeze(0).to(device)
    expl={c: shap.GradientExplainer(Wrap(model,c), data=bg) for c in range(1, num_classes_stage1)}
    stats={c:OnlineMoments() for c in range(1,6)}
    Xbuf =defaultdict(list); Sbuf=defaultdict(list)

    for idx in tqdm(idxs,desc="Compute SHAP"):
        x, y, _ = ds[idx]
        present=set(np.unique(y.numpy())) & set(range(1, num_classes_stage1))
        for c in present:
            if len(Xbuf[c])>=max_samples_per_class: continue
            xx=x.unsqueeze(0).to(device)
            sv=_safe_shap(expl[c],xx,ns=16) 
            val=sv[0] if isinstance(sv,list) else sv
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            val = np.squeeze(val)
            if val.shape[-1]==93 and val.shape[0]!=93:    
                val = np.transpose(val, (2,0,1))
            stats[c].update(val)                         
            Sbuf[c].append(val.mean((1,2)))              
            Xbuf[c].append(xx.cpu().numpy()[0].mean((1,2)))
        if all(len(Xbuf[k])>=max_samples_per_class for k in range(1, num_classes_stage1)):
            break
    cname={1:"WarmFront",2:"ColdFront",3:"Stationary",4:"Occluded",5:"Complex"}
    for c in range(1, CFG["STAGE1"]["num_classes"]):
        if Xbuf[c]:
            _save_summary(cname[c], stats[c],
                          np.vstack(Xbuf[c]), np.vstack(Sbuf[c]),
                          out_root)
    print("==========  SHAP 解析 完了 ==========\n")

def format_time(seconds):
    """
    概要:
        秒数を「X時間 Y分 Z.ZZ秒」の形式に整形して返す。

    入力:
        - seconds (float): 経過秒

    処理:
        - 3600/60で分解し、時間/分/秒を構成

    出力:
        - s (str): 整形済み文字列
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒"
    elif minutes > 0:
        return f"{int(minutes)}分 {seconds:.2f}秒"
    else:
        return f"{seconds:.2f}秒"

def main():
    """
    概要:
        パイプライン全体（Stage1→SHAP→Stage2→Stage3→可視化→評価→動画→Stage4）を順次実行するエントリポイント。

    入力:
        なし

    処理:
        - 各 Stage のランナー関数を時間計測しながら呼び出し
        - 最終的に総実行時間を出力

    出力:
        - なし（副作用として成果物の保存/ログ出力）
    """
    total_start_time = time.time()
    print_memory_usage("Start main")
    stage1_start = time.time()
    run_stage1()
    stage1_end = time.time()
    print(f"Stage1 実行時間: {format_time(stage1_end - stage1_start)}")
    shap_start = time.time()
    run_stage1_shap_evaluation_cpu(use_gpu=CFG["SHAP"]["use_gpu"],
                                   max_samples_per_class=CFG["SHAP"]["max_samples_per_class"],
                                   out_root=CFG["SHAP"]["out_root"])
    shap_end = time.time()
    print(f"Stage1 SHAP分析 実行時間: {format_time(shap_end - shap_start)}")
    stage2_start = time.time()
    run_stage2()
    stage2_end = time.time()
    print(f"Stage2 実行時間: {format_time(stage2_end - stage2_start)}")
    stage3_start = time.time()
    run_stage3()
    stage3_end = time.time()
    print(f"Stage3 実行時間: {format_time(stage3_end - stage3_start)}")
    vis_start = time.time()
    run_visualization()
    vis_end = time.time()
    print(f"可視化処理 実行時間: {format_time(vis_end - vis_start)}")
    eval_start = time.time()
    run_evaluation()
    eval_end = time.time()
    print(f"評価処理 実行時間: {format_time(eval_end - eval_start)}")
    video_start = time.time()
    create_comparison_videos(
        image_folder=CFG["VIDEO"]["image_folder"],
        output_folder=CFG["VIDEO"]["output_folder"],
        frame_rate=CFG["VIDEO"]["frame_rate"],
        low_res_scale=CFG["VIDEO"]["low_res_scale"],
        low_res_frame_rate=CFG["VIDEO"]["low_res_frame_rate"]
    )
    video_end = time.time()
    print(f"動画作成 実行時間: {format_time(video_end - video_start)}")
    stage4_start = time.time()
    run_stage4()
    stage4_end = time.time()
    print(f"Stage4 実行時間: {format_time(stage4_end - stage4_start)}")
    print_memory_usage("End main")
    print("All Stages Done.")
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"プログラム全体の実行時間: {format_time(total_elapsed_time)}")

if __name__ == "__main__":
    main()
