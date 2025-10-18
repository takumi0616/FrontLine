"""
ファイル概要（main_v4_config.py）:
- 役割:
  v4 パイプライン全体の共通設定（CFG）と、環境初期化（スレッド数・乱数種・デバイス）および
  ユーティリティ関数（メモリ使用量出力・経過時間整形）を提供するモジュール。
- CFG の構成:
  - PATHS: 入出力ディレクトリ（GSM/GT/各ステージの成果物/モデル保存先/可視化出力先）
  - IMAGE: 入力画像サイズ（ORIG_H, ORIG_W）
  - STAGE{1,1_5,2,2_5,3,3_5,4,4_5}: 各ステージの学習・推論・論理整形に必要なハイパーパラメータや入力チャネル数
  - VISUALIZATION: 可視化の色設定・気圧偏差の範囲・等値線
  - EVAL: 評価用の年など
- 初期化処理:
  - OS環境変数によるスレッド数制御
  - 乱数シード固定（numpy/torch/cuda）
  - デバイス（CPU/GPU）選択
- 提供関数:
  - print_memory_usage: 現在プロセスのメモリ使用量（MB）を表示
  - format_time: 秒数を人間可読な文字列に変換
- 注意:
  - v3 の実装には手を加えず、v4 のステージ分割要件に合わせ本モジュールの設定を定義する。
"""

import os
import sys
import gc
import psutil
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import time

# Ensure local imports (e.g. swin_unet) work when running from repo root
sys.path.append(str(Path(__file__).parent.resolve()))

# ======================================
# Global configuration (centralized params)
# ======================================

# 共通のSwinハイパーパラメータのベース
MODEL_BASE = {
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
    "ape": False,
    "patch_norm": True,
    "use_checkpoint": False,
    "final_upsample": "expand_first",
}

CFG = {
    "SEED": 0,
    "THREADS": 4,
    "PATHS": {
        "nc_gsm_dir": "./128_128/nc_gsm9",               # GSM（31変数×時刻）
        "nc_0p5_dir": "./128_128/nc_0p5_bulge_v2",       # 前線GT（5ch: warm/cold/stationary/occluded/warm_cold）

        "stage1_out_dir": "./v4_result/stage1_nc",       # Stage1: 接合=5（二値）の確率出力
        "stage1_5_out_dir": "./v4_result/stage1_5_nc",   # Stage1.5: 接合の論理整形結果（mask/class_map）
        "stage2_out_dir": "./v4_result/stage2_nc",       # Stage2: 温暖/寒冷（3クラス: none,warm,cold）
        "stage2_5_out_dir": "./v4_result/stage2_5_nc",   # Stage2.5: 接合連結制約の適用結果
        "stage3_out_dir": "./v4_result/stage3_nc",       # Stage3: 閉塞（二値）の確率出力
        "stage3_5_out_dir": "./v4_result/stage3_5_nc",   # Stage3.5: 付着制約の適用結果
        "stage4_out_dir": "./v4_result/stage4_nc",       # Stage4: 停滞（二値）の確率出力
        "stage4_5_out_dir": "./v4_result/stage4_5_nc",   # Stage4.5: サイズ/寒冷付着の論理調整＋最終アセンブル
        "final_out_dir": "./v4_result/final_nc",         # まとめ出力（必要時）

        "model_s1_save_dir": "./v4_result/stage1_model",
        "model_s2_save_dir": "./v4_result/stage2_model",
        "model_s3_save_dir": "./v4_result/stage3_model",
        "model_s4_save_dir": "./v4_result/stage4_model",

        "output_visual_dir": "./v4_result/visualizations",
    },
    "IMAGE": {
        "ORIG_H": 128,
        "ORIG_W": 128,
    },

    # Stage1: 6クラス（0:none,1:warm,2:cold,3:stationary,4:occluded,5:junction）で学習
    "STAGE1": {
        "num_classes": 6,           # 0:none, 1:warm, 2:cold, 3:stationary, 4:occluded, 5:junction
        "in_chans": 93,             # GSM 31変数×(t-6,t,t+6)
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "test_months": (2023, 1, 2023, 12),
        "dataset_cache_size": 50,
        "file_cache_size": 10,
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        "model": {
            **MODEL_BASE,
            "ape": True,            # Stage1 は絶対位置埋め込みをON（v3踏襲）
        }
    },
    # Stage1.5: 接合の論理整形
    "STAGE1_5": {
        # 2x2程度の塊を想定。過大な塊は中心で2x2へ縮退、1画素単独は削除
        "target_block_size": 2,
        "min_keep_area": 2,         # 面積<2は削除（1画素ノイズ除去）
        "max_area_to_shrink": 8,    # 面積>8なら2x2へ縮退
        "connectivity": 8
    },

    # Stage2: 温暖/寒冷 を3クラス（none,warm,cold）で学習。入力は GSM + 接合(1ch)
    "STAGE2": {
        "num_classes": 3,           # 0:none, 1:warm, 2:cold
        "in_chans": 94,             # 93 (GSM) + 1 (junction mask)
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "dataset_cache_size": 50,
        "file_cache_size": 10,
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_val": 1,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.99),
        },
        "model": {
            **MODEL_BASE,
        },
    },
    # Stage2.5: 「接合と繋がる」温暖・寒冷のみ残す。接合も温暖/寒冷双方に繋がるもののみ残す
    "STAGE2_5": {
        "connectivity": 8,
        "keep_only_connected_to_junction": True,
        "junction_must_touch_both": True
    },

    # Stage3: 閉塞 を二値（none/occluded）で学習。入力は GSM + 接合(1) + 温暖(1) + 寒冷(1)
    "STAGE3": {
        "num_classes": 2,           # 0:none, 1:occluded
        "in_chans": 96,             # 93 + 1(junc) + 1(warm) + 1(cold)
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "dataset_cache_size": 50,
        "file_cache_size": 10,
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_val": 1,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        "model": {
            **MODEL_BASE,
        },
    },
    # Stage3.5: 閉塞は温暖/寒冷/接合のいずれかに付着するもののみ残す
    "STAGE3_5": {
        "connectivity": 8,
        "must_attach_to_any_of": ["warm", "cold", "junction"]
    },

    # Stage4: 停滞 を二値（none/stationary）で学習。入力は GSM + 接合 + 温暖 + 寒冷 + 閉塞
    "STAGE4": {
        "num_classes": 2,           # 0:none, 1:stationary
        "in_chans": 97,             # 93 + 1(junc) + 1(warm) + 1(cold) + 1(occluded)
        "epochs": 50,
        "train_months": (2014, 1, 2022, 12),
        "dataset_cache_size": 50,
        "file_cache_size": 10,
        "dataloader": {
            "batch_size_train": 16,
            "batch_size_val": 1,
            "batch_size_test": 1,
            "num_workers": 4
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        "model": {
            **MODEL_BASE,
        },
    },
    # Stage4.5: 小さい停滞の削除、寒冷に付着した停滞は寒冷へ変更。最終アセンブル
    "STAGE4_5": {
        "connectivity": 8,
        "min_component_area": 4,
        "stationary_touching_cold_is_cold": True      # 停滞∩寒冷近傍 → 寒冷へ再分類
    },

    "VISUALIZATION": {
        "class_colors": {
            0: "#FFFFFF",  # 背景（白）
            1: "#FF0000",  # 温暖前線（赤）
            2: "#0000FF",  # 寒冷前線（青）
            3: "#008015",  # 停滞前線（緑）
            4: "#800080",  # 閉塞前線（紫）
            5: "#FFA500"   # 接合（橙）
        },
        "pressure_vmin": -40,   # 可視化背景：海面更正気圧偏差の最小値（hPa）
        "pressure_vmax": 40,    # 可視化背景：海面更正気圧偏差の最大値（hPa）
        "pressure_levels": 21,  # 等値線レベル数
        "parallel_factor": 4    # 並列可視化時の分割係数（未使用でも互換のため定義）
    },

    "EVAL": {
        "year": 2023
    }
}

# Set thread envs early
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = str(CFG["THREADS"])

# Device and seed setup
seed = CFG["SEED"]
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Derived constants and common paths
ORIG_H = CFG["IMAGE"]["ORIG_H"]
ORIG_W = CFG["IMAGE"]["ORIG_W"]

nc_gsm_dir = CFG["PATHS"]["nc_gsm_dir"]
nc_0p5_dir = CFG["PATHS"]["nc_0p5_dir"]

stage1_out_dir = CFG["PATHS"]["stage1_out_dir"]
stage1_5_out_dir = CFG["PATHS"]["stage1_5_out_dir"]
stage2_out_dir = CFG["PATHS"]["stage2_out_dir"]
stage2_5_out_dir = CFG["PATHS"]["stage2_5_out_dir"]
stage3_out_dir = CFG["PATHS"]["stage3_out_dir"]
stage3_5_out_dir = CFG["PATHS"]["stage3_5_out_dir"]
stage4_out_dir = CFG["PATHS"]["stage4_out_dir"]
stage4_5_out_dir = CFG["PATHS"]["stage4_5_out_dir"]
final_out_dir = CFG["PATHS"]["final_out_dir"]

model_s1_save_dir = CFG["PATHS"]["model_s1_save_dir"]
model_s2_save_dir = CFG["PATHS"]["model_s2_save_dir"]
model_s3_save_dir = CFG["PATHS"]["model_s3_save_dir"]
model_s4_save_dir = CFG["PATHS"]["model_s4_save_dir"]

output_visual_dir = CFG["PATHS"]["output_visual_dir"]

# Apply torch threads
try:
    torch.set_num_threads(CFG["THREADS"])
except Exception:
    pass


def print_memory_usage(msg: str = ""):
    """
    関数概要:
      現在の Python プロセスが使用している常駐集合サイズ（RSS）を MB 単位で標準出力へ表示する。

    入力:
      - msg (str): 先頭に付与する任意のラベル文字列（例: "Start Stage1"）

    処理:
      - psutil.Process(os.getpid()) で自身のプロセス情報を取得し、memory_info().rss を MB へ変換。
      - フォーマット済みのログ文字列を print する。

    出力:
      - 返り値なし（print の副作用のみ）。例: "[Memory] Start Stage1 memory usage: 1234.56 MB"
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] {msg} memory usage: {mem_info:.2f} MB")


def format_time(seconds: float) -> str:
    """
    関数概要:
      秒数を人間可読な文字列へ整形して返す（例: "2時間 03分 04.56秒", "03分 04.56秒", "04.56秒"）。

    入力:
      - seconds (float): 表示対象の秒数（処理時間など）

    処理:
      - divmod を用いて時間・分・秒へ分解し、0 の単位は省略して短い表現を返す。

    出力:
      - str: 可読化された時間表記
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒"
    elif minutes > 0:
        return f"{int(minutes)}分 {seconds:.2f}秒"
    else:
        return f"{seconds:.2f}秒"


# Informative prints
if torch.cuda.is_available():
    try:
        print("GPU is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("GPU is available!")
else:
    print("GPU is not available.")
print(f"Random seed set as {seed}")
print("Using device:", device)

# Safe atomic NetCDF write with retries
def atomic_save_netcdf(ds, out_path: str, engine: str = "netcdf4", retries: int = 3, sleep_sec: float = 0.5) -> bool:
    """
    原子的なNetCDF保存を行うヘルパー。tmpファイルに書いてからリネームすることで途中中断による破損を防止。
    失敗時は一定回数リトライする。
    戻り値: True=成功, False=失敗
    """
    import os as _os
    import traceback as _tb

    tmp_path = out_path + ".tmp"
    for i in range(max(1, int(retries))):
        try:
            # 既存tmpを消す
            if _os.path.exists(tmp_path):
                try:
                    _os.remove(tmp_path)
                except Exception:
                    pass
            # 書き込み
            ds.to_netcdf(tmp_path, engine=engine)
            # アトミック置換
            _os.replace(tmp_path, out_path)
            return True
        except Exception as e:
            print(f"[atomic_save_netcdf] attempt {i+1}/{retries} failed: {e}")
            print(_tb.format_exc())
            try:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
            except Exception:
                pass
            try:
                time.sleep(sleep_sec)
            except Exception:
                pass
    print(f"[atomic_save_netcdf] giving up after {retries} attempts: {out_path}")
    return False

__all__ = [
    "CFG", "device", "ORIG_H", "ORIG_W",
    "nc_gsm_dir", "nc_0p5_dir",
    "stage1_out_dir", "stage1_5_out_dir",
    "stage2_out_dir", "stage2_5_out_dir",
    "stage3_out_dir", "stage3_5_out_dir",
    "stage4_out_dir", "stage4_5_out_dir",
    "final_out_dir",
    "model_s1_save_dir", "model_s2_save_dir", "model_s3_save_dir", "model_s4_save_dir",
    "output_visual_dir",
    "print_memory_usage", "format_time", "atomic_save_netcdf",
]
