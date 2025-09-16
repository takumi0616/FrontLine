import os
import sys
import gc
import psutil
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn

# Ensure local imports (e.g. swin_unet) work when running from repo root
sys.path.append(str(Path(__file__).parent.resolve()))

# ======================================
# Global configuration (centralized params)
# ======================================
CFG = {
    "SEED": 0,
    "THREADS": 4,
    "PATHS": {
        "nc_gsm_dir": "./128_128/nc_gsm9",
        "nc_0p5_dir": "./128_128/nc_0p5_bulge_v2",
        "stage1_out_dir": "./v3_result/stage1_nc",
        "stage2_out_dir": "./v3_result/stage2_nc",
        "stage3_out_dir": "./v3_result/stage3_nc",
        "model_s1_save_dir": "./v3_result/stage1_model",
        "model_s2_save_dir": "./v3_result/stage2_model",
        "output_visual_dir": "./v3_result/visualizations",
        "stage4_svg_dir": "./v3_result/stage4_svg",
    },
    "IMAGE": {
        "ORIG_H": 128,
        "ORIG_W": 128,
    },
    "STAGE1": {
        "num_classes": 6,
        "in_chans": 93,
        "epochs": 2,
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
        "epochs": 2,
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
        },
        "diffusion": {
            "base_dim": 64,
            "dim_mults": [1, 2, 2, 2],
            "dropout": 0.0,
            "objective": "pred_v",
            "beta_schedule": "sigmoid",
            "timesteps": 1000,
            "sampling_timesteps": 20,
            "steps": 20,
            "ensemble": 4,
            "t_start_frac": 0.5,
            "class_weights": [0.7, 1.10, 1.20, 1.10, 1.05, 1.10],
            "blend_lambda": 0.20,
            "auto_normalize": True,
            "flash_attn": False
        }
    },
    "STAGE3": {
        "lap_thresh": -0.005
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
        "image_folder": "./v3_result/visualizations/",
        "output_folder": "./v3_result/",
        "frame_rate": 4,
        "low_res_scale": 4,
        "low_res_frame_rate": 2
    },
    "SHAP": {
        "use_gpu": True,
        "max_samples_per_class": 500,
        "out_root": "./v3_result/shap_stage1",
        "free_mem_threshold_gb": 4.0
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
stage2_out_dir = CFG["PATHS"]["stage2_out_dir"]
stage3_out_dir = CFG["PATHS"]["stage3_out_dir"]
model_s1_save_dir = CFG["PATHS"]["model_s1_save_dir"]
model_s2_save_dir = CFG["PATHS"]["model_s2_save_dir"]
output_visual_dir = CFG["PATHS"]["output_visual_dir"]
stage4_svg_dir = CFG["PATHS"]["stage4_svg_dir"]

# Apply torch threads
try:
    torch.set_num_threads(CFG["THREADS"])
except Exception:
    pass

def print_memory_usage(msg: str = ""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] {msg} memory usage: {mem_info:.2f} MB")

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒"
    elif minutes > 0:
        return f"{int(minutes)}分 {seconds:.2f}秒"
    else:
        return f"{seconds:.2f}秒"

# Informative prints (kept for behavioral parity with original script)
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

__all__ = [
    "CFG", "device", "ORIG_H", "ORIG_W",
    "nc_gsm_dir", "nc_0p5_dir",
    "stage1_out_dir", "stage2_out_dir", "stage3_out_dir",
    "model_s1_save_dir", "model_s2_save_dir",
    "output_visual_dir", "stage4_svg_dir",
    "print_memory_usage", "format_time",
]
