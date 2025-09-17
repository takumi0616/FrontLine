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
    "SEED": 0,  # 乱数シード（再現性のためのグローバルシード）
    "THREADS": 4,  # スレッド数（NumPy/PyTorchなどのスレッド数を制御）
    "PATHS": {
        "nc_gsm_dir": "./128_128/nc_gsm9",  # GSMデータ（31変数×時刻）ディレクトリ
        "nc_0p5_dir": "./128_128/nc_0p5_bulge_v2",  # 正解前線（5ch binary）ディレクトリ
        "stage1_out_dir": "./v3_result/stage1_nc",  # Stage1の確率出力保存先
        "stage2_out_dir": "./v3_result/stage2_nc",  # Stage2の確率出力保存先
        "stage3_out_dir": "./v3_result/stage3_nc",  # Stage3のスケルトン出力保存先
        "model_s1_save_dir": "./v3_result/stage1_model",  # Stage1モデル保存先
        "model_s2_save_dir": "./v3_result/stage2_model",  # Stage2モデル保存先
        "output_visual_dir": "./v3_result/visualizations",  # 可視化画像の出力先
        "stage4_svg_dir": "./v3_result/stage4_svg",  # Stage4（SVG）の出力先
    },
    "IMAGE": {
        "ORIG_H": 128,  # 入力画像の高さ（ピクセル）
        "ORIG_W": 128,  # 入力画像の幅（ピクセル）
    },
    "STAGE1": {
        "num_classes": 6,  # クラス数（0=背景, 1..5=前線）
        "in_chans": 93,  # 入力チャネル数（GSM31変数×3時刻）
        "epochs": 2,  # 学習エポック数
        "train_months": (2014, 1, 2022, 12),  # 学習データの年月範囲（開始年,開始月,終了年,終了月）
        "test_months":  (2023, 1, 2023, 12),  # テストデータの年月範囲
        "dataset_cache_size": 50,  # Datasetのサンプルキャッシュ上限（個）
        "file_cache_size": 10,  # Datasetでオープン保持するNetCDFファイルの上限（個）
        "dataloader": {
            "batch_size_train": 16,  # 学習時のバッチサイズ
            "batch_size_test": 1,  # テスト時のバッチサイズ
            "num_workers": 4  # DataLoaderのワーカ数
        },
        "optimizer": {
            "lr": 1e-4,  # 学習率
            "weight_decay": 0.05,  # L2正則化係数
            "betas": (0.9, 0.999),  # AdamWのベータ（モーメント係数）
        },
        "model": {
            "img_size": 128,  # 画像サイズ（正方）
            "patch_size": 2,  # パッチサイズ（Swinのパッチ分割）
            "embed_dim": 192,  # 埋め込み次元（モデル幅）
            "depths": [2, 2, 2, 2],  # エンコーダ各段のブロック数
            "depths_decoder": [1, 2, 2, 2],  # デコーダ各段のブロック数
            "num_heads": [3, 6, 12, 24],  # 各段のヘッド数
            "window_size": 16,  # Swinのウィンドウサイズ
            "mlp_ratio": 4.0,  # MLP拡張比
            "qkv_bias": True,  # QKVにバイアスを入れるか
            "qk_scale": None,  # QKスケール（Noneで自動）
            "drop_rate": 0.0,  # 全体のドロップアウト率
            "attn_drop_rate": 0.0,  # Attentionのドロップアウト率
            "drop_path_rate": 0.1,  # StochasticDepth（DropPath）率
            "norm_layer": nn.LayerNorm,  # 正規化層の種類
            "ape": True,  # 絶対位置埋め込み（Absolute Position Embedding）の使用
            "patch_norm": True,  # パッチ後に正規化を行うか
            "use_checkpoint": False,  # チェックポイント（メモリ節約）を使うか
            "final_upsample": "expand_first"  # 最終アップサンプルの方式
        }
    },
    "STAGE2": {
        "num_classes": 6,  # クラス数（Stage2側の出力クラス）
        "in_chans": 1,  # 入力チャネル（Swin版のStage2で使用、拡散では6ch確率を扱う）
        "epochs": 2,  # 学習エポック数
        "train_months": (2014, 1, 2022, 12),  # 学習データの年月範囲
        "dataset_cache_size": 50,  # FrontalRefinementDatasetのサンプルキャッシュ上限（個）
        "dataloader": {
            "batch_size_train": 16,  # 学習時のバッチサイズ
            "batch_size_val": 1,  # 検証時のバッチサイズ
            "batch_size_test": 1,  # 推論時のバッチサイズ
            "num_workers": 4  # DataLoaderのワーカ数
        },
        "optimizer": {
            "lr": 1e-4,  # 学習率
            "weight_decay": 0.05,  # L2正則化係数
            "betas": (0.9, 0.99),  # AdamWのベータ（拡散モデル側の推奨設定）
        },
        "model": {
            "img_size": 128,  # 画像サイズ
            "patch_size": 2,  # パッチサイズ（Swin Stage2で使用）
            "embed_dim": 96,  # 埋め込み次元（Swin Stage2幅）
            "depths": [2, 2, 2, 2],  # エンコーダ各段のブロック数（Swin Stage2）
            "depths_decoder": [1, 2, 2, 2],  # デコーダ各段のブロック数（Swin Stage2）
            "num_heads": [3, 6, 12, 24],  # ヘッド数（Swin Stage2）
            "window_size": 16,  # ウィンドウサイズ（Swin Stage2）
            "mlp_ratio": 4.0,  # MLP拡張比（Swin Stage2）
            "qkv_bias": True,  # QKVにバイアス（Swin Stage2）
            "qk_scale": None,  # QKスケール（Swin Stage2）
            "drop_rate": 0.0,  # ドロップアウト率（Swin Stage2）
            "attn_drop_rate": 0.0,  # Attentionのドロップアウト率（Swin Stage2）
            "drop_path_rate": 0.1,  # DropPath率（Swin Stage2）
            "norm_layer": nn.LayerNorm,  # 正規化層（Swin Stage2）
            "ape": False,  # 絶対位置埋め込み（Swin Stage2）
            "patch_norm": True,  # パッチ正規化（Swin Stage2）
            "use_checkpoint": False,  # チェックポイント使用（Swin Stage2）
            "final_upsample": "expand_first"  # アップサンプル方式（Swin Stage2）
        },
        "augment": {
            "n_augment": 10,  # 劣化サンプル数（各時刻に対し擬似劣化を何通り作るか）
            "prob_dilation": 0.8,  # 膨張適用確率
            "prob_create_gaps": 0.8,  # ギャップ生成の適用確率
            "prob_random_pixel_change": 0.8,  # ランダム画素置換の適用確率
            "prob_add_fake_front": 0.8,  # 偽前線追加の適用確率
            "dilation_kernel_range": (2, 3),  # 膨張カーネルサイズの範囲
            "num_gaps_range": (2, 4),  # 生成するギャップ数の範囲
            "gap_size_range": (3, 5),  # 各ギャップの大きさの範囲
            "num_pix_to_change_range": (20, 100),  # ランダムに変更する画素数の範囲
            "num_fake_front_range": (2, 10)  # 偽前線を追加する回数の範囲
        },
        "diffusion": {
            "base_dim": 64,  # 拡散モデルのUNet基底チャネル幅
            "dim_mults": [1, 2, 2, 2],  # UNetの段階ごとの幅倍率
            "dropout": 0.0,  # UNet内部のドロップアウト率
            "objective": "pred_v",  # 学習目標（v-parameterization）
            "beta_schedule": "sigmoid",  # ノイズスケジュール
            "timesteps": 1000,  # 学習時の拡散ステップ数
            "sampling_timesteps": 20,  # 推論時のステップ数（DDIM）
            "steps": 20,  # correct_from_probs系で使うサンプリングステップ数
            "ensemble": 4,  # アンサンブル数（生成サンプル数）
            "t_start_frac": 0.5,  # 逆拡散の開始時刻（全体ステップ比）
            "class_weights": [0.7, 1.10, 1.20, 1.10, 1.05, 1.10],  # 後処理でのクラス重み（背景0の抑制など）
            "blend_lambda": 0.20,  # Stage1分布とのブレンド比（連続性温存のため）
            "auto_normalize": True,  # 自動正規化（ライブラリ側の前処理）
            "flash_attn": False  # Flash-Attentionを使うか（Trueにすると高速化の可能性）
        }
    },
    "STAGE3": {
        "lap_thresh": -0.005  # ラプラシアンによるリッジ検出の閾値（小さくするほど線が太く残る）
    },
    "STAGE4": {
        "smoothing_window": 3  # SVG ポリライン平滑化窓サイズ（3で微平滑、0/1で無効）
    },
    "VISUALIZATION": {
        "class_colors": {
            0: "#FFFFFF",  # 背景（白）
            1: "#FF0000",  # 温暖前線（赤）
            2: "#0000FF",  # 寒冷前線（青）
            3: "#008015",  # 停滞前線（緑）
            4: "#800080",  # 閉塞前線（紫）
            5: "#FFA500"  # 接合（橙）
        },
        "pressure_vmin": -40,  # 可視化背景：海面更正気圧偏差の最小値（hPa）
        "pressure_vmax": 40,  # 可視化背景：海面更正気圧偏差の最大値（hPa）
        "pressure_levels": 21,  # 等値線レベル数
        "parallel_factor": 4  # 並列可視化時のスレッド分割係数（CPUコア/係数がプロセス数）
    },
    "VIDEO": {
        "image_folder": "./v3_result/visualizations/",  # フレーム画像の入力フォルダ
        "output_folder": "./v3_result/",  # 動画の出力フォルダ
        "frame_rate": 4,  # 通常動画のフレームレート
        "low_res_scale": 4,  # 低解像度動画の縮小倍率
        "low_res_frame_rate": 2  # 低解像度動画のフレームレート
    },
    "SHAP": {
        "use_gpu": True,  # SHAP解析にGPUを使うか
        "max_samples_per_class": 500,  # 各クラスで集計する最大サンプル数
        "out_root": "./v3_result/shap_stage1",  # SHAP結果の保存ルート
        "free_mem_threshold_gb": 4.0,  # GPU空きメモリ閾値（GB）これ未満ならCPUに切替
        "months": (2023, 1, 2023, 12),  # SHAP対象の年月範囲（Stage1のテスト月と同じにするのが標準）
        "nsamples_default": 16  # GradientExplainerの初期サンプル数（OOM時は自動的に半減）
    },
    "EVAL": {
        "year": 2023  # 評価対象の年（evaluationでのファイル選別に利用）
    }
}

# Set thread envs early
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = str(CFG["THREADS"])  # 各種数値計算ライブラリのスレッド数を統一設定

# Device and seed setup
seed = CFG["SEED"]  # 乱数シード
np.random.seed(seed)  # NumPyのシード設定
torch.manual_seed(seed)  # PyTorchのシード設定
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # CUDAのシード設定
    torch.backends.cudnn.deterministic = True  # CuDNNを決定論モードに
    torch.backends.cudnn.benchmark = False  # ベンチマーク無効（再現性優先）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用デバイス

# Derived constants and common paths
ORIG_H = CFG["IMAGE"]["ORIG_H"]  # 画像高さ（ショートカット）
ORIG_W = CFG["IMAGE"]["ORIG_W"]  # 画像幅（ショートカット）

nc_gsm_dir = CFG["PATHS"]["nc_gsm_dir"]  # GSM入力パス（ショートカット）
nc_0p5_dir = CFG["PATHS"]["nc_0p5_dir"]  # GT前線パス（ショートカット）
stage1_out_dir = CFG["PATHS"]["stage1_out_dir"]  # Stage1出力パス
stage2_out_dir = CFG["PATHS"]["stage2_out_dir"]  # Stage2出力パス
stage3_out_dir = CFG["PATHS"]["stage3_out_dir"]  # Stage3出力パス
model_s1_save_dir = CFG["PATHS"]["model_s1_save_dir"]  # Stage1モデル保存先
model_s2_save_dir = CFG["PATHS"]["model_s2_save_dir"]  # Stage2モデル保存先
output_visual_dir = CFG["PATHS"]["output_visual_dir"]  # 可視化出力先
stage4_svg_dir = CFG["PATHS"]["stage4_svg_dir"]  # SVG出力先

# Apply torch threads
try:
    torch.set_num_threads(CFG["THREADS"])  # PyTorchのスレッド数設定
except Exception:
    pass

def print_memory_usage(msg: str = ""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] {msg} memory usage: {mem_info:.2f} MB")  # 現在プロセスのメモリ使用量表示

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒"  # h m s の文字列
    elif minutes > 0:
        return f"{int(minutes)}分 {seconds:.2f}秒"  # m s の文字列
    else:
        return f"{seconds:.2f}秒"  # s の文字列

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
