"""
概要:
    Stage1（Swin-UNet）モデルに対して SHAP 解析（特徴重要度の説明）を実行するためのユーティリティ群。
    - モデルのロード、GPUメモリの確認、安全なSHAP計算（OOM耐性）、逐次統計集計、可視化出力までを一貫して提供
    - 解析は「各クラスのロジット（空間平均）」を目的関数として GradientExplainer でSHAPを算出する設計

構成:
    - メモリ関連ユーティリティ: _rss_mb(), _gpu_mb(), _mem()
    - モデルロード: _load_model()
    - GSM変数名取得: _gsm_vars(), VAR_NAMES_93（31変数×3時刻の名称リスト）
    - 逐次統計: OnlineMoments（|SHAP|平均・分散、SHAP平均）
    - 保存と可視化: _save_summary（CSV, beeswarm/bar/waterfall）
    - 実行関数: run_stage1_shap_evaluation_cpu（クラス別にSHAPを計算・集計・出力）
    - GPU選択と安全計算: _pick_gpu（空きメモリから最適GPUを選択）, _safe_shap（OOM時にnsamplesを半減）

使い方（例）:
    from main_v3_shap import run_stage1_shap_evaluation_cpu
    run_stage1_shap_evaluation_cpu()

注意:
    - CFG["SHAP"] 内の各種設定（対象月、出力先、サンプル数、GPU使用可否など）に依存
    - 背景サンプルはデータセットの先頭サンプル1件を用い、クラスごとに別Explainerを生成
"""

import os
import gc
import time
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import (
    CFG, device, print_memory_usage, format_time,
    nc_gsm_dir, model_s1_save_dir
)
from main_v3_utils import get_available_months
from main_v3_datasets import FrontalDatasetStage1
from main_v3_models import SwinUnetModel


def _rss_mb():
    """
    概要:
        プロセスRSS（常駐メモリ）をMBで返すためのプレースホルダ。
        現状は使用されておらず、常に0.0を返す軽量スタブ。

    入力:
        なし

    出力:
        - (float): 0.0（MB）
    """
    return torch.tensor(0.0).item()  # placeholder to keep parity; not used directly


def _gpu_mb():
    """
    概要:
        現在のCUDAデバイスで割り当て済みのGPUメモリ量（MB）を返す。

    入力:
        なし

    処理:
        - torch.cuda.is_available() を確認し、利用可能な場合は
          torch.cuda.memory_allocated() をMBへ換算

    出力:
        - (float): 割り当てGPUメモリ（MB）。CUDA非対応時は 0
    """
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0


def _mem(tag):
    """
    概要:
        メモリ使用状況の簡易ログ出力（GPUメモリのみ）。

    入力:
        - tag (str): 識別用のタグ（どの時点かを示すラベル）

    出力:
        なし（標準出力へログを出す副作用）
    """
    print(f"[Mem] {tag:18s}  GPU:{_gpu_mb():7.1f}MB")


def _load_model(ckpt: str, dev):
    """
    概要:
        Stage1 用 SwinUnetModel をチェックポイントからロードし、推論モードで返す。
        state_dict 形式、または {'model_state_dict': ...} を受け取り、'module.' 接頭辞も除去対応。

    入力:
        - ckpt (str): モデルのチェックポイントパス（.pth など）
        - dev (torch.device): 配置するデバイス（例: cpu / cuda:0）

    処理:
        - torch.load で辞書を読み込み、OrderedDict か 'model_state_dict' を抽出
        - DataParallel 由来の 'module.' プレフィクスを剥がす
        - SwinUnetModel に読み込んで eval モードに設定、requires_grad=False にする

    出力:
        - net (SwinUnetModel): 推論用に準備されたモデルインスタンス
    """
    net = SwinUnetModel(
        num_classes=CFG["STAGE1"]["num_classes"],
        in_chans=CFG["STAGE1"]["in_chans"],
        model_cfg=CFG["STAGE1"]["model"],
    )
    obj = torch.load(ckpt, map_location="cpu")
    sd = obj if isinstance(obj, OrderedDict) else obj.get("model_state_dict", obj)
    sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
    net.load_state_dict(sd, strict=True)
    net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


_gsm_base = None
def _gsm_vars():
    """
    概要:
        GSM の基本31変数名を nc_gsm_dir 配下の任意の gsm*.nc から取得する。

    入力:
        なし（グローバル設定 nc_gsm_dir を参照）

    処理:
        - 最初に見つかった gsm*.nc を開き、data_vars を列挙
        - 見つからない場合は var0..var30 のダミー名を返す

    出力:
        - List[str]: 31 変数名のリスト
    """
    global _gsm_base
    if _gsm_base is None:
        import xarray as xr, glob
        files = sorted(glob.glob(os.path.join(nc_gsm_dir, "gsm*.nc")))
        if not files:
            _gsm_base = [f"var{i}" for i in range(31)]
            return _gsm_base
        with xr.open_dataset(files[0]) as ds:
            _gsm_base = list(ds.data_vars)
    return _gsm_base


VAR_NAMES_93 = [f"{v}_{t}" for t in ("t-6h", "t0", "t+6h") for v in _gsm_vars()]


class OnlineMoments:
    """
    概要:
        1パスでチャンネルごとの |SHAP| 平均、SHAP平均、|SHAP| 標準偏差を逐次集計するユーティリティ。
        入力はチャネル先頭（C,H,W）を想定。

    入力:
        - n_feat (int): チャンネル数（既定 93）

    処理:
        - Welford 法に基づく逐次更新で平均・分散を安定集計
        - update(arr) に (C,H,W) あるいは (H,W,C) を渡すと自動で C,H,W に整形して集計

    出力:
        - ave_abs(): 各チャンネルの |SHAP| の平均 (np.ndarray, shape=(C,))
        - ave():     各チャンネルの SHAP の平均 (np.ndarray, shape=(C,))
        - std():     各チャンネルの |SHAP| の標準偏差 (np.ndarray, shape=(C,))
    """
    def __init__(self, n_feat=93):
        self.n = 0
        self.mu = np.zeros(n_feat, dtype=np.float64)
        self.M2 = np.zeros(n_feat, dtype=np.float64)
        self.raw = np.zeros(n_feat, dtype=np.float64)

    def _to_CHW(self, arr: np.ndarray) -> np.ndarray:
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected ndim: {arr.ndim}, shape={arr.shape}")
        if arr.shape[-1] == self.mu.size and arr.shape[0] != self.mu.size:
            arr = np.transpose(arr, (2, 0, 1))  # (..., C) -> (C, H, W)
        if arr.shape[0] != self.mu.size:
            raise ValueError(f"Channel数が合いません: {arr.shape}")
        return arr

    def update(self, arr):
        """
        概要:
            入力アレイのチャンネル要約（|SHAP|平均、SHAP平均）を逐次集計に反映する。

        入力:
            - arr (np.ndarray | torch.Tensor): 形状 (C,H,W) または (H,W,C)

        処理:
            - 必要に応じて (H,W,C)->(C,H,W) に転置
            - |arr| の空間平均と arr の空間平均を算出
            - Welford の更新式で平均・二乗和 M2 を更新

        出力:
            なし（内部状態の更新）
        """
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = self._to_CHW(arr)
        abs_m = np.abs(arr).mean(axis=(1, 2))
        sig_m = arr.mean(axis=(1, 2))
        self.n += 1
        delta = abs_m - self.mu
        self.mu += delta / self.n
        self.M2 += delta * (abs_m - self.mu)
        self.raw += sig_m

    def ave_abs(self):
        """
        概要:
            各チャンネルの |SHAP| の平均を返す。

        出力:
            - np.ndarray: 形状 (C,) の平均ベクトル
        """
        return self.mu
    def ave(self):
        """
        概要:
            各チャンネルの SHAP 値の平均を返す。

        出力:
            - np.ndarray: 形状 (C,) の平均ベクトル
        """
        return self.raw / max(self.n, 1)
    def std(self):
        """
        概要:
            各チャンネルの |SHAP| の標準偏差を返す。

        出力:
            - np.ndarray: 形状 (C,) の標準偏差ベクトル
        """
        return np.sqrt(self.M2 / max(self.n - 1, 1))


def _save_summary(cls_tag, stats: OnlineMoments, X, S, out_dir):
    """
    概要:
        1クラス分のSHAP集計結果を CSV と可視化図（beeswarm/bar/waterfall）として保存する。

    入力:
        - cls_tag (str): クラス名（例: "WarmFront"）
        - stats (OnlineMoments): 逐次集計済みの統計（|SHAP|平均/標準偏差, SHAP平均）
        - X (np.ndarray): 形状 (N, 93)。各サンプルの入力特徴の空間平均
        - S (np.ndarray): 形状 (N, 93)。各サンプルの SHAP 値の空間平均
        - out_dir (str): 出力ディレクトリ

    処理:
        - DataFrame にまとめて AveAbs_SHAP 降順でCSV出力
        - shap.summary_plot による beeswarm/bar を生成・保存
        - shap.plots.waterfall による1サンプルの寄与可視化を保存

    出力:
        なし（ファイル出力の副作用）
    """
    os.makedirs(out_dir, exist_ok=True)
    df = (
        pd.DataFrame(
            {
                "variable": VAR_NAMES_93,
                "AveAbs_SHAP": stats.ave_abs(),
                "Ave_SHAP": stats.ave(),
                "Std_SHAP": stats.std(),
            }
        )
        .sort_values("AveAbs_SHAP", ascending=False)
    )
    csv = os.path.join(out_dir, f"class{cls_tag}_summary.csv")
    df.to_csv(csv, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Beeswarm
        plt.figure()
        shap.summary_plot(S, X, feature_names=VAR_NAMES_93, show=False, plot_size=(9, 4))
        plt.title(f"{cls_tag}  summary (beeswarm)")
        plt.tight_layout()
        plt.savefig(csv.replace(".csv", "_beeswarm.png"), dpi=200)
        plt.close()

        # Bar
        plt.figure()
        shap.summary_plot(S, X, feature_names=VAR_NAMES_93, plot_type="bar", show=False)
        plt.title(f"{cls_tag}  mean(|SHAP|)")
        plt.tight_layout()
        plt.savefig(csv.replace(".csv", "_bar.png"), dpi=200)
        plt.close()

        # Waterfall for one sample
        expl = shap.Explanation(values=S[0], base_values=0, data=X[0], feature_names=VAR_NAMES_93)
        plt.figure()
        shap.plots.waterfall(expl, max_display=20, show=False)
        plt.title(f"{cls_tag}  waterfall (sample0)")
        plt.tight_layout()
        plt.savefig(csv.replace(".csv", "_waterfall.png"), dpi=200)
        plt.close()


def _pick_gpu(th=CFG["SHAP"]["free_mem_threshold_gb"]):
    """
    概要:
        空きメモリが最大のGPUデバイスIDを返す（ただし空きが閾値GB未満なら None）。

    入力:
        - th (float): 空きメモリの閾値（GB）。これ未満なら GPU を使わない

    出力:
        - (int|None): 使用するGPUのdevice index。条件を満たさない場合は None
    """
    if not torch.cuda.is_available():
        return None
    best = -1
    bid = None
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free /= 1024**3
        if free > best:
            best, bid = free, i
    return bid if best >= th else None


def _safe_shap(expl, x, ns=16):
    """
    概要:
        expl.shap_values(x, nsamples=ns) を実行し、OOM 発生時は nsamples を半減して再試行する安全ラッパ。

    入力:
        - expl: shap.Explainer（GradientExplainer など）
        - x (Tensor|ndarray): 入力（通常は (1,C,H,W)）
        - ns (int): nsamples 初期値（OOM時は 1 まで段階的に半減）

    出力:
        - shap_values: expl.shap_values の戻り値（Explainer/引数に依存。list か ndarray 等）
    """
    while True:
        try:
            return expl.shap_values(x, nsamples=ns)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            ns //= 2
            if ns < 1:
                raise
            print(f"[SHAP] OOM → nsamples={ns} で再試行")
            torch.cuda.empty_cache()


def run_stage1_shap_evaluation_cpu(
    use_gpu: bool = CFG["SHAP"]["use_gpu"],
    max_samples_per_class: int = CFG["SHAP"]["max_samples_per_class"],
    out_root: str = CFG["SHAP"]["out_root"],
):
    """
    概要:
        Stage1（Swin-UNet）に対するSHAP解析を実行し、クラス別に特徴重要度を集計・可視化する。

    入力:
        - use_gpu (bool): GPUが使用可能かつ空きメモリが閾値以上ならGPUを使用
        - max_samples_per_class (int): 各クラスで集計する最大サンプル数
        - out_root (str): 出力ルートディレクトリ（CSV/PNG群を保存）

    処理:
        - 解析対象の月（CFG["SHAP"]["months"]）で FrontalDatasetStage1 を構築
        - Stage1 学習済みモデルをロードし、クラス c ごとに
          f(x)=mean_{H,W} logit_c を出力するラッパを用意
        - GradientExplainer を各クラスに対して生成し、各サンプルで SHAP を計算
        - OnlineMoments で |SHAP|平均/標準偏差・SHAP平均を集計
        - _save_summary でクラス別 CSV と、beeswarm/bar/waterfall 図を保存

    出力:
        なし（副作用として out_root 配下にCSVと図が保存される）
    """
    print("\n========== Stage-1 SHAP 解析 ==========")
    y1, m1, y2, m2 = CFG["SHAP"].get("months", (2023, 1, 2023, 12))
    months = get_available_months(y1, m1, y2, m2)
    ds = FrontalDatasetStage1(
        months,
        nc_gsm_dir,
        CFG["PATHS"]["nc_0p5_dir"],
        cache_size=CFG["STAGE1"].get("dataset_cache_size", 50),
        file_cache_size=CFG["STAGE1"].get("file_cache_size", 10),
    )
    idxs = list(range(len(ds)))
    np.random.shuffle(idxs)

    # pick device for SHAP (can be different from global device)
    gid = _pick_gpu() if use_gpu else None
    dev = torch.device(f"cuda:{gid}") if gid is not None else torch.device("cpu")
    print(f"使用デバイス : {dev}")

    ckpt_path = os.path.join(model_s1_save_dir, "model_final.pth")
    if not os.path.exists(ckpt_path):
        print(f"[SHAP] モデルが見つかりません: {ckpt_path}")
        return
    model = _load_model(ckpt_path, dev)

    class Wrap(torch.nn.Module):
        def __init__(self, net, cid):
            super().__init__()
            self.net, self.cid = net, cid

        def forward(self, x):
            """
            概要:
                下位ネットワークのクラスロジット出力から、対象クラスcidの空間平均ロジットを返す。
                SHAPのExplainerが扱いやすいスカラー出力（(N,1,1)）に整形するためのラッパ。

            入力:
                - x (torch.Tensor): 形状 (N, C_in=93, H, W) の入力テンソル

            処理:
                - net(x): (N, num_classes, H, W) のロジットを取得
                - [:, cid]: 対象クラスcidの (N, H, W) を抽出
                - mean((1,2), keepdim=True): 空間方向(H, W)に平均して (N, 1, 1) を得る

            出力:
                - out (torch.Tensor): 形状 (N, 1, 1) の平均ロジット
            """
            out = self.net(x)[:, self.cid].mean((1, 2), keepdim=True)
            return out

    # background sample for GradientExplainer
    bg, _, _ = ds[idxs[0]]
    bg = bg.unsqueeze(0).to(dev)

    num_classes_stage1 = CFG["STAGE1"]["num_classes"]
    expl = {c: shap.GradientExplainer(Wrap(model, c), data=bg) for c in range(1, num_classes_stage1)}
    stats = {c: OnlineMoments() for c in range(1, num_classes_stage1)}
    Xbuf = defaultdict(list)
    Sbuf = defaultdict(list)

    for idx in tqdm(idxs, desc="Compute SHAP"):
        x, y, _ = ds[idx]
        present = set(np.unique(y.numpy())) & set(range(1, num_classes_stage1))
        for c in present:
            if len(Xbuf[c]) >= max_samples_per_class:
                continue
            xx = x.unsqueeze(0).to(dev)
            sv = _safe_shap(expl[c], xx, ns=CFG["SHAP"].get("nsamples_default", 16))
            val = sv[0] if isinstance(sv, list) else sv
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            val = np.squeeze(val)  # -> (H,W,C) or (C,H,W)
            if val.ndim == 3 and val.shape[-1] == CFG["STAGE1"]["in_chans"] and val.shape[0] != CFG["STAGE1"]["in_chans"]:
                val = np.transpose(val, (2, 0, 1))  # (H,W,C)->(C,H,W)
            stats[c].update(val)                            # accumulate per-channel stats
            Sbuf[c].append(val.mean((1, 2)))                # (C,)
            Xbuf[c].append(xx.detach().cpu().numpy()[0].mean((1, 2)))  # (C,)
        if all(len(Xbuf[k]) >= max_samples_per_class for k in range(1, num_classes_stage1)):
            break

    cname = {1: "WarmFront", 2: "ColdFront", 3: "Stationary", 4: "Occluded", 5: "Complex"}
    for c in range(1, num_classes_stage1):
        if Xbuf[c]:
            X = np.vstack(Xbuf[c])
            S = np.vstack(Sbuf[c])
            _save_summary(cname[c], stats[c], X, S, out_root)

    print("==========  SHAP 解析 完了 ==========\n")


__all__ = ["run_stage1_shap_evaluation_cpu"]
