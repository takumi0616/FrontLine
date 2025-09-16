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
    return torch.tensor(0.0).item()  # placeholder to keep parity; not used directly


def _gpu_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0


def _mem(tag):
    print(f"[Mem] {tag:18s}  GPU:{_gpu_mb():7.1f}MB")


def _load_model(ckpt: str, dev):
    """
    Load SwinUnetModel for Stage1 from checkpoint path.
    Accepts either plain state_dict (OrderedDict) or dict with 'model_state_dict'.
    Removes 'module.' prefix if present.
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
    Derive GSM variable names (base 31 vars) from any gsm*.nc under nc_gsm_dir.
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
    Track per-channel mean(|SHAP|), mean(SHAP), std(|SHAP|) in one pass.
    Assumes channel-first arrays (C,H,W).
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

    def ave_abs(self): return self.mu
    def ave(self):     return self.raw / max(self.n, 1)
    def std(self):     return np.sqrt(self.M2 / max(self.n - 1, 1))


def _save_summary(cls_tag, stats: OnlineMoments, X, S, out_dir):
    """
    Save CSV + beeswarm/bar/waterfall plots for a given class.
    X: (N, 93) mean input features per sample
    S: (N, 93) mean SHAP values per sample
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
    Call expl.shap_values with fallback halving nsamples on OOM.
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
    Stage1 SHAP 解析:
      - 2023年の FrontalDatasetStage1 からサンプルを抽出
      - 予測クラスごとに SHAP を GradientExplainer で算出し、統計集計
      - CSV と図を保存
    """
    print("\n========== Stage-1 SHAP 解析 ==========")
    months = get_available_months(2023, 1, 2023, 12)
    ds = FrontalDatasetStage1(months, nc_gsm_dir, CFG["PATHS"]["nc_0p5_dir"])
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
            # return mean class logit over spatial dims for class cid
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
            sv = _safe_shap(expl[c], xx, ns=16)
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
