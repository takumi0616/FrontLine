"""
概要:
    v4 向けの包括的な評価モジュール。
    - v3(main_v3_evaluation.py) の評価指標を v4 に導入し、各ステージごとに「出現するクラスのみ」で評価する
      （例: stage1/1_5 は {0,5}、stage3_5 は {0,4}、stage4_5 は {0,1,2,3,4,5} など）
    - 指標: 混同行列（行正規化 heatmap 付き）、Accuracy / Macro Precision / Recall / F1 / Cohen's Kappa
    - Front vs None の AUC / AP（Stage1/Stage2 の確率から計算）
    - 最終成果（stage4_5 の final_*.nc）を用いた季節別通過頻度・距離統計（v3 仕様の導入）

出力物（デフォルト保存先は v4_result 配下: output_visual_dir の親ディレクトリ）:
    - v4_evaluation_confusion_matrices.png      : 各ステージの混同行列（正規化）を 2x4 の図にまとめたもの
    - v4_evaluation_metrics.csv                 : 各ステージの基本指標（Accuracy, Macro P/R/F1, Kappa）
    - v4_evaluation_summary.log                 : 上記の概要と補助情報（AUC/AP, 季節/距離統計の要約など）
    - front_none_metrics.csv                    : Front vs None の AUC/AP（Stage1/Stage2）
    - seasonal_monthly_rates.csv                : 月次の通過頻度（final_*.nc から）
    - seasonal_rates.csv                        : 季節別の通過頻度（final_*.nc から）
    - distance_stats.csv                        : 予測(front>0)→GT(front) 最近傍距離 (km) の統計（final_*.nc から）

注意:
    - 年の指定は CFG["EVAL"]["year"] に従う
    - v4 の入出力仕様に合わせて、各ステージで評価対象クラスを切り替える
      stage1     : {0,5}（prob_*.nc 内の class_map_0_5 or argmax==5）
      stage1_5   : {0,5}（junction_*.nc の "junction" を 5/0 へ）
      stage2     : {0,1,2,5}（prob_*.nc の class_map_combined）
      stage2_5   : {0,1,2,5}（refined_*.nc の class_map(1/2) + junction(5)）
      stage3     : {0,1,2,4,5}（prob_*.nc の class_map_combined）
      stage3_5   : {0,4}（occluded_*.nc の class_map>0 を 4/0 へ）
      stage4     : {0,1,2,3,4,5}（prob_*.nc の class_map_combined）
      stage4_5   : {0,1,2,3,4,5}（final_out_dir の final_*.nc の class_map）
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)

from .main_v4_config import (
    CFG, format_time,
    nc_0p5_dir,
    stage1_out_dir, stage1_5_out_dir,
    stage2_out_dir, stage2_5_out_dir,
    stage3_out_dir, stage3_5_out_dir,
    stage4_out_dir, stage4_5_out_dir, final_out_dir,
    output_visual_dir,
)


# =========================
# 基本指標（v3 と同等）
# =========================
def compute_metrics(y_true, y_pred, labels):
    """
    概要:
        予測ラベルと正解ラベルから分類指標を計算して返す（v3 と同仕様）。
    出力:
        (acc, macro_prec, macro_rec, macro_f1, kappa)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    acc = np.mean(y_true == y_pred) * 100.0
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_prec *= 100.0
    macro_rec *= 100.0
    macro_f1 *= 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, macro_prec, macro_rec, macro_f1, kappa


# =========================
# ヘルパー
# =========================
def _extract_time_token(fname: str, prefix: str) -> str:
    return fname.replace(prefix, "").replace(".nc", "")


def _gt_class_map_for_time(token: str, h: Optional[int] = None, w: Optional[int] = None) -> Optional[np.ndarray]:
    """
    GT の 5ch (warm, cold, stationary, occluded, warm_cold) を 0..5 class_map に集約。
    token: "YYYYMMDDHHMM"
    サイズが指定されていれば [:h,:w] でクリップ。
    """
    try:
        t_dt = pd.to_datetime(token, format="%Y%m%d%H%M")
        month_str = token[:6]
        gt_path = os.path.join(nc_0p5_dir, f"{month_str}.nc")
        if not os.path.exists(gt_path):
            return None
        ds = xr.open_dataset(gt_path)
        ds_times = pd.to_datetime(ds["time"].values) if "time" in ds else None
        if (ds_times is not None) and (t_dt in ds_times):
            arr5 = ds.sel(time=t_dt).to_array().values  # (5,H,W)
        else:
            if ds_times is None or len(ds_times) == 0:
                ds.close()
                return None
            diffs = np.abs(ds_times - t_dt)
            idx = diffs.argmin()
            if diffs[idx] <= pd.Timedelta(hours=3):
                arr5 = ds.sel(time=ds["time"].values[idx]).to_array().values
            else:
                ds.close()
                return None
        H = arr5.shape[1]
        W = arr5.shape[2]
        if h is not None:
            H = min(H, h)
        if w is not None:
            W = min(W, w)
        arr5 = arr5[:, :H, :W]
        gt = np.zeros((H, W), dtype=np.int64)
        # 1..5 を順に上書き
        for c in range(5):
            mask = (arr5[c] == 1)
            gt[mask] = c + 1
        ds.close()
        return gt
    except Exception:
        return None


def _read_stage_pred(stage: str, token: str) -> Optional[np.ndarray]:
    """
    各ステージの予測を 0..5 へ正規化した class_map で返す（2D）。
    返り値 None は読み出し失敗。
    """
    try:
        if stage == "stage1":
            p = os.path.join(stage1_out_dir, f"prob_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "class_map_0_5" in ds:
                    cm = ds["class_map_0_5"].isel(time=0).values
                    cm = np.squeeze(np.asarray(cm))
                    pred = cm.astype(np.int64)
                else:
                    probs = ds["probabilities"].isel(time=0).values  # (H,W,6)
                    cls = np.argmax(probs, axis=-1).astype(np.int64)
                    pred = np.where(cls == 5, 5, 0)
                return pred

        if stage == "stage1_5":
            p = os.path.join(stage1_5_out_dir, f"junction_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "junction" in ds:
                    a = ds["junction"].isel(time=0).values if "time" in ds["junction"].dims else ds["junction"].values
                    a = np.squeeze(np.asarray(a))
                    pred = np.where(a > 0, 5, 0).astype(np.int64)
                elif "class_map" in ds:
                    a = ds["class_map"].values.astype(np.int64)
                    a = np.squeeze(a)
                    pred = np.where(a > 0, 5, 0).astype(np.int64)
                else:
                    return None
                return pred

        if stage == "stage2":
            p = os.path.join(stage2_out_dir, f"prob_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "class_map_combined" in ds:
                    cm = ds["class_map_combined"].isel(time=0).values
                    cm = np.squeeze(np.asarray(cm)).astype(np.int64)
                    return cm
                return None

        if stage == "stage2_5":
            p = os.path.join(stage2_5_out_dir, f"refined_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "class_map" not in ds:
                    return None
                cm = ds["class_map"].isel(time=0).values if "time" in ds["class_map"].dims else ds["class_map"].values
                cm = np.squeeze(np.asarray(cm)).astype(np.int64)
                pred = np.zeros_like(cm, dtype=np.int64)
                pred[cm == 1] = 1
                pred[cm == 2] = 2
                if "junction" in ds:
                    j = ds["junction"]
                    ja = j.isel(time=0).values if "time" in j.dims else j.values
                    ja = np.squeeze(np.asarray(ja))
                    pred[ja > 0] = 5
                return pred

        if stage == "stage3":
            p = os.path.join(stage3_out_dir, f"prob_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "class_map_combined" in ds:
                    cm = ds["class_map_combined"].isel(time=0).values
                    cm = np.squeeze(np.asarray(cm)).astype(np.int64)
                    return cm
                return None

        if stage == "stage3_5":
            # occluded（0/1）と Stage2.5 refined の warm/cold/junction を合成して 0/1/2/4/5 を生成
            p_occ = os.path.join(stage3_5_out_dir, f"occluded_{token}.nc")
            if not os.path.exists(p_occ):
                return None
            # occluded 読み出し
            with xr.open_dataset(p_occ) as ds:
                a = None
                if "class_map" in ds:
                    a = ds["class_map"].isel(time=0).values if "time" in ds["class_map"].dims else ds["class_map"].values
                elif "occluded" in ds:
                    a = ds["occluded"].isel(time=0).values if "time" in ds["occluded"].dims else ds["occluded"].values
                if a is None:
                    return None
                a = np.squeeze(np.asarray(a))
                occ = (a > 0).astype(np.uint8)
                H, W = occ.shape

            # refined (stage2_5) から warm/cold/junction
            p_wc = os.path.join(stage2_5_out_dir, f"refined_{token}.nc")
            warm = np.zeros_like(occ, dtype=np.uint8)
            cold = np.zeros_like(occ, dtype=np.uint8)
            junc = np.zeros_like(occ, dtype=np.uint8)
            if os.path.exists(p_wc):
                try:
                    with xr.open_dataset(p_wc) as ds2:
                        if "class_map" in ds2:
                            cm = ds2["class_map"].isel(time=0).values if "time" in ds2["class_map"].dims else ds2["class_map"].values
                            cm = np.squeeze(np.asarray(cm))
                            # サイズ整形
                            if cm.ndim != 2:
                                cm = np.reshape(cm, (H, W))
                            else:
                                hh = min(H, cm.shape[0]); ww = min(W, cm.shape[1])
                                tmp = np.zeros((H, W), dtype=cm.dtype)
                                tmp[:hh, :ww] = cm[:hh, :ww]
                                cm = tmp
                            warm = (cm == 1).astype(np.uint8)
                            cold = (cm == 2).astype(np.uint8)
                        if "junction" in ds2:
                            j = ds2["junction"]
                            ja = j.isel(time=0).values if "time" in j.dims else j.values
                            ja = np.squeeze(np.asarray(ja))
                            if ja.ndim != 2:
                                tmpj = np.zeros((H, W), dtype=np.uint8)
                                hh = min(H, ja.shape[-2] if ja.ndim >= 2 else 0)
                                ww = min(W, ja.shape[-1] if ja.ndim >= 2 else 0)
                                if hh > 0 and ww > 0:
                                    tmpj[:hh, :ww] = (ja[..., :hh, :ww] > 0).astype(np.uint8)
                                ja = tmpj
                            else:
                                hh = min(H, ja.shape[0]); ww = min(W, ja.shape[1])
                                tmpj = np.zeros((H, W), dtype=np.uint8)
                                tmpj[:hh, :ww] = (ja[:hh, :ww] > 0).astype(np.uint8)
                                ja = tmpj
                            junc = ja.astype(np.uint8)
                except Exception:
                    pass

            # 合成（上書き禁止の順）：5(junc) > 1(warm) > 2(cold) > 4(occ) > 0
            pred = np.zeros((H, W), dtype=np.int64)
            pred[junc == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
            mask = (pred == 0) & (occ == 1)
            pred[mask] = 4
            return pred

        if stage == "stage4":
            p = os.path.join(stage4_out_dir, f"prob_{token}.nc")
            if not os.path.exists(p):
                return None
            with xr.open_dataset(p) as ds:
                if "class_map_combined" in ds:
                    cm = ds["class_map_combined"].isel(time=0).values
                    cm = np.squeeze(np.asarray(cm)).astype(np.int64)
                    return cm
                return None

        if stage == "stage4_5":
            # 最終成果は final_out_dir の final_*.nc を使用（なければ stage4_5_out_dir にフォールバック）
            p = os.path.join(final_out_dir, f"final_{token}.nc")
            if not os.path.exists(p):
                p = os.path.join(stage4_5_out_dir, f"final_{token}.nc")
                if not os.path.exists(p):
                    return None
            with xr.open_dataset(p) as ds:
                if "class_map" in ds:
                    cm = ds["class_map"].isel(time=0).values if "time" in ds["class_map"].dims else ds["class_map"].values
                    cm = np.squeeze(np.asarray(cm)).astype(np.int64)
                    return cm
                return None

        return None
    except Exception:
        return None


def _stage_dir_prefix(stage: str) -> Tuple[str, str]:
    if stage == "stage1":
        return stage1_out_dir, "prob_"
    if stage == "stage1_5":
        return stage1_5_out_dir, "junction_"
    if stage == "stage2":
        return stage2_out_dir, "prob_"
    if stage == "stage2_5":
        return stage2_5_out_dir, "refined_"
    if stage == "stage3":
        return stage3_out_dir, "prob_"
    if stage == "stage3_5":
        return stage3_5_out_dir, "occluded_"
    if stage == "stage4":
        return stage4_out_dir, "prob_"
    if stage == "stage4_5":
        return final_out_dir, "final_"
    raise ValueError(stage)


def _labels_for_stage(stage: str) -> List[int]:
    if stage in ["stage1", "stage1_5"]:
        return [0, 5]
    if stage in ["stage2", "stage2_5"]:
        return [0, 1, 2, 5]
    if stage == "stage3":
        return [0, 1, 2, 4, 5]
    if stage == "stage3_5":
        # stage3_5 は refined (stage2_5) の warm/cold/junction と occluded を合成して 0/1/2/4/5 で可視化・評価
        return [0, 1, 2, 4, 5]
    if stage in ["stage4", "stage4_5"]:
        return [0, 1, 2, 3, 4, 5]
    return [0, 1, 2, 3, 4, 5]


def _reduce_gt_for_stage(stage: str, gt_cm: np.ndarray) -> np.ndarray:
    """
    ステージの評価対象クラスみに GT を射影（その他は 0）。
    """
    gt_cm = np.asarray(gt_cm).astype(np.int64)
    out = np.zeros_like(gt_cm, dtype=np.int64)
    if stage in ["stage1", "stage1_5"]:
        out[gt_cm == 5] = 5
    elif stage in ["stage2", "stage2_5"]:
        out[gt_cm == 1] = 1
        out[gt_cm == 2] = 2
        out[gt_cm == 5] = 5
    elif stage == "stage3":
        out[gt_cm == 1] = 1
        out[gt_cm == 2] = 2
        out[gt_cm == 4] = 4
        out[gt_cm == 5] = 5
    elif stage == "stage3_5":
        # stage3_5 も warm/cold/junction/occluded を評価（0/1/2/4/5）
        out[gt_cm == 1] = 1
        out[gt_cm == 2] = 2
        out[gt_cm == 4] = 4
        out[gt_cm == 5] = 5
    elif stage in ["stage4", "stage4_5"]:
        for c in [1, 2, 3, 4, 5]:
            out[gt_cm == c] = c
    else:
        out = gt_cm.copy()
    return out


def _list_tokens_for_stage(stage: str, year: int) -> List[str]:
    """
    指定ステージのファイルから YYYY 年のトークンを抽出して返す（昇順）。
    """
    base_dir, prefix = _stage_dir_prefix(stage)
    if not os.path.exists(base_dir):
        return []
    toks: List[str] = []
    for f in os.listdir(base_dir):
        if not (f.startswith(prefix) and f.endswith(".nc")):
            continue
        token = _extract_time_token(f, prefix)
        if token.startswith(str(year)):
            toks.append(token)
    toks.sort()
    return toks


# =========================
# 季節別通過頻度・距離統計（final_*.nc を対象）
# =========================
def compute_seasonal_crossing_rates_v4(
    final_nc_dir: str,
    out_csv_month: str,
    out_csv_season: str,
):
    """
    final_*.nc の class_map を用いて月次/季節別の「front 存在頻度」を集計。
    front = (class_map > 0)
    """
    os.makedirs(os.path.dirname(out_csv_month), exist_ok=True)
    if not os.path.exists(final_nc_dir):
        print(f"[V4-Seasonal] final dir not found: {final_nc_dir}")
        return
    files = sorted([f for f in os.listdir(final_nc_dir) if f.startswith("final_") and f.endswith(".nc")])
    if not files:
        print("[V4-Seasonal] No final_*.nc")
        return

    by_month: Dict[Tuple[int, int], List[Tuple[pd.Timestamp, str]]] = {}
    sample_lat, sample_lon = None, None
    for f in files:
        p = os.path.join(final_nc_dir, f)
        with xr.open_dataset(p) as ds:
            tval = ds["time"].values[0] if "time" in ds else None
            t_dt = pd.to_datetime(tval) if tval is not None else pd.to_datetime(_extract_time_token(f, "final_"), format="%Y%m%d%H%M")
            if sample_lat is None:
                try:
                    sample_lat = ds["lat"].values
                    sample_lon = ds["lon"].values
                except Exception:
                    sample_lat, sample_lon = None, None
        key = (t_dt.year, t_dt.month)
        by_month.setdefault(key, []).append((t_dt, p))

    monthly_rows = []
    for (y, m), items in sorted(by_month.items()):
        items.sort(key=lambda x: x[0])
        # 2D形状
        if sample_lat is not None and sample_lon is not None:
            H, W = len(sample_lat), len(sample_lon)
        else:
            # 1本開いて取得
            with xr.open_dataset(items[0][1]) as ds0:
                H = ds0.dims.get("lat", 0)
                W = ds0.dims.get("lon", 0)
        last_date = np.empty((H, W), dtype=object)
        last_date[:] = None
        count = np.zeros((H, W), dtype=np.int32)
        dates_in_month = sorted({d.date() for d, _ in items})
        days_in_month = len(dates_in_month) if len(dates_in_month) > 0 else 1

        for dtime, p in items:
            with xr.open_dataset(p) as ds:
                cm = ds["class_map"].isel(time=0).values if "time" in ds["class_map"].dims else ds["class_map"].values
                cm = np.squeeze(np.asarray(cm)).astype(np.int64)
            # 整形
            if cm.ndim != 2:
                cm = cm.reshape((H, W))
            front = (cm > 0).astype(np.uint8)
            cur_date = dtime.date()
            need = (last_date == None)  # noqa: E711
            need |= (last_date != cur_date)
            inc = (front == 1) & need
            count[inc] += 1
            last_date[front == 1] = cur_date

        rate = count.astype(np.float32) / max(1, days_in_month)
        monthly_rows.append({"year": y, "month": m, "rate_mean": float(np.mean(rate))})

    df_month = pd.DataFrame(monthly_rows)
    df_month.to_csv(out_csv_month, index=False)
    print(f"[V4-Seasonal] Monthly rates -> {out_csv_month}")

    def season_of(mm):
        if mm in [12, 1, 2]:
            return "DJF"
        if mm in [3, 4, 5]:
            return "MAM"
        if mm in [6, 7, 8]:
            return "JJA"
        return "SON"

    season_rows = []
    for year in sorted({r["year"] for r in monthly_rows}):
        by_season: Dict[str, List[float]] = {}
        for r in monthly_rows:
            if r["year"] != year:
                continue
            s = season_of(r["month"])
            by_season.setdefault(s, []).append(r["rate_mean"])
        for s, vals in by_season.items():
            if len(vals) > 0:
                season_rows.append({"year": year, "season": s, "rate_mean": float(np.mean(vals))})
    df_season = pd.DataFrame(season_rows)
    df_season.to_csv(out_csv_season, index=False)
    print(f"[V4-Seasonal] Seasonal rates -> {out_csv_season}")


def compute_distance_stats_v4(
    final_nc_dir: str,
    gt_dir: str,
    out_csv: str,
):
    """
    final_*.nc の class_map>0 を「予測 front」、GT(5ch) を front 二値化し、予測→GT の最近傍距離 (km) を統計化。
    v3 の compute_distance_stats 相当の v4 版。
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.exists(final_nc_dir):
        print(f"[V4-Distance] final dir not found: {final_nc_dir}")
        return
    files = sorted([f for f in os.listdir(final_nc_dir) if f.startswith("final_") and f.endswith(".nc")])
    if not files:
        print("[V4-Distance] No final_*.nc")
        return

    with xr.open_dataset(os.path.join(final_nc_dir, files[0])) as ds0:
        lat = ds0["lat"].values if "lat" in ds0 else None
        lon = ds0["lon"].values if "lon" in ds0 else None
    if lat is None or lon is None:
        print("[V4-Distance] No lat/lon in final nc, skip.")
        return
    lat_sorted = np.sort(np.unique(lat))
    lon_sorted = np.sort(np.unique(lon))
    dlat = np.median(np.abs(np.diff(lat_sorted))) if len(lat_sorted) > 1 else 1.0
    dlon = np.median(np.abs(np.diff(lon_sorted))) if len(lon_sorted) > 1 else 1.0
    lat_mean = float(np.mean(lat))
    dy_km = 111.0 * float(dlat)
    dx_km = 111.0 * float(dlon) * max(0.1, np.cos(np.deg2rad(lat_mean)))
    pix_km = float((dx_km + dy_km) / 2.0)

    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        print("[V4-Distance] scipy.ndimage.distance_transform_edt not available. Skip distance stats.")
        return

    rows = []
    for f in files:
        token = _extract_time_token(f, "final_")
        p = os.path.join(final_nc_dir, f)
        try:
            with xr.open_dataset(p) as ds:
                cm = ds["class_map"].isel(time=0).values if "time" in ds["class_map"].dims else ds["class_map"].values
                cm = np.squeeze(np.asarray(cm)).astype(np.int64)
            pred_bin = (cm > 0).astype(np.uint8)
            gt_cm = _gt_class_map_for_time(token, h=cm.shape[0], w=cm.shape[1])
            if gt_cm is None:
                continue
            gt_bin = (gt_cm > 0).astype(np.uint8)
            dist_to_gt = distance_transform_edt(1 - gt_bin)
            d_pred_to_gt = dist_to_gt[pred_bin == 1].astype(np.float32)
            if d_pred_to_gt.size == 0:
                continue
            dkm = d_pred_to_gt * pix_km
            rows.append(
                {
                    "time": pd.to_datetime(token, format="%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M"),
                    "mean_km": float(np.mean(dkm)),
                    "median_km": float(np.median(dkm)),
                    "p90_km": float(np.percentile(dkm, 90.0)),
                    "count": int(dkm.size),
                }
            )
        except Exception:
            continue

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"[V4-Distance] Stats -> {out_csv}")
    else:
        print("[V4-Distance] No distance rows computed.")


# =========================
# Front vs None AUC/AP（Stage1/Stage2）
# =========================
def _compute_front_none_auc_ap(year: int, out_csv: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Stage1/2 の確率から front vs none の AUC/AP を計算して CSV 出力。
    - Stage1: probs(H,W,6) の front=max(probs[...,1:6])、かつ argmax==0 の画素は front=0 に上書き（v3 踏襲）
    - Stage2: probs(H,W,3) の front=max(probs[...,1:])、同様に argmax==0 の画素は front=0
    y_true は GT class_map>0
    """
    s1_files = sorted([f for f in os.listdir(stage1_out_dir) if f.startswith(f"prob_{year}") and f.endswith(".nc")]) if os.path.exists(stage1_out_dir) else []
    s2_files = sorted([f for f in os.listdir(stage2_out_dir) if f.startswith(f"prob_{year}") and f.endswith(".nc")]) if os.path.exists(stage2_out_dir) else []
    if not s1_files or not s2_files:
        print("[AUC/AP] Stage1 or Stage2 prob files missing for year", year)
        return None, None, None, None

    any_prob_s1_list, any_prob_s2_list, y_true_front_list = [], [], []

    for f in sorted(set(s1_files) & set(s2_files)):
        token = _extract_time_token(f, "prob_")
        try:
            with xr.open_dataset(os.path.join(stage1_out_dir, f)) as ds1:
                probs_s1 = ds1["probabilities"].isel(time=0).values  # (H,W,6)
            with xr.open_dataset(os.path.join(stage2_out_dir, f)) as ds2:
                probs_s2 = ds2["probabilities"].isel(time=0).values  # (H,W,3)
            # any front prob
            any_s1 = np.max(probs_s1[..., 1:6], axis=-1)
            any_s2 = np.max(probs_s2[..., 1:3], axis=-1)
            arg1 = np.argmax(probs_s1, axis=-1)
            arg2 = np.argmax(probs_s2, axis=-1)
            any_s1[arg1 == 0] = 0.0
            any_s2[arg2 == 0] = 0.0
            gt = _gt_class_map_for_time(token, h=any_s1.shape[0], w=any_s1.shape[1])
            if gt is None:
                continue
            y_front = (gt > 0).astype(np.uint8)
            any_prob_s1_list.append(any_s1.reshape(-1))
            any_prob_s2_list.append(any_s2.reshape(-1))
            y_true_front_list.append(y_front.reshape(-1))
        except Exception:
            continue

    if len(y_true_front_list) == 0:
        print("[AUC/AP] No common tokens for Stage1/2 to compute AUC/AP.")
        return None, None, None, None

    y_true_front_all = np.concatenate(y_true_front_list, axis=0)
    any_prob_s1_all = np.concatenate(any_prob_s1_list, axis=0)
    any_prob_s2_all = np.concatenate(any_prob_s2_list, axis=0)
    mask_valid = np.isfinite(any_prob_s1_all) & np.isfinite(any_prob_s2_all)
    y_true_front_all = y_true_front_all[mask_valid]
    any_prob_s1_all = any_prob_s1_all[mask_valid]
    any_prob_s2_all = any_prob_s2_all[mask_valid]

    try:
        auc_s1 = roc_auc_score(y_true_front_all, any_prob_s1_all) if y_true_front_all.sum() > 0 else float("nan")
        auc_s2 = roc_auc_score(y_true_front_all, any_prob_s2_all) if y_true_front_all.sum() > 0 else float("nan")
        ap_s1 = average_precision_score(y_true_front_all, any_prob_s1_all) if y_true_front_all.sum() > 0 else float("nan")
        ap_s2 = average_precision_score(y_true_front_all, any_prob_s2_all) if y_true_front_all.sum() > 0 else float("nan")
    except Exception:
        auc_s1 = auc_s2 = ap_s1 = ap_s2 = float("nan")

    try:
        df_auc = pd.DataFrame({"metric": ["ROC_AUC", "AP"], "Stage1": [auc_s1, ap_s1], "Stage2": [auc_s2, ap_s2]})
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_auc.to_csv(out_csv, index=False)
        print(f"[AUC/AP] Front/None AUC/AP -> {out_csv}")
    except Exception:
        pass

    return auc_s1, ap_s1, auc_s2, ap_s2


# =========================
# ステージ別評価
# =========================
def _evaluate_stage(stage: str, year: int) -> Optional[Dict[str, object]]:
    """
    単一ステージについて、該当年の token を対象に
      - pred（0..5）読み出し
      - GT の射影（ステージ用）
      - 混同行列・指標の算出
    を行い、結果を dict で返す（空なら None）。
    """
    tokens = _list_tokens_for_stage(stage, year)
    if len(tokens) == 0:
        return None
    labels = _labels_for_stage(stage)
    y_pred_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []

    for token in tokens:
        pred = _read_stage_pred(stage, token)
        if pred is None:
            continue
        gt_full = _gt_class_map_for_time(token, h=pred.shape[0], w=pred.shape[1])
        if gt_full is None:
            continue
        gt = _reduce_gt_for_stage(stage, gt_full)
        y_pred_all.append(pred.reshape(-1))
        y_true_all.append(gt.reshape(-1))

    if len(y_true_all) == 0:
        return None

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    # 混同行列（raw と 正規化）
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    acc, mp, mr, mf, kappa = compute_metrics(y_true_all, y_pred_all, labels)

    return {
        "stage": stage,
        "labels": labels,
        "cm": cm,
        "cm_norm": cm_norm,
        "acc": acc,
        "macro_prec": mp,
        "macro_rec": mr,
        "macro_f1": mf,
        "kappa": kappa,
        "n_pixels": int(y_true_all.size),
        "n_tokens": int(len(tokens)),
        # 後段で front-only 指標を出すための元配列
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }


def _draw_confusion_grid(stage_results: List[Dict[str, object]], save_path: str):
    """
    2x4 の grid で stage1,1_5,2,2_5,3,3_5,4,4_5 の正規化 CM を並べて保存（raw counts を annot）。
    """
    order = ["stage1", "stage1_5", "stage2", "stage2_5", "stage3", "stage3_5", "stage4", "stage4_5"]
    stage_map = {r["stage"]: r for r in stage_results if r is not None}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, st in enumerate(order):
        ax = axes[i]
        res = stage_map.get(st, None)
        if res is None:
            ax.axis("off")
            ax.set_title(f"{st} (no data)")
            continue
        cm_norm = res["cm_norm"]
        cm_raw = res["cm"]
        labels = res["labels"]
        sns.heatmap(
            cm_norm,
            annot=cm_raw,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=1.0,
            cbar=False
        )
        ax.set_title(f"{st}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("GT")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[Eval] Confusion grid -> {save_path}")


def _compute_presence_match_stage4_5(year: int, out_csv: str):
    """
    Time-wise Presence-set match を stage4_5(final) に対して計算し、各時刻ごとに CSV 出力する。
    - present_pred: 予測に出現したクラスの集合（1..5）
    - present_gt  : 正解に出現したクラスの集合（1..5）
    - match (0/1) : present_pred == present_gt のとき 1
    戻り値: (match_count, total, percent[%])
    """
    tokens = _list_tokens_for_stage("stage4_5", year)
    if not tokens:
        print("[Presence] No final tokens for stage4_5 to compute presence match.")
        return 0, 0, 0.0
    rows = []
    match_count = 0
    for token in tokens:
        pred = _read_stage_pred("stage4_5", token)
        if pred is None:
            continue
        gt = _gt_class_map_for_time(token, h=pred.shape[0], w=pred.shape[1])
        if gt is None:
            continue
        # 1..5 の存在クラス
        sp = sorted(set(int(c) for c in np.unique(pred) if (c >= 1 and c <= 5)))
        sg = sorted(set(int(c) for c in np.unique(gt) if (c >= 1 and c <= 5)))
        match = int(set(sp) == set(sg))
        match_count += match
        rows.append({
            "time": pd.to_datetime(token, format="%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M"),
            "match": match,
            "present_pred": ",".join(map(str, sp)),
            "present_gt": ",".join(map(str, sg)),
        })
    total = len(rows)
    percent = (match_count / total * 100.0) if total > 0 else 0.0
    try:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[Presence] stage4_5 presence match -> {out_csv}")
    except Exception as e:
        print(f"[Presence] write csv failed: {e}")
    return match_count, total, percent


def run_evaluation_v4():
    """
    v4 ステージ一式に対する総合評価を実施し、PNG/CSV/LOG を出力する。
    - ステージ別混同行列/指標
    - Front vs None AUC/AP（Stage1/2）
    - final からの季節別通過頻度 / 距離統計
    """
    t0 = time.time()
    year = CFG.get("EVAL", {}).get("year", 2023)
    out_root = os.path.dirname(output_visual_dir)
    os.makedirs(out_root, exist_ok=True)

    stages = ["stage1", "stage1_5", "stage2", "stage2_5", "stage3", "stage3_5", "stage4", "stage4_5"]
    results: List[Optional[Dict[str, object]]] = []
    for st in stages:
        try:
            r = _evaluate_stage(st, year)
        except Exception as e:
            print(f"[Eval] Stage {st} failed: {e}")
            r = None
        results.append(r)

    # 図
    try:
        _draw_confusion_grid([r for r in results if r is not None], os.path.join(out_root, "v4_evaluation_confusion_matrices.png"))
    except Exception as e:
        print(f"[Eval] draw confusion failed: {e}")

    # CSV: metrics
    try:
        rows_csv = []
        for r in results:
            if r is None:
                continue
            rows_csv.append({
                "stage": r["stage"],
                "year": year,
                "n_tokens": r["n_tokens"],
                "n_pixels": r["n_pixels"],
                "accuracy": r["acc"],
                "macro_precision": r["macro_prec"],
                "macro_recall": r["macro_rec"],
                "macro_f1": r["macro_f1"],
                "kappa": r["kappa"],
            })
        if rows_csv:
            dfm = pd.DataFrame(rows_csv)
            dfm.to_csv(os.path.join(out_root, "v4_evaluation_metrics.csv"), index=False)
            print(f"[Eval] metrics CSV -> {os.path.join(out_root, 'v4_evaluation_metrics.csv')}")
    except Exception as e:
        print(f"[Eval] metrics CSV failed: {e}")

    # AUC/AP
    try:
        auc_s1, ap_s1, auc_s2, ap_s2 = _compute_front_none_auc_ap(
            year=year,
            out_csv=os.path.join(out_root, "front_none_metrics.csv"),
        )
    except Exception as e:
        print(f"[Eval] AUC/AP failed: {e}")
        auc_s1 = ap_s1 = auc_s2 = ap_s2 = None

    # seasonal / distance（final）/ presence match（final）
    try:
        compute_seasonal_crossing_rates_v4(
            final_out_dir,
            out_csv_month=os.path.join(out_root, "seasonal_monthly_rates.csv"),
            out_csv_season=os.path.join(out_root, "seasonal_rates.csv"),
        )
    except Exception as e:
        print(f"[Eval] seasonal failed: {e}")
    try:
        compute_distance_stats_v4(
            final_out_dir,
            nc_0p5_dir,
            out_csv=os.path.join(out_root, "distance_stats.csv"),
        )
    except Exception as e:
        print(f"[Eval] distance failed: {e}")
    try:
        presence_match, presence_total, presence_pct = _compute_presence_match_stage4_5(
            year=year,
            out_csv=os.path.join(out_root, "stage4_5_presence_match.csv"),
        )
    except Exception as e:
        print(f"[Eval] presence failed: {e}")
        presence_match, presence_total, presence_pct = 0, 0, 0.0

    # LOG（0-5 全クラス）
    try:
        log_path = os.path.join(out_root, "v4_evaluation_summary.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== V4 Evaluation Summary (classes 0-5) ===\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write(f"Year: {year}\n\n")
            for r in results:
                if r is None:
                    continue
                f.write(f"[{r['stage']}]\n")
                f.write(f"  tokens={r['n_tokens']}, pixels={r['n_pixels']}\n")
                f.write(f"  Acc={r['acc']:.2f}, MacroP={r['macro_prec']:.2f}, MacroR={r['macro_rec']:.2f}, MacroF1={r['macro_f1']:.2f}, Kappa={r['kappa']:.4f}\n\n")
            if (auc_s1 is not None) and (ap_s1 is not None) and (auc_s2 is not None) and (ap_s2 is not None):
                f.write(f"Front/None AUC/AP: S1 AUC={auc_s1}, AP={ap_s1}; S2 AUC={auc_s2}, AP={ap_s2}\n")
            f.write("\nArtifacts:\n")
            f.write(f"  Confusion Figure: {os.path.join(out_root, 'v4_evaluation_confusion_matrices.png')}\n")
            f.write(f"  Metrics CSV: {os.path.join(out_root, 'v4_evaluation_metrics.csv')}\n")
            f.write(f"  AUC/AP CSV: {os.path.join(out_root, 'front_none_metrics.csv')}\n")
            f.write(f"  Seasonal Monthly CSV: {os.path.join(out_root, 'seasonal_monthly_rates.csv')}\n")
            f.write(f"  Seasonal CSV: {os.path.join(out_root, 'seasonal_rates.csv')}\n")
            f.write(f"  Distance CSV: {os.path.join(out_root, 'distance_stats.csv')}\n")
            if 'presence_total' in locals() and presence_total and presence_total > 0:
                f.write(f"  Presence match (stage4_5): {presence_pct:.2f}% ({presence_match}/{presence_total})\n")
        print(f"[Eval] summary log -> {log_path}")
    except Exception as e:
        print(f"[Eval] summary log failed: {e}")

    # LOG（front-only: 1-5）
    try:
        log_path_front = os.path.join(out_root, "v4_evaluation_summary_only_front.log")
        with open(log_path_front, "w", encoding="utf-8") as f2:
            f2.write("=== V4 Evaluation Summary (front-only: classes 1-5) ===\n")
            f2.write(f"Generated at: {datetime.now()}\n")
            f2.write(f"Year: {year}\n\n")
            for r in results:
                if r is None:
                    continue
                y_true_all = r.get("y_true_all", None)
                y_pred_all = r.get("y_pred_all", None)
                if y_true_all is None or y_pred_all is None:
                    continue
                mask = (y_true_all != 0)
                if mask.sum() == 0:
                    continue
                labels_front = [c for c in r["labels"] if c != 0]
                acc_f, mp_f, mr_f, mf_f, kappa_f = compute_metrics(y_true_all[mask], y_pred_all[mask], labels_front)
                f2.write(f"[{r['stage']}]\n")
                f2.write(f"  Acc(front-only)={acc_f:.2f}, MacroP={mp_f:.2f}, MacroR={mr_f:.2f}, MacroF1={mf_f:.2f}, Kappa={kappa_f:.4f}\n\n")
        # 追記: Presence match（stage4_5）の全体割合
        if 'presence_total' in locals() and presence_total and presence_total > 0:
            try:
                with open(log_path_front, "a", encoding="utf-8") as f3:
                    f3.write(f"Presence match (stage4_5): {presence_pct:.2f}% ({presence_match}/{presence_total})\n")
            except Exception:
                pass
        print(f"[Eval] front-only summary log -> {log_path_front}")
    except Exception as e:
        print(f"[Eval] front-only summary log failed: {e}")

    print(f"[Eval] done in {format_time(time.time() - t0)}")


if __name__ == "__main__":
    run_evaluation_v4()


__all__ = [
    "compute_metrics",
    "compute_seasonal_crossing_rates_v4",
    "compute_distance_stats_v4",
    "run_evaluation_v4",
]
