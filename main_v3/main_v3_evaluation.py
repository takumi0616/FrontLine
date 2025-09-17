import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import (
    CFG, format_time,
    stage1_out_dir, stage2_out_dir, stage3_out_dir,
    model_s1_save_dir, model_s2_save_dir,
    output_visual_dir, nc_0p5_dir
)


def compute_metrics(y_true, y_pred, labels):
    acc = np.mean(y_true == y_pred) * 100.0
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_prec *= 100.0
    macro_rec *= 100.0
    macro_f1 *= 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, macro_prec, macro_rec, macro_f1, kappa


def compute_seasonal_crossing_rates(
    stage3_nc_dir,
    out_csv_month=os.path.join(os.path.dirname(output_visual_dir), "seasonal_monthly_rates.csv"),
    out_csv_season=os.path.join(os.path.dirname(output_visual_dir), "seasonal_rates.csv"),
):
    """
    Stage3（骨格 class_map）を用いて、月別・季節別の通過頻度を算出する。
    方法（簡易版）:
      - 各時刻の class_map>0 の画素を「その日の通過」とみなす
      - 同一セルで同日中は重複カウントしない（24hブランキングに近似）
      - 月内の「日次カウント/日数」をセル平均して月次レートとする
      - 季節（DJF, MAM, JJA, SON）は月次から集計
    出力:
      - 月次: year, month, rate_mean
      - 季節: year, season(DJF/MAM/JJA/SON), rate_mean
    """
    os.makedirs(os.path.dirname(out_csv_month), exist_ok=True)
    files = sorted([f for f in os.listdir(stage3_nc_dir) if f.startswith("skeleton_") and f.endswith(".nc")])
    if not files:
        print("[Seasonal] No stage3 skeleton files found.")
        return

    by_month = defaultdict(list)
    sample_lat, sample_lon = None, None
    for f in files:
        p = os.path.join(stage3_nc_dir, f)
        ds = xr.open_dataset(p)
        tval = ds["time"].values[0]
        t_dt = pd.to_datetime(tval)
        if sample_lat is None:
            sample_lat = ds["lat"].values
            sample_lon = ds["lon"].values
        ds.close()
        by_month[(t_dt.year, t_dt.month)].append((t_dt, p))

    monthly_rows = []
    for (y, m), items in sorted(by_month.items()):
        items.sort(key=lambda x: x[0])
        H = len(sample_lat)
        W = len(sample_lon)
        last_date = np.full((H, W), None, dtype=object)
        count = np.zeros((H, W), dtype=np.int32)
        dates_in_month = sorted({d.date() for d, _ in items})
        days_in_month = len(dates_in_month) if len(dates_in_month) > 0 else 1

        for dtime, path in items:
            ds = xr.open_dataset(path)
            cls = ds["class_map"].isel(time=0).values.astype(np.int64)
            ds.close()
            front = (cls > 0).astype(np.uint8)
            cur_date = dtime.date()
            need = np.ones((H, W), dtype=bool)
            never = (last_date == None)  # noqa: E711
            need &= (never | (last_date != cur_date))
            inc = (front == 1) & need
            count[inc] += 1
            last_date[front == 1] = cur_date

        rate = count.astype(np.float32) / max(1, days_in_month)
        monthly_rows.append({"year": y, "month": m, "rate_mean": float(np.mean(rate))})

    df_month = pd.DataFrame(monthly_rows)
    df_month.to_csv(out_csv_month, index=False)
    print(f"[Seasonal] Monthly rates -> {out_csv_month}")

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
        by_season = defaultdict(list)
        for r in monthly_rows:
            if r["year"] != year:
                continue
            s = season_of(r["month"])
            by_season[s].append(r["rate_mean"])
        for s, vals in by_season.items():
            if len(vals) > 0:
                season_rows.append({"year": year, "season": s, "rate_mean": float(np.mean(vals))})
    df_season = pd.DataFrame(season_rows)
    df_season.to_csv(out_csv_season, index=False)
    print(f"[Seasonal] Seasonal rates -> {out_csv_season}")


def compute_distance_stats(
    stage3_nc_dir,
    gt_dir,
    out_csv=os.path.join(os.path.dirname(output_visual_dir), "distance_stats.csv"),
):
    """
    距離評価（空間統計）:
      - 各時刻で GT（5chの任意フロント）を2値化
      - 予測（Stage3）骨格のセルから GT までの最近傍距離（ピクセル）を距離変換で算出
      - 代表統計（mean/median/p90）を km 換算で出力
    換算: グリッド解像度から近似。dx ≈ Δlon*111km*cos(lat_mean), dy ≈ Δlat*111km。
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    files = sorted([f for f in os.listdir(stage3_nc_dir) if f.startswith("skeleton_") and f.endswith(".nc")])
    if not files:
        print("[Distance] No stage3 skeleton files found.")
        return

    rows = []

    ds0 = xr.open_dataset(os.path.join(stage3_nc_dir, files[0]))
    lat = ds0["lat"].values
    lon = ds0["lon"].values
    ds0.close()
    lat_sorted = np.sort(np.unique(lat))
    lon_sorted = np.sort(np.unique(lon))
    dlat = np.median(np.abs(np.diff(lat_sorted))) if len(lat_sorted) > 1 else 1.0
    dlon = np.median(np.abs(np.diff(lon_sorted))) if len(lon_sorted) > 1 else 1.0
    lat_mean = float(np.mean(lat))
    dy_km = 111.0 * float(dlat)
    dx_km = 111.0 * float(dlon) * max(0.1, np.cos(np.deg2rad(lat_mean)))
    pix_km = float((dx_km + dy_km) / 2.0)

    from scipy.ndimage import distance_transform_edt

    for f in tqdm(files, desc="[Distance]"):
        p = os.path.join(stage3_nc_dir, f)
        ds = xr.open_dataset(p)
        tval = ds["time"].values[0]
        t_dt = pd.to_datetime(tval)
        pred = ds["class_map"].isel(time=0).values
        ds.close()

        month_str = t_dt.strftime("%Y%m")
        gt_path = os.path.join(gt_dir, f"{month_str}.nc")
        if not os.path.exists(gt_path):
            continue
        dsg = xr.open_dataset(gt_path)
        if t_dt in dsg["time"]:
            gsel = dsg.sel(time=t_dt).to_array().values
        else:
            diff = np.abs(dsg["time"].values - np.datetime64(t_dt))
            idx = diff.argmin()
            if diff[idx] <= np.timedelta64(3, "h"):
                gsel = dsg.sel(time=dsg["time"][idx]).to_array().values
            else:
                dsg.close()
                continue
        dsg.close()
        gt_bin = (gsel.sum(axis=0) > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        dist_to_gt = distance_transform_edt(1 - gt_bin)
        d_pred_to_gt = dist_to_gt[pred_bin == 1].astype(np.float32)
        if d_pred_to_gt.size == 0:
            continue
        dkm = d_pred_to_gt * pix_km
        rows.append(
            {
                "time": t_dt.strftime("%Y-%m-%d %H:%M"),
                "mean_km": float(np.mean(dkm)),
                "median_km": float(np.median(dkm)),
                "p90_km": float(np.percentile(dkm, 90.0)),
                "count": int(dkm.size),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"[Distance] Stats -> {out_csv}")
    else:
        print("[Distance] No distance rows computed.")


def run_evaluation():
    year = CFG.get("EVAL", {}).get("year", 2023)
    print(f"[Evaluation] Start evaluation for {year} data (6 classes).")

    ratio_buf_s1 = {c: [] for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_buf_s2 = {c: [] for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_buf_s3 = {c: [] for c in range(1, CFG["STAGE1"]["num_classes"])}

    stage1_files = sorted([f for f in os.listdir(stage1_out_dir) if f.startswith(f"prob_{year}") and f.endswith(".nc")])
    stage2_files = sorted([f for f in os.listdir(stage2_out_dir) if f.startswith(f"refined_{year}") and f.endswith(".nc")])
    stage3_files = sorted([f for f in os.listdir(stage3_out_dir) if f.startswith(f"skeleton_{year}") and f.endswith(".nc")])

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
    any_prob_s1_list = []
    any_prob_s2_list = []
    y_true_front_list = []
    if len(common_keys) == 0:
        print(f"[Evaluation] Not found any {year} common times among stage1/2/3.")
        return

    stage1_pred_list = []
    stage2_pred_list = []
    stage3_pred_list = []
    gt_list = []

    for key in common_keys:
        ds1 = xr.open_dataset(stage1_dict[key])
        probs_s1 = ds1["probabilities"].isel(time=0).values  # (H,W,6)
        ds1.close()
        pred_s1 = np.argmax(probs_s1, axis=-1)

        ds2 = xr.open_dataset(stage2_dict[key])
        probs_s2 = ds2["probabilities"].isel(time=0).values
        ds2.close()
        pred_s2 = np.argmax(probs_s2, axis=-1)

        ds3 = xr.open_dataset(stage3_dict[key])
        pred_s3 = ds3["class_map"].isel(time=0).values
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
        t_dt = pd.to_datetime(key, format="%Y%m%d%H%M")
        if t_dt in ds_gt["time"]:
            front_data = ds_gt.sel(time=t_dt).to_array().values
        else:
            diff_ = np.abs(ds_gt["time"].values - np.datetime64(t_dt))
            idx_ = diff_.argmin()
            if diff_[idx_] <= np.timedelta64(3, "h"):
                front_data = ds_gt.sel(time=ds_gt["time"][idx_]).to_array().values
            else:
                print(f"No GT time close for {key} in {gtf}")
                ds_gt.close()
                gt_list.append(np.zeros_like(pred_s1))
                continue
        ds_gt.close()
        gt_map = np.zeros_like(pred_s1)
        for c in range(5):
            mask = (front_data[c, :, :] == 1)
            gt_map[mask] = c + 1
        gt_list.append(gt_map)

        any_s1 = np.max(probs_s1[..., 1:6], axis=-1)
        any_s2 = np.max(probs_s2[..., 1:6], axis=-1)
        none_mask_s1 = (np.argmax(probs_s1, axis=-1) == 0)
        none_mask_s2 = (np.argmax(probs_s2, axis=-1) == 0)
        any_s1[none_mask_s1] = 0.0
        any_s2[none_mask_s2] = 0.0
        any_prob_s1_list.append(any_s1.reshape(-1))
        any_prob_s2_list.append(any_s2.reshape(-1))
        y_true_front_list.append((gt_map.reshape(-1) != 0).astype(np.uint8))

        for c in range(1, CFG["STAGE1"]["num_classes"]):
            gt_cnt = int((gt_map == c).sum())
            if gt_cnt > 0:
                ratio_buf_s1[c].append(float((pred_s1 == c).sum() / gt_cnt))
                ratio_buf_s2[c].append(float((pred_s2 == c).sum() / gt_cnt))
                ratio_buf_s3[c].append(float((pred_s3 == c).sum() / gt_cnt))

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

    df_metrics_full = pd.DataFrame(
        {
            "Accuracy (%)": [acc1, acc2, acc3],
            "Macro Precision (%)": [mp1, mp2, mp3],
            "Macro Recall (%)": [mr1, mr2, mr3],
            "Macro F1 (%)": [mf1, mf2, mf3],
            "Cohen Kappa": [kappa1, kappa2, kappa3],
        },
        index=["Stage1", "Stage2", "Stage3"],
    )

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

    df_metrics_filtered = pd.DataFrame(
        {
            "Accuracy (%)": [acc1_f, acc2_f, acc3_f],
            "Macro Precision (%)": [mp1_f, mp2_f, mp3_f],
            "Macro Recall (%)": [mr1_f, mr2_f, mr3_f],
            "Macro F1 (%)": [mf1_f, mf2_f, mf3_f],
            "Cohen Kappa": [kappa1_f, kappa2_f, kappa3_f],
        },
        index=["Stage1", "Stage2", "Stage3"],
    )

    ratio_s1_cls = {c: (float(np.mean(ratio_buf_s1[c])) if len(ratio_buf_s1[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_s2_cls = {c: (float(np.mean(ratio_buf_s2[c])) if len(ratio_buf_s2[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}
    ratio_s3_cls = {c: (float(np.mean(ratio_buf_s3[c])) if len(ratio_buf_s3[c]) > 0 else 0.0) for c in range(1, CFG["STAGE1"]["num_classes"])}

    rmse_s1_cls, rmse_s2_cls, rmse_s3_cls = {}, {}, {}
    for c in range(1, CFG["STAGE1"]["num_classes"]):
        gt_bin = (gt_all == c).astype(np.float32)
        pred_s1_bin = (stage1_all == c).astype(np.float32)
        pred_s2_bin = (stage2_all == c).astype(np.float32)
        pred_s3_bin = (stage3_all == c).astype(np.float32)
        rmse_s1_cls[c] = np.sqrt(np.mean((pred_s1_bin - gt_bin) ** 2))
        rmse_s2_cls[c] = np.sqrt(np.mean((pred_s2_bin - gt_bin) ** 2))
        rmse_s3_cls[c] = np.sqrt(np.mean((pred_s3_bin - gt_bin) ** 2))

    # --- Extra summary stats for figure/log ---
    try:
        # Compute AUC/AP (Front vs None) in-place for summary embedding
        if len(y_true_front_list) > 0:
            y_true_front_all_fig = np.concatenate(y_true_front_list, axis=0)
            any_prob_s1_all_fig = np.concatenate(any_prob_s1_list, axis=0)
            any_prob_s2_all_fig = np.concatenate(any_prob_s2_list, axis=0)
            mask_valid_fig = np.isfinite(any_prob_s1_all_fig) & np.isfinite(any_prob_s2_all_fig)
            y_true_front_all_fig = y_true_front_all_fig[mask_valid_fig]
            any_prob_s1_all_fig = any_prob_s1_all_fig[mask_valid_fig]
            any_prob_s2_all_fig = any_prob_s2_all_fig[mask_valid_fig]
            auc_s1_fig = roc_auc_score(y_true_front_all_fig, any_prob_s1_all_fig) if y_true_front_all_fig.sum() > 0 else float("nan")
            auc_s2_fig = roc_auc_score(y_true_front_all_fig, any_prob_s2_all_fig) if y_true_front_all_fig.sum() > 0 else float("nan")
            ap_s1_fig = average_precision_score(y_true_front_all_fig, any_prob_s1_all_fig) if y_true_front_all_fig.sum() > 0 else float("nan")
            ap_s2_fig = average_precision_score(y_true_front_all_fig, any_prob_s2_all_fig) if y_true_front_all_fig.sum() > 0 else float("nan")
        else:
            auc_s1_fig = auc_s2_fig = ap_s1_fig = ap_s2_fig = float("nan")
    except Exception:
        auc_s1_fig = auc_s2_fig = ap_s1_fig = ap_s2_fig = float("nan")

    # Precompute seasonal and distance CSVs to allow summary inclusion
    try:
        compute_seasonal_crossing_rates(
            stage3_out_dir,
            out_csv_month=os.path.join(os.path.dirname(output_visual_dir), "seasonal_monthly_rates.csv"),
            out_csv_season=os.path.join(os.path.dirname(output_visual_dir), "seasonal_rates.csv"),
        )
    except Exception:
        pass
    try:
        compute_distance_stats(
            stage3_out_dir,
            nc_0p5_dir,
            out_csv=os.path.join(os.path.dirname(output_visual_dir), "distance_stats.csv"),
        )
    except Exception:
        pass

    # Read seasonal / distance summaries if available
    seasonal_month_csv = os.path.join(os.path.dirname(output_visual_dir), "seasonal_monthly_rates.csv")
    seasonal_csv = os.path.join(os.path.dirname(output_visual_dir), "seasonal_rates.csv")
    distance_csv = os.path.join(os.path.dirname(output_visual_dir), "distance_stats.csv")

    seasonal_month_mean = None
    seasonal_means_by_season = None
    dist_mean_mean = dist_median_mean = dist_p90_mean = None

    try:
        if os.path.exists(seasonal_month_csv):
            dfm = pd.read_csv(seasonal_month_csv)
            if "rate_mean" in dfm.columns and len(dfm) > 0:
                seasonal_month_mean = float(np.mean(dfm["rate_mean"]))
    except Exception:
        seasonal_month_mean = None

    try:
        if os.path.exists(seasonal_csv):
            dfs = pd.read_csv(seasonal_csv)
            if {"season", "rate_mean"}.issubset(set(dfs.columns)) and len(dfs) > 0:
                seasonal_means_by_season = {
                    s: float(np.mean(dfs.loc[dfs["season"] == s, "rate_mean"]))
                    for s in ["DJF", "MAM", "JJA", "SON"]
                    if (dfs["season"] == s).any()
                }
    except Exception:
        seasonal_means_by_season = None

    try:
        if os.path.exists(distance_csv):
            dfd = pd.read_csv(distance_csv)
            if {"mean_km", "median_km", "p90_km"}.issubset(set(dfd.columns)) and len(dfd) > 0:
                dist_mean_mean = float(np.mean(dfd["mean_km"]))
                dist_median_mean = float(np.mean(dfd["median_km"]))
                dist_p90_mean = float(np.mean(dfd["p90_km"]))
    except Exception:
        dist_mean_mean = dist_median_mean = dist_p90_mean = None
    # --- End extra stats ---

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1.2, 1.3], hspace=0.5)

    def normalize_cm(cm):
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        return cm_norm

    cm_s1_norm = normalize_cm(cm_s1)
    cm_s2_norm = normalize_cm(cm_s2)
    cm_s3_norm = normalize_cm(cm_s3)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm_s1_norm,
        annot=cm_s1,
        fmt="d",
        cmap="Blues",
        xticklabels=label_all,
        yticklabels=label_all,
        ax=ax1,
        vmin=0,
        vmax=1.0,
    )
    ax1.set_title("Stage1 Confusion Matrix (All Classes)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        cm_s2_norm,
        annot=cm_s2,
        fmt="d",
        cmap="Blues",
        xticklabels=label_all,
        yticklabels=label_all,
        ax=ax2,
        vmin=0,
        vmax=1.0,
    )
    ax2.set_title("Stage2 Confusion Matrix (All Classes)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(
        cm_s3_norm,
        annot=cm_s3,
        fmt="d",
        cmap="Blues",
        xticklabels=label_all,
        yticklabels=label_all,
        ax=ax3,
        vmin=0,
        vmax=1.0,
    )
    ax3.set_title("Stage3 Confusion Matrix (All Classes)")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    ax_table_full = fig.add_subplot(gs[1, :])
    ax_table_full.axis("off")
    table_data_full = df_metrics_full.round(2).values
    row_labels_full = df_metrics_full.index.tolist()
    col_labels_full = df_metrics_full.columns.tolist()
    table_full = ax_table_full.table(
        cellText=table_data_full,
        rowLabels=row_labels_full,
        colLabels=col_labels_full,
        cellLoc="center",
        loc="center",
    )
    table_full.auto_set_font_size(False)
    table_full.set_fontsize(10)
    ax_table_full.set_title("Evaluation Metrics (All Classes)", fontweight="bold", pad=20)

    ax_table_filtered = fig.add_subplot(gs[2, :])
    ax_table_filtered.axis("off")
    table_data_filtered = df_metrics_filtered.round(2).values
    row_labels_filtered = df_metrics_filtered.index.tolist()
    col_labels_filtered = df_metrics_filtered.columns.tolist()
    table_filtered = ax_table_filtered.table(
        cellText=table_data_filtered,
        rowLabels=row_labels_filtered,
        colLabels=col_labels_filtered,
        cellLoc="center",
        loc="center",
    )
    table_filtered.auto_set_font_size(False)
    table_filtered.set_fontsize(10)
    ax_table_filtered.set_title("Evaluation Metrics (Front Only: Classes 1-5)", fontweight="bold", pad=20)

    ratio_text = (
        "Pixel-count ratio (pred/GT) 〈mean, cls1-〉\n"
        + "  S1: "
        + ", ".join([f"C{c}:{ratio_s1_cls[c]:.2f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
        + "\n"
        + "  S2: "
        + ", ".join([f"C{c}:{ratio_s2_cls[c]:.2f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
        + "\n"
        + "  S3: "
        + ", ".join([f"C{c}:{ratio_s3_cls[c]:.2f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
    )
    rmse_text = (
        "RMSE (cls1-)\n"
        + "  S1: "
        + ", ".join([f"C{c}:{rmse_s1_cls[c]:.3f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
        + "\n"
        + "  S2: "
        + ", ".join([f"C{c}:{rmse_s2_cls[c]:.3f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
        + "\n"
        + "  S3: "
        + ", ".join([f"C{c}:{rmse_s3_cls[c]:.3f}" for c in range(1, CFG["STAGE1"]["num_classes"])])
    )

    # Build enriched summary text for the figure footer
    extra_lines = []
    try:
        extra_lines.append(
            f"Front/None AUC/AP  S1: AUC={auc_s1_fig:.4f}, AP={ap_s1_fig:.4f} | "
            f"S2: AUC={auc_s2_fig:.4f}, AP={ap_s2_fig:.4f}"
        )
    except Exception:
        pass
    if seasonal_month_mean is not None:
        extra_lines.append(f"Seasonal crossing (monthly mean): {seasonal_month_mean:.4f}")
    if isinstance(seasonal_means_by_season, dict) and len(seasonal_means_by_season) > 0:
        parts = []
        for s in ["DJF", "MAM", "JJA", "SON"]:
            if s in seasonal_means_by_season:
                parts.append(f"{s}:{seasonal_means_by_season[s]:.4f}")
        if parts:
            extra_lines.append("Seasonal crossing by season: " + ", ".join(parts))
    if (dist_mean_mean is not None) and (dist_median_mean is not None) and (dist_p90_mean is not None):
        extra_lines.append(
            f"Distance stats mean (km): mean={dist_mean_mean:.2f}, median={dist_median_mean:.2f}, p90={dist_p90_mean:.2f}"
        )

    summary_text = (
        f"Time-wise Presence-set match  (%): S1 {ratio_s1:.2f}, S2 {ratio_s2:.2f}, S3 {ratio_s3:.2f}\n"
        f"{ratio_text}\n{rmse_text}"
        + ("\n" + "\n".join(extra_lines) if extra_lines else "")
    )
    fig.text(0.5, 0.02, summary_text, ha="center", va="bottom", fontsize=10)

    out_fig = os.path.join(os.path.dirname(output_visual_dir), "evaluation_summary.png")
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.savefig(out_fig, dpi=300)
    plt.close()

    try:
        v1_root = os.path.dirname(output_visual_dir)
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
            f.write("[Stage1]\n")
            f.write(pd.DataFrame(cm_s1).to_string(index=True, header=True))
            f.write("\n")
            f.write("[Stage2]\n")
            f.write(pd.DataFrame(cm_s2).to_string(index=True, header=True))
            f.write("\n")
            f.write("[Stage3]\n")
            f.write(pd.DataFrame(cm_s3).to_string(index=True, header=True))
            f.write("\n\n")

            f.write("Evaluation Metrics (All Classes)\n")
            f.write(df_metrics_full.round(4).to_string())
            f.write("\n\n")

            f.write("Evaluation Metrics (Front Only: Classes 1-5)\n")
            f.write(df_metrics_filtered.round(4).to_string())
            f.write("\n\n")

            # Additional Stats
            try:
                f.write("Additional Stats\n")
                f.write(
                    f"  Front/None AUC/AP: "
                    f"S1 AUC={auc_s1_fig:.4f}, AP={ap_s1_fig:.4f}; "
                    f"S2 AUC={auc_s2_fig:.4f}, AP={ap_s2_fig:.4f}\n"
                )
            except Exception:
                pass
            try:
                if seasonal_month_mean is not None:
                    f.write(f"  Seasonal crossing monthly mean: {seasonal_month_mean:.4f}\n")
                if isinstance(seasonal_means_by_season, dict) and seasonal_means_by_season:
                    parts = []
                    for s in ["DJF", "MAM", "JJA", "SON"]:
                        if s in seasonal_means_by_season:
                            parts.append(f"{s}:{seasonal_means_by_season[s]:.4f}")
                    if parts:
                        f.write("  Seasonal crossing by season: " + ", ".join(parts) + "\n")
            except Exception:
                pass
            try:
                if (dist_mean_mean is not None) and (dist_median_mean is not None) and (dist_p90_mean is not None):
                    f.write(
                        f"  Distance stats mean (km): mean={dist_mean_mean:.2f}, "
                        f"median={dist_median_mean:.2f}, p90={dist_p90_mean:.2f}\n"
                    )
            except Exception:
                pass
            f.write("\n")

            f.write("Loss History (from training)\n")
            try:
                s1_csv = os.path.join(model_s1_save_dir, "loss_history.csv")
                if os.path.exists(s1_csv):
                    df1 = pd.read_csv(s1_csv)
                    f.write("[Stage1]\n")
                    f.write(df1.to_string(index=False))
                    f.write("\n")
                else:
                    f.write("[Stage1] loss_history.csv not found.\n")
            except Exception as e:
                f.write(f"[Stage1] Error reading loss history: {e}\n")
            try:
                s2_csv = os.path.join(model_s2_save_dir, "loss_history.csv")
                if os.path.exists(s2_csv):
                    df2 = pd.read_csv(s2_csv)
                    f.write("[Stage2]\n")
                    f.write(df2.to_string(index=False))
                    f.write("\n")
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

    try:
        if len(y_true_front_list) > 0:
            y_true_front_all = np.concatenate(y_true_front_list, axis=0)
            any_prob_s1_all = np.concatenate(any_prob_s1_list, axis=0)
            any_prob_s2_all = np.concatenate(any_prob_s2_list, axis=0)
            mask_valid = np.isfinite(any_prob_s1_all) & np.isfinite(any_prob_s2_all)
            y_true_front_all = y_true_front_all[mask_valid]
            any_prob_s1_all = any_prob_s1_all[mask_valid]
            any_prob_s2_all = any_prob_s2_all[mask_valid]
            auc_s1 = roc_auc_score(y_true_front_all, any_prob_s1_all) if y_true_front_all.sum() > 0 else float("nan")
            auc_s2 = roc_auc_score(y_true_front_all, any_prob_s2_all) if y_true_front_all.sum() > 0 else float("nan")
            ap_s1 = average_precision_score(y_true_front_all, any_prob_s1_all) if y_true_front_all.sum() > 0 else float("nan")
            ap_s2 = average_precision_score(y_true_front_all, any_prob_s2_all) if y_true_front_all.sum() > 0 else float("nan")
            df_auc = pd.DataFrame({"metric": ["ROC_AUC", "AP"], "Stage1": [auc_s1, ap_s1], "Stage2": [auc_s2, ap_s2]})
            auc_path = os.path.join(os.path.dirname(output_visual_dir), "front_none_metrics.csv")
            df_auc.to_csv(auc_path, index=False)
            print(f"[Evaluation] Front/None AUC/AP -> {auc_path}")
        else:
            print("[Evaluation] No data for Front/None AUC.")
    except Exception as e:
        print(f"[Evaluation] AUC computation failed: {e}")

    try:
        compute_seasonal_crossing_rates(
            stage3_out_dir,
            out_csv_month=os.path.join(os.path.dirname(output_visual_dir), "seasonal_monthly_rates.csv"),
            out_csv_season=os.path.join(os.path.dirname(output_visual_dir), "seasonal_rates.csv"),
        )
    except Exception as e:
        print(f"[Evaluation] Seasonal rates failed: {e}")

    try:
        compute_distance_stats(
            stage3_out_dir,
            nc_0p5_dir,
            out_csv=os.path.join(os.path.dirname(output_visual_dir), "distance_stats.csv"),
        )
    except Exception as e:
        print(f"[Evaluation] Distance stats failed: {e}")

    print(f"[Evaluation] Done. Figure -> {os.path.join(os.path.dirname(output_visual_dir), 'evaluation_summary.png')}")


__all__ = [
    "compute_metrics",
    "compute_seasonal_crossing_rates",
    "compute_distance_stats",
    "run_evaluation",
]
