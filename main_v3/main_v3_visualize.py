import os
import gc
import time
import glob
import cv2
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import multiprocessing
import subprocess
import shutil
import tempfile

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import (
    CFG, print_memory_usage, format_time,
    nc_gsm_dir, nc_0p5_dir,
    stage1_out_dir, stage2_out_dir, stage3_out_dir,
    output_visual_dir
)


def create_comparison_videos(
    image_folder: str = CFG["VIDEO"]["image_folder"],
    output_folder: str = CFG["VIDEO"]["output_folder"],
    frame_rate: int = CFG["VIDEO"]["frame_rate"],
    low_res_scale: int = CFG["VIDEO"]["low_res_scale"],
    low_res_frame_rate: int = CFG["VIDEO"]["low_res_frame_rate"],
):
    """
    comparison_YYYYMMDDHHMM.png を月毎＆通年の動画に連結し、低解像度版も作成
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    image_files = glob.glob(os.path.join(image_folder, "comparison_*.png"))
    image_files.sort(key=lambda x: os.path.basename(x))
    from collections import defaultdict
    monthly_images = defaultdict(list)

    for img_file in image_files:
        timestamp = os.path.basename(img_file).replace("comparison_", "").replace(".png", "")
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

            output_video = os.path.join(output_folder, f"comparison_{month}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

            for image in images:
                img = cv2.imread(image)
                video.write(img)

            video.release()
            print(f"[動画作成] {month} の動画を保存しました。")
        else:
            print(f"[動画作成] {month} の画像がありません。")

    if all_image_files:
        year = CFG.get("EVAL", {}).get("year", 2023)
        all_image_files.sort(key=lambda x: os.path.basename(x))
        frame = cv2.imread(all_image_files[0])
        height, width, layers = frame.shape

        output_video_all = os.path.join(output_folder, f"comparison_{year}_full_year.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
            temp_image_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(temp_image_path, img_small)

        output_video_low = os.path.join(output_folder, f"comparison_{year}_full_year_low.mp4")

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-r", str(low_res_frame_rate),
            "-i", os.path.join(temp_dir, "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vcodec", "libx264",
            "-crf", "30",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            output_video_low,
        ]

        subprocess.run(ffmpeg_cmd)
        shutil.rmtree(temp_dir)

        print("[動画作成] 1月から12月までの統合動画（低画質版）を保存しました。")
    else:
        print("[動画作成] 年間動画用の画像ファイルが見つかりませんでした。")


def process_single_time(args):
    """
    並列実行される単一時刻の可視化ワーカー
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    (time_str,
     stage1_nc_path,
     stage2_nc_path,
     stage3_nc_path,
     original_nc_dir,
     nc_gsm_alt,
     out_dir,
     class_colors,
     cmap,
     norm,
     pressure_levels,
     pressure_norm,
     cmap_pressure) = args

    output_filename = f"comparison_{time_str}.png"
    output_path = os.path.join(out_dir, output_filename)

    if os.path.exists(output_path):
        print(f"{output_filename} は既に存在します。スキップします。")
        return

    ds_s1 = xr.open_dataset(stage1_nc_path)
    probs_s1 = ds_s1["probabilities"].isel(time=0).values
    class_map_s1 = np.argmax(probs_s1, axis=-1)
    lat = ds_s1["lat"].values
    lon = ds_s1["lon"].values
    ds_s1.close()
    del ds_s1, probs_s1
    gc.collect()

    ds_s2 = xr.open_dataset(stage2_nc_path)
    probs_s2 = ds_s2["probabilities"].isel(time=0).values
    class_map_s2 = np.argmax(probs_s2, axis=-1)
    ds_s2.close()
    del ds_s2, probs_s2
    gc.collect()

    ds_s3 = xr.open_dataset(stage3_nc_path)
    class_map_s3 = ds_s3["class_map"].isel(time=0).values
    ds_s3.close()
    del ds_s3
    gc.collect()

    month_str = time_str[:6]
    original_file = os.path.join(original_nc_dir, f"{month_str}.nc")
    if not os.path.exists(original_file):
        print(f"元の前線データが見つかりません: {original_file}")
        return
    ds_orig = xr.open_dataset(original_file)

    time_dt = pd.to_datetime(time_str, format="%Y%m%d%H%M")
    if time_dt in ds_orig["time"]:
        orig_data = ds_orig.sel(time=time_dt)
    else:
        time_diff = np.abs(ds_orig["time"] - time_dt)
        min_time_diff = time_diff.min()
        if min_time_diff <= np.timedelta64(3, "h"):
            nearest_time = ds_orig["time"].values[time_diff.argmin()]
            orig_data = ds_orig.sel(time=nearest_time)
            print(f"時間が一致しないため、最も近い時間 {nearest_time} を使用します。")
        else:
            print(f"時間 {time_str} が元のデータに存在しません: {original_file}")
            ds_orig.close()
            return

    class_map_orig = np.zeros((len(lat), len(lon)), dtype=np.int64)
    var_names = {1: "warm", 2: "cold", 3: "stationary", 4: "occluded", 5: "warm_cold"}
    for cid, varn in var_names.items():
        if varn in orig_data.data_vars:
            mask = orig_data[varn].values
            class_map_orig[mask == 1] = cid
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
            ds_times = pd.to_datetime(ds_nc7["time"].values)
            if time_dt in ds_times:
                lowcenter_time_idx = int(np.where(ds_times == time_dt)[0][0])
            else:
                timediffs = np.abs(ds_times - time_dt)
                minidx = timediffs.argmin()
                if timediffs[minidx] <= pd.Timedelta(hours=3):
                    lowcenter_time_idx = minidx
                    print(f"[低気圧中心] 時間が一致せず,最も近い {ds_nc7['time'].values[minidx]} を使用")
                else:
                    print(f"[低気圧中心] 時間不一致、一致するデータなし: {time_str}")
            if lowcenter_time_idx is not None:
                lowcenter_arr = ds_nc7["surface_low_center"].isel(time=lowcenter_time_idx).values
                low_mask = (lowcenter_arr == 1)
                low_center_exists = True
            ds_nc7.close()
        except Exception as e:
            print(f"[低気圧中心] 読み取り失敗 ({nc7_file}, {time_str}): {e}")
            if "ds_nc7" in locals():
                ds_nc7.close()
    else:
        print(f"[低気圧中心] ファイルが見つかりません: {nc7_file} ({time_str})")

    gsm_file = os.path.join(nc_gsm_alt, f"gsm{month_str}.nc")
    if not os.path.exists(gsm_file):
        print(f"GSMデータが見つかりません: {gsm_file}")
        return
    ds_gsm = xr.open_dataset(gsm_file)
    if time_dt in ds_gsm["time"]:
        gsm_dat = ds_gsm.sel(time=time_dt)
    else:
        time_diff = np.abs(ds_gsm["time"] - time_dt)
        min_time_diff = time_diff.min()
        if min_time_diff <= np.timedelta64(3, "h"):
            nearest_time = ds_gsm["time"].values[time_diff.argmin()]
            gsm_dat = ds_gsm.sel(time=nearest_time)
            print(f"GSM時間が一致しないため、最も近い時間 {nearest_time} を使用します。")
        else:
            print(f"時間 {time_str} がGSMデータに存在しません: {gsm_file}")
            ds_gsm.close()
            return
    if "surface_prmsl" in gsm_dat:
        surface_prmsl = gsm_dat["surface_prmsl"].values
    else:
        print(f"'surface_prmsl' 変数が存在しません: {gsm_file}")
        ds_gsm.close()
        return
    ds_gsm.close()
    del ds_gsm, gsm_dat
    gc.collect()

    area_mean = np.nanmean(surface_prmsl)
    pressure_dev = surface_prmsl - area_mean

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(24, 6))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.1)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax0 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = plt.subplot(gs[1], projection=ccrs.PlateCarree())
    ax2 = plt.subplot(gs[2], projection=ccrs.PlateCarree())
    ax3 = plt.subplot(gs[3], projection=ccrs.PlateCarree())
    axes = [ax0, ax1, ax2, ax3]

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")
        ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"))
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        ax.tick_params(labelsize=8)

    pressure_vmin = CFG["VISUALIZATION"]["pressure_vmin"]
    pressure_vmax = CFG["VISUALIZATION"]["pressure_vmax"]
    pressure_levels = np.linspace(pressure_vmin, pressure_vmax, CFG["VISUALIZATION"]["pressure_levels"])
    pressure_norm = mcolors.Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
    cmap_pressure = plt.get_cmap("RdBu_r")

    for ax in axes:
        ax.contourf(
            lon_grid, lat_grid, pressure_dev,
            levels=pressure_levels, cmap=cmap_pressure, extend="both",
            norm=pressure_norm, transform=ccrs.PlateCarree(), zorder=0
        )
        ax.contour(
            lon_grid, lat_grid, pressure_dev,
            levels=pressure_levels, colors="black", linestyles="--", linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=1
        )

    class_colors = CFG["VISUALIZATION"]["class_colors"]
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im1 = ax0.pcolormesh(
        lon_grid, lat_grid, class_map_s1,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax0.set_title(f"Stage1 結果\n{time_str}")

    im2 = ax1.pcolormesh(
        lon_grid, lat_grid, class_map_s2,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax1.set_title(f"Stage2 結果\n{time_str}")

    im3 = ax2.pcolormesh(
        lon_grid, lat_grid, class_map_s3,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax2.set_title(f"Stage3 結果（スケルトン化）\n{time_str}")

    im4 = ax3.pcolormesh(
        lon_grid, lat_grid, class_map_orig,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax3.set_title(f"元の前線データ\n{time_str}")

    # low centers (optional)
    if low_center_exists and (low_mask is not None) and (low_mask.shape == (lat.size, lon.size)):
        y_idx, x_idx = np.where(low_mask)
        low_lats = lat[y_idx]
        low_lons = lon[x_idx]
        for ax in axes:
            ax.plot(low_lons, low_lats, "rx", markersize=8, markeredgewidth=2, zorder=6, label="低気圧中心")
    else:
        print(f"低気圧中心データ取得不可: {nc7_file}, {time_str}")

    cax = plt.subplot(gs[4])
    sm = plt.cm.ScalarMappable(cmap=cmap_pressure, norm=pressure_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("海面更正気圧の偏差 (hPa)")

    plt.subplots_adjust(wspace=0.1)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    del class_map_s1, class_map_s2, class_map_s3, class_map_orig, lon_grid, lat_grid, pressure_dev
    gc.collect()


def visualize_results(
    stage1_nc_dir: str,
    stage2_nc_dir: str,
    stage3_nc_dir: str,
    original_nc_dir: str,
    output_dir: str
):
    """
    Stage1/2/3および元データを並べた比較画像 comparison_*.png を生成
    """
    print("可視化処理を開始します。")
    os.makedirs(output_dir, exist_ok=True)

    stage1_files = sorted([f for f in os.listdir(stage1_nc_dir) if f.endswith(".nc")])
    stage2_files = sorted([f for f in os.listdir(stage2_nc_dir) if f.endswith(".nc")])
    stage3_files = sorted([f for f in os.listdir(stage3_nc_dir) if f.endswith(".nc")])

    stage1_dict = {f.split("_")[1].split(".")[0]: os.path.join(stage1_nc_dir, f) for f in stage1_files}
    stage2_dict = {f.split("_")[1].split(".")[0]: os.path.join(stage2_nc_dir, f) for f in stage2_files}
    stage3_dict = {f.split("_")[1].split(".")[0]: os.path.join(stage3_nc_dir, f) for f in stage3_files}

    common_times = sorted(set(stage1_dict.keys()) & set(stage2_dict.keys()) & set(stage3_dict.keys()))
    print(f"共通の時間数: {len(common_times)}")

    class_colors = CFG["VISUALIZATION"]["class_colors"]
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    pressure_vmin = CFG["VISUALIZATION"]["pressure_vmin"]
    pressure_vmax = CFG["VISUALIZATION"]["pressure_vmax"]
    pressure_levels = np.linspace(pressure_vmin, pressure_vmax, CFG["VISUALIZATION"]["pressure_levels"])
    pressure_norm = mcolors.Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
    cmap_pressure = plt.get_cmap("RdBu_r")

    nc_gsm_alt = nc_gsm_dir

    inputs = []
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
            cmap_pressure,
        ))

    if not inputs:
        print("可視化対象がありません。")
        return

    print("キャッシュ・地図データ生成のため、最初の1件のみシリアル処理します。")
    process_single_time(inputs[0])

    num_processes = max(1, multiprocessing.cpu_count() // CFG["VISUALIZATION"]["parallel_factor"])
    print(f"{num_processes}個のプロセスで並列処理を開始します。")
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_single_time, inputs), total=len(inputs), desc="可視化処理中"))
    print("可視化処理が完了しました。")


def run_visualization():
    print_memory_usage("Start Visualization")
    vis_start = time.time()

    visualize_start = time.time()
    visualize_results(
        stage1_nc_dir=stage1_out_dir,
        stage2_nc_dir=stage2_out_dir,
        stage3_nc_dir=stage3_out_dir,
        original_nc_dir=nc_0p5_dir,
        output_dir=output_visual_dir,
    )
    visualize_end = time.time()
    print(f"[Visualization] 結果の可視化処理時間: {format_time(visualize_end - visualize_start)}")

    cleanup_start = time.time()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    cleanup_end = time.time()
    print(f"[Visualization] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    vis_end = time.time()
    print(f"[Visualization] 全体の実行時間: {format_time(vis_end - vis_start)}")
    print_memory_usage("After Visualization")


__all__ = ["create_comparison_videos", "visualize_results", "run_visualization"]
