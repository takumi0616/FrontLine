import os
import gc
import numpy as np
import pandas as pd
import xarray as xr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import stage3_out_dir, stage4_svg_dir


def smooth_polyline(points, window_size=3):
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
    各前線クラス（1～5）ごとに二値マスクから骨格抽出し、ポリライン群に変換する。
    極小領域で Skeleton が失敗する場合は重心点のダブルポイントを返す。
    """
    from skan import Skeleton
    polylines = []
    for c in range(1, 6):
        mask = (class_map == c).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        try:
            skel = Skeleton(mask)
            for i in range(skel.n_paths):
                coords = skel.path_coordinates(i)  # (N, 2) with (row, col) in image index
                coords_int = np.rint(coords).astype(int)
                # 注意: 元コード準拠で lon/lat は反転参照
                points_geo = [(lon[::-1][col], lat[::-1][row]) for (row, col) in coords_int]
                polylines.append((c, points_geo))
        except ValueError as e:
            # 非常に小さい領域で生じるエラーを回避
            ys, xs = np.nonzero(mask)
            if len(ys) > 0:
                centroid_row = int(np.round(np.mean(ys)))
                centroid_col = int(np.round(np.mean(xs)))
                pt = (lon[::-1][centroid_col], lat[::-1][centroid_row])
                polylines.append((c, [pt, pt]))
            else:
                # 空マスク
                continue
    return polylines


def save_polylines_as_svg(polylines, viewBox, output_path, smoothing_window=3):
    """
    前線別色の polyline を持つシンプルな SVG を出力する。
    viewBox = (min_lon, min_lat, width, height)
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

    # グリッド目盛り（簡易）
    lon_interval = width / 10.0 if width != 0 else 1
    lat_interval = height / 10.0 if height != 0 else 1
    for lx in np.arange(min_lon, min_lon + width + lon_interval, lon_interval):
        svg_lines.append('<line x1="{0:.4f}" y1="{1:.4f}" x2="{0:.4f}" y2="{2:.4f}" stroke="#CCCCCC" stroke-width="0.1" />'.format(lx, min_lat, min_lat+height))
        svg_lines.append('<text x="{0:.4f}" y="{1:.4f}" font-size="0.5" fill="#000000" text-anchor="middle" dy="0.5">{0:.2f}</text>'.format(lx, min_lat+height))
    for ly in np.arange(min_lat, min_lat + height + lat_interval, lat_interval):
        svg_lines.append('<line x1="{0:.4f}" y1="{1:.4f}" x2="{2:.4f}" y2="{1:.4f}" stroke="#CCCCCC" stroke-width="0.1" />'.format(min_lon, ly, min_lon+width))
        svg_lines.append('<text x="{0:.4f}" y="{1:.4f}" font-size="0.5" fill="#000000" text-anchor="start" dx="0.2" dy="0.3">{1:.2f}</text>'.format(min_lon, ly))

    for cls_id, points in polylines:
        pts = smooth_polyline(points, window_size=smoothing_window) if smoothing_window and smoothing_window > 1 else points
        points_str = " ".join("{0:.4f},{1:.4f}".format(pt[0], pt[1]) for pt in pts)
        color = class_colors.get(cls_id, "#000000")
        svg_lines.append('<polyline fill="none" stroke="{0}" stroke-width="0.5" points="{1}" />'.format(color, points_str))

    svg_lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in svg_lines:
            f.write(line + "\n")
    print("SVG saved:", output_path)


def evaluate_stage4(stage3_nc_dir, output_svg_dir):
    """
    Stage3 の骨格 class_map をポリライン化し、SVG として保存する。
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

        # viewBox 定義（lon/lat 反転参照に合わせる）
        lon_fixed = lon[::-1]
        lat_fixed = lat[::-1]
        min_lon_val = float(np.min(lon_fixed))
        max_lon_val = float(np.max(lon_fixed))
        min_lat_val = float(np.min(lat_fixed))
        max_lat_val = float(np.max(lat_fixed))
        viewBox = (min_lon_val, min_lat_val, max_lon_val - min_lon_val, max_lat_val - min_lat_val)

        output_path = os.path.join(output_svg_dir, f"skeleton_{date_str}.svg")
        save_polylines_as_svg(polylines, viewBox, output_path, smoothing_window=3)

        del class_map, lat, lon, polylines
        gc.collect()


def run_stage4():
    evaluate_stage4(stage3_out_dir, stage4_svg_dir)
    print("【Stage4 Improved】 SVG 出力処理が完了しました。")


__all__ = ["smooth_polyline", "extract_polylines_using_skan", "save_polylines_as_svg", "evaluate_stage4", "run_stage4"]
