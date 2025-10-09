"""
概要:
    Stage4（SVG 出力）のモジュール。
    - Stage3 で得た骨格クラスマップ（class_map, H×W, 値域0..5）をポリラインへ変換し、SVG で可視化する
    - 線の平滑化、色分け、簡易グリッド描画、viewBox の自動設定などを行う

構成:
    - smooth_polyline(points, window_size): ポリラインの移動平均平滑化
    - extract_polylines_using_skan(class_map, lat, lon): class_map をポリライン群へ変換（skan を使用）
    - save_polylines_as_svg(polylines, viewBox, output_path, smoothing_window): ポリラインを SVG で保存
    - evaluate_stage4(stage3_nc_dir, output_svg_dir): skeleton_*.nc を一括処理して SVG 群を出力
    - run_stage4(): CFG の出力先設定を用いた実行エントリ

注意:
    - 緯度・経度配列と画像添字の向きに差異があるため、元実装準拠で lon/lat は反転参照している箇所がある
    - SVG の座標系は viewBox（min_lon, min_lat, width, height）で定義される
"""
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import CFG, stage3_out_dir, stage4_svg_dir


def smooth_polyline(points, window_size=3):
    """
    概要:
        ポリラインの座標列に移動平均的な平滑化を適用し、折れの激しい線を滑らかにする。

    入力:
        - points (List[Tuple[float,float]]): 座標列 [(x, y), ...]（地理座標を想定）
        - window_size (int): 平滑化窓のサイズ（奇数を推奨）。各点の前後 half 幅で平均化

    処理:
        - 各点 i についてインデックス範囲 [i-half, i+half] を取り、その区間の x, y の平均を計算
        - 端点では範囲を画像内にクリップし、点数は維持したまま平滑化

    出力:
        - smoothed (List[Tuple[float,float]]): 平滑化後の座標列
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
        Stage3 の骨格 class_map（0..5）の各クラスごとに、二値マスクから骨格パスを抽出し、
        地理座標系（lon, lat）でのポリライン群へ変換する。

    入力:
        - class_map (np.ndarray): 形状 (H, W) の整数ラベル（0=背景, 1..5=前線）
        - lat (np.ndarray): 形状 (H,) の緯度配列
        - lon (np.ndarray): 形状 (W,) の経度配列

    処理:
        - 各クラス c=1..5 について class_map==c を二値化
        - skan.Skeleton により骨格パスを列挙し、パス上の画像座標 (row, col) を最寄り整数に丸める
        - 元コードとの互換のため lon[::-1], lat[::-1] を用いて (x=lon, y=lat) に変換
        - 極小領域で骨格抽出が失敗した場合は、重心点のダブルポイント [pt, pt] を代用

    出力:
        - polylines (List[Tuple[int, List[Tuple[float,float]]]]):
            [(cls_id, [(x1,y1), (x2,y2), ...]), ...] のリスト
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
    概要:
        ポリライン群をクラス別色で描画したシンプルな SVG を生成・保存する。

    入力:
        - polylines (List[Tuple[int, List[Tuple[float,float]]]]):
            [(cls_id, [(x,y), ...]), ...] 形式のポリライン集合
        - viewBox (Tuple[float,float,float,float]):
            (min_lon, min_lat, width, height) 形式の SVG viewBox
        - output_path (str): 出力ファイルパス（.svg）
        - smoothing_window (int): ポリラインの平滑化窓。None/1 で平滑化なし

    処理:
        - CFG["VISUALIZATION"]["class_colors"] に従いクラス別の stroke 色を選択
        - graticule（簡易経緯線）と数値注記を描画
        - 指定があれば smooth_polyline で座標列を平滑化し、polyline 要素を連結

    出力:
        なし（副作用として SVG ファイルを output_path に保存）
    """
    # 色指定は設定（VISUALIZATION.class_colors）から取得
    class_colors = CFG["VISUALIZATION"]["class_colors"]
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
    概要:
        skeleton_*.nc（Stage3 出力）の class_map を読み取り、各タイムスタンプごとに SVG を生成する。

    入力:
        - stage3_nc_dir (str): Stage3 の出力 .nc ディレクトリ（skeleton_*.nc）
        - output_svg_dir (str): 出力 SVG ディレクトリ

    処理:
        - skeleton_*.nc を時系列順に走査
        - class_map, lat, lon を取得して extract_polylines_using_skan でポリライン化
        - lat/lon の範囲から viewBox を自動計算
        - save_polylines_as_svg で各時刻の SVG を "skeleton_YYYYMMDDHHMM.svg" として保存

    出力:
        なし（副作用として SVG ファイル群を保存）
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
        save_polylines_as_svg(
            polylines,
            viewBox,
            output_path,
            smoothing_window=CFG["STAGE4"]["smoothing_window"],  # 平滑化ウィンドウ（ポリラインの滑らかさ）
        )

        del class_map, lat, lon, polylines
        gc.collect()


def run_stage4():
    """
    概要:
        Stage4 のエントリ関数。CFG の標準パスに基づき、Stage3 の skeleton を SVG へ変換する。

    入力:
        なし（main_v3_config.CFG の出力設定を使用）

    処理:
        - evaluate_stage4(stage3_out_dir, stage4_svg_dir) を呼び出し一括生成
        - 終了メッセージを出力

    出力:
        なし（副作用として SVG を生成）
    """
    evaluate_stage4(stage3_out_dir, stage4_svg_dir)
    print("【Stage4 Improved】 SVG 出力処理が完了しました。")


__all__ = ["smooth_polyline", "extract_polylines_using_skan", "save_polylines_as_svg", "evaluate_stage4", "run_stage4"]
