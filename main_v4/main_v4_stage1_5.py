"""
概要:
    Stage1.5（接合=5 の論理整形）
    - 入力: Stage1 の確率出力 (C=6: none/warm/cold/stationary/occluded/junction)
    - 処理:
        1) junction 二値化（argmax==5 を基本。必要なら確率閾値で強化可能）
        2) 連結成分ごとに面積評価
           - 面積 < min_keep_area → 削除（散在する1画素ノイズ除去）
           - 面積 > max_area_to_shrink → 成分の重心付近を 2x2 の塊に縮退
           - それ以外 → 現状維持（「ほぼ2x2」も許容）
    - 出力: junction マスク (H,W) を NetCDF に保存（変数名: "junction"）
            ファイル名: "junction_YYYYMMDDHHMM.nc"

設定:
    CFG["STAGE1_5"] を参照
"""

import os
import gc
import time
import numpy as np
import pandas as pd

from .main_v4_config import (
    CFG, ORIG_H, ORIG_W, print_memory_usage, format_time,
    stage1_out_dir, stage1_5_out_dir, atomic_save_netcdf,
)
# 可視化ユーティリティ:
# 各ステージ完了後に、そのステージの成果物を PNG で出力するための関数。
from .main_v4_visualize import run_visualization_for_stage

def _list_prob_files(nc_dir: str):
    """
    関数概要:
      指定ディレクトリから Stage1 出力の確率ファイル（ファイル名が 'prob_*.nc'）のみを抽出して昇順リストで返す。
    入力:
      - nc_dir (str): 走査対象ディレクトリ（CFG["PATHS"]["stage1_out_dir"] を想定）
    処理:
      - os.listdir(nc_dir) の中から 'prob_' で始まり '.nc' で終わるファイルをフィルタし、ソートして返す。
    出力:
      - List[str]: 'prob_YYYYMMDDHHMM.nc' のファイル名リスト（昇順）
    """
    return sorted([f for f in os.listdir(nc_dir) if f.startswith("prob_") and f.endswith(".nc")])

def _load_probs_2(path: str):
    """
    関数概要:
      Stage1 の確率 NetCDF（prob_*.nc）から、(H,W,C=6) の probabilities と座標 lat/lon、time を読み出す。
    入力:
      - path (str): prob_*.nc のフルパス
    処理:
      - ds["probabilities"].isel(time=0) で (H,W,C=6) を取得（time 次元がある前提）
      - lat/lon 座標と time[0] をあわせて取得
    出力:
      - Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
        (probs(H,W,6), lat(H,), lon(W,), 時刻Timestamp)
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    probs = ds["probabilities"].isel(time=0).values  # (H,W,C=6)
    lat = ds["lat"].values
    lon = ds["lon"].values
    tval = ds["time"].values[0]
    ds.close()
    return probs, lat, lon, pd.to_datetime(tval)

def _component_centroid(ys, xs):
    """
    関数概要:
      連結成分の画素座標リストから「重心に最も近い整数座標（丸め）」を返す。
    入力:
      - ys (np.ndarray | List[int]): 成分に属する画素の y 座標配列
      - xs (np.ndarray | List[int]): 成分に属する画素の x 座標配列
    処理:
      - y, x の平均値を丸め（np.round）て int へ変換し、(cy, cx) として返す。
    出力:
      - Tuple[int,int]: (cy, cx) の整数座標
    """
    cy = int(np.round(np.mean(ys)))
    cx = int(np.round(np.mean(xs)))
    return cy, cx

def _shrink_to_2x2(h: int, w: int, cy: int, cx: int) -> np.ndarray:
    """
    2x2 の塊を (cy, cx) を左上基準として配置（画像端ではクリップ）
    """
    out = np.zeros((h, w), dtype=np.uint8)
    for dy in [0, 1]:
        for dx in [0, 1]:
            yy = np.clip(cy + dy, 0, h - 1)
            xx = np.clip(cx + dx, 0, w - 1)
            out[yy, xx] = 1
    return out

def _refine_junction_mask(jmask: np.ndarray,
                          min_keep_area: int,
                          max_area_to_shrink: int,
                          connectivity: int) -> np.ndarray:
    """
    連結成分の面積に基づいて整形
    """
    from skimage.measure import label, regionprops
    lbl = label(jmask.astype(np.uint8), connectivity=2 if connectivity >= 8 else 1)
    out = np.zeros_like(jmask, dtype=np.uint8)
    H, W = out.shape

    for reg in regionprops(lbl):
        coords = reg.coords  # (N,2) with (y,x)
        area = len(coords)
        if area < min_keep_area:
            # drop
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        if area > max_area_to_shrink:
            cy, cx = _component_centroid(ys, xs)
            out |= _shrink_to_2x2(H, W, cy, cx)
        else:
            out[ys, xs] = 1
    return out

def process_one_file(prob_path: str,
                     min_keep_area: int,
                     max_area_to_shrink: int,
                     connectivity: int):
    """
    関数概要:
      1 つの Stage1 確率出力（prob_*.nc）から junction(=class5) を argmax で二値化し、
      連結成分の面積ルール（小領域削除・過大領域の 2x2 縮退）で整形したマスクを NetCDF（junction_*.nc）として保存する。
    入力:
      - prob_path (str): prob_*.nc のフルパス
      - min_keep_area (int): 面積がこの値未満の成分はノイズとして削除（0）
      - max_area_to_shrink (int): 面積がこの値を超える成分は重心付近の 2x2 へ縮退
      - connectivity (int): 連結性（4 or 8 相当）。8 以上なら connectivity=2 を採用（skimage.label の仕様）
    処理:
      1) _load_probs_2 で (H,W,6) probabilities と座標/時刻を読み出す
      2) argmax により class=5（junction）を二値化
      3) _refine_junction_mask で面積規則に基づく整形（小領域除去/2x2 縮退/現状維持）
      4) (time, lat, lon) 次元を持つ Dataset に "junction" 変数として保存
    出力:
      - 返り値なし。stage1_5_out_dir/junction_YYYYMMDDHHMM.nc を出力する副作用
    """
    import xarray as xr
    probs, lat, lon, t_dt = _load_probs_2(prob_path)
    # argmax で junction (class=5)
    cls = np.argmax(probs, axis=-1).astype(np.int64)  # (H,W)
    jmask = (cls == 5).astype(np.uint8)

    # 整形
    jmask_ref = _refine_junction_mask(
        jmask[:ORIG_H, :ORIG_W],
        min_keep_area=min_keep_area,
        max_area_to_shrink=max_area_to_shrink,
        connectivity=connectivity
    )

    # 保存（出力済みスキップ + アトミック書き込み）
    os.makedirs(stage1_5_out_dir, exist_ok=True)
    out = os.path.join(stage1_5_out_dir, f"junction_{t_dt.strftime('%Y%m%d%H%M')}.nc")
    if os.path.exists(out):
        print(f"[V4-Stage1.5] Skip existing output: {os.path.basename(out)}")
    else:
        da = xr.DataArray(
            jmask_ref.astype(np.uint8),
            dims=["lat", "lon"],
            coords={"lat": lat[:ORIG_H], "lon": lon[:ORIG_W]}
        )
        ds = xr.Dataset({"junction": da}).expand_dims("time")
        ds["time"] = [t_dt]
        ok = atomic_save_netcdf(ds, out, engine="netcdf4", retries=3, sleep_sec=0.5)
        if not ok:
            print(f"[V4-Stage1.5] Failed to save: {out}")
        del ds, da
    del probs, jmask, jmask_ref
    gc.collect()

def run_stage1_5():
    """
    関数概要:
      Stage1 の全確率出力（prob_*.nc）を走査し、junction の論理整形（小領域除去/2x2 縮退）を一括で実行する。
      各時刻について junction_*.nc（"junction" 変数, (time,lat,lon)）を stage1_5_out_dir へ保存する。
      さらに、処理完了後に可視化 PNG（stage1_5 タグ配下）を出力する。
    入力:
      - なし（CFG["PATHS"] と CFG["STAGE1_5"] を参照）
    処理:
      - _list_prob_files で prob_*.nc を列挙
      - 各ファイルに対し process_one_file(prob_path, min_keep_area, max_area_to_shrink, connectivity) を実行
      - run_visualization_for_stage("stage1_5") を呼び出し、junction の結果を可視化
    出力:
      - 返り値なし（ファイル出力とログ出力・PNG出力）
    """
    print_memory_usage("Start V4 Stage1.5")
    t0 = time.time()
    files = _list_prob_files(stage1_out_dir)
    if not files:
        print(f"[V4-Stage1.5] No Stage1 prob files in: {stage1_out_dir}")
        return

    min_keep_area = CFG["STAGE1_5"]["min_keep_area"]
    max_area_to_shrink = CFG["STAGE1_5"]["max_area_to_shrink"]
    connectivity = CFG["STAGE1_5"]["connectivity"]

    for f in files:
        process_one_file(
            os.path.join(stage1_out_dir, f),
            min_keep_area=min_keep_area,
            max_area_to_shrink=max_area_to_shrink,
            connectivity=connectivity
        )
    print_memory_usage("After V4 Stage1.5")
    print(f"[V4-Stage1.5] done in {format_time(time.time() - t0)}")
    # 追加: Stage1.5 の出力を可視化（output_visual_dir/stage1_5 配下に保存）
    run_visualization_for_stage("stage1_5")


__all__ = ["run_stage1_5"]
