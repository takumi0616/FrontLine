"""
概要:
    Stage3.5（閉塞の論理整形）
    - 入力:
        * Stage3 の確率出力 (C=2: none/occluded) → argmax で occluded マスクを得る
        * Stage2.5 の温暖/寒冷 class_map (0:none,1:warm,2:cold)
        * Stage2.5 の junction マスク (0/1)
    - 論理:
        「閉塞前線は温暖または寒冷、あるいは繋ぎ目(=5)に付着している必要がある」
        → occluded の各連結成分が warm/cold/junction のいずれかに接していない場合は削除
    - 出力:
        * class_map (H,W) int64（0:none, 1:occluded）を NetCDF (time,lat,lon) で保存
          ファイル名: occluded_YYYYMMDDHHMM.nc

要件対応（原文抜粋）:
    - 「閉塞前線は温暖前線=1と寒冷前線=2の繋ぎ目あたり、もしくは温暖前線と寒冷前世の繋ぎ目=5から発生している必要」
    - 「stage3.5では温暖、寒冷、繋ぎ目のどれかにくっついている部分のみの閉塞前線以外は削除」
"""

import os
import gc
import time
import numpy as np
import pandas as pd

from .main_v4_config import (
    CFG, print_memory_usage, format_time,
    stage2_5_out_dir, stage3_out_dir, stage3_5_out_dir,
    atomic_save_netcdf,
)
# 可視化ユーティリティ:
# 本ステージ単体で実行した場合でも、整形出力（occluded_*.nc）を可視化PNGとして保存するために使用。
from .main_v4_visualize import run_visualization_for_stage

def _map_stage3_probs():
    """
    Stage3 prob_* の時刻キー -> パス
    """
    d = {}
    for f in sorted(os.listdir(stage3_out_dir)):
        if f.startswith("prob_") and f.endswith(".nc"):
            key = f.replace("prob_", "").replace(".nc", "")
            d[key] = os.path.join(stage3_out_dir, f)
    return d

def _map_stage2_5():
    """
    Stage2.5 refined_* の時刻キー -> パス
    """
    d = {}
    for f in sorted(os.listdir(stage2_5_out_dir)):
        if f.endswith(".nc"):
            key = f.replace(".nc", "")
            if key.startswith("refined_"):
                key = key.replace("refined_", "")
            d[key] = os.path.join(stage2_5_out_dir, f)
    return d


def _load_stage3_occluded_prob(nc_path: str):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    probs = ds["probabilities"].isel(time=0).values  # (H,W,2)
    lat = ds["lat"].values
    lon = ds["lon"].values
    tval = ds["time"].values[0]
    ds.close()
    cls = np.argmax(probs, axis=-1).astype(np.int64)  # 0:none,1:occluded
    occ = (cls == 1).astype(np.uint8)
    return occ, lat, lon, pd.to_datetime(tval)

def _load_stage2_5_warm_cold(nc_path: str):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    try:
        if "class_map" in ds:
            cm = ds["class_map"]
            arr = cm.isel(time=0).values if "time" in cm.dims else cm.values
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            arr = np.squeeze(arr)
            # ensure 2D
            if arr.ndim != 2:
                h = int(ds.sizes.get("lat", 0))
                w = int(ds.sizes.get("lon", 0))
                tmp = np.zeros((h, w), dtype=arr.dtype if isinstance(arr, np.ndarray) else np.uint8)
                try:
                    hh = min(h, arr.shape[-2] if arr.ndim >= 2 else 0)
                    ww = min(w, arr.shape[-1] if arr.ndim >= 2 else 0)
                    if hh > 0 and ww > 0:
                        tmp[:hh, :ww] = arr[..., :hh, :ww]
                except Exception:
                    pass
                arr = tmp
            warm = (arr == 1).astype(np.uint8)
            cold = (arr == 2).astype(np.uint8)
        else:
            wv = ds["warm"] if "warm" in ds else None
            cv = ds["cold"] if "cold" in ds else None
            if wv is not None and cv is not None:
                w_arr = wv.isel(time=0).values if "time" in wv.dims else wv.values
                c_arr = cv.isel(time=0).values if "time" in cv.dims else cv.values
                w_arr = np.squeeze(np.asarray(w_arr))
                c_arr = np.squeeze(np.asarray(c_arr))
                if w_arr.ndim != 2 or c_arr.ndim != 2:
                    h = int(ds.sizes.get("lat", 0)); w = int(ds.sizes.get("lon", 0))
                    warm = np.zeros((h, w), dtype=np.uint8)
                    cold = np.zeros((h, w), dtype=np.uint8)
                else:
                    warm = (w_arr > 0.5).astype(np.uint8)
                    cold = (c_arr > 0.5).astype(np.uint8)
            else:
                h = int(ds.sizes.get("lat", 0)); w = int(ds.sizes.get("lon", 0))
                warm = np.zeros((h, w), dtype=np.uint8)
                cold = np.zeros((h, w), dtype=np.uint8)
        return warm, cold
    finally:
        ds.close()

def _load_stage2_5_junction(nc_path: str):
    """
    Stage2.5 refined_* 内の "junction" を読み出す。無い場合は class_map では代用せず、0/1の安全な0マスクを返す。
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    try:
        if "junction" in ds:
            j = ds["junction"]
            arr = j.isel(time=0).values if "time" in j.dims else j.values
        else:
            arr = None

        if arr is None:
            h = int(ds.sizes.get("lat", 0)); w = int(ds.sizes.get("lon", 0))
            jmask = np.zeros((h, w), dtype=np.uint8)
        else:
            a = np.asarray(arr)
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            a = np.squeeze(a)
            if a.ndim != 2:
                h = int(ds.sizes.get("lat", 0)); w = int(ds.sizes.get("lon", 0))
                tmp = np.zeros((h, w), dtype=np.uint8)
                try:
                    hh = min(h, a.shape[-2] if a.ndim >= 2 else 0)
                    ww = min(w, a.shape[-1] if a.ndim >= 2 else 0)
                    if hh > 0 and ww > 0:
                        tmp[:hh, :ww] = (a[..., :hh, :ww] > 0).astype(np.uint8)
                except Exception:
                    pass
                jmask = tmp
            else:
                jmask = (a > 0).astype(np.uint8)
        return jmask
    finally:
        ds.close()

def _filter_occluded_by_attachment(occ: np.ndarray, warm: np.ndarray, cold: np.ndarray, junc: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    occluded 成分が warm/cold/junc のどれかに接していれば保持、そうでなければ除去
    """
    from skimage.measure import label, regionprops
    import cv2

    H, W = occ.shape
    # それぞれ1px膨張して接触許容
    k = np.ones((3, 3), dtype=np.uint8) if connectivity >= 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    warm_d = cv2.dilate(warm.astype(np.uint8), k, iterations=1)
    cold_d = cv2.dilate(cold.astype(np.uint8), k, iterations=1)
    junc_d = cv2.dilate(junc.astype(np.uint8), k, iterations=1)

    lbl = label(occ.astype(np.uint8), connectivity=2 if connectivity >= 8 else 1)
    out = np.zeros_like(occ, dtype=np.uint8)

    for reg in regionprops(lbl):
        coords = reg.coords
        ys = coords[:, 0]; xs = coords[:, 1]
        touch = (warm_d[ys, xs] > 0).any() or (cold_d[ys, xs] > 0).any() or (junc_d[ys, xs] > 0).any()
        if touch:
            out[ys, xs] = 1
    return out

def process_one_time(s3_prob_path: str, s2_5_path: str, s1_5_path: str, out_dir: str, connectivity: int):
    """
    関数概要:
      単一時刻について、Stage3 の確率出力（none/occluded）と Stage2.5 の warm/cold/junction を用い、
      「付着制約（warm/cold/junction のいずれかに接している occluded 成分のみ残す）」を適用して保存する。

    入力:
      - s3_prob_path (str): Stage3 の確率出力ファイル（prob_*.nc, C=2）
      - s2_5_path (str): Stage2.5 の refined 出力（refined_*.nc, class_map と（あれば）junction）
      - s1_5_path (str): 本実装では s2_5_path と同一（junction を Stage2.5 側から読むため）
      - out_dir (str): 出力ディレクトリ（stage3_5_out_dir）
      - connectivity (int): 連結近傍（4 or 8 相当）

    処理:
      1) _load_stage3_occluded_prob で occluded の 0/1 マスクを取得
      2) _load_stage2_5_warm_cold で warm/cold を取得
      3) _load_stage2_5_junction で junction を取得（無い場合は 0 マスク）
      4) _filter_occluded_by_attachment により、occluded 各成分が warm/cold/junction のいずれかに接しているかを評価
      5) 接していない成分を除外した 0/1 マスクを class_map として occluded_*.nc に保存（time,lat,lon）

    出力:
      - 返り値なし（ファイル保存の副作用）。保存先: out_dir/occluded_YYYYMMDDHHMM.nc
    """
    import xarray as xr
    occ, lat, lon, t_dt = _load_stage3_occluded_prob(s3_prob_path)
    warm, cold = _load_stage2_5_warm_cold(s2_5_path)
    junc = _load_stage2_5_junction(s1_5_path)

    H = min(occ.shape[0], warm.shape[0], cold.shape[0], junc.shape[0], len(lat))
    W = min(occ.shape[1], warm.shape[1], cold.shape[1], junc.shape[1], len(lon))
    occ = occ[:H, :W]; warm = warm[:H, :W]; cold = cold[:H, :W]; junc = junc[:H, :W]
    lat = lat[:H]; lon = lon[:W]

    occ_ref = _filter_occluded_by_attachment(occ, warm, cold, junc, connectivity=connectivity)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"occluded_{t_dt.strftime('%Y%m%d%H%M')}.nc")
    # 出力済みスキップ
    if os.path.exists(out):
        print(f"[V4-Stage3.5] Skip existing output: {os.path.basename(out)}")
    else:
        da = xr.DataArray(occ_ref.astype(np.int64), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        ds = xr.Dataset({"class_map": da}).expand_dims("time")
        ds["time"] = [t_dt]
        ok = atomic_save_netcdf(ds, out, engine="netcdf4", retries=3, sleep_sec=0.5)
        if not ok:
            print(f"[V4-Stage3.5] Failed to save: {out}")
        del ds, da
    del occ_ref, occ, warm, cold, junc
    gc.collect()

def run_stage3_5():
    """
    関数概要:
      Stage3.5（閉塞の論理整形）を一括実行し、occluded_*.nc を stage3_5_out_dir へ保存する。
      実行後に、Stage3.5 の整形結果（occluded_*.nc）を可視化PNGとして出力する。

    入力:
      - なし（内部で CFG を参照）

    処理:
      - Stage3 の prob_*.nc と Stage2.5 の refined_*.nc の共通時刻を抽出
      - 各時刻について process_one_time を呼び、付着制約（warm/cold/junc のどれかに接している）を適用
      - run_visualization_for_stage("stage3_5") を呼び出し、occluded 出力を可視化

    出力:
      - 返り値なし（occluded_*.nc の保存・ログ出力・PNG出力）
    """
    print_memory_usage("Start V4 Stage3.5")
    t0 = time.time()

    map3 = _map_stage3_probs()
    map2 = _map_stage2_5()
    # Stage2.5 refined_* には junction も含むため、Stage1.5 への依存を外す
    keys = sorted(set(map3.keys()) & set(map2.keys()))
    if not keys:
        print(f"[V4-Stage3.5] No common times between Stage3 and Stage2.5")
        return

    connectivity = CFG["STAGE3_5"]["connectivity"]

    for k in keys:
        # junction も Stage2.5 refined_* 内の "junction" を用いる（第3引数にも map2[k] を渡す）
        process_one_time(map3[k], map2[k], map2[k], stage3_5_out_dir, connectivity=connectivity)

    print_memory_usage("After V4 Stage3.5")
    print(f"[V4-Stage3.5] done in {format_time(time.time() - t0)}")
    # 追加: Stage3.5 の整形結果を可視化（output_visual_dir/stage3_5 配下に保存）
    run_visualization_for_stage("stage3_5")


__all__ = ["run_stage3_5"]
