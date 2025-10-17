"""
概要:
    Stage2.5（温暖/寒冷の論理整形）
    - 入力:
        * Stage2 の確率出力 (C=3: none/warm/cold) → argmax で class_map を得る
        * Stage1.5 の junction マスク (0/1)
    - 論理:
        1) 温暖(=1)・寒冷(=2)の union（front_mask）に対して連結成分ラベリング
           → 繋ぎ目(junction)に接続していない front 成分は全て削除（noneにする）
        2) junction についても、各 junction 成分が「温暖にも寒冷にも」接している場合のみ残す
           → 片側しか付いていない junction 成分は削除（0）
    - 出力:
        * class_map (H,W) int64（0:none, 1:warm, 2:cold）を NetCDF (time,lat,lon) で保存
          ファイル名: refined_YYYYMMDDHHMM.nc

要件対応（原文抜粋）:
    - 「温暖前線と寒冷前世の繋ぎ目=5と繋がっている部分のみを残す（繋ぎ目から前線ありのマスを通って繋がっていること）」
    - 「温暖前線のみのもの、寒冷前線のみのものは削除」
    - 「繋ぎ目=5は、温暖と寒冷のどっちとも繋がっているもののみ残し、片側のみは削除」
"""

import os
import gc
import time
import numpy as np
import pandas as pd

from .main_v4_config import (
    CFG, print_memory_usage, format_time,
    stage1_5_out_dir, stage2_out_dir, stage2_5_out_dir,
    atomic_save_netcdf,
)
# 可視化ユーティリティ:
# 本ステージ単体で実行した場合でも、整形出力（refined_*.nc）を可視化PNGとして保存するために使用。
from .main_v4_visualize import run_visualization_for_stage

def _scan_probs_and_junc():
    """
    関数概要:
      Stage2 の確率出力（prob_*.nc）と Stage1.5 の junction（junction_*.nc）から、
      共通の時刻キー（YYYYMMDDHHMM）を列挙し、対応するファイルパスのペアを返す。

    入力:
      - なし（CFG["PATHS"]["stage2_out_dir"], CFG["PATHS"]["stage1_5_out_dir"] を参照）

    処理:
      - stage2_out_dir 内の prob_*.nc から時刻トークンを抽出
      - stage1_5_out_dir 内の junction_*.nc から時刻トークンを抽出
      - 両者の共通トークンだけを抽出して (key, stage2_path, stage1_5_path) のタプルに整形

    出力:
      - List[Tuple[str, str, str]]: [(YYYYMMDDHHMM, prob_path, junction_path), ...] 昇順
    """
    s2 = {}
    s1_5 = {}
    for f in sorted(os.listdir(stage2_out_dir)):
        if f.startswith("prob_") and f.endswith(".nc"):
            key = f.replace("prob_", "").replace(".nc", "")
            s2[key] = os.path.join(stage2_out_dir, f)
    for f in sorted(os.listdir(stage1_5_out_dir)):
        if f.startswith("junction_") and f.endswith(".nc"):
            key = f.replace("junction_", "").replace(".nc", "")
            s1_5[key] = os.path.join(stage1_5_out_dir, f)
    keys = sorted(set(s2.keys()) & set(s1_5.keys()))
    return [(k, s2[k], s1_5[k]) for k in keys]

def _load_stage2_class_map(nc_path: str):
    """
    関数概要:
      Stage2 の確率 NetCDF（prob_*.nc）を読み、argmax により 3クラスの class_map（0/1/2）を生成する。

    入力:
      - nc_path (str): Stage2 確率ファイル（prob_*.nc）のパス

    処理:
      - ds["probabilities"].isel(time=0) で (H,W,3) を取得
      - argmax で 0/1/2 の class_map を得る
      - lat/lon と time[0] も読み出す

    出力:
      - Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
        (class_map(H,W) int64, lat(H,), lon(W,), 時刻Timestamp)
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    probs = ds["probabilities"].isel(time=0).values  # (H,W,3)
    lat = ds["lat"].values
    lon = ds["lon"].values
    tval = ds["time"].values[0]
    ds.close()
    cls = np.argmax(probs, axis=-1).astype(np.int64)  # 0,1,2
    return cls, lat, lon, pd.to_datetime(tval)

def _load_junction_mask(nc_path: str):
    """
    関数概要:
      Stage1.5 の junction_*（"junction" 変数 or "class_map" 代替）から 0/1 の 2D マスクを読み出す。

    入力:
      - nc_path (str): junction_*.nc のパス（Stage1.5 出力）

    処理:
      - "junction" 変数があれば time=0 を優先して取り出し、2D に squeeze
      - "class_map" しかない場合は >0 を junction=1 として 0/1 化
      - その他の場合は先頭変数を読み、0.5 を閾値に 0/1 化
      - lat/lon も返却（形状整合のため）

    出力:
      - Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (junction(H,W) uint8, lat(H,), lon(W,))
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if "junction" in ds:
        arr = ds["junction"].isel(time=0).values.astype(np.uint8) if "time" in ds["junction"].dims else ds["junction"].values.astype(np.uint8)
    elif "class_map" in ds:
        arr = (ds["class_map"].values.astype(np.int64) > 0).astype(np.uint8)
    else:
        var = list(ds.data_vars)[0]
        v = ds[var]
        arr = (v.isel(time=0).values if "time" in v.dims else v.values)
        arr = (arr > 0.5).astype(np.uint8)
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    return arr, lat, lon

def _keep_fronts_connected_to_junction(cls_map: np.ndarray, junc: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    関数概要:
      front 成分（warm=1, cold=2）の各連結成分が junction と接触しているかを判定し、
      接触していない成分を none=0 へ落とすフィルタリングを行う。

    入力:
      - cls_map (np.ndarray): (H,W) int64 の 3値マップ（0:none,1:warm,2:cold）
      - junc (np.ndarray): (H,W) uint8 の junction マスク（0/1）
      - connectivity (int): 近傍接続の種類（4 or 8 相当）。8 以上なら膨張/ラベリングに 8 近傍を使用

    処理:
      - warm/cold を合成して front マスクを作成
      - junction を 1 ピクセル膨張し、ラベリングした front 成分と接触しているかを判定
      - 接触成分のみ元のクラスを維持（非接触は none=0）

    出力:
      - np.ndarray: (H,W) int64 の 3値マップ（junction フィルタ後）
    """
    from skimage.measure import label, regionprops
    import cv2

    H, W = cls_map.shape
    warm = (cls_map == 1).astype(np.uint8)
    cold = (cls_map == 2).astype(np.uint8)
    front = np.clip(warm + cold, 0, 1).astype(np.uint8)

    # junction 近傍にフラグ（接触）: junction を膨張して front と重なる成分を拾いやすくする
    k = np.ones((3, 3), dtype=np.uint8) if connectivity >= 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    junc_d = cv2.dilate(junc.astype(np.uint8), k, iterations=1)

    lbl = label(front, connectivity=2 if connectivity >= 8 else 1)
    keep_mask = np.zeros_like(front, dtype=np.uint8)

    for reg in regionprops(lbl):
        coords = reg.coords  # (N,2)
        ys = coords[:, 0]; xs = coords[:, 1]
        # この front 成分が junction と接しているか？（膨張juncと重なるか）
        touching = (junc_d[ys, xs] > 0).any()
        if touching:
            keep_mask[ys, xs] = 1

    # 残す front 成分は元クラスを維持。残さない成分は none=0
    out = np.zeros_like(cls_map, dtype=np.int64)
    out[(keep_mask == 1) & (warm == 1)] = 1
    out[(keep_mask == 1) & (cold == 1)] = 2
    return out

def _filter_junction_must_touch_both(junc: np.ndarray, cls_map_refined: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    関数概要:
      junction の各連結成分が「warm と cold の両者に接している」場合のみ残し、
      片側にしか接していない成分は削除（0）するフィルタリング。

    入力:
      - junc (np.ndarray): (H,W) uint8 の junction マスク（0/1）
      - cls_map_refined (np.ndarray): (H,W) int64 の front マップ（前段の接続フィルタ後）
      - connectivity (int): 近傍接続（4 or 8 相当）

    処理:
      - warm/cold を 1 ピクセル膨張
      - junction の連結成分ごとに warm/cold のどちらに触れているかを評価
      - 両方に触れている場合のみ残す

    出力:
      - np.ndarray: (H,W) uint8 の junction マスク（0/1）フィルタ後
    """
    from skimage.measure import label, regionprops
    import cv2

    warm = (cls_map_refined == 1).astype(np.uint8)
    cold = (cls_map_refined == 2).astype(np.uint8)

    k = np.ones((3, 3), dtype=np.uint8) if connectivity >= 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    warm_d = cv2.dilate(warm, k, iterations=1)
    cold_d = cv2.dilate(cold, k, iterations=1)

    lbl = label(junc.astype(np.uint8), connectivity=2 if connectivity >= 8 else 1)
    out = np.zeros_like(junc, dtype=np.uint8)

    for reg in regionprops(lbl):
        coords = reg.coords
        ys = coords[:, 0]; xs = coords[:, 1]
        touch_w = (warm_d[ys, xs] > 0).any()
        touch_c = (cold_d[ys, xs] > 0).any()
        if touch_w and touch_c:
            out[ys, xs] = 1  # 残す
    return out

def process_one_time(s2_path: str, junc_path: str, out_dir: str, connectivity: int):
    """
    関数概要:
      単一時刻に対して Stage2 確率出力からの class_map と Stage1.5 junction を用い、
      front の junction 接続フィルタ + junction の両側接触フィルタを適用した結果を保存する。

    入力:
      - s2_path (str): prob_*.nc（Stage2 確率出力）
      - junc_path (str): junction_*.nc（Stage1.5 の論理整形出力）
      - out_dir (str): 保存ディレクトリ（stage2_5_out_dir）
      - connectivity (int): 近傍接続

    処理:
      1) _load_stage2_class_map で class_map/lat/lon/time を取得
      2) _load_junction_mask で junction（0/1）を取得
      3) _keep_fronts_connected_to_junction で front を junction 接続でフィルタ
      4) _filter_junction_must_touch_both で junction を両側接触フィルタ
      5) class_map と junction を同一ファイルに保存（refined_*.nc）

    出力:
      - 返り値なし（ファイル保存の副作用）。保存変数: "class_map"(0/1/2), "junction"(0/1)
    """
    import xarray as xr
    cls_map, lat, lon, t_dt = _load_stage2_class_map(s2_path)
    junc, lat_j, lon_j = _load_junction_mask(junc_path)
    # 一応形状の整合
    H = min(len(lat), junc.shape[0]); W = min(len(lon), junc.shape[1])
    cls_map = cls_map[:H, :W]
    junc = junc[:H, :W]
    lat = lat[:H]; lon = lon[:W]

    # 1) front を junction 接続でフィルタ
    refined = _keep_fronts_connected_to_junction(cls_map, junc, connectivity=connectivity)

    # 2) junction 成分の両側接触によるフィルタ（junc自体の保存は optional）
    junc_filtered = _filter_junction_must_touch_both(junc, refined, connectivity=connectivity)

    # 保存（class_map と junction を同一ファイルに格納）
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"refined_{t_dt.strftime('%Y%m%d%H%M')}.nc")
    if os.path.exists(out):
        print(f"[V4-Stage2.5] Skip existing output: {os.path.basename(out)}")
    else:
        da_cls = xr.DataArray(
            refined.astype(np.int64),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon}
        )
        da_junc = xr.DataArray(
            junc_filtered.astype(np.uint8),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon}
        )
        ds = xr.Dataset({"class_map": da_cls, "junction": da_junc}).expand_dims("time")
        ds["time"] = [t_dt]
        ok = atomic_save_netcdf(ds, out, engine="netcdf4", retries=3, sleep_sec=0.5)
        if not ok:
            print(f"[V4-Stage2.5] Failed to save: {out}")
        del ds, da_cls, da_junc
    del refined, junc_filtered, cls_map, junc
    gc.collect()

def run_stage2_5():
    """
    関数概要:
      Stage2 の全確率出力と Stage1.5 junction の共通時刻を抽出し、process_one_time にて
      front/junction の論理整形（接続・両側接触）を一括で実行する。
      実行後に、Stage2.5 の整形結果（refined_*.nc）を可視化PNGとして出力する。

    入力:
      - なし（内部で CFG を参照）

    処理:
      - _scan_probs_and_junc で (prob_path, junction_path) のペアを列挙
      - 各ペアについて process_one_time(..., connectivity=CFG["STAGE2_5"]["connectivity"]) を呼ぶ
      - run_visualization_for_stage("stage2_5") を呼び出し、refined 出力を可視化

    出力:
      - 返り値なし（refined_*.nc のファイル保存・ログ出力・PNG出力）
    """
    print_memory_usage("Start V4 Stage2.5")
    t0 = time.time()
    pairs = _scan_probs_and_junc()
    if not pairs:
        print(f"[V4-Stage2.5] No common times in {stage2_out_dir} and {stage1_5_out_dir}")
        return

    connectivity = CFG["STAGE2_5"]["connectivity"]

    for key, s2p, juncp in pairs:
        process_one_time(s2p, juncp, stage2_5_out_dir, connectivity=connectivity)

    print_memory_usage("After V4 Stage2.5")
    print(f"[V4-Stage2.5] done in {format_time(time.time() - t0)}")
    # 追加: Stage2.5 の整形結果を可視化（output_visual_dir/stage2_5 配下に保存）
    run_visualization_for_stage("stage2_5")


__all__ = ["run_stage2_5"]
