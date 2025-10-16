"""
概要:
    Stage4.5（停滞の論理整形＋最終アセンブル）
    - 入力:
        * Stage4 の確率出力 (C=2: none/stationary) → argmax で stationary マスク
        * Stage3.5 の閉塞 class_map (0:none,1:occluded)
        * Stage2.5 の温暖/寒冷 class_map (0:none,1:warm,2:cold)
        * Stage2.5 の junction マスク (0/1)
    - 論理:
        1) 停滞の連結成分に対して「小さな塊」を削除 (面積 < min_component_area)
        2) 「寒冷に付着した停滞は寒冷へ」: 停滞成分が寒冷近傍に接している場合、その該当画素は寒冷(=2)へ再分類
    - アセンブル（最終6クラス）:
        class優先順位: junction(5) > occluded(4) > stationary(3) > cold(2) > warm(1) > none(0)
        を基本に合成し、final_class_map を生成
    - 出力:
        * stage4_5_out_dir/stationary_YYYYMMDDHHMM.nc            （変数名: "class_map" 0/1）
        * final_out_dir/final_YYYYMMDDHHMM.nc                     （変数名: "class_map" 0..5）

要件対応（原文抜粋）:
    - 「stege4.5 … 1マスのみの停滞や小さな塊を削除」
    - 「寒冷前線ピクセルに停滞前線がくっついた場合は寒冷前線に変更」
    - 「上記の各ステージでは、前のステージの結果を入力」→ Stage1.5/2.5/3.5 を消費
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from typing import Tuple

from .main_v4_config import (
    CFG, print_memory_usage, format_time,
    stage1_5_out_dir, stage2_5_out_dir, stage3_5_out_dir, stage4_out_dir,
    stage4_5_out_dir, final_out_dir,
)
# 可視化ユーティリティ:
# 本ステージ単体で実行した場合でも、整形出力（final_*.nc）を可視化PNGとして保存するために使用。
from .main_v4_visualize import run_visualization_for_stage

def _map_stage4_probs():
    """
    関数概要:
      Stage4 の確率出力（prob_*.nc）を列挙し、時刻トークン（YYYYMMDDHHMM）からファイルパスへのマップを作成する。

    入力:
      - なし（CFG["PATHS"]["stage4_out_dir"] を参照）

    処理:
      - ディレクトリ内の 'prob_*.nc' を走査し、'prob_' と拡張子を除いた部分をキーとして辞書に格納

    出力:
      - Dict[str, str]: {YYYYMMDDHHMM: /path/to/prob_YYYYMMDDHHMM.nc}
    """
    d = {}
    for f in sorted(os.listdir(stage4_out_dir)):
        if f.startswith("prob_") and f.endswith(".nc"):
            key = f.replace("prob_", "").replace(".nc", "")
            d[key] = os.path.join(stage4_out_dir, f)
    return d

def _map_stage3_5():
    """
    関数概要:
      Stage3.5 の論理整形出力（occluded_*.nc）を列挙し、時刻トークン（YYYYMMDDHHMM）からパスへのマップを作成する。

    入力:
      - なし（CFG["PATHS"]["stage3_5_out_dir"] を参照）

    処理:
      - 'occluded_*.nc' の接頭辞/拡張子を取り除き、時刻トークンをキーとする辞書を返す

    出力:
      - Dict[str, str]: {YYYYMMDDHHMM: /path/to/occluded_YYYYMMDDHHMM.nc}
    """
    d = {}
    for f in sorted(os.listdir(stage3_5_out_dir)):
        if f.endswith(".nc"):
            key = f.replace(".nc", "")
            if key.startswith("occluded_"):
                key = key.replace("occluded_", "")
            d[key] = os.path.join(stage3_5_out_dir, f)
    return d

def _map_stage2_5():
    """
    関数概要:
      Stage2.5 の refined 出力（refined_*.nc）を列挙し、時刻トークン（YYYYMMDDHHMM）からパスへのマップを作成する。
      本出力には warm/cold の class_map に加え、junction（両側接触のみ残したもの）が含まれる想定。

    入力:
      - なし（CFG["PATHS"]["stage2_5_out_dir"] を参照）

    出力:
      - Dict[str, str]: {YYYYMMDDHHMM: /path/to/refined_YYYYMMDDHHMM.nc}
    """
    d = {}
    for f in sorted(os.listdir(stage2_5_out_dir)):
        if f.endswith(".nc"):
            key = f.replace(".nc", "")
            if key.startswith("refined_"):
                key = key.replace("refined_", "")
            d[key] = os.path.join(stage2_5_out_dir, f)
    return d

def _map_stage1_5():
    """
    関数概要:
      Stage1.5 の接合マスク（junction_*.nc）を列挙し、時刻トークン（YYYYMMDDHHMM）からパスへのマップを作成する。
      注記: Stage4.5 の実処理では Stage2.5 の junction を優先して使用するため、本マップは通常使用しない。

    入力:
      - なし（CFG["PATHS"]["stage1_5_out_dir"] を参照）

    出力:
      - Dict[str, str]: {YYYYMMDDHHMM: /path/to/junction_YYYYMMDDHHMM.nc}
    """
    d = {}
    for f in sorted(os.listdir(stage1_5_out_dir)):
        if f.startswith("junction_") and f.endswith(".nc"):
            key = f.replace("junction_", "").replace(".nc", "")
            d[key] = os.path.join(stage1_5_out_dir, f)
    return d


def _load_stage4_stationary_prob(nc_path: str):
    """
    関数概要:
      Stage4 の確率 NetCDF（prob_*.nc, C=2: none/stationary）を読み、argmax による stationary の 0/1 マスクを生成する。

    入力:
      - nc_path (str): Stage4 確率ファイル（prob_*.nc）のパス

    処理:
      - ds["probabilities"].isel(time=0) から (H,W,2) を取得し、argmax==1 を stationary=1 として 0/1 化
      - lat/lon と time[0] も読み出す

    出力:
      - Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
        (stationary(H,W) uint8, lat(H,), lon(W,), 時刻 Timestamp)
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    probs = ds["probabilities"].isel(time=0).values  # (H,W,2)
    lat = ds["lat"].values
    lon = ds["lon"].values
    tval = ds["time"].values[0]
    ds.close()
    cls = np.argmax(probs, axis=-1).astype(np.int64)  # 0:none,1:stationary
    st = (cls == 1).astype(np.uint8)
    return st, lat, lon, pd.to_datetime(tval)

def _load_stage3_5_occluded(nc_path: str):
    """
    関数概要:
      Stage3.5 の occluded 出力（occluded_*.nc）を読み、0/1 の閉塞マスクを返す。

    入力:
      - nc_path (str): occluded_*.nc のパス

    処理:
      - "class_map" があれば >0 を 1 として 0/1 化
      - 代替として "occluded" 変数があれば 0.5 閾値で 0/1 化
      - どちらもなければゼロ配列（サイズは lat/lon 次元から決定）

    出力:
      - np.ndarray: (H,W) uint8 の 0/1 マスク
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if "class_map" in ds:
        occ = (ds["class_map"].values.astype(np.int64) > 0).astype(np.uint8)
    else:
        # fallback: try a variable named "occluded"
        if "occluded" in ds:
            occ = (ds["occluded"].values > 0.5).astype(np.uint8)
        else:
            h = ds.dims.get("lat", 0); w = ds.dims.get("lon", 0)
            occ = np.zeros((h, w), dtype=np.uint8)
    ds.close()
    return occ

def _load_stage2_5_warm_cold(nc_path: str):
    """
    関数概要:
      Stage2.5 refined 出力から warm/cold の 0/1 マスクを読み出す。

    入力:
      - nc_path (str): refined_*.nc のパス

    処理:
      - "class_map" があれば ==1 を warm, ==2 を cold として 0/1 化
      - 代替として "warm"/"cold" 変数があれば 0.5 閾値で 0/1 化
      - どれも無い場合はゼロ配列でフォールバック

    出力:
      - Tuple[np.ndarray, np.ndarray]: (warm(H,W) uint8, cold(H,W) uint8)
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if "class_map" in ds:
        cm = ds["class_map"].values.astype(np.int64)
        warm = (cm == 1).astype(np.uint8)
        cold = (cm == 2).astype(np.uint8)
    else:
        if "warm" in ds and "cold" in ds:
            warm = (ds["warm"].values > 0.5).astype(np.uint8)
            cold = (ds["cold"].values > 0.5).astype(np.uint8)
        else:
            h = ds.dims.get("lat", 0); w = ds.dims.get("lon", 0)
            warm = np.zeros((h, w), dtype=np.uint8)
            cold = np.zeros((h, w), dtype=np.uint8)
    ds.close()
    return warm, cold

def _load_stage1_5_junction(nc_path: str):
    """
    関数概要:
      Stage1.5 の junction_*.nc から 0/1 の junction マスクを読み出す。
      注記: 本 v4 では Stage4.5 の実処理において「Stage2.5 の junction を優先して使用」するため、
            実引数として refined_*.nc が渡される場合がある。この場合 "junction" 変数を優先的に読み出し、
            無い場合は安全に全0（代替として class_map を用いて junction を推定しない）。

    入力:
      - nc_path (str): junction_*.nc または refined_*.nc のパス

    出力:
      - np.ndarray: (H,W) uint8 の 0/1 junction マスク
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if "junction" in ds:
        j = ds["junction"]
        jmask = (j.isel(time=0).values if "time" in j.dims else j.values).astype(np.uint8)
    elif "class_map" in ds:
        # Stage2.5 の refined_* に junction が含まれないケースでは推定しない（全0で安全側フォールバック）
        jmask = (ds["class_map"].values.astype(np.int64) * 0).astype(np.uint8)
    else:
        var = list(ds.data_vars)[0]
        v = ds[var]
        jmask = (v.isel(time=0).values if "time" in v.dims else v.values)
        jmask = (jmask > 0.5).astype(np.uint8)
    ds.close()
    return jmask


def _remove_small_components(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    """
    関数概要:
      与えられた 0/1 マスクに対して連結成分をラベリングし、面積が min_area 未満の小領域を除去する。

    入力:
      - mask (np.ndarray): (H,W) の 0/1 マスク
      - min_area (int): 残す最小画素数（これ未満は削除）
      - connectivity (int): 近傍接続（4 or 8 相当）

    出力:
      - np.ndarray: 小領域除去後の 0/1 マスク
    """
    from skimage.measure import label, regionprops
    lbl = label(mask.astype(np.uint8), connectivity=2 if connectivity >= 8 else 1)
    out = np.zeros_like(mask, dtype=np.uint8)
    for reg in regionprops(lbl):
        coords = reg.coords
        if len(coords) >= min_area:
            ys = coords[:, 0]; xs = coords[:, 1]
            out[ys, xs] = 1
    return out

def _stationary_touching_cold_to_cold(sta: np.ndarray, cold: np.ndarray, connectivity: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    停滞が寒冷に接している画素を「寒冷へ再分類」するための補助関数。
    戻り値:
      - sta_keep: 寒冷に接していない停滞（引き続き停滞=1として維持）0/1
      - sta_to_cold_pixels: 寒冷に接しているため寒冷へ再分類すべき画素（0/1）
    """
    import cv2
    k = np.ones((3, 3), dtype=np.uint8) if connectivity >= 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    cold_d = cv2.dilate(cold.astype(np.uint8), k, iterations=1)
    touching = (sta == 1) & (cold_d == 1)
    sta_keep = ((sta == 1) & (cold_d == 0)).astype(np.uint8)
    sta_to_cold_pixels = touching.astype(np.uint8)
    return sta_keep, sta_to_cold_pixels


def _assemble_final(warm: np.ndarray, cold: np.ndarray, sta: np.ndarray, occ: np.ndarray, junc: np.ndarray) -> np.ndarray:
    """
    確定順に積み上げる合成（後段で上書きしない）
    - まず接合=5 を固定（以降の段階で上書き不可）
    - 次に温暖=1、寒冷=2 を固定（以降の段階で上書き不可）
    - 次に閉塞=4（ただし既に埋まっている画素は上書きしない）
    - 最後に停滞=3（ただし既に埋まっている画素は上書きしない）
    """
    H, W = warm.shape
    final = np.zeros((H, W), dtype=np.int64)

    # 1) 接合=5 を先に確定
    final[junc == 1] = 5

    # 2) 温暖=1 を final==0 の場所にのみ配置（上書き禁止）
    mask = (final == 0) & (warm == 1)
    final[mask] = 1

    # 3) 寒冷=2 を final==0 の場所にのみ配置（上書き禁止）
    mask = (final == 0) & (cold == 1)
    final[mask] = 2

    # 4) 閉塞=4 を final==0 の場所にのみ配置（上書き禁止）
    mask = (final == 0) & (occ == 1)
    final[mask] = 4

    # 5) 停滞=3 を final==0 の場所にのみ配置（上書き禁止）
    mask = (final == 0) & (sta == 1)
    final[mask] = 3

    return final


def process_one_time(s4_prob_path: str, s3_5_path: str, s2_5_path: str, s1_5_path: str,
                     min_area: int, connectivity: int, sta_touch_cold_to_cold: bool):
    """
    1時刻を処理して stationary_refined + final_class_map を保存
    """
    import xarray as xr
    st, lat, lon, t_dt = _load_stage4_stationary_prob(s4_prob_path)
    occ = _load_stage3_5_occluded(s3_5_path)
    warm, cold = _load_stage2_5_warm_cold(s2_5_path)
    junc = _load_stage1_5_junction(s1_5_path)

    # サイズ調整（最小共通）
    H = min(st.shape[0], occ.shape[0], warm.shape[0], cold.shape[0], junc.shape[0], len(lat))
    W = min(st.shape[1], occ.shape[1], warm.shape[1], cold.shape[1], junc.shape[1], len(lon))
    st = st[:H, :W]; occ = occ[:H, :W]; warm = warm[:H, :W]; cold = cold[:H, :W]; junc = junc[:H, :W]
    lat = lat[:H]; lon = lon[:W]

    # 1) 小領域削除
    st_ref = _remove_small_components(st, min_area=min_area, connectivity=connectivity)

    # 2) 寒冷に付着した停滞は寒冷へ再分類（cold を拡張）
    if sta_touch_cold_to_cold:
        st_ref, sta_to_cold = _stationary_touching_cold_to_cold(st_ref, cold, connectivity=connectivity)
        # cold を「接触していた停滞画素」で拡張
        cold_aug = np.clip(cold.astype(np.uint8) + sta_to_cold.astype(np.uint8), 0, 1).astype(np.uint8)
    else:
        cold_aug = cold

    # アセンブル（優先度適用）
    final = _assemble_final(warm, cold_aug, st_ref, occ, junc)

    # 保存: stationary refined
    os.makedirs(stage4_5_out_dir, exist_ok=True)
    da_s = xr.DataArray(st_ref.astype(np.int64), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    ds_s = xr.Dataset({"class_map": da_s}).expand_dims("time")
    ds_s["time"] = [t_dt]
    out_s = os.path.join(stage4_5_out_dir, f"stationary_{t_dt.strftime('%Y%m%d%H%M')}.nc")
    ds_s.to_netcdf(out_s, engine="netcdf4")
    del ds_s, da_s

    # 保存: final class map
    os.makedirs(final_out_dir, exist_ok=True)
    da_f = xr.DataArray(final.astype(np.int64), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    ds_f = xr.Dataset({"class_map": da_f}).expand_dims("time")
    ds_f["time"] = [t_dt]
    out_f = os.path.join(final_out_dir, f"final_{t_dt.strftime('%Y%m%d%H%M')}.nc")
    ds_f.to_netcdf(out_f, engine="netcdf4")
    del ds_f, da_f

    gc.collect()


def run_stage4_5():
    """
    関数概要:
      Stage4.5（停滞の論理整形＋最終アセンブル）を一括実行し、stationary_*.nc と final_*.nc を保存する。
      実行後に、最終成果物（final_*.nc）を可視化PNG（stage4_5 タグ配下）として出力する。

    入力:
      - なし（内部で CFG を参照）

    処理:
      - Stage4 の prob_*.nc, Stage3.5 の occluded_*.nc, Stage2.5 の refined_*.nc の共通時刻を抽出
      - 各時刻について process_one_time を呼び、小領域除去や「停滞→寒冷再分類」、最終合成を実施
      - run_visualization_for_stage("stage4_5") を呼び出し、final 出力を可視化
        （注: stage4_5 可視化は final_*.nc を対象。必要に応じて stationary_*.nc の個別可視化を追加可能）

    出力:
      - 返り値なし（stationary_*.nc と final_*.nc の保存・ログ出力・PNG出力）
    """
    print_memory_usage("Start V4 Stage4.5")
    t0 = time.time()

    map4 = _map_stage4_probs()
    map3 = _map_stage3_5()
    map2 = _map_stage2_5()
    # Stage2.5 refined_* に junction も含まれるため、Stage1.5 への依存を外す
    keys = sorted(set(map4.keys()) & set(map3.keys()) & set(map2.keys()))
    if not keys:
        print(f"[V4-Stage4.5] No common times among Stage4, Stage3.5 and Stage2.5")
        return

    min_area = CFG["STAGE4_5"]["min_component_area"]
    connectivity = CFG["STAGE4_5"]["connectivity"]
    sta_to_cold = CFG["STAGE4_5"]["stationary_touching_cold_is_cold"]

    for k in keys:
        # junction も Stage2.5 refined_* 内の "junction" を用いる（第4引数にも map2[k] を渡す）
        process_one_time(
            map4[k], map3[k], map2[k], map2[k],
            min_area=min_area,
            connectivity=connectivity,
            sta_touch_cold_to_cold=sta_to_cold
        )

    print_memory_usage("After V4 Stage4.5")
    print(f"[V4-Stage4.5] done in {format_time(time.time() - t0)}")
    # 追加: Stage4.5 の最終成果物（final_*.nc）を可視化（output_visual_dir/stage4_5 配下に保存）
    run_visualization_for_stage("stage4_5")


__all__ = ["run_stage4_5"]
