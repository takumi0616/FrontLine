"""
概要:
    v3 の可視化機能を v4 に導入。
    - 各ステージ（1,1.5,2,2.5,3,3.5,4,4.5）終了ごとに、そのステージの成果物と GT を比較する可視化 PNG を出力する。
    - 背景に海面更正気圧の偏差（GSM）を重畳、低気圧中心（存在すれば）を赤×で表示。
    - 画像は output_visual_dir/{stage_name}/comparison_{stage_name}_{YYYYMMDDHHMM}.png に保存。

注意:
    - v4 の各ステージ出力はクラス数が異なるため、v3 と同じ 4 パネルではなく「Pred vs GT」の 2 パネルで表示する。
    - カラーマップは CFG["VISUALIZATION"]["class_colors"] に従う。v4 側の予測クラスを 0..5 へマッピングして色づけする。
    - GT は v3 同様に 5ch (warm, cold, stationary, occluded, warm_cold) から 0..5 へ集約。

関数:
    - run_visualization_for_stage(stage_name: str): 一括可視化（stage_name in {"stage1","stage1_5","stage2","stage2_5","stage3","stage3_5","stage4","stage4_5"}）
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .main_v4_config import (
    CFG, print_memory_usage, format_time,
    nc_gsm_dir, nc_0p5_dir, output_visual_dir,
    stage1_out_dir, stage1_5_out_dir,
    stage2_out_dir, stage2_5_out_dir,
    stage3_out_dir, stage3_5_out_dir,
    stage4_out_dir, stage4_5_out_dir, final_out_dir,
)

def _setup_matplotlib_fonts_and_warnings():
    """
    日本語表示は japanize-matplotlib を優先して有効化し、グリフ欠落のUserWarningも抑止する。
    japanize が使用できない場合は、CJKフォント自動選択ロジックでフォールバックする。
    """
    try:
        import warnings as _warnings
        _warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Glyph .* missing from font\(s\) .*",
        )
        import matplotlib.pyplot as _plt

        # 1) japanize-matplotlib を優先
        try:
            import japanize_matplotlib  # noqa: F401
        except Exception:
            # 2) フォールバック: 環境にあるCJKフォントを自動選択
            import matplotlib.font_manager as _fm
            installed = {f.name for f in _fm.fontManager.ttflist}
            candidates = [
                "Noto Sans CJK JP",
                "Noto Sans JP",
                "IPAexGothic",
                "IPAPGothic",
                "TakaoGothic",
                "VL Gothic",
                "Source Han Sans JP",
                "Yu Gothic",
                "Hiragino Sans",
                "MS Gothic",
                "Arial Unicode MS",
            ]
            chosen = None
            for name in candidates:
                if name in installed:
                    chosen = name
                    break
            if chosen:
                _plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
            else:
                _plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]

        # マイナス記号の文字化け対策
        _plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        # ベストエフォート。失敗しても可視化処理自体は継続させる
        pass

# モジュール読み込み時に一度だけ適用
_setup_matplotlib_fonts_and_warnings()

def _to_1d_lat_lon(lat, lon):
    """
    lat/lon が 2次元格子として保存されている場合でも、可視化処理が期待する 1次元座標へ正規化する。
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    try:
        if lat.ndim == 2 and lat.shape[1] == 1:
            lat = lat[:, 0]
        elif lat.ndim == 2 and lat.shape[0] == 1:
            lat = lat[0, :]
    except Exception:
        pass
    try:
        if lon.ndim == 2 and lon.shape[0] == 1:
            lon = lon[0, :]
        elif lon.ndim == 2 and lon.shape[1] == 1:
            lon = lon[:, 0]
    except Exception:
        pass
    if lat.ndim != 1:
        lat = np.squeeze(lat)
        if lat.ndim != 1:
            lat = lat.ravel()
    if lon.ndim != 1:
        lon = np.squeeze(lon)
        if lon.ndim != 1:
            lon = lon.ravel()
    return lat, lon


def _align_to_hw(arr, H, W, fill=0):
    """
    任意形状の配列 arr を (H, W) の2次元に切り詰め＋ゼロ埋め（fill値）で整形するヘルパー。
    - 3Dで先頭が1のときは squeeze
    - 1D かつ要素数が H*W と一致すれば reshape
    - それ以外は左上へ可能範囲で貼り付け、残りを fill で埋める
    """
    a = np.asarray(arr)
    if a.ndim == 0:
        out = np.zeros((H, W), dtype=a.dtype)
        out[:] = fill
        return out
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    a = np.squeeze(a)
    if a.ndim == 1:
        if a.size == H * W:
            a = a.reshape(H, W)
        else:
            tmp = np.full((H, W), fill, dtype=a.dtype)
            lim = min(a.size, H * W)
            tmp.ravel()[:lim] = a.ravel()[:lim]
            return tmp
    if a.ndim != 2:
        tmp = np.full((H, W), fill, dtype=a.dtype)
        hh = min(H, a.shape[0] if a.ndim >= 1 else 0)
        ww = min(W, a.shape[1] if a.ndim >= 2 else 0)
        try:
            tmp[:hh, :ww] = a[:hh, :ww]
        except Exception:
            pass
        return tmp
    out = np.full((H, W), fill, dtype=a.dtype)
    hh = min(H, a.shape[0]); ww = min(W, a.shape[1])
    out[:hh, :ww] = a[:hh, :ww]
    return out


def _list_nc(dir_path: str, prefix: str = "", suffix: str = ".nc"):
    """
    関数概要:
      指定ディレクトリ配下の NetCDF ファイルを列挙し、任意の接頭辞/接尾辞でフィルタした上で
      ファイル名の昇順リストを返すユーティリティ。

    入力:
      - dir_path (str): 走査対象のディレクトリパス
      - prefix (str): ファイル名がこの接頭辞で始まるもののみ採用（空文字なら無条件）
      - suffix (str): ファイル名がこの接尾辞で終わるもののみ採用（デフォルトは ".nc"）

    処理:
      - ディレクトリが存在しない場合は空リストを返す
      - os.listdir で走査し、suffix と prefix 条件でフィルタ、昇順ソートして返す

    出力:
      - List[str]: 条件に合致したファイル名の昇順リスト
    """
    if not os.path.exists(dir_path):
        return []
    files = []
    for f in os.listdir(dir_path):
        if f.endswith(suffix) and (prefix == "" or f.startswith(prefix)):
            files.append(f)
    return sorted(files)


def _time_token_from_prob_name(fname: str, prefix="prob_"):
    """
    関数概要:
      Stage の確率出力ファイル名から時刻トークン（YYYYMMDDHHMM）を抽出する。

    入力:
      - fname (str): 例 "prob_YYYYMMDDHHMM.nc" 形式のファイル名
      - prefix (str): 接頭辞（デフォルト "prob_"）

    処理:
      - 接頭辞と拡張子 ".nc" を取り除いてトークンを抽出

    出力:
      - str: "YYYYMMDDHHMM"
    """
    return fname.replace(prefix, "").replace(".nc", "")


def _time_token_from_generic_name(fname: str, prefix: str):
    """
    関数概要:
      任意のステージ出力ファイル名から、与えられた接頭辞を除去して時刻トークン（YYYYMMDDHHMM）を抽出する。

    入力:
      - fname (str): 例 "junction_YYYYMMDDHHMM.nc" や "refined_YYYYMMDDHHMM.nc" 等
      - prefix (str): 接頭辞（例 "junction_", "refined_" など）

    出力:
      - str: "YYYYMMDDHHMM"
    """
    return fname.replace(prefix, "").replace(".nc", "")


def _load_pressure_and_lowcenter(month_str: str, time_dt: pd.Timestamp):
    """
    GSM から海面更正気圧を読み、偏差と低気圧中心(mask)を返す。
    低気圧中心は "surface_low_center" 変数があれば使用。
    """
    prmsl = None
    low_mask = None
    low_center_exists = False

    gsm_file = os.path.join(nc_gsm_dir, f"gsm{month_str}.nc")
    if not os.path.exists(gsm_file):
        return None, None, low_center_exists

    try:
        ds = xr.open_dataset(gsm_file)
        ds_times = pd.to_datetime(ds["time"].values)
        if time_dt in ds_times:
            dat = ds.sel(time=time_dt)
        else:
            timediffs = np.abs(ds_times - time_dt)
            midx = timediffs.argmin()
            if timediffs[midx] <= pd.Timedelta(hours=3):
                dat = ds.sel(time=ds["time"].values[midx])
            else:
                ds.close()
                return None, None, low_center_exists

        if "surface_prmsl" in dat:
            prmsl = dat["surface_prmsl"].values
        if "surface_low_center" in ds:
            try:
                if time_dt in ds_times:
                    lowcenter_arr = ds["surface_low_center"].sel(time=time_dt).values
                else:
                    lowcenter_arr = ds["surface_low_center"].sel(time=ds["time"].values[midx]).values
                low_mask = (lowcenter_arr == 1)
                low_center_exists = True
            except Exception:
                low_mask = None
                low_center_exists = False
        ds.close()
    except Exception:
        prmsl = None
        low_mask = None
        low_center_exists = False

    if prmsl is None:
        return None, None, low_center_exists

    area_mean = np.nanmean(prmsl)
    pressure_dev = prmsl - area_mean
    return pressure_dev, low_mask, low_center_exists

def _get_gsm_lat_lon(time_dt: pd.Timestamp):
    """
    GSM の当該月ファイルから (lat, lon) の 1D 座標を取得する（可視化用の堅牢化フォールバック）。
    見つからない/読み込めない場合は (None, None) を返す。
    """
    try:
        month_str = time_dt.strftime("%Y%m")
        gsm_file = os.path.join(nc_gsm_dir, f"gsm{month_str}.nc")
        if not os.path.exists(gsm_file):
            return None, None
        ds = xr.open_dataset(gsm_file)
        lat = np.asarray(ds["lat"].values) if "lat" in ds else None
        lon = np.asarray(ds["lon"].values) if "lon" in ds else None
        ds.close()
        return lat, lon
    except Exception:
        return None, None


def _normalize_lat_lon(lat: np.ndarray, lon: np.ndarray, H: int, W: int, time_dt: pd.Timestamp):
    """
    予測配列 pred_cm の形状 (H, W) を正とし、lat/lon を 1D で長さ (H,)/(W,) に正規化する。
    - まず与えられた lat/lon を使用（1D かつ長さ一致なら採用）
    - 一致しない場合は GSM 側の座標から取得し、必要に応じて [:H], [:W] でトリム
    - それでもダメなら単調増加の擬似座標を生成（0..H-1, 0..W-1）
    """
    def _is_ok(a, n):
        try:
            a = np.asarray(a)
            return (a.ndim == 1) and (len(a) == n)
        except Exception:
            return False

    lat_ok = _is_ok(lat, H)
    lon_ok = _is_ok(lon, W)

    if lat_ok and lon_ok:
        return np.asarray(lat), np.asarray(lon)

    # GSM からフォールバック
    glat, glon = _get_gsm_lat_lon(time_dt)
    if _is_ok(glat, H) and _is_ok(glon, W):
        return np.asarray(glat), np.asarray(glon)
    if glat is not None and glon is not None:
        # 長さが違う場合は最小限の切り詰め
        glat = np.asarray(glat)
        glon = np.asarray(glon)
        lat2 = glat[:H] if glat.ndim == 1 else np.squeeze(glat)[:H]
        lon2 = glon[:W] if glon.ndim == 1 else np.squeeze(glon)[:W]
        if _is_ok(lat2, H) and _is_ok(lon2, W):
            return np.asarray(lat2), np.asarray(lon2)

    # 最終フォールバック: インデックス座標
    return np.linspace(0.0, float(H - 1), H), np.linspace(0.0, float(W - 1), W)


def _gt_class_map_for_time(time_dt: pd.Timestamp, lat: np.ndarray, lon: np.ndarray):
    """
    GT: 5ch (warm, cold, stationary, occluded, warm_cold) -> 0..5 class_map
    """
    month_str = time_dt.strftime("%Y%m")
    gtf = os.path.join(nc_0p5_dir, f"{month_str}.nc")
    h, w = len(lat), len(lon)
    if not os.path.exists(gtf):
        return np.zeros((h, w), dtype=np.int64)

    ds = xr.open_dataset(gtf)
    if time_dt in ds["time"]:
        arr5 = ds.sel(time=time_dt).to_array().values  # (5,H,W)
    else:
        diff = np.abs(ds["time"].values - np.datetime64(time_dt))
        idx = diff.argmin()
        if diff[idx] <= np.timedelta64(3, "h"):
            arr5 = ds.sel(time=ds["time"][idx]).to_array().values
        else:
            ds.close()
            return np.zeros((h, w), dtype=np.int64)
    ds.close()
    gt = np.zeros((h, w), dtype=np.int64)
    # 1:warm,2:cold,3:stationary,4:occluded,5:warm_cold
    if arr5.shape[1] != h or arr5.shape[2] != w:
        arr5 = arr5[:, :h, :w]
    for c in range(5):
        mask = (arr5[c] == 1)
        gt[mask] = c + 1
    return gt


def _pred_class_map_for_stage(stage_name: str, nc_path: str):
    """
    v4 各ステージの出力 .nc を読み、0..5 の class_map に正規化して返す。
    仕様: Stage1/1.5 で junction=5 を確定、Stage2/2.5 で warm=1/cold=2/junction=5 を確定。
         以降のステージ表示では、確定済みを必ず重畳して見えるよう合成する（上書き禁止）。
    """
    ds = xr.open_dataset(nc_path)
    time_val = ds["time"].values[0] if "time" in ds else None
    t_dt = pd.to_datetime(time_val) if time_val is not None else None
    lat = ds["lat"].values
    lon = ds["lon"].values
    # 強制的に1次元座標へ正規化（2D格子に保存されているケース対策）
    lat, lon = _to_1d_lat_lon(lat, lon)

    h, w = len(lat), len(lon)
    pred = np.zeros((h, w), dtype=np.int64)

    def _prob_argmax(dsvar):
        prob = dsvar.isel(time=0).values  # (H,W,C)
        return np.argmax(prob, axis=-1).astype(np.int64)

    # 便利関数: 指定 time の Stage1.5/Stage2.5 を読み warm/cold/junction を取得（なければゼロ）
    def _load_fixed_layers(t_dt_local: pd.Timestamp):
        jmask = np.zeros((h, w), dtype=np.uint8)
        warm = np.zeros((h, w), dtype=np.uint8)
        cold = np.zeros((h, w), dtype=np.uint8)
        if t_dt_local is None:
            # (h,w) へ整形して返す
            warm_al = _align_to_hw(warm, h, w)
            cold_al = _align_to_hw(cold, h, w)
            jmask_al = _align_to_hw(jmask, h, w)
            return warm_al, cold_al, jmask_al
        token = t_dt_local.strftime("%Y%m%d%H%M")
        # Stage2.5 warm/cold (+ junction if available)
        wcpath = os.path.join(stage2_5_out_dir, f"refined_{token}.nc")
        if os.path.exists(wcpath):
            with xr.open_dataset(wcpath) as wd:
                if "class_map" in wd:
                    cm = wd["class_map"]
                    arr = cm.isel(time=0).values if "time" in cm.dims else cm.values
                    arr = np.asarray(arr)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim != 2:
                        arr = np.squeeze(arr)
                    warm = (arr == 1).astype(np.uint8)
                    cold = (arr == 2).astype(np.uint8)
                else:
                    if "warm" in wd and "cold" in wd:
                        w_arr = wd["warm"]
                        c_arr = wd["cold"]
                        w_arr = w_arr.isel(time=0).values if "time" in w_arr.dims else w_arr.values
                        c_arr = c_arr.isel(time=0).values if "time" in c_arr.dims else c_arr.values
                        warm = (np.squeeze(w_arr) > 0.5).astype(np.uint8)
                        cold = (np.squeeze(c_arr) > 0.5).astype(np.uint8)
                # Prefer Stage2.5 junction if present
                if "junction" in wd:
                    jv = wd["junction"]
                    jarr = jv.isel(time=0).values if "time" in jv.dims else jv.values
                    jmask = (np.squeeze(jarr) > 0).astype(np.uint8)
        # Fallback: Stage1.5 junction
        if jmask.sum() == 0:
            jpath = os.path.join(stage1_5_out_dir, f"junction_{token}.nc")
            if os.path.exists(jpath):
                with xr.open_dataset(jpath) as jd:
                    if "junction" in jd:
                        j = jd["junction"]
                        jarr = j.isel(time=0).values if "time" in j.dims else j.values
                        jmask = (np.squeeze(jarr) > 0).astype(np.uint8)
                    elif "class_map" in jd:
                        jmask = (jd["class_map"].values.astype(np.int64) > 0).astype(np.uint8)
        # (h,w) へ整形して返す
        warm = _align_to_hw(warm, h, w)
        cold = _align_to_hw(cold, h, w)
        jmask = _align_to_hw(jmask, h, w)
        return warm, cold, jmask

    try:
        if stage_name == "stage1":
            cls = _prob_argmax(ds["probabilities"])
            pred = np.where(cls == 5, 5, 0)
        elif stage_name == "stage1_5":
            if "junction" in ds:
                j = ds["junction"]
                jmask = (j.isel(time=0).values if "time" in j.dims else j.values).astype(np.uint8)
            elif "class_map" in ds:
                jmask = (ds["class_map"].values.astype(np.int64) > 0).astype(np.uint8)
            else:
                var = list(ds.data_vars)[0]
                v = ds[var]
                jmask = (v.isel(time=0).values if "time" in v.dims else v.values)
                jmask = (jmask > 0.5).astype(np.uint8)
            pred = np.where(jmask == 1, 5, 0)
        elif stage_name == "stage2":
            # 確定: junction(5) + warm(1)/cold(2)
            cls = _prob_argmax(ds["probabilities"])
            warm = (cls == 1).astype(np.uint8)
            cold = (cls == 2).astype(np.uint8)
            # junction は Stage2.5 を優先（なければ Stage1.5 にフォールバック）
            wfix, cfix, jmask = _load_fixed_layers(t_dt)
            # 表示は確定順で重畳（上書き禁止）
            pred = np.zeros((h, w), dtype=np.int64)
            pred[jmask == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
        elif stage_name == "stage2_5":
            # class_map: 0/1/2 + junction 変数が同梱（必ず2Dに正規化）
            v = ds["class_map"]
            arr = v.isel(time=0).values if "time" in v.dims else v.values
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            cm = arr.astype(np.int64)
            warm = (cm == 1).astype(np.uint8)
            cold = (cm == 2).astype(np.uint8)
            if "junction" in ds:
                j = ds["junction"]
                jarr = j.isel(time=0).values if "time" in j.dims else j.values
                jmask = (np.squeeze(np.asarray(jarr)) > 0).astype(np.uint8)
            else:
                # 念のため Stage1.5 から読む
                _, _, jmask = _load_fixed_layers(t_dt)
            pred = np.zeros((h, w), dtype=np.int64)
            pred[jmask == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
        elif stage_name == "stage3":
            # occluded 予測 + 既確定の junction/warm/cold を重畳
            cls = _prob_argmax(ds["probabilities"])
            occ = (cls == 1).astype(np.uint8)
            warm, cold, jmask = _load_fixed_layers(t_dt)
            pred = np.zeros((h, w), dtype=np.int64)
            pred[jmask == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
            mask = (pred == 0) & (occ == 1)
            pred[mask] = 4
        elif stage_name == "stage3_5":
            # class_map（閉塞0/1）を2Dに正規化
            v = ds["class_map"] if "class_map" in ds else None
            if v is not None:
                arr = v.isel(time=0).values if "time" in v.dims else v.values
            else:
                # フォールバック（通常到達しない想定）
                var = list(ds.data_vars)[0]
                vv = ds[var]
                arr = vv.isel(time=0).values if "time" in vv.dims else vv.values
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            occ = (arr > 0).astype(np.uint8)
            # (h,w) に整形
            occ = _align_to_hw(occ, h, w)
            warm, cold, jmask = _load_fixed_layers(t_dt)
            pred = np.zeros((h, w), dtype=np.int64)
            pred[jmask == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
            mask = (pred == 0) & (occ == 1)
            pred[mask] = 4
        elif stage_name == "stage4":
            # stationary 予測 + 既確定の junction/warm/cold + 可能なら stage3_5 occluded を重畳
            cls = _prob_argmax(ds["probabilities"])
            sta = (cls == 1).astype(np.uint8)
            warm, cold, jmask = _load_fixed_layers(t_dt)
            # occluded は可能なら stage3_5 から（time 次元を落として2次元に正規化）
            occ = np.zeros((h, w), dtype=np.uint8)
            if t_dt is not None:
                ocp = os.path.join(stage3_5_out_dir, f"occluded_{t_dt.strftime('%Y%m%d%H%M')}.nc")
                if os.path.exists(ocp):
                    with xr.open_dataset(ocp) as od:
                        if "class_map" in od:
                            v = od["class_map"]
                            arr = v.isel(time=0).values if "time" in v.dims else v.values
                            arr = np.squeeze(np.asarray(arr))
                            if arr.ndim == 2:
                                occ = (arr > 0).astype(np.uint8)
                        elif "occluded" in od:
                            v = od["occluded"]
                            arr = v.isel(time=0).values if "time" in v.dims else v.values
                            arr = np.squeeze(np.asarray(arr))
                            if arr.ndim == 2:
                                occ = (arr > 0.5).astype(np.uint8)
            # (h,w) に整形
            occ = _align_to_hw(occ, h, w)
            pred = np.zeros((h, w), dtype=np.int64)
            pred[jmask == 1] = 5
            mask = (pred == 0) & (warm == 1)
            pred[mask] = 1
            mask = (pred == 0) & (cold == 1)
            pred[mask] = 2
            mask = (pred == 0) & (occ == 1)
            pred[mask] = 4
            mask = (pred == 0) & (sta == 1)
            pred[mask] = 3
        elif stage_name == "stage4_5":
            # 最終合成（0..5）を2Dに正規化
            v = ds["class_map"]
            arr = v.isel(time=0).values if "time" in v.dims else v.values
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            # ensure 2D shape aligned to (h,w)
            arr = _align_to_hw(arr, h, w)
            pred = arr.astype(np.int64)
        else:
            pred = np.zeros((h, w), dtype=np.int64)
    finally:
        ds.close()

    # 予測配列の形状に lat/lon を合わせる（final_*.nc の座標が壊れている場合の堅牢化）
    try:
        if isinstance(pred, np.ndarray) and (pred.ndim == 2):
            h_pred, w_pred = pred.shape
            lat, lon = _normalize_lat_lon(lat, lon, h_pred, w_pred, t_dt if t_dt is not None else pd.to_datetime("1970-01-01"))
    except Exception:
        pass

    return pred, lat, lon, t_dt


def _stage_dirs(stage_name: str):
    """
    関数概要:
      可視化対象ステージ名に応じて、入力ディレクトリ・ファイル接頭辞・出力ディレクトリ名（タグ）を決定する。

    入力:
      - stage_name (str): "stage1","stage1_5","stage2","stage2_5","stage3","stage3_5","stage4","stage4_5" のいずれか

    処理:
      - 各ステージの成果物の保存先（*_out_dir）と、ファイル名接頭辞（prob_/junction_/refined_/occluded_/final_）を返す
      - "stage4_5" は最終成果物として final_out_dir/final_*.nc を対象に表示

    出力:
      - Tuple[str, str, str]: (入力ディレクトリ, ファイル接頭辞, 出力タグ名)
    """
    if stage_name == "stage1":
        return stage1_out_dir, "prob_", "stage1"
    if stage_name == "stage1_5":
        return stage1_5_out_dir, "junction_", "stage1_5"
    if stage_name == "stage2":
        return stage2_out_dir, "prob_", "stage2"
    if stage_name == "stage2_5":
        return stage2_5_out_dir, "refined_", "stage2_5"
    if stage_name == "stage3":
        return stage3_out_dir, "prob_", "stage3"
    if stage_name == "stage3_5":
        return stage3_5_out_dir, "occluded_", "stage3_5"
    if stage_name == "stage4":
        return stage4_out_dir, "prob_", "stage4"
    if stage_name == "stage4_5":
        # 最終成果物は final_out_dir/final_*.nc を主対象に表示
        return final_out_dir, "final_", "stage4_5"
    raise ValueError(f"Unknown stage_name: {stage_name}")


def _draw_and_save(stage_name: str, time_str: str, pred_cm: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   out_dir_stage: str):
    """
    Pred vs GT の 2 パネルを保存。背景に気圧偏差、低気圧中心（任意）を重畳。
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    gt_cm = _gt_class_map_for_time(pd.to_datetime(time_str, format="%Y%m%d%H%M"), lat, lon)

    # 背景（pressure deviation, low centers）
    month_str = time_str[:6]
    pdev, low_mask, low_exists = _load_pressure_and_lowcenter(month_str, pd.to_datetime(time_str, format="%Y%m%d%H%M"))

    # pred_cm の形状に lat/lon を正規化（座標が壊れている場合の対策）
    H, W = pred_cm.shape
    lat, lon = _normalize_lat_lon(lat, lon, H, W, pd.to_datetime(time_str, format="%Y%m%d%H%M"))
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    class_colors = CFG["VISUALIZATION"]["class_colors"]
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    pressure_vmin = CFG["VISUALIZATION"]["pressure_vmin"]
    pressure_vmax = CFG["VISUALIZATION"]["pressure_vmax"]
    pressure_levels = np.linspace(pressure_vmin, pressure_vmax, CFG["VISUALIZATION"]["pressure_levels"])
    pressure_norm = mcolors.Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
    cmap_pressure = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(16, 6))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

    ax0 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = plt.subplot(gs[1], projection=ccrs.PlateCarree())
    cax = plt.subplot(gs[2])

    # extent がゼロ幅になると Cartopy が特異になるため微小幅を付与
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    if lon_min == lon_max:
        lon_max = lon_min + 1e-6
    if lat_min == lat_max:
        lat_max = lat_min + 1e-6
    extent = [lon_min, lon_max, lat_min, lat_max]
    for ax in [ax0, ax1]:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")
        ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"))
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        ax.tick_params(labelsize=8)

        # 背景の形状が予測と一致する場合のみ重畳（不一致なら安全にスキップ）
        if pdev is not None and isinstance(pdev, np.ndarray) and pdev.shape == pred_cm.shape:
            ax.contourf(
                lon_grid, lat_grid, pdev,
                levels=pressure_levels, cmap=cmap_pressure, extend="both",
                norm=pressure_norm, transform=ccrs.PlateCarree(), zorder=0
            )
            ax.contour(
                lon_grid, lat_grid, pdev,
                levels=pressure_levels, colors="black", linestyles="--", linewidths=1.0,
                transform=ccrs.PlateCarree(), zorder=1
            )

    # Pred
    ax0.pcolormesh(
        lon_grid, lat_grid, pred_cm,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax0.set_title(f"{stage_name} Pred\n{time_str}")

    # GT
    ax1.pcolormesh(
        lon_grid, lat_grid, gt_cm,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        alpha=0.6, zorder=2
    )
    ax1.set_title(f"Ground Truth\n{time_str}")

    # 低気圧中心
    if pdev is not None and low_exists and (low_mask is not None) and (low_mask.shape == pred_cm.shape):
        y_idx, x_idx = np.where(low_mask)
        low_lats = lat[y_idx]
        low_lons = lon[x_idx]
        for ax in [ax0, ax1]:
            ax.plot(low_lons, low_lats, "rx", markersize=6, markeredgewidth=1.5, zorder=6, label="低気圧中心")

    sm = plt.cm.ScalarMappable(cmap=cmap_pressure, norm=pressure_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("海面更正気圧の偏差 (hPa)")

    os.makedirs(out_dir_stage, exist_ok=True)
    out_path = os.path.join(out_dir_stage, f"comparison_{stage_name}_{time_str}.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    gc.collect()


def run_visualization_for_stage(stage_name: str):
    """
    指定ステージの .nc を可視化し PNG を保存。
    stage_name in {"stage1","stage1_5","stage2","stage2_5","stage3","stage3_5","stage4","stage4_5"}
    """
    print_memory_usage(f"Start Visualization for {stage_name}")
    t0 = time.time()

    in_dir, prefix, stage_tag = _stage_dirs(stage_name)
    files = _list_nc(in_dir, prefix=prefix)
    if not files:
        print(f"[v4-visualize] No files for {stage_name} in: {in_dir}")
        return

    out_dir_stage = os.path.join(output_visual_dir, stage_tag)
    for f in files:
        time_str = f.replace(prefix, "").replace(".nc", "")
        nc_path = os.path.join(in_dir, f)
        try:
            pred_cm, lat, lon, t_dt = _pred_class_map_for_stage(stage_name, nc_path)
            if t_dt is None:
                # 可能な限りファイル名から復元
                t_dt = pd.to_datetime(time_str, format="%Y%m%d%H%M")
            _draw_and_save(stage_tag, t_dt.strftime("%Y%m%d%H%M"), pred_cm, lat, lon, out_dir_stage)
        except Exception as e:
            print(f"[v4-visualize] Failed to visualize {stage_name} {f}: {e}")
            continue

    print_memory_usage(f"After Visualization for {stage_name}")
    print(f"[v4-visualize] {stage_name} done in {format_time(time.time() - t0)}")


def run_visualization_final():
    """
    Final用の複合比較図を生成する。
    各時刻ごとに以下9枚を1枚にまとめ、v4_result/visualizations/final へ保存する。
      上段: stage1, stage1_5, stage2, stage2_5
      中段: stage3, stage3_5, stage4, stage4_5
      下段: 中央に Ground Truth（GT）
    背景には海面更正気圧の偏差を重畳し、低気圧中心があれば赤×で表示する。
    出力ファイル名: comparison_final_YYYYMMDDHHMM.png
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print_memory_usage("Start Visualization for final (9-up grid)")
    t0 = time.time()

    # final_out_dir の final_*.nc を基準に時刻列挙
    if not os.path.exists(final_out_dir):
        print(f"[v4-visualize] No final nc directory: {final_out_dir}")
        return
    files = [f for f in os.listdir(final_out_dir) if f.startswith("final_") and f.endswith(".nc")]
    files = sorted(files)
    if not files:
        print(f"[v4-visualize] No files for final in: {final_out_dir}")
        return

    out_dir = os.path.join(output_visual_dir, "final")
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        token = f.replace("final_", "").replace(".nc", "")
        time_dt = pd.to_datetime(token, format="%Y%m%d%H%M")

        # 各ステージの予測class_mapを取得
        stages_top = ["stage1", "stage1_5", "stage2", "stage2_5"]
        stages_mid = ["stage3", "stage3_5", "stage4"]
        titles_top = ["Stage1", "Stage1.5", "Stage2", "Stage2.5"]
        titles_mid = ["Stage3", "Stage3.5", "Stage4"]

        # 各ステージの予測class_mapを取得（辞書に格納）
        pred_by_stage = {}
        def _stage_nc_dir_and_prefix(st: str):
            if st == "stage1":   return stage1_out_dir, "prob_"
            if st == "stage1_5": return stage1_5_out_dir, "junction_"
            if st == "stage2":   return stage2_out_dir, "prob_"
            if st == "stage2_5": return stage2_5_out_dir, "refined_"
            if st == "stage3":   return stage3_out_dir, "prob_"
            if st == "stage3_5": return stage3_5_out_dir, "occluded_"
            if st == "stage4":   return stage4_out_dir, "prob_"
            if st == "stage4_5": return final_out_dir, "final_"
            return None, None

        stages_all = stages_top + stages_mid + ["stage4_5"]
        for st in stages_all:
            nc_dir, prefix = _stage_nc_dir_and_prefix(st)
            if (nc_dir is None) or (not os.path.exists(os.path.join(nc_dir, f"{prefix}{token}.nc"))):
                pred_by_stage[st] = None
                continue
            try:
                pred_cm, plat, plon, _ = _pred_class_map_for_stage(st, os.path.join(nc_dir, f"{prefix}{token}.nc"))
                # 予測に合わせて座標を正規化
                H, W = pred_cm.shape
                nlat, nlon = _normalize_lat_lon(plat, plon, H, W, time_dt)
                pred_by_stage[st] = (pred_cm, nlat, nlon)
            except Exception as e:
                print(f"[v4-visualize] Failed to read stage {st} for {token}: {e}")
                pred_by_stage[st] = None

        # まず final のファイルが壊れていても座標を確実に取得できるよう lat/lon を最初に決める
        # stage4_5 か final の座標を優先に、ダメなら GSM で補完
        try:
            pred_dummy, lat, lon, _ = _pred_class_map_for_stage("stage4_5", os.path.join(final_out_dir, f))
        except Exception:
            lat, lon = _get_gsm_lat_lon(time_dt)
            if lat is None or lon is None:
                print(f"[v4-visualize] Could not determine lat/lon for {token}")
                continue

        # 参照座標決定用の候補（すでに pred_by_stage を構築済みのためそれを利用）
        pred_maps = [pred_by_stage.get(st, None) for st in (stages_top + stages_mid + ["stage4_5"])]

        # GT class map
        # pred_mapsに含まれる最初の有効な座標に合わせる
        ref = next((x for x in pred_maps if x is not None), None)
        if ref is None:
            # 何も描けない
            print(f"[v4-visualize] No valid predictions for {token}")
            continue
        _, plat, plon = ref
        gt_cm = _gt_class_map_for_time(time_dt, plat, plon)

        # 背景の気圧偏差・低気圧中心
        pdev, low_mask, low_exists = _load_pressure_and_lowcenter(token[:6], time_dt)

        # 描画
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        class_colors = CFG["VISUALIZATION"]["class_colors"]
        cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
        bounds = np.arange(len(class_colors) + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        pressure_vmin = CFG["VISUALIZATION"]["pressure_vmin"]
        pressure_vmax = CFG["VISUALIZATION"]["pressure_vmax"]
        pressure_levels = np.linspace(pressure_vmin, pressure_vmax, CFG["VISUALIZATION"]["pressure_levels"])
        pressure_norm = mcolors.Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
        cmap_pressure = plt.get_cmap("RdBu_r")

        fig = plt.figure(figsize=(16, 12))
        from matplotlib import gridspec
        # 行間(hspace)を拡大、下段2枚の横間(wspace)のみさらに詰める（subgridspecで個別制御）
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1], wspace=0.1, hspace=0.30)

        # 上段: 4枚
        axes_top = [plt.subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(4)]
        # 中段: 3枚（右端1列は未使用）
        axes_mid = [plt.subplot(gs[1, i], projection=ccrs.PlateCarree()) for i in range(3)]
        # 下段: 左に Stage4.5、右に GT（下段のみ wspace を小さくして“より近づける”）
        bottom_sub = gs[2, :].subgridspec(1, 2, wspace=0.02)
        ax_bottom_left = plt.subplot(bottom_sub[0, 0], projection=ccrs.PlateCarree())
        ax_bottom_right = plt.subplot(bottom_sub[0, 1], projection=ccrs.PlateCarree())

        # 描画用 extent
        lat_ref, lon_ref = plat, plon
        lon_min, lon_max = float(np.min(lon_ref)), float(np.max(lon_ref))
        lat_min, lat_max = float(np.min(lat_ref)), float(np.max(lat_ref))
        if lon_min == lon_max: lon_max = lon_min + 1e-6
        if lat_min == lat_max: lat_max = lat_min + 1e-6
        extent = [lon_min, lon_max, lat_min, lat_max]

        # 背景描画 + 前線予測の重畳
        def draw_panel(ax, pred_info, title):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")
            ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.5)
            ax.add_feature(cfeature.RIVERS.with_scale("10m"))
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            ax.tick_params(labelsize=8)

            # 背景
            if pdev is not None and isinstance(pdev, np.ndarray):
                lon_grid, lat_grid = np.meshgrid(lon_ref, lat_ref)
                ax.contourf(
                    lon_grid, lat_grid, pdev,
                    levels=pressure_levels, cmap=cmap_pressure, extend="both",
                    norm=pressure_norm, transform=ccrs.PlateCarree(), zorder=0
                )
                ax.contour(
                    lon_grid, lat_grid, pdev,
                    levels=pressure_levels, colors="black", linestyles="--", linewidths=1.0,
                    transform=ccrs.PlateCarree(), zorder=1
                )
            # 予測
            if pred_info is not None:
                pred_cm, plat_i, plon_i = pred_info
                # 念のため座標整合
                H, W = pred_cm.shape
                plat_i, plon_i = _normalize_lat_lon(plat_i, plon_i, H, W, time_dt)
                lon_grid, lat_grid = np.meshgrid(plon_i, plat_i)
                ax.pcolormesh(
                    lon_grid, lat_grid, pred_cm,
                    cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                    alpha=0.6, zorder=2
                )
            # 低気圧中心
            if pdev is not None and low_exists and (low_mask is not None) and (low_mask.shape == (len(lat_ref), len(lon_ref))):
                y_idx, x_idx = np.where(low_mask)
                low_lats = lat_ref[y_idx]
                low_lons = lon_ref[x_idx]
                ax.plot(low_lons, low_lats, "rx", markersize=6, markeredgewidth=1.5, zorder=6)

            ax.set_title(title)

        # 上段4, 中段3
        for i, ax in enumerate(axes_top):
            draw_panel(ax, pred_by_stage.get(stages_top[i], None), titles_top[i] + f"\n{token}")
        for j, ax in enumerate(axes_mid):
            draw_panel(ax, pred_by_stage.get(stages_mid[j], None), titles_mid[j] + f"\n{token}")

        # 下段 左: Stage4.5, 右: GT
        draw_panel(ax_bottom_left, pred_by_stage.get("stage4_5", None), "Stage4.5\n" + token)

        def draw_gt(ax):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")
            ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.5)
            ax.add_feature(cfeature.RIVERS.with_scale("10m"))
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            ax.tick_params(labelsize=8)
            lon_grid, lat_grid = np.meshgrid(lon_ref, lat_ref)
            if pdev is not None and isinstance(pdev, np.ndarray):
                ax.contourf(
                    lon_grid, lat_grid, pdev,
                    levels=pressure_levels, cmap=cmap_pressure, extend="both",
                    norm=pressure_norm, transform=ccrs.PlateCarree(), zorder=0
                )
                ax.contour(
                    lon_grid, lat_grid, pdev,
                    levels=pressure_levels, colors="black", linestyles="--", linewidths=1.0,
                    transform=ccrs.PlateCarree(), zorder=1
                )
            ax.pcolormesh(
                lon_grid, lat_grid, gt_cm,
                cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                alpha=0.6, zorder=2
            )
            if pdev is not None and low_exists and (low_mask is not None) and (low_mask.shape == (len(lat_ref), len(lon_ref))):
                y_idx, x_idx = np.where(low_mask)
                low_lats = lat_ref[y_idx]
                low_lons = lon_ref[x_idx]
                ax.plot(low_lons, low_lats, "rx", markersize=6, markeredgewidth=1.5, zorder=6)
            ax.set_title(f"Ground Truth\n{token}")

        draw_gt(ax_bottom_right)

        save_path = os.path.join(out_dir, f"comparison_final_{token}.png")
        try:
            plt.savefig(save_path, dpi=220, bbox_inches="tight")
        except Exception as e:
            print(f"[v4-visualize] Save failed: {save_path}: {e}")
        plt.close()
        gc.collect()

    print_memory_usage("After Visualization for final (9-up grid)")
    print(f"[v4-visualize] final grid done in {format_time(time.time() - t0)}")


def create_lowres_videos_for_all_stages():
    """
    可視化フォルダごとに低解像度の通年動画(YYYY=CFG['EVAL']['year'])を1本ずつ作成する。
    出力ファイル:
      - stage1    -> v4_result/visualizations/stage_1_comparison_{year}_full_year_low.mp4
      - stage1_5  -> v4_result/visualizations/stage_1_5_comparison_{year}_full_year_low.mp4
      - stage2    -> v4_result/visualizations/stage_2_comparison_{year}_full_year_low.mp4
      - stage2_5  -> v4_result/visualizations/stage_2_5_comparison_{year}_full_year_low.mp4
      - stage3    -> v4_result/visualizations/stage_3_comparison_{year}_full_year_low.mp4
      - stage3_5  -> v4_result/visualizations/stage_3_5_comparison_{year}_full_year_low.mp4
      - stage4    -> v4_result/visualizations/stage_4_comparison_{year}_full_year_low.mp4
      - stage4_5  -> v4_result/visualizations/stage_4_5_comparison_{year}_full_year_low.mp4
      - final     -> v4_result/visualizations/final_comparison_{year}_full_year_low.mp4
    """
    import glob
    import cv2
    import tempfile
    import subprocess
    import shutil

    def _sorted_images(folder: str):
        files = glob.glob(os.path.join(folder, "comparison_*.png"))
        # ソートキーは末尾の時刻トークン
        def _key(p):
            base = os.path.basename(p).replace(".png", "")
            # token = 最後の "_" の右側
            token = base.rsplit("_", 1)[-1]
            return token
        files.sort(key=_key)
        return files

    def _make_lowres_video(image_folder: str, output_path_low: str, low_res_scale: int = 3, low_res_frame_rate: int = 1):
        imgs = _sorted_images(image_folder)
        if not imgs:
            print(f"[video] No images in {image_folder}, skip")
            return
        frame0 = cv2.imread(imgs[0])
        if frame0 is None:
            print(f"[video] Failed to read first image: {imgs[0]}")
            return
        h, w = frame0.shape[:2]
        low_w = max(2, (w // low_res_scale) // 2 * 2)
        low_h = max(2, (h // low_res_scale) // 2 * 2)

        tmpdir = tempfile.mkdtemp(prefix="v4vis_")
        try:
            for i, p in enumerate(imgs):
                img = cv2.imread(p)
                if img is None:
                    continue
                img_small = cv2.resize(img, (low_w, low_h))
                cv2.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), img_small)
            # ffmpeg で連結（yuv420p/padで非偶数回避）
            cmd = [
                "ffmpeg", "-y",
                "-r", str(low_res_frame_rate),
                "-i", os.path.join(tmpdir, "frame_%06d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-vcodec", "libx264",
                "-crf", "30",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                output_path_low,
            ]
            try:
                subprocess.run(cmd, check=False)
                print(f"[video] Wrote low-res video: {output_path_low}")
            except Exception as e:
                print(f"[video] ffmpeg failed: {e}")
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    year = CFG.get("EVAL", {}).get("year", 2023)
    out_root = output_visual_dir

    plans = [
        ("stage1",   os.path.join(out_root, "stage1"),   os.path.join(out_root, f"stage_1_comparison_{year}_full_year_low.mp4")),
        ("stage1_5", os.path.join(out_root, "stage1_5"), os.path.join(out_root, f"stage_1_5_comparison_{year}_full_year_low.mp4")),
        ("stage2",   os.path.join(out_root, "stage2"),   os.path.join(out_root, f"stage_2_comparison_{year}_full_year_low.mp4")),
        ("stage2_5", os.path.join(out_root, "stage2_5"), os.path.join(out_root, f"stage_2_5_comparison_{year}_full_year_low.mp4")),
        ("stage3",   os.path.join(out_root, "stage3"),   os.path.join(out_root, f"stage_3_comparison_{year}_full_year_low.mp4")),
        ("stage3_5", os.path.join(out_root, "stage3_5"), os.path.join(out_root, f"stage_3_5_comparison_{year}_full_year_low.mp4")),
        ("stage4",   os.path.join(out_root, "stage4"),   os.path.join(out_root, f"stage_4_comparison_{year}_full_year_low.mp4")),
        ("stage4_5", os.path.join(out_root, "stage4_5"), os.path.join(out_root, f"stage_4_5_comparison_{year}_full_year_low.mp4")),
        ("final",    os.path.join(out_root, "final"),    os.path.join(out_root, f"final_comparison_{year}_full_year_low.mp4")),
    ]

    for tag, img_dir, out_path in plans:
        if not os.path.exists(img_dir):
            print(f"[video] image dir not found for {tag}: {img_dir}, skip")
            continue
        _make_lowres_video(img_dir, out_path)

    print("[video] All requested low-res videos processed.")


__all__ = ["run_visualization_for_stage", "run_visualization_final", "create_lowres_videos_for_all_stages"]
