"""
ファイル概要（main_v4_datasets.py）:
- 役割:
  v4 パイプラインで用いる PyTorch Dataset 群（各 Stage の学習・推論データセット）と、補助ユーティリティ関数群
  （NetCDF の走査・GSM/GT 読み出し・3時刻条件チェック等）を提供する。
- 入出力の基本方針:
  入力:
    - GSM: 31 変数 × 3 時刻（t-6h, t, t+6h）を連結して 93 チャネル（_read_gsm_93）
    - 前線 GT: warm, cold, stationary, occluded, warm_cold の 5 チャネルを時刻選択し (5,H,W)
    - 途中生成物（Stage1.5/2.5/3.5）: junction/warm/cold/occluded を含む NetCDF（class_map や専用変数）
  出力:
    - 各 Dataset の __getitem__ は (x, y, timestr) を返す（推論用では y はダミー）
      x: モデル入力テンソル（C,H,W）、y: 学習ターゲット（クラスIDの2D）、timestr: サンプルの時刻（文字列）
- 時刻取り扱い:
  - 各サンプルは「GSM 側の t に対して t-6h, t, t+6h の3時刻が存在する」もののみ採用（_ensure_3times_exist）
  - 学習用 Dataset では「GSM と前線GTの共通時刻」からサンプルを構築
- 空間サイズ:
  - すべて ORIG_H, ORIG_W に切り出して整合（GSM/GT/途中生成物の端数をクリップ）

注意:
  - Stage3/Stage4 の推論・Stage3.5/Stage4.5・可視化においては「Stage2.5 後の junction（両側接触のみ残す）」を一貫して使用する。
  - Stage2.5 refined ファイルに junction が欠落する場合のフォールバックは「0 マスク」（class_map では代用しない）。
"""

import os
import gc
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset

from .main_v4_config import (
    ORIG_H, ORIG_W,
    CFG,
    nc_gsm_dir, nc_0p5_dir,
    stage1_5_out_dir, stage2_5_out_dir, stage3_5_out_dir,
    print_memory_usage,
)


def get_available_months(start_year: int, start_month: int, end_year: int, end_month: int) -> List[str]:
    """
    関数概要:
      開始年月から終了年月までの各月を 'YYYYMM' 文字列で列挙して返す（両端含む）。
    入力:
      - start_year (int), start_month (int): 開始年・月
      - end_year (int),   end_month (int)  : 終了年・月
    処理:
      - pandas の Timestamp と MonthBegin オフセットを用いて、月初で時系列を進めながら各月を収集する。
    出力:
      - List[str]: 'YYYYMM' 形式の昇順リスト
    """
    months = []
    current = pd.Timestamp(start_year, start_month, 1)
    end = pd.Timestamp(end_year, end_month, 1)
    while current <= end:
        months.append(current.strftime("%Y%m"))
        current = (current + pd.offsets.MonthBegin(1))
    return months


def _open_cached(file_cache: Dict[str, xr.Dataset], path: str, max_size: int) -> xr.Dataset:
    """
    関数概要:
      指定パスの NetCDF を xarray で開き、簡易 LRU 風キャッシュに保持して返す（メモリ/IO 最適化）。
    入力:
      - file_cache (Dict[str, xr.Dataset]): これまでに開いた Dataset を保持する辞書（キー: パス）
      - path (str): 開きたい NetCDF のパス
      - max_size (int): キャッシュ最大保持数（超過時は最古のエントリを close→削除）
    処理:
      - 既にキャッシュ済みならそれを返す。
      - 未キャッシュなら xr.open_dataset で開いてキャッシュに登録（サイズ超過なら先頭を削除）。
    出力:
      - xr.Dataset: 開いた（またはキャッシュから取得した）Dataset オブジェクト
    """
    if path in file_cache:
        return file_cache[path]
    ds = xr.open_dataset(path)
    if len(file_cache) >= max_size:
        # pop first inserted key
        old_key = next(iter(file_cache.keys()))
        try:
            file_cache[old_key].close()
        except Exception:
            pass
        del file_cache[old_key]
    file_cache[path] = ds
    return ds


def _ensure_3times_exist(ds_gsm: xr.Dataset, t_now: np.datetime64) -> Optional[Tuple[np.datetime64, np.datetime64]]:
    """
    関数概要:
      GSM データセットの時刻軸において、t_now の前後6時間（t-6h, t, t+6h）が全て存在するかをチェックする。
    入力:
      - ds_gsm (xr.Dataset): GSM の xarray データセット（time, lat, lon, variables）
      - t_now (np.datetime64): チェック対象の中心時刻
    処理:
      - pandas を用いて t-6h, t+6h を計算し、ds_gsm["time"] に全て含まれるかを検査。
    出力:
      - Optional[Tuple[np.datetime64, np.datetime64]]:
        3時刻が全て存在する場合は (t_prev, t_next) を返す。存在しない場合は None。
    """
    t_prev = pd.to_datetime(t_now) - pd.Timedelta(hours=6)
    t_next = pd.to_datetime(t_now) + pd.Timedelta(hours=6)
    t_prev_np = np.datetime64(t_prev)
    t_next_np = np.datetime64(t_next)
    if (t_prev_np in ds_gsm["time"]) and (t_next_np in ds_gsm["time"]) and (t_now in ds_gsm["time"]):
        return t_prev_np, t_next_np
    return None


def _read_gsm_93(ds_gsm: xr.Dataset, t_prev: np.datetime64, t_now: np.datetime64, t_next: np.datetime64) -> np.ndarray:
    """
    関数概要:
      GSM の 31 変数を 3 時刻（t-6h, t, t+6h）で取り出し、チャネル方向に連結して (93,H,W) を作る。
    入力:
      - ds_gsm (xr.Dataset): GSM データセット
      - t_prev, t_now, t_next (np.datetime64): 3時刻（_ensure_3times_exist で妥当性確認済み）
    処理:
      - sel(time=*) → to_array() → values で (C=31,H,W) を3つ取得し、axis=0 方向に連結して 93 チャネルにする。
    出力:
      - np.ndarray: (93,H,W) の float32 配列（モデル入力用）
    """
    data_prev = ds_gsm.sel(time=t_prev).to_array().load().values
    data_now = ds_gsm.sel(time=t_now).to_array().load().values
    data_next = ds_gsm.sel(time=t_next).to_array().load().values
    gsm = np.concatenate([data_prev, data_now, data_next], axis=0).astype(np.float32)
    return gsm


def _gt_to_front_channels(ds_front: xr.Dataset, t_now: np.datetime64) -> np.ndarray:
    """
    関数概要:
      前線 GT の 5 変数（warm, cold, stationary, occluded, warm_cold）を、指定時刻で (5,H,W) にまとめる。
    入力:
      - ds_front (xr.Dataset): 前線 GT（5 変数を含む Dataset）
      - t_now (np.datetime64): 抽出する時刻
    処理:
      - sel(time=t_now).to_array().values で変数をスタックし、float32 へ変換。
    出力:
      - np.ndarray: (5,H,W) の float32 配列（チャンネル順は [warm, cold, stationary, occluded, warm_cold]）
    """
    arr5 = ds_front.sel(time=t_now).to_array().load().values.astype(np.float32)  # (5,H,W)
    return arr5


def _read_time_from_file(nc_path: str) -> Optional[pd.Timestamp]:
    """
    .nc 内の time[0] を pd.Timestamp で返す（無ければ None）
    """
    try:
        ds = xr.open_dataset(nc_path)
        t = ds["time"].values[0] if "time" in ds else None
        ds.close()
        if t is None:
            return None
        return pd.to_datetime(t)
    except Exception:
        return None


def _scan_nc_with_time(nc_dir: str) -> List[Tuple[str, pd.Timestamp]]:
    """
    関数概要:
      指定ディレクトリ配下の .nc ファイルを列挙し、ファイル先頭の time[0] を読み取って (path, Timestamp) の一覧を返す。
    入力:
      - nc_dir (str): 走査対象ディレクトリ
    処理:
      - .nc 以外はスキップ。_read_time_from_file で time[0] を取得できたものだけを採用。
    出力:
      - List[Tuple[str, pd.Timestamp]]: (ファイルパス, 先頭時刻) の昇順リスト
    備考:
      - time を持たないファイルは推論パイプラインの結合条件として扱えないため除外する。
    """
    if not os.path.exists(nc_dir):
        return []
    out = []
    for f in sorted(os.listdir(nc_dir)):
        if not f.endswith(".nc"):
            continue
        p = os.path.join(nc_dir, f)
        t = _read_time_from_file(p)
        if t is not None:
            out.append((p, t))
    return out


class V4DatasetStage1Train(Dataset):
    """
    クラス概要（Stage1 学習用）:
      junction=5（二値: none/junction）を学習するための Dataset。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (93,H,W) float32（GSM 31 変数 × 3 時刻の連結）
        y: (H,W) int64（0:none, 1:junction）GT は warm_cold==1 を 1 とする2値
        timestr: サンプルの時刻（文字列）
    処理フロー:
      - GSM/GT の共通時刻のうち、GSM 側で t±6h が存在するサンプルのみ採用。
      - _read_gsm_93 で (93,H,W) を構築。GT の warm_cold から2値ターゲットを生成。
      - ORIG_H/ORIG_W で空間をクリップして返す。
    キャッシュ:
      - file_cache: xarray.Dataset の LRU 的キャッシュ
      - sample_cache: 前回アクセスしたサンプルの N 件保持
    """
    def __init__(self, months: List[str], cache_size: int = 50, file_cache_size: int = 10):
        self.months = months
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.file_cache: Dict[str, xr.Dataset] = {}
        self.sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage1Train prepare_index")
        for month in self.months:
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            front_file = os.path.join(nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(gsm_file) or not os.path.exists(front_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            # 共通時刻
            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)
            for t in times:
                ok = _ensure_3times_exist(ds_gsm, t)
                if ok is None:
                    continue
                t_prev, t_next = ok
                self.data_index.append({
                    "gsm_file": gsm_file,
                    "front_file": front_file,
                    "t": t,
                    "t_prev": t_prev,
                    "t_next": t_next,
                    "t_dt": pd.to_datetime(t)
                })
            ds_gsm.close(); ds_front.close()
            del ds_gsm, ds_front
            gc.collect()
        print_memory_usage("After V4DatasetStage1Train prepare_index")
        print(f"[V4S1Train] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        ds_front = xr.open_dataset(it["front_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        arr5 = _gt_to_front_channels(ds_front, it["t"])
        ds_gsm.close(); ds_front.close()
        # target: junction only (warm_cold)
        junc = (arr5[4] == 1).astype(np.int64)  # (H,W) 0/1
        # clip to ORIG_H/W in case input larger
        gsm = gsm[:, :ORIG_H, :ORIG_W]
        junc = junc[:ORIG_H, :ORIG_W]
        return gsm, junc, it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            gsm, tgt, t_dt = self.sample_cache[idx]
        else:
            gsm, tgt, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (gsm, tgt, t_dt)
        x = torch.from_numpy(gsm)  # (93,H,W) float
        y = torch.from_numpy(tgt).long()  # (H,W) long {0,1}
        return x, y, str(t_dt)


class V4DatasetStage1Test(Dataset):
    """
    クラス概要（Stage1 推論用）:
      junction=5（二値）モデルの推論入力（GSM のみ 93ch）を供給する Dataset。y はダミー。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (93,H,W) float32（GSM のみ）
        y: (H,W) int64（常に 0）ダミー
        timestr: サンプルの時刻（文字列）
    処理:
      - GSM の全時刻から、t±6h が存在するもののみ採用し、_read_gsm_93 で (93,H,W) を構築。
    """
    def __init__(self, months: List[str], cache_size: int = 50, file_cache_size: int = 10):
        self.months = months
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.file_cache: Dict[str, xr.Dataset] = {}
        self.sample_cache: Dict[int, Tuple[np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage1Test prepare_index")
        for month in self.months:
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            if not os.path.exists(gsm_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            times = np.sort(ds_gsm["time"].values)
            for t in times:
                ok = _ensure_3times_exist(ds_gsm, t)
                if ok is None:
                    continue
                t_prev, t_next = ok
                self.data_index.append({
                    "gsm_file": gsm_file,
                    "t": t,
                    "t_prev": t_prev,
                    "t_next": t_next,
                    "t_dt": pd.to_datetime(t)
                })
            ds_gsm.close()
            del ds_gsm
            gc.collect()
        print_memory_usage("After V4DatasetStage1Test prepare_index")
        print(f"[V4S1Test] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        ds_gsm.close()
        gsm = gsm[:, :ORIG_H, :ORIG_W]
        return gsm, it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            gsm, t_dt = self.sample_cache[idx]
        else:
            gsm, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (gsm, t_dt)
        x = torch.from_numpy(gsm).float()
        y = torch.zeros(ORIG_H, ORIG_W, dtype=torch.long)
        return x, y, str(t_dt)


class V4DatasetStage2Train(Dataset):
    """
    クラス概要（Stage2 学習用）:
      warm/cold（3クラス: 0:none, 1:warm, 2:cold）を学習する Dataset。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (94,H,W) float32 = GSM(93) + GT junction(1)
        y: (H,W) int64（0/1/2）warm==1→1、cold==1→2、重複は cold を優先
        timestr: 時刻（文字列）
    処理:
      - GSM/GT 共通時刻かつ t±6h が存在するサンプルを採用。
      - junction は GT warm_cold から 0/1 マスクを作成。
    """
    def __init__(self, months: List[str], cache_size: int = 50, file_cache_size: int = 10):
        self.months = months
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.file_cache: Dict[str, xr.Dataset] = {}
        self.sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage2Train prepare_index")
        for month in self.months:
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            front_file = os.path.join(nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(gsm_file) or not os.path.exists(front_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)
            for t in times:
                ok = _ensure_3times_exist(ds_gsm, t)
                if ok is None:
                    continue
                t_prev, t_next = ok
                self.data_index.append({
                    "gsm_file": gsm_file,
                    "front_file": front_file,
                    "t": t,
                    "t_prev": t_prev,
                    "t_next": t_next,
                    "t_dt": pd.to_datetime(t)
                })
            ds_gsm.close(); ds_front.close()
            del ds_gsm, ds_front
            gc.collect()
        print_memory_usage("After V4DatasetStage2Train prepare_index")
        print(f"[V4S2Train] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        ds_front = xr.open_dataset(it["front_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])                   # (93,H,W)
        arr5 = _gt_to_front_channels(ds_front, it["t"])                                   # (5,H,W)
        ds_gsm.close(); ds_front.close()
        junction = (arr5[4] == 1).astype(np.float32)                                      # (H,W)
        warm = (arr5[0] == 1)
        cold = (arr5[1] == 1)
        y = np.zeros((ORIG_H, ORIG_W), dtype=np.int64)
        y[warm[:ORIG_H, :ORIG_W]] = 1
        # warm と cold の重複は cold に上書き（頻度は低い想定）
        mask_cold = cold[:ORIG_H, :ORIG_W]
        y[mask_cold] = 2
        x = np.concatenate([gsm[:, :ORIG_H, :ORIG_W], junction[:ORIG_H, :ORIG_W][None, ...]], axis=0)  # (94,H,W)
        return x.astype(np.float32), y, it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, y, t_dt = self.sample_cache[idx]
        else:
            x, y, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, y, t_dt)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), str(t_dt)


class V4DatasetStage2Test(Dataset):
    """
    クラス概要（Stage2 推論用）:
      warm/cold（3クラス）の推論入力を供給する Dataset。junction は Stage1.5 結果を使用。y はダミー。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (94,H,W) float32 = GSM(93) + Stage1.5 junction(1)
        y: (H,W) int64（常に 0）
        timestr: 時刻（文字列）
    処理:
      - Stage1.5 の junction_*.nc を走査し、各時刻トークンから対応する月の GSM を読み、t±6h 条件を満たすものを採用。
    """
    def __init__(self, cache_size: int = 50, file_cache_size: int = 10):
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.sample_cache: Dict[int, Tuple[np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage2Test prepare_index")
        junc_files = _scan_nc_with_time(stage1_5_out_dir)
        for p, t_dt in junc_files:
            month = t_dt.strftime("%Y%m")
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            if not os.path.exists(gsm_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            t_np = np.datetime64(t_dt)
            ok = _ensure_3times_exist(ds_gsm, t_np)
            ds_gsm.close()
            if ok is None:
                continue
            t_prev, t_next = ok
            self.data_index.append({
                "gsm_file": gsm_file,
                "junc_file": p,
                "t": t_np,
                "t_prev": t_prev,
                "t_next": t_next,
                "t_dt": t_dt
            })
        print_memory_usage("After V4DatasetStage2Test prepare_index")
        print(f"[V4S2Test] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _read_junction_mask(self, path: str) -> np.ndarray:
        """
        Stage1.5 の junction_* から 2次元 (H,W) の 0/1 マスクを返す。
        time 次元がある場合は time=0 を選択し、3次元 (1,H,W) や (time,H,W) の可能性をsqueezeする。
        """
        ds = xr.open_dataset(path)
        try:
            if "junction" in ds:
                j = ds["junction"]
                if "time" in j.dims:
                    arr = j.isel(time=0).values
                else:
                    arr = j.values
            elif "class_map" in ds:
                cm = ds["class_map"]
                if "time" in cm.dims:
                    arr = cm.isel(time=0).values
                else:
                    arr = cm.values
                arr = (arr.astype(np.int64) > 0).astype(np.float32)
                # ensure 2D return
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim != 2:
                    arr = np.squeeze(arr)
                return arr.astype(np.float32)
            else:
                # fallback: first data_var
                var = list(ds.data_vars)[0]
                v = ds[var]
                arr = v.isel(time=0).values if "time" in v.dims else v.values
            # squeeze to 2D
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            # normalize to {0,1}
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = (arr > 0).astype(np.float32)
            return arr
        finally:
            ds.close()

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        ds_gsm.close()
        junc = self._read_junction_mask(it["junc_file"])  # (H,W)
        x = np.concatenate([gsm[:, :ORIG_H, :ORIG_W], junc[:ORIG_H, :ORIG_W][None, ...]], axis=0)
        return x.astype(np.float32), it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, t_dt = self.sample_cache[idx]
        else:
            x, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, t_dt)
        y = torch.zeros(ORIG_H, ORIG_W, dtype=torch.long)
        return torch.from_numpy(x).float(), y, str(t_dt)


class V4DatasetStage3Train(Dataset):
    """
    クラス概要（Stage3 学習用）:
      occluded（二値: 0/1）を学習する Dataset。入力は GSM(93) + GT junction(1) + GT warm(1) + GT cold(1)。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (96,H,W) float32 = 93(GSM) + 1(junc GT) + 1(warm GT) + 1(cold GT)
        y: (H,W) int64 = occluded GT（0/1）
        timestr: 時刻（文字列）
    処理:
      - 学習は GT のみで構成（推論時は Stage2.5 出力を使用する点に注意）。
    """
    def __init__(self, months: List[str], cache_size: int = 50, file_cache_size: int = 10):
        self.months = months
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage3Train prepare_index")
        for month in self.months:
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            front_file = os.path.join(nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(gsm_file) or not os.path.exists(front_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)
            for t in times:
                ok = _ensure_3times_exist(ds_gsm, t)
                if ok is None: continue
                t_prev, t_next = ok
                self.data_index.append({
                    "gsm_file": gsm_file,
                    "front_file": front_file,
                    "t": t, "t_prev": t_prev, "t_next": t_next,
                    "t_dt": pd.to_datetime(t)
                })
            ds_gsm.close(); ds_front.close()
            del ds_gsm, ds_front
            gc.collect()
        print_memory_usage("After V4DatasetStage3Train prepare_index")
        print(f"[V4S3Train] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        ds_front = xr.open_dataset(it["front_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        arr5 = _gt_to_front_channels(ds_front, it["t"])
        ds_gsm.close(); ds_front.close()
        junc = (arr5[4] == 1).astype(np.float32)
        warm = (arr5[0] == 1).astype(np.float32)
        cold = (arr5[1] == 1).astype(np.float32)
        y = (arr5[3] == 1).astype(np.int64)  # occluded
        x = np.concatenate([
            gsm[:, :ORIG_H, :ORIG_W],
            junc[:ORIG_H, :ORIG_W][None, ...],
            warm[:ORIG_H, :ORIG_W][None, ...],
            cold[:ORIG_H, :ORIG_W][None, ...],
        ], axis=0)  # (96,H,W)
        return x.astype(np.float32), y[:ORIG_H, :ORIG_W], it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, y, t_dt = self.sample_cache[idx]
        else:
            x, y, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, y, t_dt)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), str(t_dt)


class V4DatasetStage3Test(Dataset):
    """
    クラス概要（Stage3 推論用）:
      occluded（二値）推論の入力を供給する Dataset。junction/warm/cold は Stage2.5（refined_*.nc）を使用。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (96,H,W) float32 = GSM(93) + Stage2.5 junction(1) + Stage2.5 warm(1) + Stage2.5 cold(1)
        y: (H,W) int64（常に 0）ダミー
        timestr: 時刻（文字列）
    注意:
      - Stage2.5 refined に junction が欠落する場合は 0 マスクでフォールバック（class_map では代用しない）。
      - GSM は t±6h 条件を満たすサンプルのみ採用。
    """
    def __init__(self, cache_size: int = 50, file_cache_size: int = 10):
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.sample_cache: Dict[int, Tuple[np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage3Test prepare_index")
        junc_files = _scan_nc_with_time(stage2_5_out_dir)
        warmcold_files = _scan_nc_with_time(stage2_5_out_dir)
        # time -> path map
        jmap = {t.strftime("%Y%m%d%H%M"): p for p, t in junc_files}
        wcmap = {t.strftime("%Y%m%d%H%M"): p for p, t in warmcold_files}
        common_keys = sorted(set(jmap.keys()) & set(wcmap.keys()))
        for key in common_keys:
            t_dt = pd.to_datetime(key, format="%Y%m%d%H%M")
            month = t_dt.strftime("%Y%m")
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            if not os.path.exists(gsm_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            t_np = np.datetime64(t_dt)
            ok = _ensure_3times_exist(ds_gsm, t_np)
            ds_gsm.close()
            if ok is None:
                continue
            t_prev, t_next = ok
            self.data_index.append({
                "gsm_file": gsm_file,
                "junc_file": jmap[key],
                "wc_file": wcmap[key],
                "t": t_np, "t_prev": t_prev, "t_next": t_next, "t_dt": t_dt
            })
        print_memory_usage("After V4DatasetStage3Test prepare_index")
        print(f"[V4S3Test] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _read_junction_mask(self, path: str) -> np.ndarray:
        """
        Read 2D (H,W) junction mask from Stage2.5 refined file.
        - junction 変数が無ければ warm/cold の class_map では代用せず、0マスクを返す（堅牢性重視）。
        """
        ds = xr.open_dataset(path)
        try:
            if "junction" in ds:
                j = ds["junction"]
                arr = j.isel(time=0).values if "time" in j.dims else j.values
                arr = np.squeeze(arr).astype(np.float32)
                if arr.ndim != 2:
                    arr = np.squeeze(arr)
                if arr.max() > 1.0:
                    arr = (arr > 0).astype(np.float32)
                return arr
            else:
                # 安全なフォールバック: 全0 (lat/lon 次元サイズがあればそれに合わせる)
                h = ds.dims.get("lat", ORIG_H)
                w = ds.dims.get("lon", ORIG_W)
                return np.zeros((h, w), dtype=np.float32)
        finally:
            ds.close()

    def _read_warm_cold(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        ds = xr.open_dataset(path)
        try:
            if "class_map" in ds:
                cm = ds["class_map"]
                arr = cm.isel(time=0).values if "time" in cm.dims else cm.values
                arr = np.asarray(arr)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim != 2:
                    arr = np.squeeze(arr)
                warm = (arr == 1).astype(np.float32)
                cold = (arr == 2).astype(np.float32)
            else:
                wv = ds["warm"] if "warm" in ds else None
                cv = ds["cold"] if "cold" in ds else None
                if wv is not None and cv is not None:
                    w_arr = wv.isel(time=0).values if "time" in wv.dims else wv.values
                    c_arr = cv.isel(time=0).values if "time" in cv.dims else cv.values
                    w_arr = np.squeeze(w_arr)
                    c_arr = np.squeeze(c_arr)
                    warm = (w_arr > 0.5).astype(np.float32)
                    cold = (c_arr > 0.5).astype(np.float32)
                else:
                    h = ds.dims.get("lat", ORIG_H)
                    w = ds.dims.get("lon", ORIG_W)
                    warm = np.zeros((h, w), dtype=np.float32)
                    cold = np.zeros((h, w), dtype=np.float32)
            return warm, cold
        finally:
            ds.close()

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        ds_gsm.close()
        junc = self._read_junction_mask(it["junc_file"])
        warm, cold = self._read_warm_cold(it["wc_file"])
        x = np.concatenate([
            gsm[:, :ORIG_H, :ORIG_W],
            junc[:ORIG_H, :ORIG_W][None, ...],
            warm[:ORIG_H, :ORIG_W][None, ...],
            cold[:ORIG_H, :ORIG_W][None, ...],
        ], axis=0)  # (96,H,W)
        return x.astype(np.float32), it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, t_dt = self.sample_cache[idx]
        else:
            x, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, t_dt)
        y = torch.zeros(ORIG_H, ORIG_W, dtype=torch.long)
        return torch.from_numpy(x).float(), y, str(t_dt)


class V4DatasetStage4Train(Dataset):
    """
    クラス概要（Stage4 学習用）:
      stationary（二値: 0/1）を学習する Dataset。入力は GSM(93) + GT junction + GT warm + GT cold + GT occluded。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (97,H,W) float32 = 93(GSM) + 1(junc GT) + 1(warm GT) + 1(cold GT) + 1(occ GT)
        y: (H,W) int64 = stationary GT（0/1）
        timestr: 時刻（文字列）
    """
    def __init__(self, months: List[str], cache_size: int = 50, file_cache_size: int = 10):
        self.months = months
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage4Train prepare_index")
        for month in self.months:
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            front_file = os.path.join(nc_0p5_dir, f"{month}.nc")
            if not os.path.exists(gsm_file) or not os.path.exists(front_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            ds_front = xr.open_dataset(front_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            times = np.intersect1d(ds_gsm["time"], ds_front["time"])
            times = np.sort(times)
            for t in times:
                ok = _ensure_3times_exist(ds_gsm, t)
                if ok is None: continue
                t_prev, t_next = ok
                self.data_index.append({
                    "gsm_file": gsm_file,
                    "front_file": front_file,
                    "t": t, "t_prev": t_prev, "t_next": t_next,
                    "t_dt": pd.to_datetime(t)
                })
            ds_gsm.close(); ds_front.close()
            del ds_gsm, ds_front
            gc.collect()
        print_memory_usage("After V4DatasetStage4Train prepare_index")
        print(f"[V4S4Train] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        ds_front = xr.open_dataset(it["front_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        arr5 = _gt_to_front_channels(ds_front, it["t"])
        ds_gsm.close(); ds_front.close()
        junc = (arr5[4] == 1).astype(np.float32)
        warm = (arr5[0] == 1).astype(np.float32)
        cold = (arr5[1] == 1).astype(np.float32)
        occl = (arr5[3] == 1).astype(np.float32)
        y = (arr5[2] == 1).astype(np.int64)  # stationary
        x = np.concatenate([
            gsm[:, :ORIG_H, :ORIG_W],
            junc[:ORIG_H, :ORIG_W][None, ...],
            warm[:ORIG_H, :ORIG_W][None, ...],
            cold[:ORIG_H, :ORIG_W][None, ...],
            occl[:ORIG_H, :ORIG_W][None, ...],
        ], axis=0)  # (97,H,W)
        return x.astype(np.float32), y[:ORIG_H, :ORIG_W], it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, y, t_dt = self.sample_cache[idx]
        else:
            x, y, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, y, t_dt)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), str(t_dt)


class V4DatasetStage4Test(Dataset):
    """
    クラス概要（Stage4 推論用）:
      stationary（二値）推論の入力を供給する Dataset。junction/warm/cold は Stage2.5、occluded は Stage3.5 を使用。
    入出力:
      - __getitem__(idx) -> (x, y, timestr)
        x: (97,H,W) float32 = GSM(93) + Stage2.5 junction + Stage2.5 warm + Stage2.5 cold + Stage3.5 occluded
        y: (H,W) int64（常に 0）ダミー
        timestr: 時刻（文字列）
    注意:
      - Stage2.5 refined に junction が欠落する場合は 0 マスクでフォールバック（class_map では代用しない）。
      - GSM は t±6h 条件を満たすサンプルのみ採用。
    """
    def __init__(self, cache_size: int = 50, file_cache_size: int = 10):
        self.cache_size = cache_size
        self.file_cache_size = file_cache_size
        self.sample_cache: Dict[int, Tuple[np.ndarray, pd.Timestamp]] = {}
        self.data_index: List[Dict[str, Any]] = []
        self.lat = None
        self.lon = None
        self._prepare_index()

    def _prepare_index(self):
        print_memory_usage("Before V4DatasetStage4Test prepare_index")
        junc_files = _scan_nc_with_time(stage2_5_out_dir)
        warmcold_files = _scan_nc_with_time(stage2_5_out_dir)
        occl_files = _scan_nc_with_time(stage3_5_out_dir)
        jmap = {t.strftime("%Y%m%d%H%M"): p for p, t in junc_files}
        wcmap = {t.strftime("%Y%m%d%H%M"): p for p, t in warmcold_files}
        omap = {t.strftime("%Y%m%d%H%M"): p for p, t in occl_files}
        common_keys = sorted(set(jmap.keys()) & set(wcmap.keys()) & set(omap.keys()))
        for key in common_keys:
            t_dt = pd.to_datetime(key, format="%Y%m%d%H%M")
            month = t_dt.strftime("%Y%m")
            gsm_file = os.path.join(nc_gsm_dir, f"gsm{month}.nc")
            if not os.path.exists(gsm_file):
                continue
            ds_gsm = xr.open_dataset(gsm_file)
            if self.lat is None or self.lon is None:
                self.lat = ds_gsm["lat"].values[:ORIG_H]
                self.lon = ds_gsm["lon"].values[:ORIG_W]
            t_np = np.datetime64(t_dt)
            ok = _ensure_3times_exist(ds_gsm, t_np)
            ds_gsm.close()
            if ok is None:
                continue
            t_prev, t_next = ok
            self.data_index.append({
                "gsm_file": gsm_file,
                "junc_file": jmap[key],
                "wc_file": wcmap[key],
                "occ_file": omap[key],
                "t": t_np, "t_prev": t_prev, "t_next": t_next, "t_dt": t_dt
            })
        print_memory_usage("After V4DatasetStage4Test prepare_index")
        print(f"[V4S4Test] samples: {len(self.data_index)}")

    def __len__(self): return len(self.data_index)

    def _read_junction_mask(self, path: str) -> np.ndarray:
        ds = xr.open_dataset(path)
        try:
            if "junction" in ds:
                j = ds["junction"]
                arr = j.isel(time=0).values if "time" in j.dims else j.values
                arr = np.asarray(arr)
                arr = np.squeeze(arr).astype(np.float32)
                if arr.max() > 1.0:
                    arr = (arr > 0).astype(np.float32)
            else:
                # 安全なフォールバック: 全0（サイズは lat/lon があればそれに合わせる）
                h = int(ds.sizes.get("lat", ORIG_H))
                w = int(ds.sizes.get("lon", ORIG_W))
                arr = np.zeros((h, w), dtype=np.float32)

            # 2D正規化 + 最終サイズを (ORIG_H, ORIG_W) に揃える（足りない部分は0で埋め、余剰は切り詰め）
            if arr.ndim != 2:
                h = int(ds.sizes.get("lat", ORIG_H))
                w = int(ds.sizes.get("lon", ORIG_W))
                arr2 = np.zeros((h, w), dtype=np.float32)
                try:
                    if arr.ndim == 1:
                        lim = min(arr.size, h * w)
                        arr2.ravel()[:lim] = arr.ravel()[:lim]
                    else:
                        hh = min(h, arr.shape[0]); ww = min(w, arr.shape[1])
                        arr2[:hh, :ww] = arr[:hh, :ww]
                except Exception:
                    pass
                arr = arr2

            # 最終整形: (ORIG_H, ORIG_W)
            out = np.zeros((ORIG_H, ORIG_W), dtype=np.float32)
            hh = min(ORIG_H, arr.shape[0]); ww = min(ORIG_W, arr.shape[1])
            out[:hh, :ww] = arr[:hh, :ww]
            return out
        finally:
            ds.close()

    def _read_warm_cold(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        ds = xr.open_dataset(path)
        try:
            if "class_map" in ds:
                cm = ds["class_map"]
                arr = cm.isel(time=0).values if "time" in cm.dims else cm.values
                arr = np.asarray(arr)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim != 2:
                    arr = np.squeeze(arr)
                warm = (arr == 1).astype(np.float32)
                cold = (arr == 2).astype(np.float32)
            else:
                wv = ds["warm"] if "warm" in ds else None
                cv = ds["cold"] if "cold" in ds else None
                if wv is not None and cv is not None:
                    w_arr = wv.isel(time=0).values if "time" in wv.dims else wv.values
                    c_arr = cv.isel(time=0).values if "time" in cv.dims else cv.values
                    w_arr = np.asarray(w_arr); c_arr = np.asarray(c_arr)
                    w_arr = np.squeeze(w_arr);  c_arr = np.squeeze(c_arr)
                    warm = (w_arr > 0.5).astype(np.float32)
                    cold = (c_arr > 0.5).astype(np.float32)
                else:
                    h = int(ds.sizes.get("lat", ORIG_H))
                    w = int(ds.sizes.get("lon", ORIG_W))
                    warm = np.zeros((h, w), dtype=np.float32)
                    cold = np.zeros((h, w), dtype=np.float32)

            # 2D正規化
            if warm.ndim != 2:
                warm = np.squeeze(warm)
            if cold.ndim != 2:
                cold = np.squeeze(cold)

            # 最終整形: (ORIG_H, ORIG_W)
            out_w = np.zeros((ORIG_H, ORIG_W), dtype=np.float32)
            out_c = np.zeros((ORIG_H, ORIG_W), dtype=np.float32)
            hh = min(ORIG_H, warm.shape[0]); ww = min(ORIG_W, warm.shape[1])
            out_w[:hh, :ww] = warm[:hh, :ww]
            hh = min(ORIG_H, cold.shape[0]); ww = min(ORIG_W, cold.shape[1])
            out_c[:hh, :ww] = cold[:hh, :ww]
            return out_w, out_c
        finally:
            ds.close()

    def _read_occluded(self, path: str) -> np.ndarray:
        """
        Stage3.5 の出力などから閉塞(occluded)の 0/1 マスクを読み出す。
        - time 次元があれば time=0 を選択して squeeze。
        - まれに 1次元（フラット）になっている場合は lat/lon の次元サイズから (H,W) へ復元を試みる。
          復元できない場合は安全にゼロ配列へフォールバックする。
        """
        ds = xr.open_dataset(path)
        try:
            if "occluded" in ds:
                v = ds["occluded"]
                arr = v.isel(time=0).values if "time" in v.dims else v.values
                arr = (arr > 0.5).astype(np.float32)
            elif "class_map" in ds:
                cm = ds["class_map"]
                arr = cm.isel(time=0).values if "time" in cm.dims else cm.values
                arr = (np.asarray(arr) > 0).astype(np.float32)
            else:
                var = list(ds.data_vars)[0]
                v = ds[var]
                arr = v.isel(time=0).values if "time" in v.dims else v.values
                arr = (arr > 0.5).astype(np.float32)

            arr = np.asarray(arr)
            arr = np.squeeze(arr)

            # 1次元の場合は lat/lon 次元から復元を試みる
            if arr.ndim == 1 and ("lat" in ds.sizes) and ("lon" in ds.sizes):
                h = int(ds.sizes["lat"])
                w = int(ds.sizes["lon"])
                if h * w == arr.size:
                    arr = arr.reshape(h, w)

            # 2D正規化（失敗時はゼロ配列へ）
            if arr.ndim != 2:
                h = int(ds.sizes.get("lat", ORIG_H))
                w = int(ds.sizes.get("lon", ORIG_W))
                arr2 = np.zeros((h, w), dtype=np.float32)
                try:
                    if arr.ndim == 1 and arr.size > 0:
                        lim = min(arr.size, h * w)
                        arr2.ravel()[:lim] = arr.ravel()[:lim]
                    elif arr.ndim >= 2:
                        hh = min(h, arr.shape[0]); ww = min(w, arr.shape[1])
                        arr2[:hh, :ww] = arr[:hh, :ww]
                except Exception:
                    pass
                arr = arr2

            # 最終整形: (ORIG_H, ORIG_W) へパディング/クリップ
            out = np.zeros((ORIG_H, ORIG_W), dtype=np.float32)
            hh = min(ORIG_H, arr.shape[0]); ww = min(ORIG_W, arr.shape[1])
            out[:hh, :ww] = arr[:hh, :ww]
            return out
        finally:
            ds.close()

    def _load_item(self, idx: int):
        it = self.data_index[idx]
        ds_gsm = xr.open_dataset(it["gsm_file"])
        gsm = _read_gsm_93(ds_gsm, it["t_prev"], it["t"], it["t_next"])
        ds_gsm.close()
        junc = self._read_junction_mask(it["junc_file"])
        warm, cold = self._read_warm_cold(it["wc_file"])
        occl = self._read_occluded(it["occ_file"])
        x = np.concatenate([
            gsm[:, :ORIG_H, :ORIG_W],
            junc[:ORIG_H, :ORIG_W][None, ...],
            warm[:ORIG_H, :ORIG_W][None, ...],
            cold[:ORIG_H, :ORIG_W][None, ...],
            occl[:ORIG_H, :ORIG_W][None, ...],
        ], axis=0)  # (97,H,W)
        return x.astype(np.float32), it["t_dt"]

    def __getitem__(self, idx: int):
        if idx in self.sample_cache:
            x, t_dt = self.sample_cache[idx]
        else:
            x, t_dt = self._load_item(idx)
            if len(self.sample_cache) >= self.cache_size:
                old = next(iter(self.sample_cache.keys()))
                del self.sample_cache[old]
            self.sample_cache[idx] = (x, t_dt)
        y = torch.zeros(ORIG_H, ORIG_W, dtype=torch.long)
        return torch.from_numpy(x).float(), y, str(t_dt)


__all__ = [
    "get_available_months",
    "V4DatasetStage1Train",
    "V4DatasetStage1Test",
    "V4DatasetStage2Train",
    "V4DatasetStage2Test",
    "V4DatasetStage3Train",
    "V4DatasetStage3Test",
    "V4DatasetStage4Train",
    "V4DatasetStage4Test",
]
