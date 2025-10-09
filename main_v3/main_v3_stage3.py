"""
概要:
    Stage3（後処理・骨格抽出）のモジュール。
    - Stage2 の確率出力 (H,W,6) から「前線の中軸」を抽出し、クラス付きスケルトンを生成して保存する
    - パイプライン: any-front 強度 → Laplacian によるリッジ検出 → medial_axis で骨格化 → 最短路で連結 → クラス付与

構成:
    - evaluate_stage3_v3(stage2_nc_dir, save_nc_dir, lap_thresh): 骨格生成のメイン処理
    - run_stage3(): CFG の設定に基づき Stage3 を一括実行

注意:
    - ラプラシアン閾値 lap_thresh は小さくするほど骨格が太く残る傾向（CFG["STAGE3"]["lap_thresh"] 参照）
    - 出力は time 次元付きの NetCDF（"skeleton_YYYYMMDDHHMM.nc"）
"""
import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import (
    print_memory_usage, format_time, CFG,
    stage2_out_dir, stage3_out_dir
)


def evaluate_stage3_v3(stage2_nc_dir, save_nc_dir, lap_thresh=-0.005):
    """
    概要:
        Stage2 の確率 (H,W,6) を入力として、前線の「骨格（スケルトン）」を抽出し、
        クラス付き class_map (H,W) を time 次元付きで NetCDF として保存する。

    入力:
        - stage2_nc_dir (str): Stage2 の確率 .nc ファイルが格納されたディレクトリ
                               期待形式: {"probabilities": (time=1, lat, lon, class=6)}
        - save_nc_dir (str): 出力先ディレクトリ（存在しなければ作成）
        - lap_thresh (float): ラプラシアンによるリッジ検出の閾値（小さいほど線が太く残る）

    処理:
        1) any-front 強度場の生成:
           - 各画素でクラス1..5の最大確率を取り、かつ argmax が 0(none) の場所は 0 に抑制
        2) リッジ検出:
           - 離散ラプラシアン（4近傍）を適用し、lap <= lap_thresh を前線候補マスクとする
        3) 骨格化:
           - skimage.morphology.medial_axis でマスクを1画素幅の骨格にする
        4) 連結補完:
           - 連結成分ごとに端点を抽出し、skimage.graph.route_through_array により
             any-front 反転のコスト場上で最短路を通して連結
        5) クラス付与:
           - 生成したパス上の各点に対し、Stage2 の argmax クラス（1..5）を割り当て
             背景に落ちた場合は近傍多数決で補完

    出力:
        なし（副作用として save_nc_dir に "skeleton_YYYYMMDDHHMM.nc" を保存）
    """
    print_memory_usage("Before evaluate_stage3_v3")
    t0 = time.time()
    os.makedirs(save_nc_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(stage2_nc_dir) if f.endswith(".nc")])

    import xarray as xr
    import cv2
    from skimage.morphology import medial_axis
    from skimage.measure import label, regionprops
    from skimage.graph import route_through_array

    def compute_any_front(prob):  # prob: (H,W,6)
        """
        概要:
            クラス1..5の最大確率を any-front 強度として算出し、none(0) が最尤な画素は 0 に抑制する。

        入力:
            - prob (np.ndarray): 形状 (H, W, 6) のクラス確率

        出力:
            - any_front (np.ndarray): 形状 (H, W) の any-front 強度（0..1）
        """
        any_front = prob[..., 1:6].max(axis=-1)  # exclude class-0 (none)
        # zero-out where class-0 is dominant
        cls = np.argmax(prob, axis=-1)
        any_front[cls == 0] = 0.0
        return any_front

    def laplacian2d(arr):
        """
        概要:
            2次元離散ラプラシアン（4近傍）を適用してリッジ強調画像を返す。

        入力:
            - arr (np.ndarray): 形状 (H, W) のスカラー場（float）

        出力:
            - lap (np.ndarray): 形状 (H, W) のラプラシアン応答
        """
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        return cv2.filter2D(arr.astype(np.float32), -1, k)

    for f in tqdm(files, desc="[Stage3-v3]"):
        p = os.path.join(stage2_nc_dir, f)
        ds = xr.open_dataset(p)
        prob = ds["probabilities"].isel(time=0).values  # (H,W,6)
        lat = ds["lat"].values
        lon = ds["lon"].values
        tval = ds["time"].values[0]
        ds.close()
        t_dt = pd.to_datetime(tval)
        date_str = t_dt.strftime("%Y%m%d%H%M")
        H, W, C = prob.shape

        any_front = compute_any_front(prob)
        lap = laplacian2d(any_front)
        mask = (lap <= lap_thresh).astype(np.uint8)
        skel = medial_axis(mask > 0).astype(np.uint8)

        # cost for MCP: lower along strong fronts
        cost = 1.0 - any_front
        cost = np.clip(cost, 1e-6, 1.0).astype(np.float32)

        lbl = label(skel, connectivity=2)
        out_map = np.zeros((H, W), dtype=np.int64)
        cls_map = np.argmax(prob, axis=-1)
        neigh_kernel = np.ones((3, 3), dtype=np.uint8)
        neigh_kernel[1, 1] = 0

        for region in regionprops(lbl):
            coords = region.coords  # [(y,x),...]
            if len(coords) == 0:
                continue
            comp = np.zeros((H, W), dtype=np.uint8)
            comp[tuple(coords.T)] = 1
            deg = cv2.filter2D(comp, -1, neigh_kernel, borderType=cv2.BORDER_CONSTANT)
            endpoints = np.argwhere((comp == 1) & (deg <= 1))
            if len(endpoints) < 2:
                pts = coords
                if len(pts) < 2:
                    y, x = pts[0]
                    out_map[y, x] = cls_map[y, x] if cls_map[y, x] >= 1 else 0
                    continue
                pts_arr = pts.astype(np.int32)
                dmax = -1
                s = pts_arr[0]
                e = pts_arr[-1]
                for i in range(len(pts_arr)):
                    for j in range(i + 1, len(pts_arr)):
                        di = np.linalg.norm(pts_arr[i] - pts_arr[j])
                        if di > dmax:
                            dmax = di
                            s = pts_arr[i]
                            e = pts_arr[j]
                endpoints = np.array([s, e])
            if len(endpoints) > 2:
                dmax = -1
                best = (endpoints[0], endpoints[1])
                for i in range(len(endpoints)):
                    for j in range(i + 1, len(endpoints)):
                        di = np.linalg.norm(endpoints[i] - endpoints[j])
                        if di > dmax:
                            dmax = di
                            best = (endpoints[i], endpoints[j])
                endpoints = np.array(best)

            start = tuple(endpoints[0])
            end = tuple(endpoints[1])
            try:
                path, _ = route_through_array(cost, start, end, fully_connected=True, geometric=True)
            except Exception:
                path = [tuple(p) for p in coords]

            for (yy, xx) in path:
                yy = int(np.clip(yy, 0, H - 1))
                xx = int(np.clip(xx, 0, W - 1))
                c = cls_map[yy, xx]
                if c >= 1:
                    out_map[yy, xx] = c
                else:
                    nb = cls_map[max(0, yy - 1):min(H, yy + 2), max(0, xx - 1):min(W, xx + 2)]
                    nb = nb[nb >= 1]
                    if nb.size > 0:
                        out_map[yy, xx] = int(np.bincount(nb).argmax())
                    else:
                        out_map[yy, xx] = 0

        da = xr.DataArray(out_map, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        ds_out = xr.Dataset({"class_map": da}).expand_dims("time")
        ds_out["time"] = [t_dt]
        outp = os.path.join(save_nc_dir, f"skeleton_{date_str}.nc")
        ds_out.to_netcdf(outp, engine="netcdf4")

    print(f"[Stage3-v3] done in {format_time(time.time() - t0)}")
    print_memory_usage("After evaluate_stage3_v3")


def run_stage3():
    """
    概要:
        Stage3 の後処理パイプラインを実行するエントリ関数。

    入力:
        なし（main_v3_config.CFG と出力ディレクトリに依存）

    処理:
        - Stage2 出力（確率 .nc）を入力に evaluate_stage3_v3 を呼び出す
        - CUDA キャッシュ解放とガーベジコレクションを実施

    出力:
        なし（副作用として skeleton_*.nc を保存）
    """
    print_memory_usage("Start Stage 3")
    evaluate_stage3_v3(stage2_nc_dir=stage2_out_dir, save_nc_dir=stage3_out_dir, lap_thresh=CFG["STAGE3"]["lap_thresh"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("After Stage 3")


# torch import used in run_stage3 cleanup
import torch  # noqa: E402

__all__ = ["evaluate_stage3_v3", "run_stage3"]
