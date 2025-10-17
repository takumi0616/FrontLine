"""
概要:
    Stage3（閉塞=4、二値: none/occluded）の学習・評価・推論モジュール。
    - 学習: 入力 = GSM(93) + GT junction(1) + GT warm(1) + GT cold(1) → (96ch), 目標 = 2クラス (0:none,1:occluded)
    - 推論: 入力 = GSM(93) + Stage2.5 junction(1) + Stage2.5 warm(1) + Stage2.5 cold(1) → (96ch)
    - 出力: probabilities(H,W,C=2) を NetCDF (time, lat, lon, class) で保存

要件との対応:
    - 「Stage3の学習では、学習期間の温暖(1), 寒冷(2)、閉塞(4)、正解の繋ぎ目=5、および気象変数データ」
    - 「Stage3の予測では、stage2.5の温暖/寒冷/繋ぎ目=5 と気象変数データ」
"""

import os
import re
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .main_v4_config import (
    CFG, device, print_memory_usage, format_time,
    model_s3_save_dir, stage3_out_dir, atomic_save_netcdf,
)
from .main_v4_datasets import (
    get_available_months,
    V4DatasetStage3Train,
    V4DatasetStage3Test,
)
from .main_v4_models import SwinUnetModel, make_combined_loss
# 可視化ユーティリティ:
# 本ステージ単体で実行した場合でも、出力確率（prob_*.nc）を可視化PNGとして保存するために使用。
from .main_v4_visualize import run_visualization_for_stage


def train_one_epoch(model, loader, optimizer, loss_fn, epoch: int, num_classes: int):
    """
    関数概要:
      Stage3（閉塞=4 の二値分類）モデルを 1 エポック分学習する。

    入力:
      - model (nn.Module): 学習対象モデル（SwinUnetModel）
      - loader (DataLoader): 学習用データローダ（V4DatasetStage3Train 由来、(x,y,timestr) を供給）
      - optimizer (torch.optim.Optimizer): 最適化手法（AdamW など）
      - loss_fn (Callable): 損失関数（CE + Dice の複合を想定）
      - epoch (int): 現在のエポック番号（0 始まり）
      - num_classes (int): クラス数（Stage3 は 2）

    処理:
      - モデルを train モードに設定
      - 各バッチ (x: (B,96,H,W), y: (B,H,W)) で forward→loss→backward→optimizer.step を実行
      - 10 バッチごとに進捗（移動平均的な loss）を tqdm に表示
      - クラスごとのピクセル精度（pred==y の割合）を集計
      - 勾配爆発対策として clip_grad_norm_ を適用

    出力:
      - float: エポック全体の平均損失
    """
    print_memory_usage(f"Before Stage3 train epoch={epoch+1}")
    model.train()
    running = 0.0
    total = 0.0
    nb = 0
    correct = [0] * num_classes
    total_pix = [0] * num_classes

    pbar = tqdm(loader, desc=f"[V4-Stage3][Train {epoch+1}]")
    for x, y, _ in pbar:
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lv = float(loss.detach().item())
        running += lv; total += lv; nb += 1
        if nb % 10 == 0:
            pbar.set_postfix({"Loss": f"{(running/10):.4f}"})
            running = 0.0

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            for c in range(num_classes):
                correct[c] += ((pred == c) & (y == c)).sum().item()
                total_pix[c] += (y == c).sum().item()

    avg = total / max(1, nb)
    print_memory_usage(f"After Stage3 train epoch={epoch+1}")
    gc.collect()
    print(f"[V4-Stage3][Train {epoch+1}] loss={avg:.4f}")
    for c in range(num_classes):
        acc = (correct[c] / total_pix[c] * 100.0) if total_pix[c] > 0 else 0.0
        print(f"  Class{c} acc: {acc:.2f}%")
    return avg


def eval_one_epoch(model, loader, loss_fn, epoch: int, num_classes: int):
    """
    関数概要:
      Stage3 モデルの 1 エポック分の検証（評価）を行う（学習は行わず、forward のみ）。

    入力:
      - model (nn.Module): 評価対象モデル
      - loader (DataLoader): 検証用データローダ（V4DatasetStage3Train の別インスタンス等）
      - loss_fn (Callable): 損失関数（CE + Dice）
      - epoch (int): 現在のエポック番号（0 始まり）
      - num_classes (int): クラス数（2）

    処理:
      - model.eval(), torch.no_grad()
      - 各バッチについて損失とクラスごとのピクセル精度を集計

    出力:
      - float: 検証セットにおける平均損失
    """
    print_memory_usage(f"Before Stage3 val epoch={epoch+1}")
    model.eval()
    total = 0.0
    nb = 0
    correct = [0] * num_classes
    total_pix = [0] * num_classes

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x)
            loss = loss_fn(logits, y)
            total += float(loss.item())
            nb += 1

            pred = torch.argmax(logits, dim=1)
            for c in range(num_classes):
                correct[c] += ((pred == c) & (y == c)).sum().item()
                total_pix[c] += (y == c).sum().item()

    avg = total / max(1, nb)
    print_memory_usage(f"After Stage3 val epoch={epoch+1}")
    gc.collect()
    print(f"[V4-Stage3][Val {epoch+1}] loss={avg:.4f}")
    for c in range(num_classes):
        acc = (correct[c] / total_pix[c] * 100.0) if total_pix[c] > 0 else 0.0
        print(f"  Class{c} acc: {acc:.2f}%")
    return avg


def export_probabilities(model, loader, save_dir: str, num_classes: int):
    """
    関数概要:
      学習済み Stage3 モデルで推論を行い、各サンプルのクラス確率 (H,W,C=2) を NetCDF に保存する。

    入力:
      - model (nn.Module): 学習済みモデル
      - loader (DataLoader): 推論用データローダ（V4DatasetStage3Test を想定）
      - save_dir (str): 出力ディレクトリ（prob_YYYYMMDDHHMM.nc を保存）
      - num_classes (int): クラス数（2）

    処理:
      - model.eval() + no_grad で各バッチを推論し softmax で確率へ変換
      - 各時刻について (H,W,C) 配列を "probabilities" 変数として保存（dims=["lat","lon","class"]）
      - "time" 次元を 1 つ持つ Dataset として保存

    出力:
      - 返り値なし（ファイル保存の副作用のみ）。保存先: save_dir/prob_YYYYMMDDHHMM.nc
    """
    import xarray as xr
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    lats = loader.dataset.lat
    lons = loader.dataset.lon

    with torch.no_grad():
        for x, _, times in tqdm(loader, desc="[V4-Stage3] Inference"):
            x = x.to(device).float()
            logits = model(x)
            prob = torch.softmax(logits, dim=1)  # (B,2,H,W)
            prob_np = prob.cpu().numpy()
            B = prob_np.shape[0]
            for i in range(B):
                tstr = times[i]
                arr = np.transpose(prob_np[i], (1, 2, 0))
                da = xr.DataArray(
                    arr,
                    dims=["lat", "lon", "class"],
                    coords={"lat": lats, "lon": lons, "class": np.arange(num_classes)},
                )
                ds = xr.Dataset({"probabilities": da}).expand_dims("time")
                try:
                    t_dt = pd.to_datetime(tstr)
                except Exception:
                    t_dt = pd.to_datetime(str(tstr))
                ds["time"] = [t_dt]
                out_name = os.path.join(save_dir, f"prob_{t_dt.strftime('%Y%m%d%H%M')}.nc")
                # 出力済みスキップ + アトミック書き込み（リトライ付き）
                if os.path.exists(out_name):
                    print(f"[V4-Stage3] Skip existing output: {os.path.basename(out_name)}")
                else:
                    ok = atomic_save_netcdf(ds, out_name, engine="netcdf4", retries=3, sleep_sec=0.5)
                    if not ok:
                        print(f"[V4-Stage3] Failed to save: {out_name}")
                del ds, da
            del prob, logits, prob_np
            gc.collect()
    print(f"[V4-Stage3] Probabilities saved -> {save_dir}")


def run_stage3():
    """
    関数概要:
      Stage3（閉塞=4 の二値分類）のフルパイプラインを実行する。
      データセット構築→モデル学習・検証→最良モデル保存→損失曲線保存→Stage2.5 入力での推論（NetCDF 出力）。

    入力:
      - なし（内部で CFG を参照）

    処理:
      1) 学習対象月を get_available_months で列挙し、V4DatasetStage3Train を train/val 用に構築
      2) SwinUnetModel を初期化し、AdamW + CE+Dice で所定エポック学習
      3) 最良モデルを保存（model_final.pth）, 損失曲線 PNG/CSV を保存
      4) V4DatasetStage3Test（GSM + Stage2.5 junction + warm + cold）で推論し、prob_*.nc を保存

    出力:
      - 返り値なし（ログ・モデル・図表・NetCDF のファイル出力）
    """
    print_memory_usage("Start V4 Stage3")
    t0 = time.time()

    # months
    y1, m1, y2, m2 = CFG["STAGE3"]["train_months"]
    months = get_available_months(y1, m1, y2, m2)

    # datasets
    ds_tr = V4DatasetStage3Train(months,
                                 cache_size=CFG["STAGE3"].get("dataset_cache_size", 50),
                                 file_cache_size=CFG["STAGE3"].get("file_cache_size", 10))
    ds_va = V4DatasetStage3Train(months,
                                 cache_size=CFG["STAGE3"].get("dataset_cache_size", 50),
                                 file_cache_size=CFG["STAGE3"].get("file_cache_size", 10))
    ld_tr = DataLoader(ds_tr,
                       batch_size=CFG["STAGE3"]["dataloader"]["batch_size_train"],
                       shuffle=True,
                       num_workers=CFG["STAGE3"]["dataloader"]["num_workers"])
    ld_va = DataLoader(ds_va,
                       batch_size=CFG["STAGE3"]["dataloader"]["batch_size_val"],
                       shuffle=False,
                       num_workers=CFG["STAGE3"]["dataloader"]["num_workers"])

    # model
    model = SwinUnetModel(
        in_chans=CFG["STAGE3"]["in_chans"],
        num_classes=CFG["STAGE3"]["num_classes"],
        model_cfg=CFG["STAGE3"]["model"],
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["STAGE3"]["optimizer"]["lr"],
        weight_decay=CFG["STAGE3"]["optimizer"]["weight_decay"],
        betas=tuple(CFG["STAGE3"]["optimizer"].get("betas", (0.9, 0.999))),
    )
    loss_fn = make_combined_loss(num_classes=CFG["STAGE3"]["num_classes"])

    # resume
    os.makedirs(model_s3_save_dir, exist_ok=True)
    start_epoch = 0
    ckpts = [f for f in os.listdir(model_s3_save_dir) if f.startswith("checkpoint_epoch_")]
    if ckpts:
        ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        last = ckpts[-1]
        obj = torch.load(os.path.join(model_s3_save_dir, last), map_location="cpu")
        state = obj["model_state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        opt.load_state_dict(obj["optimizer_state_dict"])
        start_epoch = int(obj["epoch"]) + 1
        print(f"[V4-Stage3] Resume from {last} (epoch {start_epoch})")
    else:
        print("[V4-Stage3] Train from scratch")

    # loop
    E = CFG["STAGE3"]["epochs"]
    best_val = float("inf")
    best_state = None
    tr_hist, va_hist = [], []
    best_ep = -1

    for ep in range(start_epoch, E):
        t_ep = time.time()
        tr = train_one_epoch(model, ld_tr, opt, loss_fn, ep, CFG["STAGE3"]["num_classes"])
        va = eval_one_epoch(model, ld_va, loss_fn, ep, CFG["STAGE3"]["num_classes"])
        tr_hist.append(tr); va_hist.append(va)
        if va < best_val:
            best_val = va
            try:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            except Exception:
                best_state = model.state_dict()
            best_ep = ep
            print(f"[V4-Stage3] New best at epoch {ep+1}: val_loss={va:.4f}")

        ck = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "train_loss": tr,
            "val_loss": va,
        }
        torch.save(ck, os.path.join(model_s3_save_dir, f"checkpoint_epoch_{ep}.pth"))
        if ep > 0:
            prev = os.path.join(model_s3_save_dir, f"checkpoint_epoch_{ep-1}.pth")
            if os.path.exists(prev):
                try:
                    os.remove(prev)
                except Exception:
                    pass
        print(f"[V4-Stage3] Epoch {ep+1} done in {format_time(time.time() - t_ep)}")

    final_path = os.path.join(model_s3_save_dir, "model_final.pth")
    if best_state is not None:
        torch.save(best_state, final_path)
        model.load_state_dict(best_state)
        print(f"[V4-Stage3] Saved best model (epoch {best_ep+1}) -> model_final.pth")
    else:
        torch.save(model.state_dict(), final_path)
        print("[V4-Stage3] Saved final model")

    try:
        plt.figure(figsize=(9, 5))
        xs = list(range(start_epoch + 1, start_epoch + len(tr_hist) + 1))
        plt.plot(xs, tr_hist, label="train_loss")
        plt.plot(xs, va_hist, label="val_loss")
        if best_ep >= 0:
            plt.axvline(x=best_ep + 1, color="g", linestyle="--", label=f"Best ({best_ep+1})")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(model_s3_save_dir, "loss_curve.png"), dpi=150); plt.close()
        pd.DataFrame({"epoch": xs, "train_loss": tr_hist, "val_loss": va_hist}).to_csv(
            os.path.join(model_s3_save_dir, "loss_history.csv"), index=False
        )
    except Exception as e:
        print(f"[V4-Stage3] Save curves failed: {e}")

    # inference using Stage2.5
    ds_te = V4DatasetStage3Test(
        cache_size=CFG["STAGE3"].get("dataset_cache_size", 50),
        file_cache_size=CFG["STAGE3"].get("file_cache_size", 10),
    )
    ld_te = DataLoader(ds_te,
                       batch_size=CFG["STAGE3"]["dataloader"]["batch_size_test"],
                       shuffle=False,
                       num_workers=CFG["STAGE3"]["dataloader"]["num_workers"])
    print(f"[V4-Stage3] Test samples (inference): {len(ds_te)}")
    export_probabilities(model, ld_te, save_dir=stage3_out_dir, num_classes=CFG["STAGE3"]["num_classes"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("After V4 Stage3")
    print(f"[V4-Stage3] done in {format_time(time.time() - t0)}")
    # 追加: Stage3 の出力確率を可視化（output_visual_dir/stage3 配下に保存）
    # main_v4.py でも可視化を呼ぶが、本モジュール単体実行時の利便性向上のためここでも実施する。
    run_visualization_for_stage("stage3")


__all__ = ["run_stage3"]
