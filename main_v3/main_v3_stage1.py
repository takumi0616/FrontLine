import os
import re
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from main_v3_config import (
    CFG, device, print_memory_usage, format_time,
    nc_gsm_dir, nc_0p5_dir, model_s1_save_dir, stage1_out_dir
)
from main_v3_utils import get_available_months
from main_v3_datasets import FrontalDatasetStage1
from main_v3_models import SwinUnetModel, combined_loss


def train_stage1_one_epoch(model, dataloader, optimizer, epoch, num_classes):
    print_memory_usage(f"Before Stage1 train epoch={epoch+1}")
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = [0] * num_classes
    total = [0] * num_classes
    batch_count = 0

    pbar = tqdm(dataloader, desc=f"[Stage1][Train Epoch {epoch+1}]")
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        total_loss += loss_val
        batch_count += 1

        _, predicted = torch.max(outputs, dim=1)
        for i in range(num_classes):
            correct[i] += ((predicted == i) & (targets == i)).sum().item()
            total[i] += (targets == i).sum().item()

        if (batch_idx + 1) % 10 == 0:
            avg_loss_10 = running_loss / 10
            pbar.set_postfix({"Loss": f"{avg_loss_10:.4f}"})
            running_loss = 0.0

    avg_epoch_loss = total_loss / max(1, batch_count)
    print_memory_usage(f"After Stage1 train epoch={epoch+1}")
    gc.collect()

    print(f"\n[Stage1][Train Epoch {epoch+1}] Loss: {avg_epoch_loss:.4f}")
    print(f"[Stage1][Train Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i] / total[i] * 100) if total[i] > 0 else 0
        print(f"  Class {i}: {acc:.2f} %")

    return avg_epoch_loss


def test_stage1_one_epoch(model, dataloader, epoch, num_classes):
    print_memory_usage(f"Before Stage1 test epoch={epoch+1}")
    model.eval()
    test_loss = 0.0
    correct = [0] * num_classes
    total = [0] * num_classes
    batch_count = 0

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            test_loss += float(loss.item())
            batch_count += 1

            _, predicted = torch.max(outputs, dim=1)
            for i in range(num_classes):
                correct[i] += ((predicted == i) & (targets == i)).sum().item()
                total[i] += (targets == i).sum().item()

    print_memory_usage(f"After Stage1 test epoch={epoch+1}")
    gc.collect()

    avg_loss = test_loss / max(1, batch_count)
    print(f"\n[Stage1][Test Epoch {epoch+1}] Loss: {avg_loss:.4f}")
    print(f"[Stage1][Test Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i] / total[i] * 100) if total[i] > 0 else 0
        print(f"  Class {i}: {acc:.2f} %")

    return avg_loss


def evaluate_stage1(model, dataloader, save_nc_dir=None):
    print_memory_usage("Before evaluate_stage1")
    evaluate_start = time.time()

    model.eval()
    all_probs = []
    all_outputs = []
    all_targets = []
    all_times = []

    inference_start = time.time()
    with torch.no_grad():
        for inputs, targets, times in tqdm(dataloader, desc="[Stage1] Evaluate"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            pred_cls = torch.argmax(prob, dim=1)

            all_probs.append(prob.cpu())
            all_outputs.append(pred_cls.cpu())
            all_targets.append(targets.cpu())
            all_times.extend(times)
    inference_end = time.time()
    print(f"[Stage1] 推論処理時間: {format_time(inference_end - inference_start)}")

    metrics_start = time.time()
    all_probs = torch.cat(all_probs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 旧実装と同等の基本指標を出力（Accuracy/Precision/Recall/F1 per class）
    try:
        from sklearn.metrics import precision_recall_fscore_support
        pred_flat = all_outputs.view(-1).numpy()
        targ_flat = all_targets.view(-1).numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            targ_flat,
            pred_flat,
            labels=list(range(CFG["STAGE1"]["num_classes"])),
            average=None,
            zero_division=0
        )
        accuracy = float((pred_flat == targ_flat).sum()) / max(1, len(targ_flat)) * 100.0
        print(f"\n[Stage1] Pixel Accuracy (all classes): {accuracy:.2f}%")
        for i in range(CFG["STAGE1"]["num_classes"]):
            print(f"  Class{i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    except Exception as e:
        print(f"[Stage1] metric computation failed (sklearn not available?): {e}")

    if save_nc_dir is not None:
        import xarray as xr
        os.makedirs(save_nc_dir, exist_ok=True)
        for i in range(all_probs.shape[0]):
            time_str = all_times[i]
            probs_np = all_probs[i].numpy()
            probs_np = np.transpose(probs_np, (1, 2, 0))  # (H,W,C)
            lat = dataloader.dataset.lat
            lon = dataloader.dataset.lon
            da = xr.DataArray(
                probs_np,
                dims=["lat", "lon", "class"],
                coords={"lat": lat, "lon": lon, "class": np.arange(CFG["STAGE1"]["num_classes"])},
            )
            ds = xr.Dataset({"probabilities": da})
            ds = ds.expand_dims("time")
            ds["time"] = [pd.to_datetime(time_str)]
            date_str = pd.to_datetime(time_str).strftime("%Y%m%d%H%M")
            ds.to_netcdf(os.path.join(save_nc_dir, f"prob_{date_str}.nc"), engine="netcdf4")
            del ds, da, probs_np
            gc.collect()
        print(f"[Stage1] Probabilities saved to {save_nc_dir}")

    metrics_end = time.time()
    print(f"[Stage1] 評価指標計算時間: {format_time(metrics_end - metrics_start)}")

    cleanup_start = time.time()
    del all_probs, all_outputs, all_targets
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage1] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    evaluate_end = time.time()
    print(f"[Stage1] 評価全体の実行時間: {format_time(evaluate_end - evaluate_start)}")
    print_memory_usage("After evaluate_stage1")


def run_stage1():
    print_memory_usage("Start Stage 1")
    stage1_start = time.time()

    # Dataset
    ds_start = time.time()
    y1, m1, y2, m2 = CFG["STAGE1"]["train_months"]
    train_months = get_available_months(y1, m1, y2, m2)
    y1, m1, y2, m2 = CFG["STAGE1"]["test_months"]
    test_months = get_available_months(y1, m1, y2, m2)

    train_dataset_s1 = FrontalDatasetStage1(
        train_months,
        nc_gsm_dir,
        nc_0p5_dir,
        cache_size=CFG["STAGE1"].get("dataset_cache_size", 50),           # サンプルキャッシュ上限（個）
        file_cache_size=CFG["STAGE1"].get("file_cache_size", 10),          # オープン済みNetCDFファイルのキャッシュ上限（個）
    )
    test_dataset_s1 = FrontalDatasetStage1(
        test_months,
        nc_gsm_dir,
        nc_0p5_dir,
        cache_size=CFG["STAGE1"].get("dataset_cache_size", 50),           # サンプルキャッシュ上限（個）
        file_cache_size=CFG["STAGE1"].get("file_cache_size", 10),          # オープン済みNetCDFファイルのキャッシュ上限（個）
    )
    train_loader_s1 = DataLoader(
        train_dataset_s1,
        batch_size=CFG["STAGE1"]["dataloader"]["batch_size_train"],
        shuffle=True,
        num_workers=CFG["STAGE1"]["dataloader"]["num_workers"],
    )
    test_loader_s1 = DataLoader(
        test_dataset_s1,
        batch_size=CFG["STAGE1"]["dataloader"]["batch_size_test"],
        shuffle=False,
        num_workers=CFG["STAGE1"]["dataloader"]["num_workers"],
    )
    ds_end = time.time()
    print(f"[Stage1] データセット準備時間: {format_time(ds_end - ds_start)}")
    print(f"[Stage1] Train dataset size: {len(train_dataset_s1)}")
    print(f"[Stage1] Test  dataset size: {len(test_dataset_s1)}")

    # Model and optimizer
    import torch.optim as optim

    model_init_start = time.time()
    model_s1 = SwinUnetModel(
        num_classes=CFG["STAGE1"]["num_classes"],
        in_chans=CFG["STAGE1"]["in_chans"],
        model_cfg=CFG["STAGE1"]["model"],
    ).to(device)
    optimizer_s1 = optim.AdamW(
        model_s1.parameters(),
        lr=CFG["STAGE1"]["optimizer"]["lr"],
        weight_decay=CFG["STAGE1"]["optimizer"]["weight_decay"],
        betas=tuple(CFG["STAGE1"]["optimizer"].get("betas", (0.9, 0.999)))  # AdamWのβ（慣性項）
    )
    model_init_end = time.time()
    print(f"[Stage1] モデル初期化時間: {format_time(model_init_end - model_init_start)}")

    # Resume checkpoint if exists
    num_epochs_stage1 = CFG["STAGE1"]["epochs"]
    ckpt_start = time.time()
    start_epoch = 0
    os.makedirs(model_s1_save_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(model_s1_save_dir) if f.startswith("checkpoint_epoch_")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(model_s1_save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_s1.load_state_dict(state_dict)
        optimizer_s1.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[Stage1] 既存のチェックポイント {latest_checkpoint} から学習を再開します（エポック {start_epoch} から）")
    else:
        print("[Stage1] 新規に学習を開始します")
    ckpt_end = time.time()
    print(f"[Stage1] チェックポイント読み込み時間: {format_time(ckpt_end - ckpt_start)}")

    # Train/Test loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs_stage1):
        epoch_start = time.time()
        train_loss = train_stage1_one_epoch(
            model_s1, train_loader_s1, optimizer_s1, epoch, CFG["STAGE1"]["num_classes"]
        )
        test_loss = test_stage1_one_epoch(model_s1, test_loader_s1, epoch, CFG["STAGE1"]["num_classes"])
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model_s1.state_dict()
            best_epoch = epoch
            print(f"[Stage1] 新しい最良モデルを見つけました（エポック {epoch+1}）: テスト損失 = {test_loss:.4f}")

        epoch_end = time.time()
        print(f"[Stage1] エポック {epoch+1} 実行時間: {format_time(epoch_end - epoch_start)}")

        # Save checkpoint
        save_start = time.time()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_s1.state_dict(),
            "optimizer_state_dict": optimizer_s1.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        checkpoint_path = os.path.join(model_s1_save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"[Stage1] チェックポイントを保存しました: checkpoint_epoch_{epoch}.pth")
        if epoch > 0:
            previous_checkpoint_path = os.path.join(model_s1_save_dir, f"checkpoint_epoch_{epoch - 1}.pth")
            if os.path.exists(previous_checkpoint_path):
                try:
                    os.remove(previous_checkpoint_path)
                    print(f"[Stage1] 前回のチェックポイントを削除しました: checkpoint_epoch_{epoch - 1}.pth")
                except Exception as e:
                    print(f"[Stage1] 旧チェックポイント削除失敗: {e}")
        save_end = time.time()
        print(f"[Stage1] チェックポイント保存時間: {format_time(save_end - save_start)}")

    training_end = time.time()
    print(f"[Stage1] 学習ループ全体の実行時間: {format_time(training_end - training_start)}")

    # Save final/best model
    final_save_start = time.time()
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(model_s1_save_dir, "model_final.pth"))
        print(f"[Stage1] 最良モデル（エポック {best_epoch+1}）を model_final.pth として保存しました")
    else:
        torch.save(model_s1.state_dict(), os.path.join(model_s1_save_dir, "model_final.pth"))
        print(f"[Stage1] 最終的なモデルを保存しました: model_final.pth")

    # Plot and save loss curves + CSV
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs = list(range(start_epoch + 1, start_epoch + len(train_losses) + 1))
        plt.plot(epochs, train_losses, "b-", label="Train Loss")
        plt.plot(epochs, test_losses, "r-", label="Test Loss")
        if best_epoch >= 0:
            plt.axvline(x=best_epoch + 1, color="g", linestyle="--", label=f"Best Epoch ({best_epoch+1})")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Stage1 Training and Test Loss")
        plt.legend()
        plt.grid(True)
        min_loss = min(min(train_losses), min(test_losses))
        max_loss = max(max(train_losses), max(test_losses))
        margin = (max_loss - min_loss) * 0.1
        plt.ylim([max(0, min_loss - margin), max_loss + margin])

        loss_curve_path = os.path.join(model_s1_save_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[Stage1] Loss曲線を保存しました: {loss_curve_path}")

        # CSV logs
        try:
            epochs_col = epochs
        except NameError:
            epochs_col = list(range(1, len(train_losses) + 1))
        df_loss_s1 = pd.DataFrame({"epoch": epochs_col, "train_loss": train_losses, "test_loss": test_losses})
        csv_path_s1 = os.path.join(model_s1_save_dir, "loss_history.csv")
        df_loss_s1.to_csv(csv_path_s1, index=False)
        try:
            root_csv_s1 = os.path.join(os.path.dirname(model_s1_save_dir), "loss_history_stage1.csv")
            df_loss_s1.to_csv(root_csv_s1, index=False)
        except Exception as e:
            print(f"[Stage1] Loss history CSV copy skipped: {e}")

    final_save_end = time.time()
    print(f"[Stage1] 最終モデル保存時間: {format_time(final_save_end - final_save_start)}")

    # Evaluate and export nc
    eval_start = time.time()
    if best_model_state is not None:
        model_s1.load_state_dict(best_model_state)
        print(f"[Stage1] 評価のために最良モデル（エポック {best_epoch+1}）をロードしました")

    evaluate_stage1(model_s1, test_loader_s1, save_nc_dir=stage1_out_dir)
    eval_end = time.time()
    print(f"[Stage1] 評価時間: {format_time(eval_end - eval_start)}")

    # Cleanup
    cleanup_start = time.time()
    del train_dataset_s1, test_dataset_s1, train_loader_s1, test_loader_s1
    del model_s1, optimizer_s1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage1] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    stage1_end = time.time()
    print(f"[Stage1] 全体の実行時間: {format_time(stage1_end - stage1_start)}")
    print_memory_usage("After Stage 1")


__all__ = [
    "train_stage1_one_epoch",
    "test_stage1_one_epoch",
    "evaluate_stage1",
    "run_stage1",
]
