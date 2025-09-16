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
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from main_v3_config import (
    CFG, device, print_memory_usage, format_time,
    nc_0p5_dir, stage1_out_dir, stage2_out_dir, model_s2_save_dir
)
from main_v3_datasets import FrontalRefinementDataset
from main_v3_models import SwinUnetModelStage2, combined_loss
from main_v3_utils import get_available_months


def train_stage2_one_epoch(model, dataloader, optimizer, epoch, num_classes):
    print_memory_usage(f"Before Stage2 train epoch={epoch+1}")
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = [0] * num_classes
    total = [0] * num_classes
    batch_count = 0

    pbar = tqdm(dataloader, desc=f"[Stage2][Train Epoch {epoch+1}]")
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        lv = float(loss.item())
        running_loss += lv
        total_loss += lv
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
    print_memory_usage(f"After Stage2 train epoch={epoch+1}")
    gc.collect()

    print(f"\n[Stage2][Train Epoch {epoch+1}] Loss: {avg_epoch_loss:.4f}")
    print(f"[Stage2][Train Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i] / total[i] * 100) if total[i] > 0 else 0
        print(f"  Class {i}: {acc:.2f} %")

    return avg_epoch_loss


def test_stage2_one_epoch(model, dataloader, epoch, num_classes):
    print_memory_usage(f"Before Stage2 test epoch={epoch+1}")
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

    print_memory_usage(f"After Stage2 test epoch={epoch+1}")
    gc.collect()

    avg_loss = test_loss / max(1, batch_count)
    print(f"\n[Stage2][Test Epoch {epoch+1}] Loss: {avg_loss:.4f}")
    print(f"[Stage2][Test Epoch {epoch+1}] Accuracy by class:")
    for i in range(num_classes):
        acc = (correct[i] / total[i] * 100) if total[i] > 0 else 0
        print(f"  Class {i}: {acc:.2f} %")

    return avg_loss


def evaluate_stage2(model, dataloader, save_nc_dir=None):
    print_memory_usage("Before evaluate_stage2")
    evaluate_start = time.time()
    model.eval()
    all_probs = []
    all_outputs = []
    all_targets = []
    all_times = []

    with torch.no_grad():
        for inputs, targets, times in tqdm(dataloader, desc="[Stage2] Evaluate"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            all_probs.append(prob.cpu())
            all_outputs.append(pred_cls.cpu())
            all_targets.append(targets.cpu())
            all_times.extend(times)

    all_probs = torch.cat(all_probs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Metrics when GT exists (train/val). In test mode targets likely zeros.
    try:
        pred_flat = all_outputs.view(-1).numpy()
        targ_flat = all_targets.view(-1).numpy()
        if targ_flat.sum() > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targ_flat, pred_flat, labels=list(range(CFG["STAGE2"]["num_classes"])), average=None, zero_division=0
            )
            accuracy = (pred_flat == targ_flat).sum() / len(targ_flat) * 100.0
            print(f"\n[Stage2] Pixel Accuracy (all classes): {accuracy:.2f}%")
            for i, cls_id in enumerate(list(range(CFG["STAGE2"]["num_classes"]))):
                print(f"  Class{cls_id}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

            cm = confusion_matrix(targ_flat, pred_flat, labels=list(range(CFG["STAGE2"]["num_classes"])))
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=list(range(CFG["STAGE2"]["num_classes"])),
                yticklabels=list(range(CFG["STAGE2"]["num_classes"])),
            )
            plt.title("[Stage2] Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig("stage2_confusion_matrix.png")
            plt.close()
    except Exception as e:
        print(f"[Stage2] metrics failed: {e}")

    if save_nc_dir is not None:
        import xarray as xr
        os.makedirs(save_nc_dir, exist_ok=True)
        lat = dataloader.dataset.lat
        lon = dataloader.dataset.lon
        for i in range(all_probs.shape[0]):
            time_str = all_times[i]
            probs_np = all_probs[i].numpy()
            probs_np = np.transpose(probs_np, (1, 2, 0))
            da = xr.DataArray(
                probs_np,
                dims=["lat", "lon", "class"],
                coords={"lat": lat, "lon": lon, "class": np.arange(CFG["STAGE2"]["num_classes"])},
            )
            ds = xr.Dataset({"probabilities": da})
            ds = ds.expand_dims("time")
            ds["time"] = [pd.to_datetime(time_str)]
            date_str = pd.to_datetime(time_str).strftime("%Y%m%d%H%M")
            ds.to_netcdf(os.path.join(save_nc_dir, f"refined_{date_str}.nc"), engine="netcdf4")
            del ds, da, probs_np
            gc.collect()
        print(f"[Stage2] Refined probabilities saved to {save_nc_dir}")

    evaluate_end = time.time()
    print(f"[Stage2] 評価全体の実行時間: {format_time(evaluate_end - evaluate_start)}")
    print_memory_usage("After evaluate_stage2")


def run_stage2():
    print_memory_usage("Start Stage 2")
    stage2_start = time.time()

    # Build datasets
    ds_start = time.time()
    y1, m1, y2, m2 = CFG["STAGE2"]["train_months"]
    train_months = get_available_months(y1, m1, y2, m2)
    aug = CFG["STAGE2"]["augment"]
    train_dataset_s2 = FrontalRefinementDataset(
        months=train_months,
        nc_0p5_dir=nc_0p5_dir,
        mode="train",
        stage1_out_dir=None,
        n_augment=aug["n_augment"],
        prob_dilation=aug["prob_dilation"],
        prob_create_gaps=aug["prob_create_gaps"],
        prob_random_pixel_change=aug["prob_random_pixel_change"],
        prob_add_fake_front=aug["prob_add_fake_front"],
        dilation_kernel_range=aug["dilation_kernel_range"],
        num_gaps_range=aug["num_gaps_range"],
        gap_size_range=aug["gap_size_range"],
        num_pix_to_change_range=aug["num_pix_to_change_range"],
        num_fake_front_range=aug["num_fake_front_range"],
    )
    val_dataset_s2 = FrontalRefinementDataset(
        months=train_months,
        nc_0p5_dir=nc_0p5_dir,
        mode="val",
        stage1_out_dir=None,
        n_augment=aug["n_augment"],
        prob_dilation=aug["prob_dilation"],
        prob_create_gaps=aug["prob_create_gaps"],
        prob_random_pixel_change=aug["prob_random_pixel_change"],
        prob_add_fake_front=aug["prob_add_fake_front"],
        dilation_kernel_range=aug["dilation_kernel_range"],
        num_gaps_range=aug["num_gaps_range"],
        gap_size_range=aug["gap_size_range"],
        num_pix_to_change_range=aug["num_pix_to_change_range"],
        num_fake_front_range=aug["num_fake_front_range"],
    )
    train_loader_s2 = DataLoader(
        train_dataset_s2,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_train"],
        shuffle=True,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
    )
    val_loader_s2 = DataLoader(
        val_dataset_s2,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_val"],
        shuffle=False,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
    )
    ds_end = time.time()
    print(f"[Stage2] データセット準備時間: {format_time(ds_end - ds_start)}")
    print(f"[Stage2] Train dataset size: {len(train_dataset_s2)}")
    print(f"[Stage2] Val   dataset size: {len(val_dataset_s2)}")

    # Model
    import torch.optim as optim

    model_init_start = time.time()
    model_s2 = SwinUnetModelStage2(
        num_classes=CFG["STAGE2"]["num_classes"],
        in_chans=CFG["STAGE2"]["in_chans"],
        model_cfg=CFG["STAGE2"]["model"],
    ).to(device)
    optimizer_s2 = optim.AdamW(
        model_s2.parameters(),
        lr=CFG["STAGE2"]["optimizer"]["lr"],
        weight_decay=CFG["STAGE2"]["optimizer"]["weight_decay"],
    )
    model_init_end = time.time()
    print(f"[Stage2] モデル初期化時間: {format_time(model_init_end - model_init_start)}")

    # Resume
    num_epochs_stage2 = CFG["STAGE2"]["epochs"]
    ckpt_start = time.time()
    start_epoch = 0
    os.makedirs(model_s2_save_dir, exist_ok=True)
    checkpoint_files = [f for f in os.listdir(model_s2_save_dir) if f.startswith("checkpoint_epoch_")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(model_s2_save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_s2.load_state_dict(state_dict)
        optimizer_s2.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[Stage2] 既存のチェックポイント {latest_checkpoint} から学習を再開します（エポック {start_epoch} から）")
    else:
        print("[Stage2] 新規に学習を開始します")
    ckpt_end = time.time()
    print(f"[Stage2] チェックポイント読み込み時間: {format_time(ckpt_end - ckpt_start)}")

    # Train/Val
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs_stage2):
        epoch_start = time.time()
        train_loss = train_stage2_one_epoch(
            model_s2, train_loader_s2, optimizer_s2, epoch, CFG["STAGE2"]["num_classes"]
        )
        val_loss = test_stage2_one_epoch(model_s2, val_loader_s2, epoch, CFG["STAGE2"]["num_classes"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_s2.state_dict()
            best_epoch = epoch
            print(f"[Stage2] 新しい最良モデルを見つけました（エポック {epoch+1}）: 検証損失 = {val_loss:.4f}")

        epoch_end = time.time()
        print(f"[Stage2] エポック {epoch+1} 実行時間: {format_time(epoch_end - epoch_start)}")

        # Save ckpt
        save_start = time.time()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_s2.state_dict(),
            "optimizer_state_dict": optimizer_s2.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        checkpoint_path = os.path.join(model_s2_save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"[Stage2] チェックポイントを保存しました: checkpoint_epoch_{epoch}.pth")
        if epoch > 0:
            previous_checkpoint_path = os.path.join(model_s2_save_dir, f"checkpoint_epoch_{epoch - 1}.pth")
            if os.path.exists(previous_checkpoint_path):
                try:
                    os.remove(previous_checkpoint_path)
                    print(f"[Stage2] 前回のチェックポイントを削除しました: checkpoint_epoch_{epoch - 1}.pth")
                except Exception as e:
                    print(f"[Stage2] 旧チェックポイント削除失敗: {e}")
        save_end = time.time()
        print(f"[Stage2] チェックポイント保存時間: {format_time(save_end - save_start)}")

    training_end = time.time()
    print(f"[Stage2] 学習ループ全体の実行時間: {format_time(training_end - training_start)}")

    # Save best/final
    final_save_start = time.time()
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(model_s2_save_dir, "model_final.pth"))
        print(f"[Stage2] 最良モデル（エポック {best_epoch+1}）を model_final.pth として保存しました")
    else:
        torch.save(model_s2.state_dict(), os.path.join(model_s2_save_dir, "model_final.pth"))
        print(f"[Stage2] 最終的なモデルを保存しました: model_final.pth")

    # Loss curves
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs = list(range(start_epoch + 1, start_epoch + len(train_losses) + 1))
        plt.plot(epochs, train_losses, "b-", label="Train Loss")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss")
        if best_epoch >= 0:
            plt.axvline(x=best_epoch + 1, color="g", linestyle="--", label=f"Best Epoch ({best_epoch+1})")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Stage2 Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        min_loss = min(min(train_losses), min(val_losses))
        max_loss = max(max(train_losses), max(val_losses))
        margin = (max_loss - min_loss) * 0.1
        plt.ylim([max(0, min_loss - margin), max_loss + margin])

        loss_curve_path = os.path.join(model_s2_save_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[Stage2] Loss曲線を保存しました: {loss_curve_path}")

        # CSV logs
        try:
            epochs_col = epochs
        except NameError:
            epochs_col = list(range(1, len(train_losses) + 1))
        df_loss_s2 = pd.DataFrame({"epoch": epochs_col, "train_loss": train_losses, "val_loss": val_losses})
        csv_path_s2 = os.path.join(model_s2_save_dir, "loss_history.csv")
        df_loss_s2.to_csv(csv_path_s2, index=False)
        try:
            root_csv_s2 = os.path.join(os.path.dirname(model_s2_save_dir), "loss_history_stage2.csv")
            df_loss_s2.to_csv(root_csv_s2, index=False)
        except Exception as e:
            print(f"[Stage2] Loss history CSV copy skipped: {e}")

    final_save_end = time.time()
    print(f"[Stage2] 最終モデル保存時間: {format_time(final_save_end - final_save_start)}")

    # Evaluate on Stage1 outputs
    eval_start = time.time()
    test_dataset_s2 = FrontalRefinementDataset(
        months=None,
        nc_0p5_dir=nc_0p5_dir,
        mode="test",
        stage1_out_dir=stage1_out_dir,
    )
    test_loader_s2 = DataLoader(
        test_dataset_s2,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_test"],
        shuffle=False,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
    )
    print(f"[Stage2] Test dataset size (Stage1結果): {len(test_dataset_s2)}")
    if best_model_state is not None:
        model_s2.load_state_dict(best_model_state)
        print(f"[Stage2] 評価のために最良モデル（エポック {best_epoch+1}）をロードしました")

    evaluate_stage2(model_s2, test_loader_s2, save_nc_dir=stage2_out_dir)
    eval_end = time.time()
    print(f"[Stage2] 評価時間: {format_time(eval_end - eval_start)}")

    # Cleanup
    cleanup_start = time.time()
    del train_dataset_s2, val_dataset_s2, train_loader_s2, val_loader_s2
    del test_dataset_s2, test_loader_s2
    del model_s2, optimizer_s2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    cleanup_end = time.time()
    print(f"[Stage2] メモリクリーンアップ時間: {format_time(cleanup_end - cleanup_start)}")

    stage2_end = time.time()
    print(f"[Stage2] 全体の実行時間: {format_time(stage2_end - stage2_start)}")
    print_memory_usage("After Stage 2")


__all__ = [
    "train_stage2_one_epoch",
    "test_stage2_one_epoch",
    "evaluate_stage2",
    "run_stage2",
]
