import os
import gc
import time
from pathlib import Path
import importlib.util as _ilu

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from main_v3_config import (
    CFG, device, ORIG_H,
    print_memory_usage, format_time,
    nc_0p5_dir, stage1_out_dir, stage2_out_dir, model_s2_save_dir
)
from main_v3_datasets import FrontalRefinementDataset, Stage2DiffusionTestDataset


def _load_diffusion_corrector():
    """
    Load DiffusionCorrector class from diffusion-model.py.

    Search order:
      1) src/FrontLine/main_v3/diffusion-model.py        (same dir as this file)
      2) src/FrontLine/diffusion-model.py                (parent dir)
      3) Absolute fallback: /home/takumi/docker_miniconda/src/FrontLine/diffusion-model.py

    Raises:
      FileNotFoundError if none are found.
    """
    candidates = [
        Path(__file__).parent / "diffusion-model.py",
        Path(__file__).parent.parent / "diffusion-model.py",
        Path("/home/takumi/docker_miniconda/src/FrontLine/diffusion-model.py"),
    ]
    last_error = None
    for mod_path in candidates:
        try:
            if not mod_path.exists():
                continue
            spec = _ilu.spec_from_file_location("diffusion_corrector_mod", str(mod_path))
            if spec is None or spec.loader is None:
                continue
            module = _ilu.module_from_spec(spec)
            spec.loader.exec_module(module)
            # verify attribute exists
            if hasattr(module, "DiffusionCorrector"):
                return module.DiffusionCorrector
        except Exception as e:
            last_error = e
            continue
    raise FileNotFoundError(
        "diffusion-model.py not found or invalid. Tried:\n  - " +
        "\n  - ".join(str(p) for p in candidates) +
        (f"\nLast error: {last_error}" if last_error else "")
    )


def _load_diffusion_module():
    """
    Load diffusion-model.py as a Python module object to access both
    DiffusionCorrector and ConditionalDiffusionCorrector classes.
    """
    candidates = [
        Path(__file__).parent / "diffusion-model.py",
        Path(__file__).parent.parent / "diffusion-model.py",
        Path("/home/takumi/docker_miniconda/src/FrontLine/diffusion-model.py"),
    ]
    last_error = None
    for mod_path in candidates:
        try:
            if not mod_path.exists():
                continue
            spec = _ilu.spec_from_file_location("diffusion_corrector_mod", str(mod_path))
            if spec is None or spec.loader is None:
                continue
            module = _ilu.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "ConditionalDiffusionCorrector"):
                return module
        except Exception as e:
            last_error = e
            continue
    raise FileNotFoundError(
        "diffusion-model.py not found or invalid (ConditionalDiffusionCorrector missing). Tried:\n  - " +
        "\n  - ".join(str(p) for p in candidates) +
        (f"\nLast error: {last_error}" if last_error else "")
    )


def run_stage2_diffusion():
    print_memory_usage("Start Stage 2 (Diffusion, Paired-Conditional)")
    stage2_start = time.time()

    # -----------------------------
    # Training/Validation datasets (use v1-style degraded pairs)
    # -----------------------------
    y1, m1, y2, m2 = CFG["STAGE2"]["train_months"]
    train_months = [f"{y}{m:02d}"
                    for y in range(y1, y2 + 1)
                    for m in range(1, 13)
                    if (y > y1 or m >= m1) and (y < y2 or m <= m2)]
    aug = CFG["STAGE2"]["augment"]

    # FrontalRefinementDataset(train/val) yields (in_cls(degraded), tgt_cls(GT), time_str)
    train_ds = FrontalRefinementDataset(
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
        cache_size=CFG["STAGE2"].get("dataset_cache_size", 50),
    )
    val_ds = FrontalRefinementDataset(
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
        cache_size=CFG["STAGE2"].get("dataset_cache_size", 50),
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_train"],
        shuffle=True,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CFG["STAGE2"]["dataloader"]["num_workers"] > 0
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_val"],
        shuffle=False,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CFG["STAGE2"]["dataloader"]["num_workers"] > 0
    )

    # -----------------------------
    # Model and optimizer (Conditional Diffusion)
    # -----------------------------
    diff_mod = _load_diffusion_module()
    cfgd = CFG["STAGE2"]["diffusion"]
    model = diff_mod.ConditionalDiffusionCorrector(
        image_size=ORIG_H,
        channels=CFG["STAGE2"]["num_classes"],
        cond_channels=CFG["STAGE2"]["num_classes"],
        base_dim=cfgd["base_dim"],
        dim_mults=tuple(cfgd["dim_mults"]),
        dropout=cfgd["dropout"],
        objective=cfgd["objective"],
        beta_schedule=cfgd["beta_schedule"],
        timesteps=cfgd["timesteps"],
        sampling_timesteps=cfgd["sampling_timesteps"],
        auto_normalize=cfgd["auto_normalize"],
        flash_attn=cfgd["flash_attn"],
        device=device
    )
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["STAGE2"]["optimizer"]["lr"],
        weight_decay=CFG["STAGE2"]["optimizer"]["weight_decay"],
        betas=tuple(CFG["STAGE2"]["optimizer"].get("betas", (0.9, 0.99)))
    )
    os.makedirs(model_s2_save_dir, exist_ok=True)

    def labels_to_one_hot(lbl: torch.Tensor, num_classes: int) -> torch.Tensor:
        # lbl: (B,H,W) long -> (B,C,H,W) float in {0,1}
        oh = torch.nn.functional.one_hot(lbl.long(), num_classes=num_classes)  # (B,H,W,C)
        return oh.permute(0, 3, 1, 2).contiguous().float()

    # -----------------------------
    # Training loop (paired: cond=degraded, x0=GT)
    # -----------------------------
    best_loss = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    print("[Stage2-Diff][Paired] Training start")
    for epoch in range(CFG["STAGE2"]["epochs"]):
        # Train
        model.train()
        ep_loss, nb = 0.0, 0
        pbar = tqdm(train_ld, desc=f"[Stage2-Diff][Epoch {epoch+1}]")
        for x_in, y_gt, _ in pbar:
            # x_in: (B,1,H,W) float with labels-like values 0..5
            # y_gt: (B,H,W) long  0..5
            cond_lbl = x_in.squeeze(1).long()
            cond_oh = labels_to_one_hot(cond_lbl, CFG["STAGE2"]["num_classes"]).to(device)
            x0_oh = labels_to_one_hot(y_gt, CFG["STAGE2"]["num_classes"]).to(device)

            opt.zero_grad(set_to_none=True)
            loss = model(x0_oh, cond=cond_oh)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lv = float(loss.detach().item())
            ep_loss += lv
            nb += 1
            if nb % 10 == 0:
                pbar.set_postfix({"loss": f"{(ep_loss / max(1, nb)):.4f}"})
        avg_tr = ep_loss / max(1, nb)
        train_losses.append(avg_tr)

        # Val
        model.eval()
        v_loss, vb = 0.0, 0
        with torch.no_grad():
            for x_in, y_gt, _ in val_ld:
                cond_lbl = x_in.squeeze(1).long()
                cond_oh = labels_to_one_hot(cond_lbl, CFG["STAGE2"]["num_classes"]).to(device)
                x0_oh = labels_to_one_hot(y_gt, CFG["STAGE2"]["num_classes"]).to(device)
                l = model(x0_oh, cond=cond_oh)
                if not torch.isfinite(l):
                    continue
                v_loss += float(l.detach().item())
                vb += 1
        avg_val = v_loss / max(1, vb)
        val_losses.append(avg_val)
        print(f"[Stage2-Diff][Epoch {epoch+1}] train_loss={avg_tr:.4f} val_loss={avg_val:.4f}")

        # checkpoint + best
        ckpt_path = os.path.join(model_s2_save_dir, f"diff_checkpoint_epoch_{epoch}.pth")
        try:
            torch.save(model.state_dict(), ckpt_path)
        except Exception as e:
            print(f"[Stage2-Diff] checkpoint save failed: {e}")
        if avg_val < best_loss:
            best_loss = avg_val
            try:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            except Exception:
                best_state = model.state_dict()

        if epoch > 0:
            prev = os.path.join(model_s2_save_dir, f"diff_checkpoint_epoch_{epoch-1}.pth")
            if os.path.exists(prev):
                try:
                    os.remove(prev)
                except Exception as e:
                    print(f"[Stage2-Diff] remove prev checkpoint failed: {e}")

    # Save final/best model and loss curves
    final_path = os.path.join(model_s2_save_dir, "model_final.pth")
    try:
        if best_state is not None:
            torch.save(best_state, final_path)
            model.load_state_dict(best_state)
        else:
            torch.save(model.state_dict(), final_path)
    except Exception as e:
        print(f"[Stage2-Diff] save final failed: {e}")

    try:
        plt.figure(figsize=(8, 4))
        epochs_axis = list(range(1, len(train_losses) + 1))
        plt.plot(epochs_axis, train_losses, label="train_loss")
        plt.plot(epochs_axis, val_losses, label="val_loss")
        best_epoch = int(np.argmin(val_losses)) + 1 if len(val_losses) > 0 else 1
        plt.axvline(x=best_epoch, color="g", linestyle="--", label=f"Best ({best_epoch})")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(model_s2_save_dir, "loss_curve.png"), dpi=150); plt.close()
        pd.DataFrame({"epoch": epochs_axis, "train_loss": train_losses, "val_loss": val_losses}) \
            .to_csv(os.path.join(model_s2_save_dir, "loss_history.csv"), index=False)
    except Exception as e:
        print(f"[Stage2-Diff] save loss curve failed: {e}")

    # -----------------------------
    # Inference (refinement) conditioned on Stage1 probabilities
    # -----------------------------
    test_ds = Stage2DiffusionTestDataset(stage1_out_dir)
    test_ld = DataLoader(
        test_ds,
        batch_size=CFG["STAGE2"]["dataloader"]["batch_size_test"],
        shuffle=False,
        num_workers=CFG["STAGE2"]["dataloader"]["num_workers"],
        pin_memory=torch.cuda.is_available()
    )
    os.makedirs(stage2_out_dir, exist_ok=True)
    print(f"[Stage2-Diff][Paired] Inference on {len(test_ds)} files")

    steps = cfgd["steps"]
    ensemble = cfgd["ensemble"]
    t_start_frac = cfgd["t_start_frac"]
    class_weights = torch.tensor(cfgd["class_weights"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    blend_lambda = cfgd["blend_lambda"]

    import xarray as xr

    with torch.no_grad():
        for probs, token in tqdm(test_ld, desc="[Stage2-Diff][Paired] Infer"):
            # probs: (B,6,H,W)  -> use as condition
            probs = probs.to(device)
            t_start = int(t_start_frac * (model.num_timesteps - 1))
            rec = model.correct_from_probs_cond(probs, steps=steps, t_start=t_start, ensemble=ensemble)  # (E*B,6,H,W)

            # ensemble average
            rec = rec.view(ensemble, -1, rec.shape[1], rec.shape[2], rec.shape[3]).mean(dim=0)  # (B,6,H,W)

            # class weights then renorm
            rec = torch.clamp(rec * class_weights, 0.0, 1.0)
            rec = rec / (rec.sum(dim=1, keepdim=True) + 1e-8)

            # optional blend with Stage1 to preserve continuity
            if blend_lambda > 0:
                s1 = torch.clamp(probs, 0.0, 1.0)
                s1 = s1 / (s1.sum(dim=1, keepdim=True) + 1e-8)
                rec = (1.0 - blend_lambda) * rec + blend_lambda * s1
                rec = rec / (rec.sum(dim=1, keepdim=True) + 1e-8)

            # save
            for b in range(rec.shape[0]):
                arr = rec[b].detach().cpu().numpy()   # (6,H,W)
                arr = np.transpose(arr, (1, 2, 0))    # (H,W,6)
                lat = test_ds.lat
                lon = test_ds.lon
                da = xr.DataArray(
                    arr, dims=["lat", "lon", "class"],
                    coords={"lat": lat, "lon": lon, "class": np.arange(CFG["STAGE2"]["num_classes"])}
                )
                ds = xr.Dataset({"probabilities": da}).expand_dims("time")
                tstr = token[b]
                if isinstance(tstr, bytes):
                    tstr = tstr.decode()
                try:
                    t_dt = pd.to_datetime(tstr, format="%Y%m%d%H%M")
                except Exception:
                    t_dt = pd.to_datetime(str(tstr))
                ds["time"] = [t_dt]
                out = os.path.join(stage2_out_dir, f"refined_{t_dt.strftime('%Y%m%d%H%M')}.nc")
                ds.to_netcdf(out, engine="netcdf4")

    stage2_end = time.time()
    print(f"[Stage2-Diff][Paired] done in {format_time(stage2_end - stage2_start)}")
    print_memory_usage("After Stage 2 (Diffusion, Paired)")


__all__ = ["run_stage2_diffusion"]
