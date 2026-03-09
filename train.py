"""Unified training script for 2D and 3D BF -> GFP prediction.

Usage:
    python train.py -c configs/unet_2d_imagenet.yaml
    python train.py -c configs/unet_3d_random.yaml --resume ckpts/unet_3d_random/latest.pth
"""

import os
import csv
import json
import shutil
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.utils import set_seed, prepare_env, save_checkpoint, load_checkpoint, make_train_val_split, get_git_hash
from src.models import build_model
from src.losses import build_loss
from src.metrics import psnr as compute_psnr
from src.data.datasets import SliceDataset, VolumeDataset
from src.data import transforms as T


def save_val_montages(model, val_loader, epoch, ckpt_dir, cfg, accelerator, n_samples=4):
    """Save montages of validation predictions on new best loss.

    For 2D: BF | GT | Pred | |Error| per sample.
    For 3D: same layout but shows 5 evenly-spaced Z slices per sample.
    """
    dims = cfg["model"]["dims"]
    montage_dir = os.path.join(ckpt_dir, "montages")
    os.makedirs(montage_dir, exist_ok=True)

    model.eval()
    collected = 0
    bf_list, gt_list, pred_list = [], [], []

    with torch.no_grad():
        for bf, fl in val_loader:
            pred = model(bf)
            # Move to CPU numpy
            bf_np = bf.cpu().numpy()
            gt_np = fl.cpu().numpy()
            pred_np = pred.cpu().numpy()

            for i in range(bf_np.shape[0]):
                if collected >= n_samples:
                    break
                bf_list.append(bf_np[i])
                gt_list.append(gt_np[i])
                pred_list.append(pred_np[i])
                collected += 1
            if collected >= n_samples:
                break

    if not bf_list:
        return

    if dims == "2d":
        # Each is (1, H, W) — show as single row per sample
        n = len(bf_list)
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[np.newaxis]
        for i in range(n):
            bf_img = bf_list[i][0]
            gt_img = gt_list[i][0]
            pr_img = pred_list[i][0]
            err = np.abs(gt_img - pr_img)

            axes[i, 0].imshow(bf_img, cmap="gray")
            axes[i, 0].set_title("BF" if i == 0 else "")
            axes[i, 1].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("GT GFP" if i == 0 else "")
            axes[i, 2].imshow(pr_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("Predicted" if i == 0 else "")
            axes[i, 3].imshow(err, cmap="hot", vmin=0, vmax=0.5)
            axes[i, 3].set_title("|Error|" if i == 0 else "")
            for ax in axes[i]:
                ax.axis("off")
    else:
        # Each is (1, H, W, D) — pick 5 Z slices from first sample
        n = min(len(bf_list), 2)  # show up to 2 volumes for 3D
        n_z = 5
        fig, axes = plt.subplots(n * 4, n_z, figsize=(4 * n_z, 4 * n * 4))
        if n * 4 == 4:
            axes = axes[np.newaxis] if axes.ndim == 1 else axes
        for vi in range(n):
            # (1, H, W, D) -> (D, H, W)
            bf_vol = bf_list[vi][0].transpose(2, 0, 1)
            gt_vol = gt_list[vi][0].transpose(2, 0, 1)
            pr_vol = pred_list[vi][0].transpose(2, 0, 1)
            D = bf_vol.shape[0]
            z_idx = np.linspace(0, D - 1, n_z, dtype=int)
            row_base = vi * 4
            for j, zi in enumerate(z_idx):
                axes[row_base + 0, j].imshow(bf_vol[zi], cmap="gray")
                if j == 0:
                    axes[row_base + 0, j].set_ylabel("BF")
                axes[row_base + 1, j].imshow(gt_vol[zi], cmap="gray", vmin=0, vmax=1)
                if j == 0:
                    axes[row_base + 1, j].set_ylabel("GT")
                axes[row_base + 2, j].imshow(pr_vol[zi], cmap="gray", vmin=0, vmax=1)
                if j == 0:
                    axes[row_base + 2, j].set_ylabel("Pred")
                err = np.abs(gt_vol[zi] - pr_vol[zi])
                axes[row_base + 3, j].imshow(err, cmap="hot", vmin=0, vmax=0.5)
                if j == 0:
                    axes[row_base + 3, j].set_ylabel("|Error|")
                axes[row_base + 0, j].set_title(f"Z={zi}" if vi == 0 else "")
                for r in range(4):
                    axes[row_base + r, j].set_xticks([])
                    axes[row_base + r, j].set_yticks([])

    plt.suptitle(f"Epoch {epoch + 1} (best)", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(montage_dir, f"epoch_{epoch + 1:04d}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    accelerator.print(f"  -> Saved montage to {out_path}")


def build_transforms(cfg, train=True):
    """Build augmentation pipeline based on config (2D or 3D)."""
    dims = cfg["model"]["dims"]
    crop = cfg["data"]["crop_size"]

    if dims == "2d":
        if train:
            return T.Compose([
                T.RandomCrop2D(crop),
                T.RandomHFlip2D(),
                T.RandomVFlip2D(),
                T.RandomRot90_2D(),
                T.IntensityJitter2D(n_input_channels=1),
                T.ToTensor2D(),
            ])
        else:
            return T.Compose([
                T.CenterCrop2D(crop),
                T.ToTensor2D(),
            ])
    else:
        depth = cfg["data"]["patch_depth"]
        if train:
            return T.Compose([
                T.RandomCrop3D(depth, crop, crop),
                T.RandomHFlip3D(),
                T.RandomVFlip3D(),
                T.RandomZFlip3D(),
                T.RandomRot90_3D(),
                T.IntensityJitter3D(n_input_channels=1),
                T.ToTensor3D(),
            ])
        else:
            return T.Compose([
                T.CenterCrop3D(depth, crop, crop),
                T.ToTensor3D(),
            ])


def build_datasets(cfg):
    """Build train and val datasets."""
    dcfg = cfg["data"]
    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")

    # Discover files
    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    assert len(bf_files) > 0, f"No .npy files found in {bf_dir}"

    # Match GFP files by stem
    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]
    gfp_files = [os.path.join(gfp_dir, f"{s}.npy") for s in stems]
    for gf in gfp_files:
        assert os.path.exists(gf), f"Missing GFP file: {gf}"

    # Train/val split
    train_stems, val_stems = make_train_val_split(
        stems, val_fraction=dcfg.get("val_fraction", 0.15), seed=cfg.get("seed", 42))

    def stems_to_paths(stem_list):
        bf = [os.path.join(bf_dir, f"{s}.npy") for s in stem_list]
        gfp = [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]
        return bf, gfp

    train_bf, train_gfp = stems_to_paths(train_stems)
    val_bf, val_gfp = stems_to_paths(val_stems)

    # Determine if TIMM normalization should be applied
    apply_timm = cfg["model"].get("encoder_weights") is not None
    dims = cfg["model"]["dims"]
    cache = dcfg.get("cache_volumes", False)

    DatasetClass = SliceDataset if dims == "2d" else VolumeDataset

    common_kwargs = dict(
        stats_dir=stats_dir,
        apply_timm=apply_timm,
        cache_volumes=cache,
    )

    if dims == "2d":
        extra_kwargs = dict(crop_size=dcfg["crop_size"])
    else:
        extra_kwargs = dict(
            patch_depth=dcfg.get("patch_depth", 32),
            crop_size=dcfg["crop_size"],
            patches_per_volume=dcfg.get("patches_per_volume", 32),
        )

    train_ds = DatasetClass(
        train_bf, train_gfp,
        transform=build_transforms(cfg, train=True),
        **common_kwargs, **extra_kwargs,
    )
    val_ds = DatasetClass(
        val_bf, val_gfp,
        transform=build_transforms(cfg, train=False),
        **common_kwargs, **extra_kwargs,
    )

    return train_ds, val_ds, train_stems, val_stems


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build LR scheduler with optional warmup."""
    tcfg = cfg["training"]
    sched_type = tcfg.get("scheduler", "cosine")
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    total_epochs = tcfg["epochs"]

    if sched_type == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs,
            eta_min=tcfg["lr"] * 0.01,
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return cosine
    elif sched_type == "none" or sched_type is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


def main(config_path, resume_from=None):
    cfg = load_config(config_path)
    cfg = validate_config(cfg)

    tcfg = cfg["training"]
    experiment_name = cfg.get("experiment_name", "default")

    # Seeding
    set_seed(cfg.get("seed", 42))

    # Environment
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    accelerator.print(f"Experiment: {experiment_name}")
    accelerator.print(f"Model: {cfg['model']['dims']} U-Net, encoder={cfg['model']['encoder']}, "
                      f"weights={cfg['model'].get('encoder_weights', 'random')}")

    # Checkpoint dir
    ckpt_dir = tcfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config snapshot
    shutil.copy2(config_path, os.path.join(ckpt_dir, "config.yaml"))

    # Datasets
    train_ds, val_ds, train_stems, val_stems = build_datasets(cfg)
    accelerator.print(f"Train: {len(train_ds)} samples ({len(train_stems)} volumes), "
                      f"Val: {len(val_ds)} samples ({len(val_stems)} volumes)")

    # Save split info
    with open(os.path.join(ckpt_dir, "split.json"), "w") as f:
        json.dump({"train": train_stems, "val": val_stems}, f, indent=2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tcfg["batch_size"],
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=tcfg.get("num_workers", 4),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=tcfg["batch_size"],
        shuffle=False, drop_last=False,
        num_workers=tcfg.get("num_workers", 4),
    )

    # Model
    model = build_model(cfg)
    accelerator.print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = build_loss(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.01),
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from:
        accelerator.print(f"Resuming from {resume_from}")
        ckpt = load_checkpoint(resume_from, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        accelerator.print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Prepare for distributed / mixed precision
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    if scheduler:
        scheduler = accelerator.prepare(scheduler)

    # Gradient accumulation
    grad_accum = tcfg.get("grad_accumulation_steps", 1)

    # CSV logger
    csv_path = os.path.join(ckpt_dir, "log.csv")
    csv_exists = os.path.exists(csv_path) and resume_from
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "lr"])

    # Training loop
    for epoch in range(start_epoch, tcfg["epochs"]):
        # === Train ===
        model.train()
        train_losses = []
        optimizer.zero_grad()
        progress = tqdm(total=len(train_loader),
                        desc=f"Epoch {epoch+1}/{tcfg['epochs']} [train]")

        for step, (bf, fl) in enumerate(train_loader):
            with accelerator.accumulate(model):
                pred = model(bf)
                loss = criterion(pred, fl) / grad_accum
                accelerator.backward(loss)

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            train_losses.append(loss.item() * grad_accum)
            progress.set_postfix(loss=f"{train_losses[-1]:.4f}")
            progress.update()
        progress.close()

        if scheduler:
            scheduler.step()

        # === Validate ===
        model.eval()
        val_losses, val_psnrs = [], []
        with torch.no_grad():
            progress = tqdm(total=len(val_loader),
                            desc=f"Epoch {epoch+1}/{tcfg['epochs']} [val]")
            for bf, fl in val_loader:
                pred = model(bf)
                loss = criterion(pred, fl)
                p = compute_psnr(pred, fl)
                val_losses.append(loss.item())
                val_psnrs.append(p.item() if isinstance(p, torch.Tensor) else p)
                progress.set_postfix(loss=f"{loss.item():.4f}",
                                     psnr=f"{val_psnrs[-1]:.1f}")
                progress.update()
            progress.close()

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        mean_psnr = np.mean([p for p in val_psnrs if p != float("inf")] or [0])
        current_lr = optimizer.param_groups[0]["lr"]

        accelerator.print(
            f"Epoch {epoch+1}: train_loss={mean_train:.4f}  "
            f"val_loss={mean_val:.4f}  val_psnr={mean_psnr:.1f}dB  "
            f"lr={current_lr:.2e}")

        # CSV log
        csv_writer.writerow([epoch + 1, f"{mean_train:.6f}", f"{mean_val:.6f}",
                             f"{mean_psnr:.2f}", f"{current_lr:.2e}"])
        csv_file.flush()

        # Save best
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            accelerator.print(f"  -> New best val loss: {best_val_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, best_val_loss, cfg,
                            os.path.join(ckpt_dir, "best.pth"), accelerator)
            save_val_montages(model, val_loader, epoch, ckpt_dir, cfg, accelerator)

        # Periodic save
        save_every = tcfg.get("save_every", 25)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, mean_val, cfg,
                            os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"), accelerator)

        # Always save latest (for resume)
        save_checkpoint(model, optimizer, epoch, mean_val, cfg,
                        os.path.join(ckpt_dir, "latest.pth"), accelerator)

    csv_file.close()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BF -> GFP model")
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Path to experiment config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.resume)
