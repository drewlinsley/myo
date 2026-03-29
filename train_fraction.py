"""Train with a fraction of the training data (for power-law / scaling experiments).

Wraps the standard train.py pipeline but subsamples training volumes after the
train/val split, so the val set stays fixed across fractions.

Usage:
    python train_fraction.py -c configs/unet_2d_imagenet_pearson.yaml --fraction 0.25
"""

import os
import csv
import json
import shutil
import argparse
from glob import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.config import load_config, validate_config
from src.utils import (set_seed, prepare_env, save_checkpoint, load_checkpoint,
                       make_train_val_split)
from src.models import build_model
from src.losses import build_loss
from src.metrics import psnr as compute_psnr
from src.data.datasets import SliceDataset, VolumeDataset

# Reuse helpers from train.py
from train import build_transforms, build_scheduler, save_val_montages


def build_datasets_with_fraction(cfg, fraction, seed=42):
    """Build train/val datasets, subsampling training volumes to `fraction`."""
    dcfg = cfg["data"]
    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")

    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    assert len(bf_files) > 0, f"No .npy files found in {bf_dir}"

    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]

    # Fixed train/val split (identical across fractions)
    train_stems, val_stems = make_train_val_split(
        stems, val_fraction=dcfg.get("val_fraction", 0.15), seed=cfg.get("seed", 42))

    # Subsample training stems
    if fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = max(1, int(len(train_stems) * fraction))
        train_stems = sorted(rng.choice(train_stems, size=n_keep, replace=False).tolist())

    def stems_to_paths(stem_list):
        bf = [os.path.join(bf_dir, f"{s}.npy") for s in stem_list]
        gfp = [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]
        return bf, gfp

    train_bf, train_gfp = stems_to_paths(train_stems)
    val_bf, val_gfp = stems_to_paths(val_stems)

    apply_timm = cfg["model"].get("encoder_weights") is not None
    dims = cfg["model"]["dims"]
    cache = dcfg.get("cache_volumes", False)
    DatasetClass = SliceDataset if dims == "2d" else VolumeDataset
    z_range = dcfg.get("z_range", None)
    gfp_norm_mode = dcfg.get("gfp_norm_mode", "volume")
    filter_empty_gfp = dcfg.get("filter_empty_gfp", False)
    empty_gfp_threshold = dcfg.get("empty_gfp_threshold", 0.01)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))

    common_kwargs = dict(
        stats_dir=stats_dir,
        apply_timm=apply_timm,
        cache_volumes=cache,
        z_range=z_range,
        gfp_norm_mode=gfp_norm_mode,
        filter_empty_gfp=filter_empty_gfp,
        empty_gfp_threshold=empty_gfp_threshold,
        percentile_clip=percentile_clip,
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


def main(config_path, fraction, resume_from=None):
    cfg = load_config(config_path)
    cfg = validate_config(cfg)

    tcfg = cfg["training"]
    experiment_name = cfg.get("experiment_name", "default")
    frac_tag = f"frac{int(fraction * 100):03d}"

    set_seed(cfg.get("seed", 42))

    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    accelerator.print(f"Experiment: {experiment_name} [{frac_tag}]")
    accelerator.print(f"Training fraction: {fraction:.0%}")

    # Checkpoint dir includes fraction tag
    ckpt_dir = os.path.join(tcfg["checkpoint_dir"] + f"_{frac_tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    shutil.copy2(config_path, os.path.join(ckpt_dir, "config.yaml"))

    # Datasets
    train_ds, val_ds, train_stems, val_stems = build_datasets_with_fraction(
        cfg, fraction, seed=cfg.get("seed", 42))
    accelerator.print(f"Train: {len(train_ds)} samples ({len(train_stems)} volumes), "
                      f"Val: {len(val_ds)} samples ({len(val_stems)} volumes)")

    with open(os.path.join(ckpt_dir, "split.json"), "w") as f:
        json.dump({"train": train_stems, "val": val_stems, "fraction": fraction}, f, indent=2)

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

    model = build_model(cfg)
    accelerator.print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_loss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.01),
    )

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    patience = tcfg.get("patience", 50)
    epochs_without_improvement = 0

    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from:
        accelerator.print(f"Resuming from {resume_from}")
        ckpt = load_checkpoint(resume_from, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    if scheduler:
        scheduler = accelerator.prepare(scheduler)

    grad_accum = tcfg.get("grad_accumulation_steps", 1)

    csv_path = os.path.join(ckpt_dir, "log.csv")
    csv_exists = os.path.exists(csv_path) and resume_from
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "lr"])

    for epoch in range(start_epoch, tcfg["epochs"]):
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

        csv_writer.writerow([epoch + 1, f"{mean_train:.6f}", f"{mean_val:.6f}",
                             f"{mean_psnr:.2f}", f"{current_lr:.2e}"])
        csv_file.flush()

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            epochs_without_improvement = 0
            accelerator.print(f"  -> New best val loss: {best_val_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, best_val_loss, cfg,
                            os.path.join(ckpt_dir, "best.pth"), accelerator)
            save_val_montages(model, val_loader, epoch, ckpt_dir, cfg, accelerator)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                accelerator.print(f"Early stopping: no improvement for {patience} epochs")
                break

        save_every = tcfg.get("save_every", 25)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, mean_val, cfg,
                            os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"), accelerator)

        save_checkpoint(model, optimizer, epoch, mean_val, cfg,
                        os.path.join(ckpt_dir, "latest.pth"), accelerator)

    csv_file.close()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with data fraction")
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Path to experiment config YAML")
    parser.add_argument("--fraction", required=True, type=float,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.fraction, args.resume)
