"""Train a two-head GFP classifier (exercise, perturbation) at a data fraction.

Same train/val split logic as train_fraction.py so scaling curves are
directly comparable to the BF->GFP sweep.

Usage:
    python train_gfp_classifier.py -c configs/gfp_classifier.yaml --fraction 0.25
"""

import os
import csv
import json
import shutil
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, validate_config
from src.utils import (set_seed, prepare_env, save_checkpoint, load_checkpoint,
                       make_train_val_split)
from src.data.classification_dataset import (
    GFPClassificationDataset, build_label_vocab,
)
from src.models.gfp_classifier import build_gfp_classifier
from src.data import transforms as T
from extract_features import load_metadata


def build_transforms(cfg, train):
    crop = cfg["data"]["crop_size"]
    if train:
        return T.Compose([
            T.RandomCrop2D(crop),
            T.RandomHFlip2D(),
            T.RandomVFlip2D(),
            T.RandomRot90_2D(),
            T.IntensityJitter2D(n_input_channels=1),
            T.ToTensor2D(),
        ])
    return T.Compose([T.CenterCrop2D(crop), T.ToTensor2D()])


def build_datasets(cfg, metadata_path, fraction, seed=42):
    dcfg = cfg["data"]
    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    gfp_dir = os.path.join(data_dir, "gfp")
    bf_dir = os.path.join(data_dir, "bf")

    # Same discovery+split logic as train_fraction.py
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]

    train_stems, val_stems = make_train_val_split(
        stems, val_fraction=dcfg.get("val_fraction", 0.15),
        seed=cfg.get("seed", 42))

    if fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = max(1, int(len(train_stems) * fraction))
        train_stems = sorted(
            rng.choice(train_stems, size=n_keep, replace=False).tolist())

    metadata = load_metadata(metadata_path)
    label_vocab = build_label_vocab(metadata, stems)

    def paths(stem_list):
        return [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))

    common = dict(stats_dir=stats_dir, metadata=metadata,
                  label_vocab=label_vocab, z_range=z_range,
                  apply_timm=apply_timm, percentile_clip=percentile_clip)

    train_ds = GFPClassificationDataset(
        paths(train_stems), transform=build_transforms(cfg, True), **common)
    val_ds = GFPClassificationDataset(
        paths(val_stems), transform=build_transforms(cfg, False), **common)

    return train_ds, val_ds, train_stems, val_stems, label_vocab


def _accuracy(logits, targets):
    """Per-head accuracy, ignoring targets == -1. Returns (correct, total)."""
    valid = targets != -1
    if valid.sum() == 0:
        return 0, 0
    preds = logits[valid].argmax(dim=1)
    return int((preds == targets[valid]).sum().item()), int(valid.sum().item())


def main(config_path, metadata_path, fraction, resume_from=None):
    cfg = load_config(config_path)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    frac_tag = f"frac{int(fraction * 100):03d}"

    set_seed(cfg.get("seed", 42))
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    ckpt_dir = tcfg["checkpoint_dir"] + f"_{frac_tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(ckpt_dir, "config.yaml"))

    train_ds, val_ds, train_stems, val_stems, label_vocab = build_datasets(
        cfg, metadata_path, fraction, seed=cfg.get("seed", 42))
    accelerator.print(f"[{frac_tag}] train slices={len(train_ds)} ({len(train_stems)} vols), "
                      f"val slices={len(val_ds)} ({len(val_stems)} vols)")
    accelerator.print(f"label_vocab: {label_vocab}")

    with open(os.path.join(ckpt_dir, "split.json"), "w") as f:
        json.dump({"train": train_stems, "val": val_stems,
                   "fraction": fraction, "label_vocab": label_vocab}, f, indent=2)

    n_ex = max(len(label_vocab["exercise"]), 2)
    n_pt = max(len(label_vocab["perturbation"]), 2)
    model = build_gfp_classifier(cfg, n_ex, n_pt)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        pin_memory=True, num_workers=tcfg.get("num_workers", 4))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg.get("num_workers", 4))

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg.get("weight_decay", 0.01))

    start_epoch, best_val = 0, float("inf")
    if resume_from:
        ckpt = load_checkpoint(resume_from, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", float("inf"))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    csv_path = os.path.join(ckpt_dir, "log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss",
                     "val_acc_exercise", "val_acc_perturbation"])

    patience = tcfg.get("patience", 20)
    no_improve = 0

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        train_losses = []
        for img, ex, pt in tqdm(train_loader, desc=f"ep{epoch+1}[train]"):
            logits_ex, logits_pt = model(img)
            loss_ex = criterion(logits_ex, ex)
            loss_pt = criterion(logits_pt, pt)
            # Only sum heads that have at least one valid target in this batch
            loss = torch.tensor(0.0, device=img.device)
            n_valid = 0
            if (ex != -1).any():
                loss = loss + loss_ex
                n_valid += 1
            if (pt != -1).any():
                loss = loss + loss_pt
                n_valid += 1
            if n_valid == 0:
                continue
            loss = loss / n_valid
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        ex_c, ex_t, pt_c, pt_t = 0, 0, 0, 0
        with torch.no_grad():
            for img, ex, pt in tqdm(val_loader, desc=f"ep{epoch+1}[val]"):
                logits_ex, logits_pt = model(img)
                loss = 0.0
                n = 0
                if (ex != -1).any():
                    loss = loss + criterion(logits_ex, ex).item()
                    n += 1
                if (pt != -1).any():
                    loss = loss + criterion(logits_pt, pt).item()
                    n += 1
                if n > 0:
                    val_losses.append(loss / n)
                c, t = _accuracy(logits_ex, ex)
                ex_c += c; ex_t += t
                c, t = _accuracy(logits_pt, pt)
                pt_c += c; pt_t += t

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        acc_ex = ex_c / max(ex_t, 1)
        acc_pt = pt_c / max(pt_t, 1)

        accelerator.print(
            f"ep{epoch+1}: train={mean_train:.4f} val={mean_val:.4f} "
            f"acc_ex={acc_ex:.3f} ({ex_c}/{ex_t}) "
            f"acc_pt={acc_pt:.3f} ({pt_c}/{pt_t})")
        writer.writerow([epoch + 1, f"{mean_train:.6f}", f"{mean_val:.6f}",
                         f"{acc_ex:.4f}", f"{acc_pt:.4f}"])
        csv_file.flush()

        if mean_val < best_val:
            best_val = mean_val
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val, cfg,
                            os.path.join(ckpt_dir, "best.pth"), accelerator)
            # Save best-epoch metrics for aggregation
            with open(os.path.join(ckpt_dir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "val_loss": best_val,
                           "val_acc_exercise": acc_ex,
                           "val_acc_perturbation": acc_pt,
                           "fraction": fraction}, f, indent=2)
        else:
            no_improve += 1
            if no_improve >= patience:
                accelerator.print(f"Early stop at epoch {epoch+1}")
                break

        save_checkpoint(model, optimizer, epoch, mean_val, cfg,
                        os.path.join(ckpt_dir, "latest.pth"), accelerator)

    csv_file.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default="data_mapping_drew.csv")
    p.add_argument("--fraction", required=True, type=float)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    main(args.config, args.metadata, args.fraction, args.resume)
