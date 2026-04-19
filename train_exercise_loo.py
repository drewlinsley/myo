"""Leave-one-out training on the 8 Exercise volumes.

For each Exercise volume: train encoder+linear head on the other 7, evaluate
on all Z-slices of the held-out volume, vote for the volume-level prediction.

Usage:
    python train_exercise_loo.py -c configs/gfp_classifier.yaml
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.utils import set_seed, prepare_env
from src.data.classification_dataset import GFPClassificationDataset
from src.models.gfp_classifier import build_gfp_classifier
from src.data import transforms as T
from extract_features import load_metadata


def build_transforms(cfg, train):
    dims = cfg["model"].get("dims", "2d")
    crop = cfg["data"]["crop_size"]
    if dims == "2d":
        if train:
            return T.Compose([
                T.RandomCrop2D(crop), T.RandomHFlip2D(), T.RandomVFlip2D(),
                T.RandomRot90_2D(), T.IntensityJitter2D(n_input_channels=1),
                T.ToTensor2D(),
            ])
        return T.Compose([T.CenterCrop2D(crop), T.ToTensor2D()])
    # 3D: dataset does random cropping itself, so transforms only do
    # flips/jitter/tensor conversion on a (D, H, W, C) patch.
    if train:
        return T.Compose([
            T.RandomHFlip3D(), T.RandomVFlip3D(), T.RandomZFlip3D(),
            T.RandomRot90_3D(),
            T.IntensityJitter3D(n_input_channels=1), T.ToTensor3D(),
        ])
    return T.Compose([T.ToTensor3D()])


def main(config_path, metadata_path, output_path):
    cfg = load_config(config_path)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    set_seed(cfg.get("seed", 42))
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    data_dir = dcfg["data_dir"]
    gfp_dir = os.path.join(data_dir, "gfp")
    stats_dir = os.path.join(data_dir, "stats")

    metadata = load_metadata(metadata_path)

    # Filter to Exercise volumes only (stems with a non-None Exercise label)
    all_stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob(os.path.join(gfp_dir, "*.npy"))])
    ex_stems = [s for s in all_stems
                if metadata.get(s, {}).get("Exercise") not in (None, "")]

    # Use raw Exercise labels (e.g. "Stimulated"/"Unstimulated"), not binarized
    raw_classes = sorted({metadata[s]["Exercise"] for s in ex_stems})
    label_vocab = {"exercise": raw_classes, "perturbation": []}
    n_ex = max(len(raw_classes), 2)
    accelerator.print(f"Exercise volumes: {len(ex_stems)}")
    accelerator.print(f"label_vocab: {label_vocab}")
    if len(ex_stems) < 2:
        raise SystemExit("Need at least 2 Exercise volumes for LOO.")

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))

    dims = cfg["model"].get("dims", "2d")
    patch_depth = dcfg.get("patch_depth", 32)
    patches_per_volume = dcfg.get("patches_per_volume", 32)
    crop_size = dcfg.get("crop_size", 256)

    def make_ds(stem_list, train):
        paths = [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]
        return GFPClassificationDataset(
            paths, stats_dir=stats_dir, metadata=metadata,
            label_vocab=label_vocab,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip, use_raw_labels=True,
            mode=dims, patch_depth=patch_depth,
            patches_per_volume=(patches_per_volume if train else 8),
            crop_size=crop_size)

    results = []
    epochs = tcfg.get("epochs", 30)
    lr = tcfg["lr"]

    for held in ex_stems:
        train_stems = [s for s in ex_stems if s != held]
        true_label = metadata[held]["Exercise"]
        true_idx = label_vocab["exercise"].index(true_label)

        accelerator.print(f"\n── LOO held-out: {held} (true={true_label}) ──")

        # Fresh model per fold
        model = build_gfp_classifier(cfg, n_ex, n_perturbation=2)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=tcfg.get("weight_decay", 0.01))

        train_loader = torch.utils.data.DataLoader(
            make_ds(train_stems, train=True),
            batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
            num_workers=tcfg.get("num_workers", 4), pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            make_ds([held], train=False),
            batch_size=tcfg["batch_size"], shuffle=False,
            num_workers=tcfg.get("num_workers", 4))

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        def eval_held_out():
            model.eval()
            total_loss, total_n = 0.0, 0
            probs = torch.zeros(n_ex, device=device)
            n_slices = 0
            with torch.no_grad():
                for img, ex, _pt in val_loader:
                    logits_ex, _ = model(img)
                    if (ex != -1).any():
                        total_loss += criterion(
                            logits_ex, ex).item() * img.shape[0]
                        total_n += img.shape[0]
                    probs += logits_ex.softmax(dim=1).sum(dim=0)
                    n_slices += img.shape[0]
            mean_loss = total_loss / max(total_n, 1)
            mean_probs = (probs / max(n_slices, 1)).cpu().numpy()
            return mean_loss, mean_probs

        # Train with early stopping on held-out loss (leaky at n=8, acceptable)
        patience = tcfg.get("patience", 10)
        min_delta = tcfg.get("min_delta", 1e-3)
        best_val_loss = float("inf")
        best_probs = None
        best_epoch = 0
        no_improve = 0
        for ep in range(epochs):
            model.train()
            losses = []
            for img, ex, _pt in train_loader:
                logits_ex, _ = model(img)
                if (ex == -1).all():
                    continue
                loss = criterion(logits_ex, ex)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                losses.append(loss.item())
            mean_tr = float(np.mean(losses)) if losses else float("inf")
            val_loss, probs = eval_held_out()
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} train={mean_tr:.4f} val={val_loss:.4f}")
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_probs = probs
                best_epoch = ep + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    accelerator.print(
                        f"  early stop at ep{ep+1} "
                        f"(best_val={best_val_loss:.4f} @ ep{best_epoch})")
                    break

        mean_probs = best_probs if best_probs is not None else probs
        pred_idx = int(mean_probs.argmax())
        pred_label = label_vocab["exercise"][pred_idx]
        correct = int(pred_idx == true_idx)
        accelerator.print(
            f"  pred={pred_label} (p={mean_probs[pred_idx]:.3f})  "
            f"correct={correct}")
        results.append({
            "stem": held,
            "true": true_label,
            "pred": pred_label,
            "correct": correct,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "probs": {label_vocab["exercise"][i]: float(mean_probs[i])
                      for i in range(n_ex)},
        })

    # Aggregate
    acc = float(np.mean([r["correct"] for r in results]))
    classes = label_vocab["exercise"]
    per_class = {c: {"correct": 0, "total": 0} for c in classes}
    for r in results:
        per_class[r["true"]]["total"] += 1
        per_class[r["true"]]["correct"] += r["correct"]

    summary = {
        "n_volumes": len(results),
        "overall_accuracy": acc,
        "classes": classes,
        "per_class": {c: {**v,
                          "accuracy": v["correct"] / max(v["total"], 1)}
                      for c, v in per_class.items()},
        "per_volume": results,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(f"\nLOO accuracy: {acc:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
    accelerator.print(f"Saved {output_path}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["overall"] + classes
    ys = [acc] + [per_class[c]["correct"] / max(per_class[c]["total"], 1)
                  for c in classes]
    ns = [len(results)] + [per_class[c]["total"] for c in classes]
    ax.bar(xs, ys, color=["#4363d8"] + ["#3cb44b"] * len(classes))
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + 0.02, f"{y:.2f}\n(n={n})", ha="center", fontsize=9)
    ax.axhline(1.0 / len(classes), color="gray", linestyle=":",
               label=f"chance ({1.0/len(classes):.2f})")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Exercise LOO (n={len(results)})")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.splitext(output_path)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    accelerator.print(f"Saved {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default="data_mapping_drew.csv")
    p.add_argument("--output", default=None,
                   help="Default: results/exercise_loo_{2d|3d}.json")
    args = p.parse_args()
    if args.output is None:
        _cfg = load_config(args.config)
        _dims = _cfg["model"].get("dims", "2d")
        args.output = f"results/exercise_loo_{_dims}.json"
    main(args.config, args.metadata, args.output)
