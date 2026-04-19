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
from src.data.classification_dataset import (
    GFPClassificationDataset, build_label_vocab, binarize,
)
from src.models.gfp_classifier import build_gfp_classifier
from src.data import transforms as T
from extract_features import load_metadata


def build_transforms(cfg, train):
    crop = cfg["data"]["crop_size"]
    if train:
        return T.Compose([
            T.RandomCrop2D(crop), T.RandomHFlip2D(), T.RandomVFlip2D(),
            T.RandomRot90_2D(), T.IntensityJitter2D(n_input_channels=1),
            T.ToTensor2D(),
        ])
    return T.Compose([T.CenterCrop2D(crop), T.ToTensor2D()])


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
                if binarize(metadata.get(s, {}).get("Exercise")) is not None]

    label_vocab = build_label_vocab(metadata, ex_stems)
    n_ex = max(len(label_vocab["exercise"]), 2)
    accelerator.print(f"Exercise volumes: {len(ex_stems)}")
    accelerator.print(f"label_vocab: {label_vocab}")
    if len(ex_stems) < 2:
        raise SystemExit("Need at least 2 Exercise volumes for LOO.")

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))

    def make_ds(stem_list, train):
        paths = [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]
        return GFPClassificationDataset(
            paths, stats_dir=stats_dir, metadata=metadata,
            label_vocab=label_vocab,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip)

    results = []
    epochs = tcfg.get("epochs", 30)
    lr = tcfg["lr"]

    for held in ex_stems:
        train_stems = [s for s in ex_stems if s != held]
        true_label = binarize(metadata[held]["Exercise"])
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

        # Train
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
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} loss={np.mean(losses):.4f}")

        # Eval on held-out volume
        model.eval()
        probs_accum = torch.zeros(n_ex, device=device)
        n_slices = 0
        with torch.no_grad():
            for img, _ex, _pt in val_loader:
                logits_ex, _ = model(img)
                probs_accum += logits_ex.softmax(dim=1).sum(dim=0)
                n_slices += img.shape[0]
        mean_probs = (probs_accum / max(n_slices, 1)).cpu().numpy()
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
    p.add_argument("--output", default="results/exercise_loo.json")
    args = p.parse_args()
    main(args.config, args.metadata, args.output)
