"""Leave-one-out classifier with optional encoder init from a BF->GFP U-Net.

For each volume in the selected task subset (Exercise or Perturbation):
  1. Fresh model: GFPTwoHeadClassifier (2D or 3D).
  2. If --init_from <unet_ckpt>: load encoder weights from that checkpoint
     (BF->GFP pretrained encoder). Heads stay randomly initialized.
  3. Train on the other N-1 volumes, evaluate on the held-out volume every
     epoch. Early-stop on held-out CE plateau.
  4. Predict via mean-softmax across all slices/patches of the held-out vol.

Usage:
    python train_loo_classifier.py \
        -c configs/gfp_classifier_3d.yaml \
        --task exercise --input bf \
        --init_from ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
        --output results/loo/frac100_exercise_bf.json
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


TASK_LABEL_COL = {"exercise": "Exercise", "perturbation": "Perturbation"}


def build_transforms(cfg, train):
    dims = cfg["model"].get("dims", "2d")
    crop = cfg["data"]["crop_size"]
    if dims == "2d":
        if train:
            return T.Compose([
                T.RandomCrop2D(crop), T.RandomHFlip2D(), T.RandomVFlip2D(),
                T.RandomRot90_2D(), T.IntensityJitter2D(n_input_channels=1),
                T.ToTensor2D()])
        return T.Compose([T.CenterCrop2D(crop), T.ToTensor2D()])
    if train:
        return T.Compose([
            T.RandomHFlip3D(), T.RandomVFlip3D(), T.RandomZFlip3D(),
            T.RandomRot90_3D(), T.IntensityJitter3D(n_input_channels=1),
            T.ToTensor3D()])
    return T.Compose([T.ToTensor3D()])


def load_encoder_from_unet(classifier, ckpt_path, device):
    """Load encoder weights from a BF->GFP U-Net checkpoint into classifier."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    encoder_state = {k[len("encoder."):]: v for k, v in state.items()
                     if k.startswith("encoder.")}
    result = classifier.encoder.load_state_dict(
        encoder_state, strict=False)
    if result is None:
        raise RuntimeError(
            f"encoder.load_state_dict returned None for {ckpt_path}; "
            "likely shape mismatch (2D ckpt into 3D encoder or vice versa)")
    missing, unexpected = result
    return len(encoder_state), len(missing), len(unexpected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--metadata", default="data_mapping_drew.csv")
    p.add_argument("--task", choices=["exercise", "perturbation"],
                   required=True)
    p.add_argument("--input", choices=["bf", "gfp"], default="bf",
                   help="Modality fed into the encoder")
    p.add_argument("--init_from", default=None,
                   help="Optional U-Net checkpoint to initialize encoder")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    set_seed(cfg.get("seed", 42))
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)  # "bf" or "gfp"

    metadata = load_metadata(args.metadata)
    label_col = TASK_LABEL_COL[args.task]

    # Filter to volumes with a label for this task
    all_stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob(os.path.join(mod_dir, "*.npy"))])
    stems = [s for s in all_stems
             if metadata.get(s, {}).get(label_col) not in (None, "")]

    raw_classes = sorted({metadata[s][label_col] for s in stems})
    vocab = {"exercise": [], "perturbation": []}
    vocab[args.task] = raw_classes
    n_cls = max(len(raw_classes), 2)
    accelerator.print(f"task={args.task} input={args.input} "
                      f"n_volumes={len(stems)} classes={raw_classes}")
    if len(stems) < 2:
        raise SystemExit(f"Need >=2 {args.task} volumes, got {len(stems)}")

    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    percentile_clip = tuple(dcfg.get("percentile_clip", [0.5, 99.5]))
    dims = cfg["model"].get("dims", "2d")

    def make_ds(stem_list, train):
        paths = [os.path.join(mod_dir, f"{s}.npy") for s in stem_list]
        return GFPClassificationDataset(
            paths, stats_dir=stats_dir, metadata=metadata, label_vocab=vocab,
            transform=build_transforms(cfg, train),
            z_range=z_range, apply_timm=apply_timm,
            percentile_clip=percentile_clip, use_raw_labels=True,
            mode=dims, patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=(dcfg.get("patches_per_volume", 32)
                                if train else 8),
            crop_size=dcfg.get("crop_size", 256), modality=args.input)

    epochs = tcfg.get("epochs", 50)
    patience = tcfg.get("patience", 10)
    min_delta = tcfg.get("min_delta", 1e-3)
    lr = tcfg["lr"]
    n_ex = n_cls if args.task == "exercise" else 2
    n_pt = n_cls if args.task == "perturbation" else 2

    results = []
    for held in stems:
        train_stems = [s for s in stems if s != held]
        true_label = metadata[held][label_col]
        true_idx = raw_classes.index(true_label)
        accelerator.print(f"\n── LOO held: {held} (true={true_label}) ──")

        model = build_gfp_classifier(cfg, n_ex, n_pt)
        if args.init_from:
            n_load, n_miss, n_extra = load_encoder_from_unet(
                model, args.init_from, "cpu")
            accelerator.print(
                f"  loaded {n_load} encoder tensors "
                f"(missing={n_miss} unexpected={n_extra})")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=tcfg.get("weight_decay", 0.01))
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        train_loader = torch.utils.data.DataLoader(
            make_ds(train_stems, True), batch_size=tcfg["batch_size"],
            shuffle=True, drop_last=True,
            num_workers=tcfg.get("num_workers", 4), pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            make_ds([held], False), batch_size=tcfg["batch_size"],
            shuffle=False, num_workers=tcfg.get("num_workers", 4))

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)

        def forward_head(logits_ex, logits_pt, ex, pt):
            if args.task == "exercise":
                return logits_ex, ex
            return logits_pt, pt

        def eval_held():
            model.eval()
            total_loss, total_n = 0.0, 0
            probs = torch.zeros(n_cls, device=device)
            n_samples = 0
            with torch.no_grad():
                for img, ex, pt in val_loader:
                    lex, lpt = model(img)
                    logits, target = forward_head(lex, lpt, ex, pt)
                    if (target != -1).any():
                        total_loss += criterion(
                            logits, target).item() * img.shape[0]
                        total_n += img.shape[0]
                    probs += logits.softmax(dim=1).sum(dim=0)
                    n_samples += img.shape[0]
            return (total_loss / max(total_n, 1),
                    (probs / max(n_samples, 1)).cpu().numpy())

        best_val = float("inf")
        best_probs = None
        best_epoch = 0
        no_improve = 0
        for ep in range(epochs):
            model.train()
            losses = []
            for img, ex, pt in train_loader:
                lex, lpt = model(img)
                logits, target = forward_head(lex, lpt, ex, pt)
                if (target == -1).all():
                    continue
                loss = criterion(logits, target)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                losses.append(loss.item())
            tr = float(np.mean(losses)) if losses else float("inf")
            vl, probs = eval_held()
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} train={tr:.4f} val={vl:.4f}")
            if vl < best_val - min_delta:
                best_val = vl
                best_probs = probs
                best_epoch = ep + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    accelerator.print(
                        f"  early stop ep{ep+1} "
                        f"(best={best_val:.4f} @ ep{best_epoch})")
                    break

        pred_probs = best_probs if best_probs is not None else probs
        pred_idx = int(pred_probs.argmax())
        pred = raw_classes[pred_idx]
        correct = int(pred_idx == true_idx)
        accelerator.print(
            f"  pred={pred} (p={pred_probs[pred_idx]:.3f}) correct={correct}")
        results.append({
            "stem": held, "true": true_label, "pred": pred,
            "correct": correct, "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "probs": {raw_classes[i]: float(pred_probs[i])
                      for i in range(n_cls)}})

    acc = float(np.mean([r["correct"] for r in results]))
    per_class = {c: {"correct": 0, "total": 0} for c in raw_classes}
    for r in results:
        per_class[r["true"]]["total"] += 1
        per_class[r["true"]]["correct"] += r["correct"]

    summary = {
        "task": args.task, "input": args.input,
        "init_from": args.init_from,
        "n_volumes": len(results), "overall_accuracy": acc,
        "classes": raw_classes,
        "per_class": {c: {**v,
                          "accuracy": v["correct"] / max(v["total"], 1)}
                      for c, v in per_class.items()},
        "per_volume": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(
        f"\nLOO acc: {acc:.3f} "
        f"({sum(r['correct'] for r in results)}/{len(results)})")
    accelerator.print(f"Saved {args.output}")

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["overall"] + raw_classes
    ys = [acc] + [per_class[c]["correct"] / max(per_class[c]["total"], 1)
                  for c in raw_classes]
    ns = [len(results)] + [per_class[c]["total"] for c in raw_classes]
    ax.bar(xs, ys, color=["#4363d8"] + ["#3cb44b"] * len(raw_classes))
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + 0.02, f"{y:.2f}\n(n={n})", ha="center", fontsize=9)
    ax.axhline(1.0 / len(raw_classes), color="gray", linestyle=":")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{args.task} LOO | input={args.input} | "
                 f"init={'scratch' if not args.init_from else os.path.basename(os.path.dirname(args.init_from))}")
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    accelerator.print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
