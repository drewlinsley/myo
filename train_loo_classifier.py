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
from src.data.classification_dataset import GFPClassificationDataset, binarize
from src.data.grouping import stem_to_group
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
    if not encoder_state:
        raise RuntimeError(f"No encoder.* keys found in {ckpt_path}")
    # Detect 2D vs 3D from any conv weight (4D = 2D conv, 5D = 3D conv).
    sample_conv = next((v for k, v in encoder_state.items()
                        if "conv" in k.lower() and v.ndim >= 4), None)
    if sample_conv is not None:
        ckpt_dims = "3d" if sample_conv.ndim == 5 else "2d"
        cls_sample = next((p for n, p in classifier.encoder.named_parameters()
                           if "conv" in n.lower() and p.ndim >= 4), None)
        cls_dims = ("3d" if cls_sample.ndim == 5 else "2d") if cls_sample is not None else "?"
        if ckpt_dims != cls_dims:
            raise RuntimeError(
                f"Dim mismatch: ckpt {ckpt_path} is {ckpt_dims} but classifier "
                f"encoder is {cls_dims}. Use a checkpoint matching the LOO config.")
    # smp_3d's ResNet encoder overrides load_state_dict to do 2D->3D weight
    # conversion and doesn't propagate the return value, so we may get None
    # even on a successful load. Treat that as success.
    result = classifier.encoder.load_state_dict(
        encoder_state, strict=False)
    if result is None:
        return len(encoder_state), 0, 0
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
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Number of label permutations for p-value (0 to skip)")
    p.add_argument("--binarize", action="store_true",
                   help="Collapse raw labels to Control/Perturbed (yes/no)")
    p.add_argument("--collapse_doses", action="store_true",
                   help="Perturbation only: collapse 'high dose'/'low dose' "
                        "into a single class per perturbation agent. "
                        "Mutually exclusive with --binarize.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config seed for reproducibility / SE runs")
    p.add_argument("--cv_unit", choices=["volume", "replicate"],
                   default="volume",
                   help="Leave-one-group-out unit; replicate groups by "
                        "(label, Tissue) per colleague's recommendation")
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="Fraction of train_stems set aside as inner val for "
                        "early stopping (so the outer held-out group is "
                        "NEVER used to select epochs). 0 disables.")
    p.add_argument("--peek_val_for_earlystop", action="store_true",
                   help="(Legacy / debug) early-stop on held-out CE — leaks "
                        "the held-out label. Off by default.")
    args = p.parse_args()
    if args.binarize and args.collapse_doses:
        raise SystemExit("--binarize and --collapse_doses are mutually exclusive")

    cfg = load_config(args.config)
    cfg = validate_config(cfg)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False))

    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)  # "bf" or "gfp"

    metadata = load_metadata(args.metadata)
    label_col = TASK_LABEL_COL[args.task]

    def collapse_dose(s):
        """Strip 'high dose'/'low dose' from a perturbation label and bin
        any control variant (BSA / DMSO) into a single 'Control' class.
        Yields {Activin A, Control, Dexamethasone} for our 6 raw classes.
        """
        if s is None or s == "":
            return s
        s = str(s).strip()
        if "control" in s.lower():
            return "Control"
        for suffix in (" high dose", " low dose"):
            if s.endswith(suffix):
                return s[: -len(suffix)]
        return s

    # Collapse-doses preprocessing: mutate Perturbation labels in metadata so
    # both the dataset (use_raw_labels=True) and label_of() see the collapsed
    # value. Preserve the original under _perturbation_orig for grouping.
    if args.collapse_doses and args.task == "perturbation":
        for stem in metadata:
            v = metadata[stem].get("Perturbation")
            if v in (None, ""):
                continue
            metadata[stem]["_perturbation_orig"] = v
            metadata[stem]["Perturbation"] = collapse_dose(v)

    label_mode = ("binary" if args.binarize
                  else "collapsed" if args.collapse_doses
                  else "raw")

    def label_of(stem):
        v = metadata.get(stem, {}).get(label_col)
        if v in (None, ""):
            return None
        return binarize(v) if args.binarize else v

    # Filter to volumes with a label for this task
    all_stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob(os.path.join(mod_dir, "*.npy"))])
    stems = [s for s in all_stems if label_of(s) is not None]

    raw_classes = sorted({label_of(s) for s in stems})
    if args.binarize and "Control" in raw_classes:
        raw_classes.remove("Control")
        raw_classes = ["Control"] + raw_classes
    vocab = {"exercise": [], "perturbation": []}
    vocab[args.task] = raw_classes
    n_cls = max(len(raw_classes), 2)
    accelerator.print(f"task={args.task} input={args.input} "
                      f"label_mode={label_mode} cv_unit={args.cv_unit} "
                      f"n_volumes={len(stems)} "
                      f"n_classes={len(raw_classes)} classes={raw_classes}")
    if len(stems) < 2:
        raise SystemExit(f"Need >=2 {args.task} volumes, got {len(stems)}")

    # Group stems by CV unit (volume = current LOO behavior).
    groups = {}
    for s in stems:
        g = stem_to_group(s, metadata, args.cv_unit, args.task)
        if g is None:
            accelerator.print(
                f"  skip {s}: no group id for cv_unit={args.cv_unit}")
            continue
        groups.setdefault(g, []).append(s)
    if len(groups) < 2:
        raise SystemExit(
            f"cv_unit={args.cv_unit} produced {len(groups)} group(s) for "
            f"task={args.task}; need >= 2.")
    for g, members in groups.items():
        glabels = {label_of(s) for s in members}
        if len(glabels) > 1:
            raise SystemExit(
                f"Group {g} mixes labels {glabels}; group-LOO requires "
                "label-homogeneous groups.")
    accelerator.print(
        f"  groups ({len(groups)}): "
        + ", ".join(f"{g}[n={len(v)}]" for g, v in groups.items()))

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
            percentile_clip=percentile_clip,
            use_raw_labels=not args.binarize,
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
    for held_g, held_stems in groups.items():
        train_stems = [s for g, ss in groups.items() if g != held_g for s in ss]
        true_label = label_of(held_stems[0])  # homogeneous (validated above)
        true_idx = raw_classes.index(true_label)

        # Inner train/val split for early stopping (no peek at outer held-out)
        rng_split = np.random.default_rng(seed + hash(held_g) % (2 ** 31))
        train_stems_arr = np.array(train_stems)
        rng_split.shuffle(train_stems_arr)
        if args.peek_val_for_earlystop or args.inner_val_frac <= 0:
            inner_train = list(train_stems_arr)
            inner_val = []
        else:
            n_iv = max(1, int(round(len(train_stems_arr) * args.inner_val_frac)))
            inner_val = list(train_stems_arr[:n_iv])
            inner_train = list(train_stems_arr[n_iv:])

        accelerator.print(
            f"\n── LOO group: {held_g} (n_held={len(held_stems)}, "
            f"n_inner_train={len(inner_train)}, n_inner_val={len(inner_val)}, "
            f"true={true_label}) ──")

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
            make_ds(inner_train, True), batch_size=tcfg["batch_size"],
            shuffle=True, drop_last=True,
            num_workers=tcfg.get("num_workers", 4), pin_memory=True)
        predict_loader = torch.utils.data.DataLoader(
            make_ds(held_stems, False), batch_size=tcfg["batch_size"],
            shuffle=False, num_workers=tcfg.get("num_workers", 4))
        inner_val_loader = (torch.utils.data.DataLoader(
            make_ds(inner_val, False), batch_size=tcfg["batch_size"],
            shuffle=False, num_workers=tcfg.get("num_workers", 4))
            if inner_val else None)

        to_prep = [model, optimizer, train_loader, predict_loader]
        if inner_val_loader is not None:
            to_prep.append(inner_val_loader)
        prepared = accelerator.prepare(*to_prep)
        model, optimizer, train_loader, predict_loader = prepared[:4]
        if inner_val_loader is not None:
            inner_val_loader = prepared[4]

        def forward_head(logits_ex, logits_pt, ex, pt):
            if args.task == "exercise":
                return logits_ex, ex
            return logits_pt, pt

        def eval_loss_on(loader):
            """CE loss on a loader (uses labels) — for early-stop signal."""
            model.eval()
            total_loss, total_n = 0.0, 0
            with torch.no_grad():
                for img, ex, pt in loader:
                    lex, lpt = model(img)
                    logits, target = forward_head(lex, lpt, ex, pt)
                    if (target != -1).any():
                        total_loss += (
                            criterion(logits, target).item() * img.shape[0])
                        total_n += img.shape[0]
            return total_loss / max(total_n, 1)

        def predict_on_held():
            """Mean softmax over all slices/patches of held_stems. Uses NO
            held-out labels — pure prediction, safe to call every epoch."""
            model.eval()
            probs = torch.zeros(n_cls, device=device)
            n_samples = 0
            with torch.no_grad():
                for img, ex, pt in predict_loader:
                    lex, lpt = model(img)
                    logits = lex if args.task == "exercise" else lpt
                    probs += logits.softmax(dim=1).sum(dim=0)
                    n_samples += img.shape[0]
            return (probs / max(n_samples, 1)).cpu().numpy()

        best_sig = float("inf")
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

            # Early-stop signal selection
            if args.peek_val_for_earlystop:
                sig = eval_loss_on(predict_loader)   # LEAK (legacy)
                sig_name = "held_val_ce"
            elif inner_val_loader is not None:
                sig = eval_loss_on(inner_val_loader)
                sig_name = "inner_val_ce"
            else:
                sig = tr
                sig_name = "train_ce"

            probs = predict_on_held()   # no held-out labels touched

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                accelerator.print(
                    f"  ep{ep+1}/{epochs} train_ce={tr:.4f} "
                    f"{sig_name}={sig:.4f}")
            if sig < best_sig - min_delta:
                best_sig = sig
                best_probs = probs
                best_epoch = ep + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    accelerator.print(
                        f"  early stop ep{ep+1} "
                        f"({sig_name}={best_sig:.4f} @ ep{best_epoch})")
                    break

        pred_probs = best_probs if best_probs is not None else probs
        pred_idx = int(pred_probs.argmax())
        pred = raw_classes[pred_idx]
        correct = int(pred_idx == true_idx)
        accelerator.print(
            f"  pred={pred} (p={pred_probs[pred_idx]:.3f}) correct={correct}")
        results.append({
            "group": held_g, "stems": held_stems,
            "true": true_label, "pred": pred,
            "correct": correct, "best_epoch": best_epoch,
            "best_signal_loss": float(best_sig),
            "probs": {raw_classes[i]: float(pred_probs[i])
                      for i in range(n_cls)}})

    acc = float(np.mean([r["correct"] for r in results]))
    per_class = {c: {"correct": 0, "total": 0} for c in raw_classes}
    for r in results:
        per_class[r["true"]]["total"] += 1
        per_class[r["true"]]["correct"] += r["correct"]

    perm_info = None
    if args.n_permutations and len(results) > 1:
        rng = np.random.default_rng(0)
        true_labels = np.array([r["true"] for r in results])
        pred_labels = np.array([r["pred"] for r in results])
        perm_accs = np.empty(args.n_permutations, dtype=float)
        for i in range(args.n_permutations):
            shuffled = rng.permutation(true_labels)
            perm_accs[i] = float(np.mean(shuffled == pred_labels))
        n_ge = int(np.sum(perm_accs >= acc))
        p_value = (n_ge + 1) / (args.n_permutations + 1)
        perm_info = {
            "n_permutations": int(args.n_permutations),
            "p_value": float(p_value),
            "perm_mean": float(perm_accs.mean()),
            "perm_std": float(perm_accs.std()),
            "n_ge_observed": n_ge,
        }
        accelerator.print(
            f"Permutation test: p={p_value:.4f} "
            f"(perm mean={perm_accs.mean():.3f} "
            f"std={perm_accs.std():.3f}, n_ge={n_ge}/{args.n_permutations})")

    n_vols_total = sum(len(r["stems"]) for r in results)
    summary = {
        "task": args.task, "input": args.input,
        "init_from": args.init_from,
        "seed": int(seed),
        "cv_unit": args.cv_unit,
        "label_mode": label_mode,
        "n_classes": len(raw_classes),
        "chance": 1.0 / len(raw_classes) if raw_classes else 0.5,
        "n_groups": len(results),
        "n_volumes": n_vols_total,
        "overall_accuracy": acc,
        "permutation_test": perm_info,
        "classes": raw_classes,
        "per_class": {c: {**v,
                          "accuracy": v["correct"] / max(v["total"], 1)}
                      for c, v in per_class.items()},
        "per_group": results,
    }
    summary["config_flags"] = {
        "inner_val_frac": float(args.inner_val_frac),
        "peek_val_for_earlystop": bool(args.peek_val_for_earlystop),
    }
    if args.cv_unit == "volume":
        # Back-compat: also expose per_volume with the original key shape
        summary["per_volume"] = [
            {"stem": r["stems"][0], "true": r["true"], "pred": r["pred"],
             "correct": r["correct"], "best_epoch": r["best_epoch"],
             "best_val_loss": r["best_signal_loss"], "probs": r["probs"]}
            for r in results]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    accelerator.print(
        f"\nLOO acc: {acc:.3f} "
        f"({sum(r['correct'] for r in results)}/{len(results)} groups, "
        f"cv_unit={args.cv_unit})")
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
    ax.set_title(f"{args.task} LOO ({args.cv_unit}) | input={args.input} | "
                 f"init={'scratch' if not args.init_from else os.path.basename(os.path.dirname(args.init_from))}")
    fig.tight_layout()
    plot_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    accelerator.print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
