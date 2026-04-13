"""Classify volumes via leave-one-out k-NN on encoder features.

For a given seg checkpoint, extracts volume-level encoder features, then
runs leave-one-out k-NN classification for each task (Exercise, Perturbation).
No train/test split — each volume is classified by its nearest neighbors
among all other volumes.  Truly zero-shot w.r.t. classifier training data.

Usage:
    python train_classifiers.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --checkpoint ckpts/unet_2d_imagenet_pearson_frac025/best.pth \
        --metadata data_mapping_drew.xlsx \
        --seg_tag frac025 \
        --output results/classifier/frac025.json

    # For untrained (0%) baseline:
    python train_classifiers.py \
        -c configs/unet_2d_imagenet_pearson.yaml \
        --no_checkpoint \
        --metadata data_mapping_drew.xlsx \
        --seg_tag frac000 \
        --output results/classifier/frac000.json
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint
from src.data.normalization import normalize
from extract_features import (
    load_metadata, extract_encoder_features, DATASET_LABEL_COL
)


K_VALUES = [1, 3, 5]

# Labels that map to "Control" (case-insensitive).  Everything else → "Perturbed".
CONTROL_KEYWORDS = {"control", "ctrl", "no", "none", "untreated", "vehicle",
                    "unstimulated", "baseline", "wt", "wild type", "dmso"}


def binarize_labels(labels):
    """Map raw labels to binary Control / Perturbed.

    Uses word-level matching so multi-word labels like "DMSO control"
    are caught when any word overlaps CONTROL_KEYWORDS.
    """
    out = []
    for l in labels:
        words = set(l.strip().lower().split())
        if words & CONTROL_KEYWORDS:
            out.append("Control")
        else:
            out.append("Perturbed")
    return out


def extract_volume_features_all(cfg, checkpoint, metadata_path, no_checkpoint,
                                layers, device, mask_percentile=10):
    """Load seg model and extract volume-level features for every volume.

    Args:
        mask_percentile: percentile of raw BF intensity used as foreground
            threshold.  Pixels below this percentile are considered background
            and excluded from spatial pooling.  Set to 0 to disable masking.
    """
    apply_timm = cfg["model"].get("encoder_weights") is not None

    if no_checkpoint:
        model = build_model(cfg)
        print("Using untrained seg model (0% baseline)")
    else:
        cfg_copy = cfg.copy()
        cfg_copy["model"] = cfg["model"].copy()
        cfg_copy["model"]["encoder_weights"] = None
        model = build_model(cfg_copy)
        ckpt = load_checkpoint(checkpoint, model)
        print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch', '?')})")

    model = model.to(device)
    model.eval()

    metadata = load_metadata(metadata_path)

    data_dir = cfg["data"]["data_dir"]
    bf_dir = os.path.join(data_dir, "bf")
    stats_dir = os.path.join(data_dir, "stats")
    z_range = cfg["data"].get("z_range", None)

    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))

    vol_features = []
    vol_meta = []

    with torch.no_grad():
        for bf_path in bf_files:
            stem = os.path.splitext(os.path.basename(bf_path))[0]
            if stem not in metadata:
                warnings.warn(f"Stem '{stem}' not in metadata — skipping")
                continue
            meta = metadata[stem]

            stats_path = os.path.join(stats_dir, f"{stem}.json")
            with open(stats_path) as f:
                stats = json.load(f)

            bf_raw = np.load(bf_path)
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]

            # Foreground mask from raw BF (before normalization)
            if mask_percentile > 0:
                thresh = np.percentile(bf_raw, mask_percentile)
                bf_mask = bf_raw > thresh  # (Z, H, W) bool
                bg_count = int((~bf_mask).sum())
                fg_count = int(bf_mask.sum())
            else:
                bf_mask = None
                bg_count = 0
                fg_count = int(bf_raw.size)

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            feats = extract_encoder_features(model, bf, device, layers,
                                             batch_size=16, mask=bf_mask)

            # Aggregate to volume level (encoder features only — no GFP
            # covariates, so we measure purely what the encoder represents)
            vol_feat = feats.mean(axis=0)

            vol_features.append(vol_feat)
            meta["_bg_pixels"] = bg_count
            meta["_fg_pixels"] = fg_count
            meta["_bg_frac"] = bg_count / max(bg_count + fg_count, 1)
            meta["_stem"] = stem
            vol_meta.append(meta)
            print(f"  {stem}: Dataset={meta['Dataset']}, D={len(vol_feat)}, "
                  f"bg={meta['_bg_frac']:.1%}")

    return np.array(vol_features, dtype=np.float32), vol_meta


def knn_leave_one_out(features, labels, k_values, metric="cosine"):
    """Leave-one-out k-NN classification at multiple k values.

    For each sample, finds its nearest neighbors among all OTHER samples
    and predicts the majority label.  No train/test split needed.

    Args:
        features: (N, D) array
        labels: list of N label strings
        k_values: list of k values to evaluate
        metric: "cosine" or "euclidean"

    Returns:
        dict with results per k value, or None if <2 classes.
    """
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        print("  <2 unique classes — skipping task")
        return None

    class_counts = np.bincount(y)
    N = len(y)

    # Compute pairwise distance matrix
    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features / norms
        sim = normed @ normed.T          # (N, N) cosine similarity
        dist = 1.0 - sim
    else:
        # Euclidean
        diff = features[:, None, :] - features[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Set self-distance to infinity so a sample can't be its own neighbor
    np.fill_diagonal(dist, np.inf)

    # Nearest neighbor indices sorted by distance
    nn_indices = np.argsort(dist, axis=1)  # (N, N)

    results = {
        "n_samples": int(N),
        "n_classes": int(n_classes),
        "classes": [str(c) for c in le.classes_],
        "class_counts": [int(c) for c in class_counts],
        "metric": metric,
        "per_k": {},
    }

    max_k = max(k_values)
    for k in k_values:
        if k > N - 1:
            print(f"  k={k}: not enough samples (N={N}), skipping")
            continue

        correct = 0
        per_sample = []
        for i in range(N):
            neighbors = nn_indices[i, :k]
            neighbor_labels = y[neighbors]
            counts = Counter(neighbor_labels)
            pred = counts.most_common(1)[0][0]
            is_correct = int(pred == y[i])
            correct += is_correct
            per_sample.append({
                "true": str(le.classes_[y[i]]),
                "pred": str(le.classes_[pred]),
                "correct": is_correct,
            })

        acc = correct / N
        results["per_k"][str(k)] = {
            "k": k,
            "accuracy": float(acc),
            "n_correct": int(correct),
            "n_total": int(N),
            "per_sample": per_sample,
        }
        print(f"  k={k}: acc={acc:.3f} ({correct}/{N})")

    return results


def _compute_distance_matrix(features, metric="cosine"):
    """Compute pairwise distance matrix with self-distance = inf."""
    if metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features / norms
        sim = normed @ normed.T
        dist = 1.0 - sim
    else:
        diff = features[:, None, :] - features[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return dist


def knn_leave_one_out_regression(features, targets, k_values,
                                 metric="cosine", stems=None, labels=None):
    """Leave-one-out k-NN regression at multiple k values.

    Prediction = mean of k nearest neighbors' target values.

    Note: same-tissue FOVs share force values, so a volume's nearest
    neighbor may be its same-tissue sibling, inflating R².

    Returns:
        dict with R², Pearson r, MAE, RMSE per k, or None if <3 samples.
    """
    N = len(targets)
    if N < 3:
        print(f"  <3 regression samples (N={N}) — skipping")
        return None

    targets = np.asarray(targets, dtype=np.float64)
    dist = _compute_distance_matrix(features, metric)
    nn_indices = np.argsort(dist, axis=1)

    results = {
        "n_samples": int(N),
        "target_col": "peak_amplitude_week_5",
        "metric": metric,
        "per_k": {},
    }

    ss_tot = np.sum((targets - targets.mean()) ** 2)

    for k in k_values:
        if k > N - 1:
            print(f"  k={k}: not enough samples (N={N}), skipping")
            continue

        preds = np.zeros(N)
        for i in range(N):
            neighbors = nn_indices[i, :k]
            preds[i] = targets[neighbors].mean()

        ss_res = np.sum((targets - preds) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        corr_matrix = np.corrcoef(targets, preds)
        pearson_r = float(np.nan_to_num(corr_matrix[0, 1]))
        mae = float(np.mean(np.abs(targets - preds)))
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))

        per_sample = []
        for i in range(N):
            entry = {"true": float(targets[i]), "pred": float(preds[i])}
            if stems is not None:
                entry["stem"] = stems[i]
            if labels is not None:
                entry["label"] = labels[i]
            per_sample.append(entry)

        results["per_k"][str(k)] = {
            "k": k,
            "r2": float(r2),
            "pearson_r": pearson_r,
            "mae": mae,
            "rmse": rmse,
            "per_sample": per_sample,
        }
        print(f"  k={k}: R²={r2:.3f}, r={pearson_r:.3f}, "
              f"MAE={mae:.3f}, RMSE={rmse:.3f}")

    return results


# Colors for scatter plots (matches extract_features.COLORS)
_SCATTER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9",
]


def plot_regression_results(results_per_k, labels_raw, stems, output_path):
    """Scatter plot of predicted vs actual force for best k (highest R²)."""
    best_k = max(results_per_k.keys(),
                 key=lambda k: results_per_k[k]["r2"])
    best = results_per_k[best_k]

    trues = np.array([s["true"] for s in best["per_sample"]])
    preds = np.array([s["pred"] for s in best["per_sample"]])

    unique_labels = sorted(set(labels_raw))
    label_to_color = {lab: _SCATTER_COLORS[i % len(_SCATTER_COLORS)]
                      for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in unique_labels:
        mask = np.array([l == lab for l in labels_raw])
        ax.scatter(trues[mask], preds[mask], c=label_to_color[lab],
                   label=lab, s=50, alpha=0.85, edgecolors="white",
                   linewidths=0.5)

    lo = min(trues.min(), preds.min()) * 0.9
    hi = max(trues.max(), preds.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.7)

    textstr = (f"k={best['k']}  R²={best['r2']:.3f}\n"
               f"r={best['pearson_r']:.3f}  RMSE={best['rmse']:.2f}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Actual peak_amplitude_week_5")
    ax.set_ylabel("Predicted peak_amplitude_week_5")
    ax.set_title("k-NN LOO Regression: Force Prediction")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_feature_force_correlation(features, targets, output_path):
    """Horizontal bar plot of top-20 feature dims most correlated with force."""
    D = features.shape[1]
    corrs = np.array([np.corrcoef(features[:, d], targets)[0, 1]
                      for d in range(D)])
    corrs = np.nan_to_num(corrs)

    top_idx = np.argsort(np.abs(corrs))[::-1][:20]
    top_corrs = corrs[top_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e6194b" if c < 0 else "#3cb44b" for c in top_corrs]
    y_pos = np.arange(len(top_idx))
    ax.barh(y_pos, top_corrs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"dim {i}" for i in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r with peak_amplitude_week_5")
    ax.set_title("Top 20 Feature Dimensions by Force Correlation")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_force_by_group(labels_raw, targets, output_path):
    """Box + strip plot of peak_amplitude_week_5 grouped by perturbation label."""
    unique_labels = sorted(set(labels_raw))
    label_to_color = {lab: _SCATTER_COLORS[i % len(_SCATTER_COLORS)]
                      for i, lab in enumerate(unique_labels)}

    # Group targets by label
    grouped = {lab: [] for lab in unique_labels}
    for lab, val in zip(labels_raw, targets):
        grouped[lab].append(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(unique_labels)))
    box_data = [grouped[lab] for lab in unique_labels]

    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False)
    for patch, lab in zip(bp["boxes"], unique_labels):
        patch.set_facecolor(label_to_color[lab])
        patch.set_alpha(0.4)

    # Overlay individual points
    for i, lab in enumerate(unique_labels):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grouped[lab]))
        ax.scatter(np.full(len(grouped[lab]), i) + jitter, grouped[lab],
                   c=label_to_color[lab], s=40, alpha=0.9, edgecolors="white",
                   linewidths=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(unique_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("peak_amplitude_week_5 (force)")
    ax.set_title("Contractile Force by Perturbation Group")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_classification_summary(results, ds_labels_raw, ds_labels, output_path):
    """Bar chart of k-NN accuracy per k + confusion-style per-class accuracy."""
    per_k = results.get("per_k", {})
    if not per_k:
        return

    k_vals = sorted(per_k.keys(), key=int)
    accs = [per_k[k]["accuracy"] for k in k_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy per k
    ax = axes[0]
    colors_k = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(k_vals)))
    ax.bar(range(len(k_vals)), accs, color=colors_k, edgecolor="white")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("LOO Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification Accuracy by k")
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=10)
    # Chance line
    n_classes = results.get("n_classes", 2)
    if n_classes > 1:
        ax.axhline(1.0 / n_classes, color="#999", linestyle=":", alpha=0.7,
                    label="chance")
        ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: per-class accuracy for best k
    ax = axes[1]
    best_k = max(k_vals, key=lambda k: per_k[k]["accuracy"])
    samples = per_k[best_k]["per_sample"]
    classes = sorted(set(s["true"] for s in samples))
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}
    for s in samples:
        class_total[s["true"]] += 1
        class_correct[s["true"]] += s["correct"]

    class_acc = [class_correct[c] / max(class_total[c], 1) for c in classes]
    class_colors = [_SCATTER_COLORS[i % len(_SCATTER_COLORS)]
                    for i in range(len(classes))]
    ax.bar(range(len(classes)), class_acc, color=class_colors, edgecolor="white")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class Accuracy (k={best_k})")
    for i, (acc, c) in enumerate(zip(class_acc, classes)):
        ax.text(i, acc + 0.02, f"{acc:.2f}\n(n={class_total[c]})",
                ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out k-NN classification on seg-encoder features")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use untrained model (0%% seg baseline)")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--seg_tag", required=True,
                        help="Label for this seg model, e.g. frac025")
    parser.add_argument("--layers", type=int, nargs="+", default=[5],
                        help="Encoder stages to extract (default: [5])")
    parser.add_argument("--k_values", type=int, nargs="+", default=K_VALUES,
                        help="k values for k-NN (default: 1 3 5)")
    parser.add_argument("--mask_percentile", type=float, default=50,
                        help="BF intensity percentile for foreground mask (0=no mask)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"],
                        default="cosine",
                        help="Distance metric (default: cosine)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir", default=None,
                        help="Directory for plots (default: same dir as --output)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.output) or "."

    if not args.no_checkpoint and args.checkpoint is None:
        parser.error("Either --checkpoint or --no_checkpoint is required")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Extracting features for seg_tag={args.seg_tag}...")
    features, vol_meta = extract_volume_features_all(
        cfg, args.checkpoint, args.metadata, args.no_checkpoint,
        args.layers, device, mask_percentile=args.mask_percentile)

    if features.size == 0 or features.ndim < 2:
        raise RuntimeError(
            f"No volumes matched metadata for seg_tag={args.seg_tag}. "
            "Check --metadata path and dataset stems."
        )

    out = {
        "seg_tag": args.seg_tag,
        "checkpoint": args.checkpoint or "none (untrained)",
        "feature_dim": int(features.shape[1]),
        "n_volumes": int(features.shape[0]),
        "tasks": {},
    }

    datasets_seen = sorted(set(m["Dataset"].lower() for m in vol_meta))
    for ds in datasets_seen:
        label_col = DATASET_LABEL_COL.get(ds)
        if label_col is None:
            warnings.warn(f"Unknown dataset '{ds}' — skipping")
            continue

        mask = np.array([m["Dataset"].lower() == ds for m in vol_meta])
        ds_features = features[mask]
        ds_labels_raw = [vol_meta[i][label_col] for i in range(len(vol_meta))
                         if mask[i]]
        ds_meta = [vol_meta[i] for i in range(len(vol_meta)) if mask[i]]

        # Binarize: Control vs Perturbed
        ds_labels = binarize_labels(ds_labels_raw)

        print(f"\n── Task: {ds.capitalize()} ({label_col}) — {int(mask.sum())} volumes ──")
        print(f"  Raw labels: {sorted(set(ds_labels_raw))}")
        print(f"  Binary:     {dict(zip(*np.unique(ds_labels, return_counts=True)))}")

        # ── Control: background pixel count per class ──
        from collections import defaultdict
        bg_by_class = defaultdict(list)
        for label, meta in zip(ds_labels, ds_meta):
            bg_by_class[label].append(meta["_bg_frac"])
        print(f"  CONTROL — background fraction (mask=0) per {label_col}:")
        bg_control = {}
        for cls in sorted(bg_by_class.keys()):
            vals = np.array(bg_by_class[cls])
            print(f"    {cls}: bg_frac = {vals.mean():.3f} ± {vals.std():.3f}  "
                  f"(n={len(vals)})")
            bg_control[cls] = {
                "mean_bg_frac": float(vals.mean()),
                "std_bg_frac": float(vals.std()),
                "n": len(vals),
            }

        results = knn_leave_one_out(
            ds_features, ds_labels, args.k_values, args.metric)
        if results is not None:
            results["bg_control"] = bg_control
            out["tasks"][ds] = {"label_col": label_col, **results}

            # Classification summary plot
            os.makedirs(args.output_dir, exist_ok=True)
            cls_plot_path = os.path.join(
                args.output_dir,
                f"{args.seg_tag}_{ds}_classification.png")
            plot_classification_summary(
                results, ds_labels_raw, ds_labels, cls_plot_path)

    # ── Regression on peak_amplitude_week_5 ────────────────────────
    out["regression"] = {}
    # Check if any volume has force data
    has_force = any(
        isinstance(m.get("peak_amplitude_week_5"), (int, float))
        for m in vol_meta
    )
    if has_force:
        os.makedirs(args.output_dir, exist_ok=True)
        for ds in datasets_seen:
            label_col = DATASET_LABEL_COL.get(ds)
            if label_col is None:
                continue

            mask = np.array([m["Dataset"].lower() == ds for m in vol_meta])
            ds_meta = [vol_meta[i] for i in range(len(vol_meta)) if mask[i]]

            # Filter to samples with valid (non-None) force values
            valid_idx = []
            targets = []
            stems = []
            labels_raw = []
            for j, meta in enumerate(ds_meta):
                force = meta.get("peak_amplitude_week_5")
                if isinstance(force, (int, float)):
                    valid_idx.append(j)
                    targets.append(float(force))
                    stems.append(meta.get("_stem", ""))
                    labels_raw.append(meta.get(label_col, ""))

            if len(targets) < 3:
                print(f"\n── Regression: {ds} — <3 valid force values, skipping ──")
                continue

            ds_features = features[mask]
            reg_features = ds_features[valid_idx]
            targets_arr = np.array(targets)

            print(f"\n── Regression: {ds} — {len(targets)} volumes with force data ──")
            reg_results = knn_leave_one_out_regression(
                reg_features, targets_arr, args.k_values, args.metric,
                stems=stems, labels=labels_raw)

            if reg_results is not None:
                out["regression"][ds] = reg_results

                # Scatter plot: predicted vs actual
                scatter_path = os.path.join(
                    args.output_dir,
                    f"{args.seg_tag}_{ds}_regression_scatter.png")
                plot_regression_results(
                    reg_results["per_k"], labels_raw, stems, scatter_path)

                # Force by group box plot
                force_path = os.path.join(
                    args.output_dir,
                    f"{args.seg_tag}_{ds}_force_by_group.png")
                plot_force_by_group(labels_raw, targets, force_path)

                # Feature-force correlation bar plot
                corr_path = os.path.join(
                    args.output_dir,
                    f"{args.seg_tag}_{ds}_feature_force_corr.png")
                plot_feature_force_correlation(
                    reg_features, targets_arr, corr_path)
    else:
        print("\nNo force data found in metadata — skipping regression.")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
