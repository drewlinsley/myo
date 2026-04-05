"""Train XGBoost classifiers on encoder features at multiple data fractions.

For a given seg checkpoint, extracts volume-level encoder features, then trains
classifiers for each task (Exercise, Perturbation) at 10%, 25%, 50%, 75%, 100%
of the classifier training data. Uses repeated stratified shuffle splits for
robust accuracy estimates.

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

from glob import glob
from src.config import load_config
from src.models.factory import build_model
from src.utils import load_checkpoint
from src.data.normalization import normalize
from extract_features import (
    load_metadata, extract_encoder_features, DATASET_LABEL_COL
)


CLASSIFIER_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
N_SPLITS = 10      # repeated stratified shuffle splits
TEST_SIZE = 0.25   # fraction held out per split


def extract_volume_features_all(cfg, checkpoint, metadata_path, no_checkpoint,
                                layers, device):
    """Load seg model and extract volume-level features for every volume."""
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
    gfp_dir = os.path.join(data_dir, "gfp")
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
            gfp_raw = np.load(os.path.join(gfp_dir, f"{stem}.npy"))
            if z_range is not None:
                z_lo = max(0, z_range[0])
                z_hi = min(bf_raw.shape[0], z_range[1])
                bf_raw = bf_raw[z_lo:z_hi]
                gfp_raw = gfp_raw[z_lo:z_hi]

            bf = normalize(bf_raw, stats["bf"]["p_low"], stats["bf"]["p_high"],
                           apply_timm=apply_timm)

            feats = extract_encoder_features(model, bf, device, layers,
                                             batch_size=16)
            Z = feats.shape[0]
            nonzero_prop = np.array(
                [(gfp_raw[z] > 0).mean() for z in range(Z)], dtype=np.float32)

            # Aggregate to volume level
            vol_feat = feats.mean(axis=0)
            vol_feat = np.concatenate([vol_feat, [float(nonzero_prop.mean())]])

            vol_features.append(vol_feat)
            vol_meta.append(meta)
            print(f"  {stem}: Dataset={meta['Dataset']}, D={len(vol_feat)}")

    return np.array(vol_features, dtype=np.float32), vol_meta


def train_classifier_sweep(features, labels, fractions, n_splits, seed=42):
    """Repeated stratified shuffle splits + training-data subsampling."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    import xgboost as xgb

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        print("  <2 unique classes — skipping task")
        return None

    class_counts = np.bincount(y)
    min_count = int(class_counts.min())
    n_samples = len(y)

    # StratifiedShuffleSplit requires each split's test set to contain at least
    # one sample from each class; sklearn enforces test_size * n_samples >= n_classes.
    n_test = int(round(TEST_SIZE * n_samples))
    use_stratified = min_count >= 2 and n_test >= n_classes
    if not use_stratified:
        print(f"  WARNING: min class count={min_count}, n_test={n_test}, "
              f"n_classes={n_classes}; using unstratified splits")

    if use_stratified:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=TEST_SIZE, random_state=seed)
        splits = list(splitter.split(features, y))
    else:
        splitter = ShuffleSplit(
            n_splits=n_splits, test_size=TEST_SIZE, random_state=seed)
        splits = list(splitter.split(features))

    results = {
        "n_samples": int(len(y)),
        "n_classes": int(n_classes),
        "classes": [str(c) for c in le.classes_],
        "class_counts": [int(c) for c in class_counts],
        "per_fraction": {},
    }

    for frac in fractions:
        accs, n_trains = [], []
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            if frac < 1.0:
                rng = np.random.RandomState(seed + split_idx * 101)
                if use_stratified:
                    sub = []
                    for cls in np.unique(y[train_idx]):
                        cls_idx = train_idx[y[train_idx] == cls]
                        n_keep = max(1, int(round(len(cls_idx) * frac)))
                        n_keep = min(n_keep, len(cls_idx))
                        sub.extend(rng.choice(cls_idx, n_keep, replace=False))
                    train_sub = np.array(sub)
                else:
                    n_keep = max(n_classes, int(round(len(train_idx) * frac)))
                    n_keep = min(n_keep, len(train_idx))
                    train_sub = rng.choice(train_idx, n_keep, replace=False)
            else:
                train_sub = train_idx

            if len(np.unique(y[train_sub])) < 2:
                continue

            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                verbosity=0, eval_metric="mlogloss", n_jobs=1,
            )
            clf.fit(features[train_sub], y[train_sub])
            acc = clf.score(features[test_idx], y[test_idx])
            accs.append(float(acc))
            n_trains.append(int(len(train_sub)))

        if accs:
            results["per_fraction"][f"{frac:.2f}"] = {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "n_splits": len(accs),
                "n_train_mean": float(np.mean(n_trains)),
            }
            print(f"  clf_frac={frac:.2f}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                  f"(n_train≈{int(np.mean(n_trains))}, {len(accs)} splits)")
        else:
            print(f"  clf_frac={frac:.2f}: all splits degenerate, skipped")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifiers on seg-encoder features")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use untrained model (0%% seg baseline)")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--seg_tag", required=True,
                        help="Label for this seg model, e.g. frac025")
    parser.add_argument("--layers", type=int, nargs="+", default=[5],
                        help="Encoder stages to extract (default: [5])")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.no_checkpoint and args.checkpoint is None:
        parser.error("Either --checkpoint or --no_checkpoint is required")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Extracting features for seg_tag={args.seg_tag}...")
    features, vol_meta = extract_volume_features_all(
        cfg, args.checkpoint, args.metadata, args.no_checkpoint,
        args.layers, device)

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
        ds_labels = [vol_meta[i][label_col] for i in range(len(vol_meta))
                     if mask[i]]

        print(f"\n── Task: {ds.capitalize()} ({label_col}) — {int(mask.sum())} volumes ──")
        results = train_classifier_sweep(
            ds_features, ds_labels, CLASSIFIER_FRACTIONS, args.n_splits, args.seed)
        if results is not None:
            out["tasks"][ds] = {"label_col": label_col, **results}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
