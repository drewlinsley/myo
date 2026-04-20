#!/bin/bash
# End-to-end pipeline:
#   1. GFP control baseline (ImageNet-only encoder, no retrain needed)
#   2. Retrain scaling-law models across fractions (with masked loss)
#   3. Run classifiers on all trained checkpoints (incl. GFP control per-ckpt)
#
# Usage: bash scripts/run_full_pipeline.sh
#        bash scripts/run_full_pipeline.sh --mask_method minimum --mask_dilate 10
#
# Note: to enable masked training loss, the config YAML must have
#       `data.mask_method: minimum` (or otsu/li/triangle) set.

set -e

CONFIG=configs/unet_3d_imagenet_pearson.yaml
METADATA=data_mapping_drew.csv
OUT_DIR=results/classifier
FRACTIONS=(0.05 0.1 0.25 0.5 1.0)

EXTRA_ARGS="$@"

mkdir -p "$OUT_DIR"

# ──────────────────────────────────────────────────────────────
# Step 1: GFP classifier scaling sweep (end-to-end, per-slice,
#         two heads: Exercise + Perturbation)
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 1: GFP Classifier Scaling Sweep #"
echo "########################################"
GFP_CLS_CONFIG=configs/gfp_classifier.yaml
for frac in "${FRACTIONS[@]}"; do
    echo "=== GFP classifier fraction=$frac ==="
    python train_gfp_classifier.py \
        -c "$GFP_CLS_CONFIG" \
        --metadata "$METADATA" \
        --fraction "$frac"
done

echo "=== Plotting GFP classifier scaling ==="
python plot_gfp_classifier_scaling.py \
    --ckpt_glob "ckpts/gfp_classifier_frac*" \
    --output "$OUT_DIR/gfp_classifier_scaling.png"

# ──────────────────────────────────────────────────────────────
# Step 2: Retrain scaling-law models across fractions
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 2: Retrain Scaling Law Models   #"
echo "########################################"
for frac in "${FRACTIONS[@]}"; do
    echo "=== Training fraction=$frac ==="
    python train_fraction.py -c "$CONFIG" --fraction "$frac"
done

# ──────────────────────────────────────────────────────────────
# Step 3: LOO fine-tuning on BF-trained encoders
# (BF → {Exercise, Perturbation} per encoder checkpoint)
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 3: LOO Fine-tuning (per task)   #"
echo "########################################"
LOO_OUT=results/loo
mkdir -p "$LOO_OUT"

# Use the matching 3D GFP-classifier config for encoder arch
LOO_CFG=configs/gfp_classifier_3d.yaml

# 0% baseline: no init_from, starts from ImageNet-only encoder
for task in exercise perturbation; do
    echo "=== frac000 LOO ($task, BF) ==="
    python train_loo_classifier.py \
        -c "$LOO_CFG" \
        --metadata "$METADATA" \
        --task "$task" --input bf \
        --output "$LOO_OUT/frac000_${task}_bf.json"
done

# Fine-tune from each BF->GFP checkpoint (Step 2 outputs only; must match LOO_CFG dims)
for ckpt in ckpts/*frac*/best.pth; do
    [ -f "$ckpt" ] || continue
    dir_name=$(basename "$(dirname "$ckpt")")
    if [[ "$dir_name" == gfp_classifier* ]]; then
        continue
    fi
    if [[ "$dir_name" =~ (frac[0-9]+) ]]; then
        tag="${BASH_REMATCH[1]}"
    else
        tag="$dir_name"
    fi
    for task in exercise perturbation; do
        echo "=== $tag LOO ($task, BF, init=$ckpt) ==="
        python train_loo_classifier.py \
            -c "$LOO_CFG" \
            --metadata "$METADATA" \
            --task "$task" --input bf \
            --init_from "$ckpt" \
            --output "$LOO_OUT/${tag}_${task}_bf.json"
    done
done

# Scaling plot: LOO accuracy vs BF->GFP training fraction
echo "=== Plotting LOO scaling curve ==="
python plot_loo_scaling.py \
    --results_dir "$LOO_OUT" --input bf \
    --output "$OUT_DIR/loo_scaling_bf.png"

echo ""
echo "Pipeline complete! Results in $OUT_DIR/"
