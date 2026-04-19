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

CONFIG=configs/unet_2d_imagenet_pearson.yaml
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
# Step 3: Run classifiers on untrained + all trained checkpoints
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 3: Run Classifiers              #"
echo "########################################"

# 0% baseline (untrained — BF through ImageNet encoder)
echo "=== frac000: untrained baseline ==="
python train_classifiers.py \
    -c "$CONFIG" \
    --no_checkpoint \
    --metadata "$METADATA" \
    --seg_tag frac000 \
    --output "$OUT_DIR/frac000.json" \
    --output_dir "$OUT_DIR" \
    $EXTRA_ARGS

# Trained checkpoints: BF and GFP-control variants
for ckpt in ckpts/*/best.pth; do
    [ -f "$ckpt" ] || continue
    dir_name=$(basename "$(dirname "$ckpt")")
    if [[ "$dir_name" =~ (frac[0-9]+) ]]; then
        tag="${BASH_REMATCH[1]}"
    else
        tag="$dir_name"
    fi

    echo "=== $tag: $ckpt ==="
    python train_classifiers.py \
        -c "$CONFIG" \
        --checkpoint "$ckpt" \
        --metadata "$METADATA" \
        --seg_tag "$tag" \
        --output "$OUT_DIR/${tag}.json" \
        --output_dir "$OUT_DIR" \
        $EXTRA_ARGS

    echo "=== ${tag}_gfp: GFP through trained encoder ==="
    python train_classifiers.py \
        -c "$CONFIG" \
        --checkpoint "$ckpt" \
        --gfp_control \
        --metadata "$METADATA" \
        --seg_tag "${tag}_gfp" \
        --output "$OUT_DIR/${tag}_gfp.json" \
        --output_dir "$OUT_DIR" \
        $EXTRA_ARGS
done

# Scaling plot
echo "=== Plotting scaling curve ==="
python plot_classification_scaling.py \
    --results_dir "$OUT_DIR" \
    --output "$OUT_DIR/classification_scaling.png"

echo ""
echo "Pipeline complete! Results in $OUT_DIR/"
