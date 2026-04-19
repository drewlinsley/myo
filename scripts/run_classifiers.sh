#!/bin/bash
# Run leave-one-out k-NN classification + regression for all available checkpoints.
#
# Auto-discovers best.pth under ckpts/*/ directories.
# Also runs the untrained (0%) baseline and the scaling plot.
#
# Usage: bash scripts/run_classifiers.sh
#        bash scripts/run_classifiers.sh --mask_method minimum --mask_dilate 10

set -e

CONFIG=configs/unet_2d_imagenet_pearson.yaml
METADATA=data_mapping_drew.csv
OUT_DIR=results/classifier

# Pass through any extra args (e.g. --mask_method minimum --mask_dilate 10)
EXTRA_ARGS="$@"

mkdir -p "$OUT_DIR"

# ── GFP control (ImageNet-only weights) ──
echo "=== gfp_control: GFP through encoder (ImageNet) ==="
python train_classifiers.py \
    -c "$CONFIG" \
    --no_checkpoint \
    --gfp_control \
    --metadata "$METADATA" \
    --seg_tag gfp_control \
    --output "$OUT_DIR/gfp_control.json" \
    --output_dir "$OUT_DIR" \
    $EXTRA_ARGS

# ── 0% baseline (untrained) ──
echo "=== frac000: untrained baseline ==="
python train_classifiers.py \
    -c "$CONFIG" \
    --no_checkpoint \
    --metadata "$METADATA" \
    --seg_tag frac000 \
    --output "$OUT_DIR/frac000.json" \
    --output_dir "$OUT_DIR" \
    $EXTRA_ARGS

# ── Trained checkpoints ──
# Expects directories like ckpts/unet_2d_imagenet_pearson_frac025/best.pth
# Extracts the frac tag from the directory name.
for ckpt in ckpts/*/best.pth; do
    [ -f "$ckpt" ] || continue

    # Extract directory name, e.g. unet_2d_imagenet_pearson_frac025
    dir_name=$(basename "$(dirname "$ckpt")")

    # Skip GFP classifier checkpoints — they're not BF->GFP U-Nets
    if [[ "$dir_name" == gfp_classifier_* ]]; then
        continue
    fi

    # Try to pull out fracXXX tag
    if [[ "$dir_name" =~ (frac[0-9]+) ]]; then
        tag="${BASH_REMATCH[1]}"
    else
        # No frac tag — use full directory name
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

    # GFP control with trained encoder
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

# ── Scaling plot ──
echo "=== Plotting scaling curve ==="
python plot_classification_scaling.py \
    --results_dir "$OUT_DIR" \
    --output "$OUT_DIR/classification_scaling.png"

echo "Done! Results in $OUT_DIR/"
