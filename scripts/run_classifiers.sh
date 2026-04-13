#!/bin/bash
# Run leave-one-out k-NN classification + regression for all available checkpoints.
#
# Auto-discovers best.pth under ckpts/*/ directories.
# Also runs the untrained (0%) baseline and the scaling plot.
#
# Usage: bash scripts/run_classifiers.sh

set -e

CONFIG=configs/unet_2d_imagenet_pearson.yaml
METADATA=data_mapping_drew.csv
OUT_DIR=results/classifier

mkdir -p "$OUT_DIR"

# ── 0% baseline (untrained) ──
echo "=== frac000: untrained baseline ==="
python train_classifiers.py \
    -c "$CONFIG" \
    --no_checkpoint \
    --metadata "$METADATA" \
    --seg_tag frac000 \
    --output "$OUT_DIR/frac000.json" \
    --output_dir "$OUT_DIR"

# ── Trained checkpoints ──
# Expects directories like ckpts/unet_2d_imagenet_pearson_frac025/best.pth
# Extracts the frac tag from the directory name.
for ckpt in ckpts/*/best.pth; do
    [ -f "$ckpt" ] || continue

    # Extract directory name, e.g. unet_2d_imagenet_pearson_frac025
    dir_name=$(basename "$(dirname "$ckpt")")

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
        --output_dir "$OUT_DIR"
done

# ── Scaling plot ──
echo "=== Plotting scaling curve ==="
python plot_classification_scaling.py \
    --results_dir "$OUT_DIR" \
    --output "$OUT_DIR/classification_scaling.png"

echo "Done! Results in $OUT_DIR/"
