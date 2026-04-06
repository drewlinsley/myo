#!/bin/bash
# Run leave-one-out k-NN classification on features from each seg-fraction checkpoint.
#
# Usage:
#   bash run_classifier_scaling.sh
#
# Outputs:
#   results/classifier/frac{000,001,025,050,100}.json
#   results/classifier/classification_scaling.png

set -e

CONFIG="configs/unet_2d_imagenet_pearson.yaml"
METADATA="data_mapping_drew.xlsx"
RESULTS_DIR="results/classifier"

FRAC_LIST="0.0 0.01 0.25 0.50 1.0"
TAG_LIST="frac000 frac001 frac025 frac050 frac100"

mkdir -p "$RESULTS_DIR"

read -ra FRACS <<< "$FRAC_LIST"
read -ra TAGS <<< "$TAG_LIST"

for i in "${!FRACS[@]}"; do
    FRAC="${FRACS[$i]}"
    SEG_TAG="${TAGS[$i]}"
    OUT_JSON="${RESULTS_DIR}/${SEG_TAG}.json"

    if [ -f "$OUT_JSON" ]; then
        echo "=== Skipping ${SEG_TAG} (${OUT_JSON} exists) ==="
        continue
    fi

    if [ "$FRAC" = "0.0" ]; then
        echo "=== k-NN on seg=${SEG_TAG} (untrained baseline) ==="
        python train_classifiers.py \
            -c "$CONFIG" \
            --no_checkpoint \
            --metadata "$METADATA" \
            --seg_tag "$SEG_TAG" \
            --output "$OUT_JSON"
    else
        CKPT="ckpts/unet_2d_imagenet_pearson_${SEG_TAG}/best.pth"
        echo "=== k-NN on seg=${SEG_TAG} (${CKPT}) ==="
        python train_classifiers.py \
            -c "$CONFIG" \
            --checkpoint "$CKPT" \
            --metadata "$METADATA" \
            --seg_tag "$SEG_TAG" \
            --output "$OUT_JSON"
    fi
done

echo "=== Plotting classification scaling ==="
python plot_classification_scaling.py \
    --results_dir "$RESULTS_DIR" \
    --output "${RESULTS_DIR}/classification_scaling.png"

echo "=== Done. Plot saved to ${RESULTS_DIR}/classification_scaling.png ==="
