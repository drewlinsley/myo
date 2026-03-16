#!/bin/bash
# Run prediction + evaluation only (no training).
# Usage: bash scripts/run_eval.sh <config> <checkpoint> <experiment_name>
# Example: bash scripts/run_eval.sh configs/unet_3d_imagenet_pearson.yaml ckpts/unet_3d_imagenet_pearson/best.pth unet_3d_imagenet_pearson

set -e

CONFIG=${1:?Usage: bash scripts/run_eval.sh <config> <checkpoint> <experiment_name>}
CHECKPOINT=${2:?Provide checkpoint path}
EXPERIMENT=${3:?Provide experiment name}
Z_RANGE="70 105"

echo "=== Predict: ${EXPERIMENT} ==="
python predict.py -c "$CONFIG" --checkpoint "$CHECKPOINT" --output_dir "predictions/${EXPERIMENT}"

echo "=== Evaluate: ${EXPERIMENT} ==="
python evaluate.py --pred_dir "predictions/${EXPERIMENT}" --data_dir data --output_dir "results/${EXPERIMENT}" --z_range $Z_RANGE

echo "Done: results/${EXPERIMENT}"
