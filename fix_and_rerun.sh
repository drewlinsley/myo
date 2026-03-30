#!/bin/bash
set -e

# Fix frac100 symlink: best.pth instead of latest.pth
rm -f ckpts/unet_2d_imagenet_pearson_frac100/best.pth
mkdir -p ckpts/unet_2d_imagenet_pearson_frac100
ln -sf "$(realpath ckpts/unet_2d_imagenet_pearson/best.pth)" ckpts/unet_2d_imagenet_pearson_frac100/best.pth

# Clear old eval results so all metrics get recomputed
rm -f results/power_law/*.json

# Re-run eval (training skipped for existing checkpoints) + plot
bash run_power_law.sh

# Re-generate comparison montages (now includes 0% column)
python compare_predictions.py -c configs/unet_2d_imagenet_pearson.yaml --output_dir comparisons/
