#!/bin/bash
set -e

# Fix frac100 symlink: best.pth instead of latest.pth
rm -f ckpts/unet_2d_imagenet_pearson_frac100/best.pth
mkdir -p ckpts/unet_2d_imagenet_pearson_frac100
ln -sf "$(realpath ckpts/unet_2d_imagenet_pearson/best.pth)" ckpts/unet_2d_imagenet_pearson_frac100/best.pth

# Clear old eval results
rm -f results/power_law/*.json

# Re-run eval + plot
bash run_power_law.sh

# Re-generate comparison montages
python compare_predictions.py -c configs/unet_2d_imagenet_pearson.yaml --output_dir comparisons/
