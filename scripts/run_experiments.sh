#!/bin/bash
# Run all 4 experiments sequentially.
# Usage: bash scripts/run_experiments.sh

set -e

Z_RANGE="70 105"

echo "=== Experiment 1: 2D U-Net + ImageNet ==="
python train.py -c configs/unet_2d_imagenet.yaml
python predict.py -c configs/unet_2d_imagenet.yaml --checkpoint ckpts/unet_2d_imagenet/best.pth --output_dir predictions/unet_2d_imagenet
python evaluate.py --pred_dir predictions/unet_2d_imagenet --data_dir data --output_dir results/unet_2d_imagenet --z_range $Z_RANGE

echo "=== Experiment 2: 3D U-Net + ImageNet ==="
python train.py -c configs/unet_3d_imagenet.yaml
python predict.py -c configs/unet_3d_imagenet.yaml --checkpoint ckpts/unet_3d_imagenet/best.pth --output_dir predictions/unet_3d_imagenet
python evaluate.py --pred_dir predictions/unet_3d_imagenet --data_dir data --output_dir results/unet_3d_imagenet --z_range $Z_RANGE

echo "=== Experiment 3: 2D U-Net + Random ==="
python train.py -c configs/unet_2d_random.yaml
python predict.py -c configs/unet_2d_random.yaml --checkpoint ckpts/unet_2d_random/best.pth --output_dir predictions/unet_2d_random
python evaluate.py --pred_dir predictions/unet_2d_random --data_dir data --output_dir results/unet_2d_random --z_range $Z_RANGE

echo "=== Experiment 4: 3D U-Net + Random ==="
python train.py -c configs/unet_3d_random.yaml
python predict.py -c configs/unet_3d_random.yaml --checkpoint ckpts/unet_3d_random/best.pth --output_dir predictions/unet_3d_random
python evaluate.py --pred_dir predictions/unet_3d_random --data_dir data --output_dir results/unet_3d_random --z_range $Z_RANGE

echo "=== Experiment 5: pix2pix-turbo ==="
python train_pix2pix.py -c configs/pix2pix_turbo.yaml
python predict_pix2pix.py -c configs/pix2pix_turbo.yaml --checkpoint ckpts/pix2pix_turbo/best.pkl --output_dir predictions/pix2pix_turbo
python evaluate.py --pred_dir predictions/pix2pix_turbo --data_dir data --output_dir results/pix2pix_turbo --z_range $Z_RANGE

echo "=== Comparison ==="
python evaluate.py --compare results/unet_2d_imagenet results/unet_2d_random results/unet_3d_imagenet results/unet_3d_random results/pix2pix_turbo

echo "All experiments complete!"
