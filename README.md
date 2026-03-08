# Myotube Fluorescence Prediction

Predict per-pixel fluorescence from brightfield image stacks using 3D U-Net models.

Given a stack of brightfield microscopy images of myotubes, this model predicts the corresponding fluorescence signal at each pixel -- enabling virtual staining without the need for fluorescent markers.

## Architecture

- **Encoder**: ResNeXt-50 (3D, pretrained from ImageNet 2D weights)
- **Decoder**: U-Net with skip connections and temporal kernel pooling
- **Task**: Per-pixel regression (brightfield -> fluorescence intensity)
- **Loss**: MSE + L1 (weighted combination)

## Installation

```bash
# Create environment
conda create --name=myo python=3.10
conda activate myo

# Install PyTorch for your system: https://pytorch.org/get-started/locally/
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Organize your data as paired `.npy` stacks:

```
data/
  brightfield/
    experiment1/
      sample_001.npy    # shape: (T, H, W) or (T, H, W, 1)
      sample_002.npy
    experiment2/
      ...
  fluorescence/
    experiment1/
      sample_001.npy    # same shape, matching filenames
      sample_002.npy
    experiment2/
      ...
```

Each `.npy` file should be a video stack with matching filenames between brightfield and fluorescence directories.

## Configuration

Edit `configs/myotube.yaml` to set:
- Data directories and file patterns
- Model architecture (encoder, decoder, channels)
- Training hyperparameters (lr, batch size, epochs, loss weights)
- Inference settings (checkpoint path, output directory)

## Training

```bash
python train.py --config=configs/myotube.yaml
```

Training features:
- Cosine annealing LR scheduler with warmup
- Mixed MSE + L1 loss for sharp predictions
- PSNR metric logging
- Full checkpoint saving (model + optimizer state for resuming)
- Multi-GPU support via HuggingFace Accelerate

## Inference

```bash
python predict.py --config=configs/myotube.yaml
```

Outputs predicted fluorescence `.npy` stacks to the configured output directory, with sliding-window temporal inference and spatial padding for arbitrary input sizes.

## Project Structure

```
myo/
  configs/
    myotube.yaml              # Main configuration
  src/
    dataloaders.py            # BrightfieldFluorescence dataset + legacy loaders
    transforms.py             # Video augmentations (crop, flip)
    video_functional.py       # Low-level video operations
    utils.py                  # Config, normalization, checkpoint loading
  segmentation_models_pytorch_3d/
    base/                     # SegmentationModel, heads, modules
    encoders/                 # ResNet/ResNeXt 3D encoders
    decoders/
      unet/                   # U-Net decoder
      unetplusplus/           # U-Net++ decoder
    losses/                   # Dice, Focal, Jaccard, etc.
    metrics/                  # IoU, F1, accuracy
    utils/                    # Training helpers, weight conversion
  train.py                    # Training entry point
  predict.py                  # Inference entry point
  requirements.txt
```

## Troubleshooting

**MPS (Apple Silicon) error**: If you see `NotImplementedError: The operator 'aten::max_pool3d_with_indices' is not currently implemented for the MPS device`, run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config=configs/myotube.yaml
```
