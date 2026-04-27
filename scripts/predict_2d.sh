#!/bin/bash
# Per-slice 2D BF->GFP predictions for every volume in data/bf/.
# Auto-resolves the best 2D checkpoint, with sensible env-var overrides.
#
# Usage:
#   bash scripts/predict_2d.sh
#   CKPT=ckpts/unet_2d_imagenet_pearson/best.pth bash scripts/predict_2d.sh
#   OUT=predictions/per_slice_v2 bash scripts/predict_2d.sh
#   STEMS="day12_093025_fixed_stim_t1_20x002 T26_101525__20x_" bash scripts/predict_2d.sh

set -e

OUT="${OUT:-predictions/per_slice}"

# Pick checkpoint: env override, else best 2D U-Net (Pearson > plain ImageNet > random),
# preferring frac100 / unsuffixed dirs.
if [ -z "$CKPT" ]; then
  for cand in ckpts/unet_2d_imagenet_pearson/best.pth \
              ckpts/unet_2d_imagenet_pearson_frac100/best.pth \
              ckpts/unet_2d_imagenet_pearson_*frac100*/best.pth \
              ckpts/unet_2d_imagenet/best.pth \
              ckpts/unet_2d_imagenet_*frac100*/best.pth \
              ckpts/unet_2d_random/best.pth; do
    if compgen -G "$cand" > /dev/null; then
      CKPT=$(ls $cand | head -1)
      break
    fi
  done
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no 2D checkpoint found. Set CKPT=path/to/best.pth." >&2
  exit 1
fi

# Use the config copy that train.py saved next to the checkpoint when present;
# otherwise fall back to the matching config in configs/.
CKPT_DIR=$(dirname "$CKPT")
CFG="${CFG:-$CKPT_DIR/config.yaml}"
if [ ! -f "$CFG" ]; then
  case "$(basename "$CKPT_DIR")" in
    *imagenet_pearson*) CFG=configs/unet_2d_imagenet_pearson.yaml ;;
    *imagenet*)         CFG=configs/unet_2d_imagenet.yaml ;;
    *random*)           CFG=configs/unet_2d_random.yaml ;;
    *)                  CFG=configs/unet_2d_imagenet_pearson.yaml ;;
  esac
fi

echo "Checkpoint : $CKPT"
echo "Config     : $CFG"
echo "Output dir : $OUT"

ARGS=(-c "$CFG" --checkpoint "$CKPT" --output_dir "$OUT")
if [ -n "$STEMS" ]; then
  ARGS+=(--stems $STEMS)
fi
if [ -n "$DENORM" ]; then
  ARGS+=(--denormalize)
fi

python predict_2d_per_slice.py "${ARGS[@]}"

echo ""
echo "Done. Per-slice predictions under $OUT/{stem}/z{idx:04d}.npy"
