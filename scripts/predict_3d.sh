#!/bin/bash
# Per-slice 3D BF->GFP predictions for every volume in data/bf/, using the
# best frac100 checkpoint by default.
#
# Auto-resolves checkpoint preference order (override via CKPT=...):
#   1. ckpts/unet_3d_imagenet_pearson_frac100/best.pth        (true 100%)
#   2. ckpts/unet_3d_imagenet_pearson_frac100_holdEx/best.pth (excludes 8 exercise vols)
#   3. ckpts/unet_3d_imagenet_pearson_frac100_holdPt/best.pth (excludes 21 perturbation vols)
#   4. any other 3D U-Net frac100 ckpt
#   5. highest-fraction 3D U-Net ckpt available
#
# Usage:
#   bash scripts/predict_3d.sh
#   CKPT=ckpts/unet_3d_imagenet_pearson_frac050_holdEx/best.pth bash scripts/predict_3d.sh
#   OUT=predictions/3d_frac100 bash scripts/predict_3d.sh
#   STEMS="day12_093025_fixed_stim_t1_20x002" bash scripts/predict_3d.sh
#   DENORM=1 bash scripts/predict_3d.sh   # save in raw GFP scale, not [0,1]

set -e

OUT="${OUT:-predictions/3d_frac100}"

if [ -z "$CKPT" ]; then
  for cand in ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
              ckpts/unet_3d_imagenet_pearson_frac100_holdEx/best.pth \
              ckpts/unet_3d_imagenet_pearson_frac100_holdPt/best.pth \
              ckpts/unet_3d_*frac100*/best.pth \
              ckpts/unet_3d_*frac050*/best.pth \
              ckpts/unet_3d_*frac025*/best.pth \
              ckpts/unet_3d_*frac010*/best.pth \
              ckpts/unet_3d_*frac005*/best.pth; do
    if compgen -G "$cand" > /dev/null; then
      CKPT=$(ls $cand | head -1)
      break
    fi
  done
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no 3D checkpoint found. Set CKPT=path/to/best.pth." >&2
  exit 1
fi

CKPT_DIR=$(dirname "$CKPT")
CFG="${CFG:-$CKPT_DIR/config.yaml}"
# Fall back to the canonical config if the in-ckpt copy can't resolve `base:`.
if [ ! -f "$CFG" ]; then
  case "$(basename "$CKPT_DIR")" in
    *imagenet_pearson*) CFG=configs/unet_3d_imagenet_pearson.yaml ;;
    *imagenet*)         CFG=configs/unet_3d_imagenet.yaml ;;
    *random*)           CFG=configs/unet_3d_random.yaml ;;
    *)                  CFG=configs/unet_3d_imagenet_pearson.yaml ;;
  esac
fi

echo "Checkpoint : $CKPT"
echo "Config     : $CFG"
echo "Output dir : $OUT"

ARGS=(-c "$CFG" --checkpoint "$CKPT" --output_dir "$OUT")
if [ -n "$STEMS" ]; then ARGS+=(--stems $STEMS); fi
if [ -n "$DENORM" ]; then ARGS+=(--denormalize); fi

python predict_per_slice.py "${ARGS[@]}"

echo ""
echo "Done. Per-slice predictions under $OUT/{stem}/z{idx:04d}.npy"
