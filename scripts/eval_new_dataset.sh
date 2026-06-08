#!/bin/bash
# Evaluate existing BF->GFP models on a NEW dataset (e.g. the 64-vol drop).
#
# Steps:
#   1. Compute per-volume percentile stats (skips existing).
#   2. Run BF->GFP regression eval (MAE / SSIM / Pearson) per vol, for the
#      best 2D and 3D checkpoints available. Auto-uses _holdEx and _holdPt
#      variants if present (they trained without the labeled vols, so they
#      are the cleanest transfer encoders).
#   3. Save per-slice masked predictions (BF foreground → kept; background → 0)
#      so artifacts at edges are zeroed.
#
# Usage:
#   DATA_DIR=data_new bash scripts/eval_new_dataset.sh
#
# Env knobs:
#   DATA_DIR        — new dataset root (default: data_new)
#   OUT_DIR         — eval JSON root  (default: results/eval_new_dataset)
#   PRED_DIR        — predictions root (default: predictions/new_dataset)
#   CKPTS_2D        — space-separated list of 2D ckpt paths (overrides auto-find)
#   CKPTS_3D        — same for 3D
#   MASK_METHOD     — minimum|otsu|li|triangle (default: minimum)
#   MASK_DILATE     — dilation iters (default: 3)
#   MASK_MIN_FRAC   — small-component cleanup (default: 0.01)
#   SKIP_PRED=1     — skip step 3 (predictions)
#   FORCE=1         — overwrite existing eval JSONs / stats / predictions

set -e

DATA_DIR="${DATA_DIR:-data_new}"
OUT_DIR="${OUT_DIR:-results/eval_new_dataset}"
PRED_DIR="${PRED_DIR:-predictions/new_dataset}"
MASK_METHOD="${MASK_METHOD:-minimum}"
MASK_DILATE="${MASK_DILATE:-3}"
MASK_MIN_FRAC="${MASK_MIN_FRAC:-0.01}"

if [ ! -d "$DATA_DIR/bf" ]; then
  echo "ERROR: $DATA_DIR/bf/ not found. Set DATA_DIR or stage the data." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$PRED_DIR"

# ────────────────────────────────────────────────────────────
# Step 1: percentile stats (skips existing unless FORCE=1)
# ────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 1: percentile stats (DATA_DIR=$DATA_DIR)"
echo "########################################"
STATS_ARGS=(--data_dir "$DATA_DIR")
[ -n "$FORCE" ] && STATS_ARGS+=(--force)
python compute_stats.py "${STATS_ARGS[@]}"

# ────────────────────────────────────────────────────────────
# Pick checkpoints: env overrides win, else auto-discover the best per arch.
# ────────────────────────────────────────────────────────────
if [ -z "$CKPTS_3D" ]; then
  CKPTS_3D=""
  for cand in ckpts/unet_3d_imagenet_pearson_frac100_holdEx/best.pth \
              ckpts/unet_3d_imagenet_pearson_frac100_holdPt/best.pth \
              ckpts/unet_3d_imagenet_pearson_frac100/best.pth; do
    [ -f "$cand" ] && CKPTS_3D="$CKPTS_3D $cand"
  done
fi
if [ -z "$CKPTS_2D" ]; then
  CKPTS_2D=""
  for cand in ckpts/unet_2d_imagenet_pearson_frac100_holdEx/best.pth \
              ckpts/unet_2d_imagenet_pearson_frac100_holdPt/best.pth \
              ckpts/unet_2d_imagenet_pearson_frac100/best.pth \
              ckpts/unet_2d_imagenet_pearson/best.pth; do
    [ -f "$cand" ] && CKPTS_2D="$CKPTS_2D $cand"
  done
fi

if [ -z "$CKPTS_3D" ] && [ -z "$CKPTS_2D" ]; then
  echo "ERROR: no 2D or 3D checkpoints found. Set CKPTS_2D/CKPTS_3D." >&2
  exit 1
fi

# ────────────────────────────────────────────────────────────
# Step 2: BF->GFP metrics on new vols
# ────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STEP 2: BF->GFP metrics on $DATA_DIR"
echo "########################################"
for CKPT in $CKPTS_3D $CKPTS_2D; do
  TAG=$(basename "$(dirname "$CKPT")")
  OUT="$OUT_DIR/${TAG}.json"
  if [ -f "$OUT" ] && [ -z "$FORCE" ]; then
    echo "skip: $OUT exists (FORCE=1 to redo)"
    continue
  fi
  echo "=== Eval $CKPT ==="
  python eval_bfgfp_metrics.py \
    --ckpt "$CKPT" --data_dir "$DATA_DIR" --output "$OUT"
done

# ────────────────────────────────────────────────────────────
# Step 3: masked per-slice predictions (one folder per ckpt)
# ────────────────────────────────────────────────────────────
if [ "${SKIP_PRED:-0}" != "1" ]; then
  echo ""
  echo "########################################"
  echo "# STEP 3: per-slice predictions (mask=$MASK_METHOD d=$MASK_DILATE r=$MASK_MIN_FRAC)"
  echo "########################################"
  for CKPT in $CKPTS_3D; do
    TAG=$(basename "$(dirname "$CKPT")")
    OUT="$PRED_DIR/${TAG}_masked"
    [ -d "$OUT" ] && [ -z "$FORCE" ] && { echo "skip: $OUT exists"; continue; }
    echo "=== Predict $CKPT ==="
    python predict_per_slice.py \
      --checkpoint "$CKPT" --data_dir "$DATA_DIR" --output_dir "$OUT" \
      --mask_background --mask_method "$MASK_METHOD" \
      --mask_dilate "$MASK_DILATE" --mask_min_frac "$MASK_MIN_FRAC"
  done
  for CKPT in $CKPTS_2D; do
    TAG=$(basename "$(dirname "$CKPT")")
    OUT="$PRED_DIR/${TAG}_masked"
    [ -d "$OUT" ] && [ -z "$FORCE" ] && { echo "skip: $OUT exists"; continue; }
    echo "=== Predict $CKPT ==="
    python predict_2d_per_slice.py \
      --checkpoint "$CKPT" --data_dir "$DATA_DIR" --output_dir "$OUT" \
      --mask_background --mask_method "$MASK_METHOD" \
      --mask_dilate "$MASK_DILATE" --mask_min_frac "$MASK_MIN_FRAC"
  done
fi

echo ""
echo "Done."
echo "Metrics : $OUT_DIR/"
echo "Predictions : $PRED_DIR/"
