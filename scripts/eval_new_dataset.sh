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
#
# Optional staging + classification:
#   STAGE_ND2=1     — run stage_nd2.py first (ND2 → bf/+gfp/.npy) using
#                     ND2_SRC (recursive root) and the default channel layout
#                     ND2_BF_CH=0, ND2_TGT_CH=-1 (last channel). DATA_DIR
#                     becomes the staging OUT.
#   CLASSIFY=1      — also run predict_classifier.py per task. Knobs:
#                     CLASSIFIER_CKPT     single .pth (e.g. from train_gfp_classifier.py)
#                     CLASSIFIER_CKPT_GLOB shell glob for LOO ensemble (e.g.
#                                          "ckpts/loo_classifier_3d_pt/*/best.pth"),
#                                          REQUIRES train_loo_classifier.py to have
#                                          been run with --save_ckpt_dir.
#                     CLS_TASKS="exercise perturbation" (default),
#                     CLS_N_CLASSES=2, CLS_OUT_DIR="results/classify_new_dataset".
#   REGRESS=1       — run predict_regression.py (force amplitude). Knobs:
#                     REGRESSOR_CKPT      single .pth
#                     REGRESSOR_CKPT_GLOB e.g. "ckpts/loo_regression_3d/*/best.pth"
#                                          REQUIRES train_loo_regression.py to
#                                          have been run with --save_ckpt_dir.
#                     REG_OUT_DIR="results/regress_new_dataset".

set -e

DATA_DIR="${DATA_DIR:-data_new}"
OUT_DIR="${OUT_DIR:-results/eval_new_dataset}"
PRED_DIR="${PRED_DIR:-predictions/new_dataset}"
MASK_METHOD="${MASK_METHOD:-minimum}"
MASK_DILATE="${MASK_DILATE:-3}"
MASK_MIN_FRAC="${MASK_MIN_FRAC:-0.01}"

# ────────────────────────────────────────────────────────────
# Step 0 (optional): stage ND2 → bf/ + gfp/
# ────────────────────────────────────────────────────────────
if [ "${STAGE_ND2:-0}" = "1" ]; then
  if [ -z "$ND2_SRC" ]; then
    echo "ERROR: STAGE_ND2=1 requires ND2_SRC=<root containing .nd2 files>" >&2
    exit 1
  fi
  echo ""
  echo "########################################"
  echo "# STEP 0: stage ND2 (src=$ND2_SRC -> $DATA_DIR)"
  echo "########################################"
  STAGE_ARGS=(--src "$ND2_SRC" --out "$DATA_DIR"
              --bf_channel "${ND2_BF_CH:-0}"
              --target_channel "${ND2_TGT_CH:--1}")
  [ -n "$FORCE" ] && STAGE_ARGS+=(--force)
  python stage_nd2.py "${STAGE_ARGS[@]}"
fi

if [ ! -d "$DATA_DIR/bf" ]; then
  echo "ERROR: $DATA_DIR/bf/ not found. Set DATA_DIR, run STAGE_ND2=1, or stage manually." >&2
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

# ────────────────────────────────────────────────────────────
# Step 4 (optional): classifier predictions per task
# ────────────────────────────────────────────────────────────
if [ "${CLASSIFY:-0}" = "1" ]; then
  CLS_OUT_DIR="${CLS_OUT_DIR:-results/classify_new_dataset}"
  CLS_TASKS="${CLS_TASKS:-exercise perturbation}"
  CLS_N_CLASSES="${CLS_N_CLASSES:-2}"
  # Resolve checkpoint source: glob ensemble > single ckpt > auto-find.
  CLS_CKPT_FLAG=""
  CLS_TAG=""
  if [ -n "$CLASSIFIER_CKPT_GLOB" ]; then
    # Pass through; predict_classifier.py also expands globs internally.
    CLS_CKPT_FLAG="--ckpts $CLASSIFIER_CKPT_GLOB"
    # Tag = leaf dir of the glob's parent (e.g. loo_classifier_3d_pt)
    CLS_TAG="$(basename "$(dirname "$(dirname "$CLASSIFIER_CKPT_GLOB")")")_ens"
  else
    if [ -z "$CLASSIFIER_CKPT" ]; then
      for cand in ckpts/gfp_classifier_3d_frac100/best.pth \
                  ckpts/gfp_classifier_3d/best.pth \
                  ckpts/gfp_classifier/best.pth; do
        [ -f "$cand" ] && CLASSIFIER_CKPT="$cand" && break
      done
    fi
    if [ -n "$CLASSIFIER_CKPT" ]; then
      CLS_CKPT_FLAG="--ckpt $CLASSIFIER_CKPT"
      CLS_TAG="$(basename "$(dirname "$CLASSIFIER_CKPT")")"
    fi
  fi
  if [ -z "$CLS_CKPT_FLAG" ]; then
    echo "WARN: CLASSIFY=1 but no classifier ckpt found. Either" >&2
    echo "      (a) train_loo_classifier.py --save_ckpt_dir <dir>, then" >&2
    echo "          set CLASSIFIER_CKPT_GLOB='<dir>/*/best.pth'; or" >&2
    echo "      (b) train_gfp_classifier.py, then set CLASSIFIER_CKPT." >&2
  else
    mkdir -p "$CLS_OUT_DIR"
    echo ""
    echo "########################################"
    echo "# STEP 4: classifier predictions ($CLS_TAG)"
    echo "########################################"
    for T in $CLS_TASKS; do
      OUT="$CLS_OUT_DIR/${CLS_TAG}_${T}.json"
      if [ -f "$OUT" ] && [ -z "$FORCE" ]; then
        echo "skip: $OUT exists (FORCE=1 to redo)"
        continue
      fi
      echo "=== Classify task=$T ==="
      python predict_classifier.py \
        $CLS_CKPT_FLAG --data_dir "$DATA_DIR" \
        --task "$T" --n_classes "$CLS_N_CLASSES" --output "$OUT"
    done
    echo "Classifier preds: $CLS_OUT_DIR/"
  fi
fi

# ────────────────────────────────────────────────────────────
# Step 5 (optional): regression predictions (force amplitude)
# ────────────────────────────────────────────────────────────
if [ "${REGRESS:-0}" = "1" ]; then
  REG_OUT_DIR="${REG_OUT_DIR:-results/regress_new_dataset}"
  REG_CKPT_FLAG=""
  REG_TAG=""
  if [ -n "$REGRESSOR_CKPT_GLOB" ]; then
    REG_CKPT_FLAG="--ckpts $REGRESSOR_CKPT_GLOB"
    REG_TAG="$(basename "$(dirname "$(dirname "$REGRESSOR_CKPT_GLOB")")")_ens"
  elif [ -n "$REGRESSOR_CKPT" ]; then
    REG_CKPT_FLAG="--ckpt $REGRESSOR_CKPT"
    REG_TAG="$(basename "$(dirname "$REGRESSOR_CKPT")")"
  else
    for cand in ckpts/loo_regression_3d ckpts/loo_regression_2d; do
      hits=$(ls "$cand"/*/best.pth 2>/dev/null | head -1)
      if [ -n "$hits" ]; then
        REG_CKPT_FLAG="--ckpts $cand/*/best.pth"
        REG_TAG="$(basename "$cand")_ens"
        break
      fi
    done
  fi
  if [ -z "$REG_CKPT_FLAG" ]; then
    echo "WARN: REGRESS=1 but no regression ckpt found. Run" >&2
    echo "      train_loo_regression.py --save_ckpt_dir <dir>, then set" >&2
    echo "      REGRESSOR_CKPT_GLOB='<dir>/*/best.pth'." >&2
  else
    mkdir -p "$REG_OUT_DIR"
    OUT="$REG_OUT_DIR/${REG_TAG}.json"
    echo ""
    echo "########################################"
    echo "# STEP 5: force regression ($REG_TAG)"
    echo "########################################"
    if [ -f "$OUT" ] && [ -z "$FORCE" ]; then
      echo "skip: $OUT exists (FORCE=1 to redo)"
    else
      python predict_regression.py $REG_CKPT_FLAG \
        --data_dir "$DATA_DIR" --output "$OUT"
    fi
    echo "Regression preds: $REG_OUT_DIR/"
  fi
fi

echo ""
echo "Done."
echo "Metrics : $OUT_DIR/"
echo "Predictions : $PRED_DIR/"
