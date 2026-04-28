#!/bin/bash
# End-to-end: redo BF->GFP power laws with held-out task evals,
# 10 seeds for SE on classification, MAE/SSIM side panels.
#
# Stages (idempotent — set FORCE=1 to invalidate existence checks):
#   1. frac=0 LOO baselines (no BF->GFP pretraining; ImageNet encoder only)
#   2. For each holdout (exercise, perturbation), each fraction:
#        a. Train BF->GFP if no checkpoint
#        b. Compute MAE/SSIM/Pearson on held-out task vols
#        c. Run LOO classifier on the held-out task with --binarize × 10 seeds
#   3. Plot 3-panel figure
#
# Usage: bash scripts/run_power_laws.sh

set -e

METADATA=data_mapping_drew.csv
CFG_BFGFP=configs/unet_3d_imagenet_pearson.yaml
CFG_LOO=configs/gfp_classifier_3d.yaml
LOO_OUT=results/loo
METRICS_OUT=results/bfgfp_metrics
PLOT_OUT=results/classifier
EXTRA_ARGS="$@"

mkdir -p "$LOO_OUT" "$METRICS_OUT" "$PLOT_OUT"

# fraction -> 3-char pct (matches train_fraction.py naming)
declare -A PCT=( [0.05]=005 [0.10]=010 [0.25]=025 [0.50]=050 [1.00]=100 )
FRACS=(0.05 0.10 0.25 0.50 1.00)
read -r -a SEEDS <<< "${SEEDS:-0}"   # default 1 seed; SEEDS="0 1 2 3 4" to override

skip_if_done() {
  local out="$1"
  if [ -f "$out" ] && [ -z "$FORCE" ]; then
    echo "skip: $out exists (FORCE=1 to redo)"
    return 0
  fi
  return 1
}

# ──────────────────────────────────────────────────────────────
# Stage 1: frac=0 LOO baselines (no BF->GFP)
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STAGE 1: frac=0 LOO baselines        #"
echo "########################################"
for HOLD in exercise perturbation; do
  for S in "${SEEDS[@]}"; do
    OUT="$LOO_OUT/frac000_${HOLD}_bf_seed${S}.json"
    skip_if_done "$OUT" && continue
    echo "=== frac000 LOO ($HOLD, seed=$S) ==="
    python train_loo_classifier.py \
      -c "$CFG_LOO" --metadata "$METADATA" \
      --task "$HOLD" --input bf --binarize \
      --seed "$S" --output "$OUT" \
      $EXTRA_ARGS
  done
done

# ──────────────────────────────────────────────────────────────
# Stage 2: held-out scaling sweeps
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STAGE 2: BF->GFP sweeps + LOO seeds  #"
echo "########################################"
for HOLD in exercise perturbation; do
  if [ "$HOLD" = exercise ]; then TAG=holdEx; else TAG=holdPt; fi
  for F in "${FRACS[@]}"; do
    P=${PCT[$F]}
    CKPT=$(ls -d ckpts/unet_3d_*frac${P}_${TAG}/best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
      echo "=== Training BF->GFP (frac=$F, holdout=$HOLD) ==="
      python train_fraction.py -c "$CFG_BFGFP" \
        --fraction "$F" --holdout "$HOLD" --metadata "$METADATA"
      CKPT=$(ls -d ckpts/unet_3d_*frac${P}_${TAG}/best.pth | head -1)
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
      echo "ERROR: no checkpoint after training for frac=$F holdout=$HOLD" >&2
      exit 1
    fi
    echo "Using ckpt: $CKPT"

    METRICS_JSON="$METRICS_OUT/$(basename "$(dirname "$CKPT")").json"
    if ! skip_if_done "$METRICS_JSON"; then
      echo "=== Eval MAE/SSIM (frac=$F, holdout=$HOLD) ==="
      python eval_bfgfp_metrics.py --ckpt "$CKPT" --output "$METRICS_JSON"
    fi

    for S in "${SEEDS[@]}"; do
      OUT="$LOO_OUT/${TAG}_frac${P}_${HOLD}_bf_seed${S}.json"
      skip_if_done "$OUT" && continue
      echo "=== LOO ${TAG} frac=$F ($HOLD, seed=$S) ==="
      python train_loo_classifier.py \
        -c "$CFG_LOO" --metadata "$METADATA" \
        --task "$HOLD" --input bf --binarize \
        --init_from "$CKPT" --seed "$S" --output "$OUT" \
        $EXTRA_ARGS
    done
  done
done

# ──────────────────────────────────────────────────────────────
# Stage 3: plot
# ──────────────────────────────────────────────────────────────
echo ""
echo "########################################"
echo "# STAGE 3: power-laws plot             #"
echo "########################################"
python plot_power_laws_v2.py \
  --loo_dir "$LOO_OUT" --metrics_dir "$METRICS_OUT" \
  --output "$PLOT_OUT/power_laws_v2.png"

echo ""
echo "Done. Plot: $PLOT_OUT/power_laws_v2.png"
