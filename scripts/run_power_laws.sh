#!/bin/bash
# End-to-end BF->GFP power laws with held-out task evals (classification + regression).
#
# Stages per architecture (idempotent — set FORCE=1 to invalidate existence checks):
#   1. frac=0 LOO baselines (no BF->GFP pretraining; ImageNet encoder only)
#   2. For each holdout (exercise, perturbation), each fraction:
#        a. Train BF->GFP if no checkpoint
#        b. Compute MAE/SSIM/Pearson on held-out task vols
#        c. Run LOO classifier on the held-out task × label modes × seeds
#   3. Classification scaling plot (one per label mode)
#   4. (Optional) LOO regression on a numeric column + Pearson/MSE scaling plots
#
# Architecture: defaults to 3D for backward compat. To also run 2D:
#   ARCHS="2d 3d" bash scripts/run_power_laws.sh
#
# Usage:
#   bash scripts/run_power_laws.sh                            # 3D only (default)
#   ARCHS="2d" bash scripts/run_power_laws.sh                 # 2D only
#   ARCHS="2d 3d" bash scripts/run_power_laws.sh              # both
#   LABEL_MODES="binary collapsed" bash scripts/run_power_laws.sh
#   SEEDS="0 1 2" bash scripts/run_power_laws.sh
#   REGRESSION=0 bash scripts/run_power_laws.sh               # skip Stage 4

set -e

METADATA=data_mapping_drew.csv
METRICS_OUT=results/bfgfp_metrics
PLOT_OUT=results/classifier
EXTRA_ARGS="$@"

mkdir -p "$METRICS_OUT" "$PLOT_OUT"

# fraction -> 3-char pct (matches train_fraction.py naming)
declare -A PCT=( [0.05]=005 [0.10]=010 [0.25]=025 [0.50]=050 [1.00]=100 )
FRACS=(0.05 0.10 0.25 0.50 1.00)
read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a CV_UNITS <<< "${CV_UNITS:-volume replicate}"
read -r -a LABEL_MODES <<< "${LABEL_MODES:-binary}"
read -r -a ARCHS <<< "${ARCHS:-3d}"

label_flag_for() {
  case "$1" in
    binary) echo "--binarize" ;;
    collapsed) echo "--collapse_doses" ;;
    *) echo "" ;;
  esac
}
label_suffix_for() {
  case "$1" in
    binary) echo "" ;;          # preserve existing filenames
    collapsed) echo "_lbl-collapsed" ;;
    *) echo "_lbl-$1" ;;
  esac
}
arch_suffix_for() {
  case "$1" in
    3d) echo "" ;;              # backward-compat: 3D outputs land where they were
    2d) echo "_arch-2d" ;;
    *) echo "_arch-$1" ;;
  esac
}
arch_loo_dir() {
  # 3D defaults to results/loo for backward compat; 2D and others use suffix dirs.
  case "$1" in
    3d) echo "results/loo" ;;
    *) echo "results/loo_$1" ;;
  esac
}
arch_reg_dir() {
  case "$1" in
    3d) echo "results/loo_reg" ;;
    *) echo "results/loo_reg_$1" ;;
  esac
}

skip_if_done() {
  local out="$1"
  if [ -f "$out" ] && [ -z "$FORCE" ]; then
    echo "skip: $out exists (FORCE=1 to redo)"
    return 0
  fi
  return 1
}

for ARCH in "${ARCHS[@]}"; do
  case "$ARCH" in
    2d) CFG_BFGFP=configs/unet_2d_imagenet_pearson.yaml
        CFG_LOO=configs/gfp_classifier.yaml ;;
    3d) CFG_BFGFP=configs/unet_3d_imagenet_pearson.yaml
        CFG_LOO=configs/gfp_classifier_3d.yaml ;;
    *) echo "ERROR: unknown ARCH=$ARCH (use 2d or 3d)" >&2; exit 1 ;;
  esac
  LOO_OUT=$(arch_loo_dir "$ARCH")
  ARCH_SFX=$(arch_suffix_for "$ARCH")
  mkdir -p "$LOO_OUT"

  echo ""
  echo "############################################################"
  echo "# ARCH=$ARCH  CFG_BFGFP=$CFG_BFGFP  CFG_LOO=$CFG_LOO"
  echo "############################################################"

  # ────────────────────────────────────────────────────────────
  # Stage 1: frac=0 LOO baselines (no BF->GFP)
  # ────────────────────────────────────────────────────────────
  echo ""
  echo "=== STAGE 1: frac=0 LOO baselines ($ARCH) ==="
  for HOLD in exercise perturbation; do
    for LBL in "${LABEL_MODES[@]}"; do
      if [ "$HOLD" = exercise ] && [ "$LBL" != binary ]; then
        echo "skip: label_mode=$LBL not applicable to exercise"
        continue
      fi
      LBL_FLAG=$(label_flag_for "$LBL")
      LBL_SFX=$(label_suffix_for "$LBL")
      for CV in "${CV_UNITS[@]}"; do
        for S in "${SEEDS[@]}"; do
          if [ "$CV" = volume ]; then
            OUT="$LOO_OUT/frac000_${HOLD}_bf${LBL_SFX}_seed${S}.json"
          else
            OUT="$LOO_OUT/frac000_${HOLD}_bf_cv-${CV}${LBL_SFX}_seed${S}.json"
          fi
          skip_if_done "$OUT" && continue
          echo "=== [$ARCH] frac000 LOO ($HOLD, cv=$CV, label=$LBL, seed=$S) ==="
          python train_loo_classifier.py \
            -c "$CFG_LOO" --metadata "$METADATA" \
            --task "$HOLD" --input bf $LBL_FLAG \
            --cv_unit "$CV" --seed "$S" --output "$OUT" \
            $EXTRA_ARGS
        done
      done
    done
  done

  # ────────────────────────────────────────────────────────────
  # Stage 2: held-out scaling sweeps
  # ────────────────────────────────────────────────────────────
  echo ""
  echo "=== STAGE 2: BF->GFP sweeps + LOO seeds ($ARCH) ==="
  for HOLD in exercise perturbation; do
    if [ "$HOLD" = exercise ]; then TAG=holdEx; else TAG=holdPt; fi
    for F in "${FRACS[@]}"; do
      P=${PCT[$F]}
      CKPT=$(ls -d ckpts/unet_${ARCH}_*frac${P}_${TAG}/best.pth 2>/dev/null | head -1)
      if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "=== [$ARCH] Training BF->GFP (frac=$F, holdout=$HOLD) ==="
        python train_fraction.py -c "$CFG_BFGFP" \
          --fraction "$F" --holdout "$HOLD" --metadata "$METADATA"
        CKPT=$(ls -d ckpts/unet_${ARCH}_*frac${P}_${TAG}/best.pth | head -1)
      fi
      if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "ERROR: no $ARCH checkpoint after training for frac=$F holdout=$HOLD" >&2
        exit 1
      fi
      echo "Using ckpt: $CKPT"

      METRICS_JSON="$METRICS_OUT/$(basename "$(dirname "$CKPT")").json"
      if ! skip_if_done "$METRICS_JSON"; then
        echo "=== [$ARCH] Eval MAE/SSIM (frac=$F, holdout=$HOLD) ==="
        python eval_bfgfp_metrics.py --ckpt "$CKPT" --output "$METRICS_JSON"
      fi

      for LBL in "${LABEL_MODES[@]}"; do
        if [ "$HOLD" = exercise ] && [ "$LBL" != binary ]; then
          echo "skip: label_mode=$LBL not applicable to exercise"
          continue
        fi
        LBL_FLAG=$(label_flag_for "$LBL")
        LBL_SFX=$(label_suffix_for "$LBL")
        for CV in "${CV_UNITS[@]}"; do
          for S in "${SEEDS[@]}"; do
            if [ "$CV" = volume ]; then
              OUT="$LOO_OUT/${TAG}_frac${P}_${HOLD}_bf${LBL_SFX}_seed${S}.json"
            else
              OUT="$LOO_OUT/${TAG}_frac${P}_${HOLD}_bf_cv-${CV}${LBL_SFX}_seed${S}.json"
            fi
            skip_if_done "$OUT" && continue
            echo "=== [$ARCH] LOO ${TAG} frac=$F ($HOLD, cv=$CV, label=$LBL, seed=$S) ==="
            python train_loo_classifier.py \
              -c "$CFG_LOO" --metadata "$METADATA" \
              --task "$HOLD" --input bf $LBL_FLAG \
              --init_from "$CKPT" --cv_unit "$CV" \
              --seed "$S" --output "$OUT" \
              $EXTRA_ARGS
          done
        done
      done
    done
  done

  # ────────────────────────────────────────────────────────────
  # Stage 3: classification plot (one per label mode)
  # ────────────────────────────────────────────────────────────
  echo ""
  echo "=== STAGE 3: classification plots ($ARCH) ==="
  for LBL in "${LABEL_MODES[@]}"; do
    LBL_SFX=$(label_suffix_for "$LBL")
    OUT_PNG="$PLOT_OUT/power_laws_v2${LBL_SFX}${ARCH_SFX}.png"
    python plot_power_laws_v2.py \
      --loo_dir "$LOO_OUT" --metrics_dir "$METRICS_OUT" \
      --label_mode "$LBL" --output "$OUT_PNG"
  done

  # ────────────────────────────────────────────────────────────
  # Stage 4 (optional): LOO regression on Wk5 amplitude (default)
  # ────────────────────────────────────────────────────────────
  if [ "${REGRESSION:-1}" != "0" ]; then
    REG_OUT=$(arch_reg_dir "$ARCH")
    mkdir -p "$REG_OUT"
    REG_TARGET="${REG_TARGET:-peak_amplitude_week_5}"
    REG_HOLD="${REG_HOLD:-perturbation}"
    if [ "$REG_HOLD" = exercise ]; then REG_TAG=holdEx; else REG_TAG=holdPt; fi
    echo ""
    echo "=== STAGE 4: LOO regression $REG_TARGET ($ARCH) ==="
    for S in "${SEEDS[@]}"; do
      OUT="$REG_OUT/frac000_${REG_TARGET}_bf_seed${S}.json"
      if skip_if_done "$OUT"; then :; else
        echo "=== [$ARCH] frac000 regression (target=$REG_TARGET, seed=$S) ==="
        python train_loo_regression.py \
          -c "$CFG_LOO" --metadata "$METADATA" \
          --target_col "$REG_TARGET" --input bf \
          --seed "$S" --output "$OUT" \
          $EXTRA_ARGS
      fi
    done
    for F in "${FRACS[@]}"; do
      P=${PCT[$F]}
      CKPT=$(ls -d ckpts/unet_${ARCH}_*frac${P}_${REG_TAG}/best.pth 2>/dev/null | head -1)
      if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "skip: no $ARCH $REG_TAG ckpt at frac=$F"
        continue
      fi
      for S in "${SEEDS[@]}"; do
        OUT="$REG_OUT/${REG_TAG}_frac${P}_${REG_TARGET}_bf_seed${S}.json"
        skip_if_done "$OUT" && continue
        echo "=== [$ARCH] regression ${REG_TAG} frac=$F (target=$REG_TARGET, seed=$S) ==="
        python train_loo_regression.py \
          -c "$CFG_LOO" --metadata "$METADATA" \
          --target_col "$REG_TARGET" --input bf \
          --init_from "$CKPT" --seed "$S" --output "$OUT" \
          $EXTRA_ARGS
      done
    done
    echo "=== [$ARCH] Regression scaling plots ==="
    python plot_regression_scaling.py \
      --loo_dir "$REG_OUT" --output_dir "$PLOT_OUT" \
      --prefix "regression_${REG_TARGET}${ARCH_SFX}"
  fi

done  # end ARCH loop

echo ""
echo "Done. Plots in $PLOT_OUT/"
