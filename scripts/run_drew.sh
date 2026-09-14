#!/bin/bash
# Everything on the drew perturbation/exercise drop, in one go:
#
#   A. frozen DINOv2 probe on every target (nulls, shuffled control,
#      family-wise correction)                        -> results/dino_sweep/
#   B. figures + XAI for every target                  -> results/figures/, results/xai/
#   C. end-to-end finetuning, OPT-IN with E2E=1         -> results/e2e_force/, results/xai_e2e/
#
# Default is A+B: results and pictures in minutes. Add the finetuning once
# the probe numbers are in hand:   E2E=1 bash scripts/run_drew.sh
# (A and B are cached, so that second invocation only pays for C.)
#
# Targets:
#   regression      peak_amplitude_week_5, peak_amplitude_week_4   (12 tissues)
#   classification  perturbed (control vs drug, 12 tissues, 2 controls)
#                   Exercise  (Stimulated vs Unstimulated, 4 tissues -- a pilot:
#                             p >= 1/6 by construction, read for direction + XAI)
#
# Neither label is plate-determined on this drop (each imaging day has its
# own control; both exercise classes share a day), so the standard protocol
# applies: leave-one-TISSUE-out, within-day permutation null.
#
# Prerequisites (once):
#   python stage_nd2.py --src data --inspect
#   python stage_nd2.py --src data --out data_drew_staged --bf_channel <bf> --target_channel <fluor>
#   python compute_stats.py --data_dir data_drew_staged
#   python check_mask_polarity.py --data_dir data_drew_staged --feature_dir ""
#
# Cost: A+B are minutes after one feature extraction. C is ~80 finetunings
# (observed + shuffled control for four targets): budget 3-6 h on a GPU;
# E2E_TARGETS="peak_amplitude_week_5 perturbed" trims it to the two that
# matter most.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA_DIR="${DATA_DIR:-data_drew_staged}"
export METADATA="${METADATA:-data_mapping_drew_aug.csv}"
export FEAT_DIR="${FEAT_DIR:-results/dino_features_drew}"
export GROUP_COLS="${GROUP_COLS:-plate,Dataset,Exercise,Perturbation,Tissue}"
FORCE_COLS="${FORCE_COLS:-peak_amplitude_week_5 peak_amplitude_week_4}"
CATEGORICAL="${CATEGORICAL:-perturbed Exercise}"
E2E="${E2E:-0}"
E2E_TARGETS="${E2E_TARGETS:-$FORCE_COLS $CATEGORICAL}"

[ -d "$DATA_DIR/gfp" ] || { echo "ERROR: $DATA_DIR/gfp missing -- stage first (see header)" >&2; exit 1; }
if [ ! -f "$METADATA" ] && [ -f data_mapping_drew.csv ]; then
  # *.csv is gitignored, so the augmented sheet may not have travelled with
  # the repo; it is a deterministic function of the source sheet, so build it.
  echo "▶ $METADATA missing -- deriving it from data_mapping_drew.csv"
  python scripts/augment_drew_mapping.py data_mapping_drew.csv
fi
[ -f "$METADATA" ] || { echo "ERROR: $METADATA missing -- run scripts/augment_drew_mapping.py" >&2; exit 1; }

is_cat() { case " $CATEGORICAL " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "################################################################"
echo "# A. frozen probe: $FORCE_COLS | $CATEGORICAL"
echo "################################################################"
FORCE_COLS="$FORCE_COLS" CATEGORICAL="$CATEGORICAL" bash scripts/all_targets.sh

echo ""; echo "################################################################"
echo "# B. figures + XAI per target"
echo "################################################################"
for t in $FORCE_COLS $CATEGORICAL; do
  echo ""; echo "▶ $t"
  if is_cat "$t"; then
    TARGET_COL="$t" TARGET_TYPE=categorical SKIP_SWEEP=1 bash scripts/explain_force.sh
  else
    TARGET_COL="$t" TARGET_TYPE=numeric SKIP_SWEEP=1 bash scripts/explain_force.sh
  fi
done

if [ "$E2E" = "1" ]; then
  echo ""; echo "################################################################"
  echo "# C. end-to-end finetuning per target: $E2E_TARGETS"
  echo "################################################################"
  for t in $E2E_TARGETS; do
    echo ""; echo "▶ e2e $t"
    if is_cat "$t"; then
      # CV_GROUP=replicate explicitly: the categorical default of plate-folds
      # is for plate-determined labels, and neither label here is one.
      TARGET_COL="$t" TARGET_TYPE=categorical CV_GROUP=replicate \
        XAI_DIR="results/xai_e2e" bash scripts/e2e_force.sh
    else
      TARGET_COL="$t" TARGET_TYPE=numeric XAI_DIR="results/xai_e2e" \
        bash scripts/e2e_force.sh
    fi
  done
else
  echo ""; echo "# C. e2e not run. When the probe results are in hand:"
  echo "#      E2E=1 bash scripts/run_drew.sh        (A+B are cached; only C runs)"
fi

echo ""
echo "################################################################"
echo "# done"
echo "#   probe tables   results/dino_sweep/<target>_*/"
echo "#   figures        results/figures/probe_<target>_*.png  (+ probe_e2e_<target>_*)"
echo "#   XAI            results/xai/  (probe)   results/xai_e2e/  (finetuned)"
echo "#"
echo "#   Exercise: 4 tissues -> p >= 1/6 by construction. Direction + XAI only."
echo "#   perturbed: majority rate 0.83 -> judge accuracy against its null."
echo "################################################################"
