#!/bin/bash
# Every target, one command: force (both timepoints) + the categorical labels.
#
#   bash scripts/all_targets.sh
#   CATEGORICAL="treated perturbed" bash scripts/all_targets.sh
#   FORCE_COLS="peak_amplitude_week1" bash scripts/all_targets.sh   # just one
#
# Each target is a SEPARATE question, so each gets its own family-wise
# correction. They are NOT pooled: correcting across targets would answer
# "is anything predictable from anything", which is not what you want to know.
#
# Features are extracted once and cached by config hash, so the second and
# later targets cost seconds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
FORCE_COLS="${FORCE_COLS:-peak_amplitude_week1 peak_amplitude_week3}"
# Leave empty to have the script tell you what is available instead of
# guessing column names that may not exist.
CATEGORICAL="${CATEGORICAL:-}"
SUMMARY="results/dino_sweep/ALL_TARGETS.txt"

mkdir -p results/dino_sweep
: > "$SUMMARY"

echo "════════════════════════════════════════════════════════════════"
echo " Step 0. what is actually in the metadata"
echo "════════════════════════════════════════════════════════════════"
python inventory_metadata.py "$METADATA" --group_cols "$GROUP_COLS" \
  --data_dir "$DATA_DIR" --output results/dino_sweep/metadata_inventory.json \
  | tee -a "$SUMMARY"

if [ -z "$CATEGORICAL" ]; then
  echo ""
  echo "  NOTE: CATEGORICAL is empty, so only the force columns will be run."
  echo "  Pick the treated/perturbed column names from the 'categorical label"
  echo "  columns' table above -- skip any row marked CONFOUNDED, because"
  echo "  plate alone already predicts it -- then rerun as:"
  echo "     CATEGORICAL=\"<col> <col>\" bash scripts/all_targets.sh"
fi

run_target () {           # $1 = column, $2 = numeric|categorical
  # tee these into $SUMMARY too, or the final per-target roll-up below has
  # family-wise p values with no target names attached to them.
  { echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " TARGET: $1   ($2)"
    echo "════════════════════════════════════════════════════════════════"
  } | tee -a "$SUMMARY"
  local task="regression"
  [ "$2" = "categorical" ] && task="classification"
  TARGET_COL="$1" TARGET_TYPE="$2" TASK="$task" \
  METADATA="$METADATA" DATA_DIR="$DATA_DIR" GROUP_COLS="$GROUP_COLS" \
  SKIP_EXTRACT="${SKIP_EXTRACT:-0}" \
    bash scripts/dino_force_sweep.sh 2>&1 | tee -a "$SUMMARY"
  # Extraction is cached by config hash; only the first target pays for it.
  SKIP_EXTRACT=1
}

for col in $FORCE_COLS; do
  run_target "$col" numeric
done
for col in $CATEGORICAL; do
  run_target "$col" categorical
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " ALL TARGETS — family-wise result per target"
echo "════════════════════════════════════════════════════════════════"
grep -E "^ TARGET:|family-wise p =|does NOT survive|survives correction|NONE IS POSSIBLE|control \(shuffled" \
  "$SUMMARY" || true
echo ""
echo "  full log: $SUMMARY"
echo "  Reminder: peak_amplitude_week3 is heavily plate-confounded"
echo "  (omega^2 ~ 0.78). Read its numbers against the plate->label and"
echo "  omega2 columns in step 0 before believing anything there."
