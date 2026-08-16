#!/bin/bash
# Audit the data funnel: raw nd2 -> staged -> z_range-usable -> force-matched
# -> train/val/test split. Answers "are we actually using all of our data?"
# and prints exactly where volumes drop out (and why).
#
# Torch-free; safe to run on the VM while training is going.
#
# Usage
# -----
#   bash scripts/audit_data.sh
#   ND2_DIR=/path/to/raw_drop bash scripts/audit_data.sh   # include raw-file check
#
# Env knobs (defaults in parens)
#   DATA_DIR    (data_phalloidin_mhc_051826_staged)   staged root
#   METADATA    (phalloidin_mhc_mapping_051426_SS edit.xlsx)
#   TARGET_COL  (peak_amplitude_week3)
#   GROUP_COLS  (plate,Tissue)
#   FILE_COL    (file)
#   ND2_DIR     (unset)          raw .nd2 drop root — enables the raw-vs-staged check
#   CONFIG      (configs/base.yaml)  supplies z_range / patch_depth
#   OUT         (results/audit/data_audit.json)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
FILE_COL="${FILE_COL:-file}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUT="${OUT:-results/audit/data_audit.json}"

flags=(--data_dir "$DATA_DIR" --config "$CONFIG" --out "$OUT"
       --target_col "$TARGET_COL" --group_cols "$GROUP_COLS"
       --file_col "$FILE_COL")
[ -f "$METADATA" ] && flags+=(--metadata "$METADATA") \
  || echo "note: metadata '$METADATA' not found — skipping force-label match"
[ -n "${ND2_DIR:-}" ] && flags+=(--nd2_dir "$ND2_DIR")

python audit_data.py "${flags[@]}"
