#!/bin/bash
# End-to-end DINOv2 finetuning on force, leave-one-replicate-out.
#
# One condition = ~22 finetunings (one per held-out replicate). Budget
# roughly 2-5 min/fold on a decent GPU: ~1-2 h for the observed run, the
# same again for the shuffled control. A real permutation null (N_PERM=19,
# p-resolution 0.05) is ~20x the observed run: overnight. Without it the
# result has NO p-value -- the frozen probe's null is not transferable to a
# different fitting procedure.
#
# What this can and cannot show: finetuning adds capacity, not statistical
# power. The ~0.54 single-config detection floor at n=22 replicates applies
# to this model exactly as it does to the frozen probe. Run it as a
# comparison against the corrected-mask frozen baseline.
#
# Usage
#   bash scripts/e2e_force.sh                       # observed + control
#   N_PERM=19 bash scripts/e2e_force.sh             # + overnight null
#   FINAL=0 bash scripts/e2e_force.sh               # skip the XAI model
#   EPOCHS=4 bash scripts/e2e_force.sh              # quick smoke run

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week1}"
DECONFOUND="${DECONFOUND:-plate}"
TUNE="${TUNE:-lora}"
TUNE_BLOCKS="${TUNE_BLOCKS:-2}"
EPOCHS="${EPOCHS:-10}"
FG_MIN="${FG_MIN:-0.2}"
Z_STRIDE="${Z_STRIDE:-1}"
SEED="${SEED:-42}"
N_PERM="${N_PERM:-0}"
SHUFFLE="${SHUFFLE:-1}"
FINAL="${FINAL:-1}"
FIG_DIR="${FIG_DIR:-results/figures}"
XAI_DIR="${XAI_DIR:-results/xai_e2e}"

RUN_KEY="$(printf '%s|%s|%s|%s|%s|%s|%s|%s|%s' \
           "$TARGET_COL" "$DECONFOUND" "$TUNE" "$TUNE_BLOCKS" "$EPOCHS" \
           "$FG_MIN" "$Z_STRIDE" "$SEED" "$N_PERM" | cksum | cut -d' ' -f1)"
OUT="results/e2e_force/${TARGET_COL}_dc-${DECONFOUND}_${TUNE}${TUNE_BLOCKS}_${RUN_KEY}"
mkdir -p "$OUT"

common=(--data_dir "$DATA_DIR" --metadata "$METADATA"
        --target_col "$TARGET_COL" --deconfound "$DECONFOUND"
        --tune "$TUNE" --tune_blocks "$TUNE_BLOCKS" --epochs "$EPOCHS"
        --fg_min "$FG_MIN" --z_stride "$Z_STRIDE" --seed "$SEED")

echo "════════════════════════════════════════════════════════════════"
echo " e2e DINOv2 -> $TARGET_COL   dc=$DECONFOUND  tune=$TUNE x$TUNE_BLOCKS"
echo " results -> $OUT"
echo " ~22 finetunings per run; this is the slow, honest version"
echo "════════════════════════════════════════════════════════════════"

# ── 1. observed LOO (+ optional permutation null, + XAI checkpoint) ──
obs="$OUT/e2e_${TUNE}${TUNE_BLOCKS}.json"
if [ ! -f "$obs" ] || [ "${FORCE:-0}" = "1" ]; then
  final_arg=()
  [ "$FINAL" = "1" ] && final_arg=(--final_fit "$OUT/e2e_final.pt")
  python train_dino_e2e.py "${common[@]}" --n_perm "$N_PERM" \
    --output "$obs" "${final_arg[@]+"${final_arg[@]}"}"
else
  echo "▶ 1. cached: $obs"
fi

# ── 2. shuffled-label control: the leak canary ──
ctrl="$OUT/_control_shuffled.json"
if [ "$SHUFFLE" = "1" ] && { [ ! -f "$ctrl" ] || [ "${FORCE:-0}" = "1" ]; }; then
  echo ""; echo "▶ 2. shuffled-label control (same cost as the observed run)"
  python train_dino_e2e.py "${common[@]}" --shuffle --output "$ctrl"
elif [ "$SHUFFLE" = "1" ]; then
  echo "▶ 2. cached: $ctrl"
else
  echo "▶ 2. SKIPPED (SHUFFLE=0) -- without the canary a fold-logic leak is"
  echo "     invisible; do not report numbers from this run"
fi

# ── 3. figures (same machinery as the frozen probe) ──
echo ""; echo "▶ 3. figures"
python plot_force_probe.py --results_dir "$OUT" --out "$FIG_DIR" \
  --tag "e2e_${TARGET_COL}"

# ── 4. XAI on the final-fit model ──
if [ "$FINAL" = "1" ] && [ -f "$OUT/e2e_final.pt" ]; then
  echo ""; echo "▶ 4. XAI: extract finetuned features + explain"
  echo "     (the final-fit model saw every label: pictures, not statistics)"
  FEAT="results/dino_features_e2e"
  python extract_dino_features.py --data_dir "$DATA_DIR" --input gfp \
    --output_dir "$FEAT" --checkpoint "$OUT/e2e_final.pt" \
    --mask_source gfp --mask_projection max --z_stride "$Z_STRIDE"
  fdir="$(ls -d "$FEAT"/gfp_tiled_volume_* 2>/dev/null | head -1)"
  python explain_dino_probe.py --readout "$OUT/e2e_final.readout.npz" \
    --feature_dir "$fdir" --data_dir "$DATA_DIR" \
    --level view --out "$XAI_DIR"
  python explain_dino_probe.py --readout "$OUT/e2e_final.readout.npz" \
    --feature_dir "$fdir" --data_dir "$DATA_DIR" \
    --checkpoint "$OUT/e2e_final.pt" \
    --level patch --n_volumes "${N_VOLUMES:-6}" --out "$XAI_DIR"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " done. Compare $FIG_DIR/probe_e2e_${TARGET_COL}_scatter.png against"
echo " the frozen probe's scatter. Without N_PERM>0 there is NO p-value"
echo " for the e2e number; with the control at chance and the observed"
echo " spearman below the frozen probe's detection floor, the honest"
echo " summary is 'no better than frozen'."
echo "════════════════════════════════════════════════════════════════"
