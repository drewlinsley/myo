#!/bin/bash
# Two-stage, apples-to-apples force prediction on the NEW data:
#   Stage 1: train a BF->GFP translation U-Net on THIS dataset.
#   Stage 2: freeze it and probe its representations for force (linear probe).
#
# Everything is held constant vs. the direct GFP->force run
# (scripts/force_from_gfp_new.sh): SAME replicate split, SAME force bins, SAME
# architecture. The ONLY change is where the encoder comes from — here it is
# trained BF->GFP on this data's TRAIN replicates and then frozen.
#
# Leakage control: the split is computed ONCE (stage 0) and both stages consume
# it. The BF->GFP model trains only on NON-TEST stems, so the force TEST
# replicates are never seen by the encoder OR the probe head.
#
# Pipeline
# --------
#   0. Emit the replicate split manifest (force.split.json + force.bfgfp_split.json).
#   1. Train BF->GFP (2D + 3D) on the non-test stems (train.py --split_json).
#   2. Force probe (2D + 3D): freeze the new encoder, train only the head on the
#      SAME split; report TEST accuracy / correlation / plots.
#
# By default the probe feeds BRIGHTFIELD (PROBE_INPUT=bf) into the frozen
# BF->GFP encoder — a fully label-free force predictor. Set PROBE_INPUT=gfp to
# feed GFP instead (direct input-match to the GFP->force model).
#
# Usage
# -----
#   bash scripts/force_two_stage.sh
#   PROBE_INPUT=gfp bash scripts/force_two_stage.sh          # GFP-input probe
#   FREEZE=0 bash scripts/force_two_stage.sh                 # fine-tune (not a probe)
#
# Env knobs (plus everything force_from_gfp_new.sh accepts)
# --------
#   DATA_DIR     staged root with bf/ + gfp/    (default: data_phalloidin_mhc_051826_staged)
#   METADATA     mapping spreadsheet            (default: phalloidin_mhc_mapping_051426_SS edit.xlsx)
#   TARGET_COL   force column                   (default: peak_amplitude_week3)
#   GROUP_COLS   replicate id columns           (default: plate,Tissue)
#   N_BINS/BIN_SCHEME/TEST_FRAC/VAL_FRAC/SEED   (defaults: 4/quantile/0.25/0/42)
#   OUT_DIR      results root                   (default: results/force_two_stage)
#   PROBE_INPUT  bf | gfp                        (default: bf)
#   FREEZE       1=linear probe, 0=fine-tune     (default: 1)
#   PROBE_L2     L2 (weight decay) on linear head (default: 0.05)
#   ONLY         both | 2d | 3d                  (default: both)
#   FORCE=1      redo existing stage outputs
#   ALLOW_PARTIAL_MATCH=1  train on matched subset if some force rows don't match

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
FILE_COL="${FILE_COL:-file}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
N_BINS="${N_BINS:-4}"   # 4-way is the practical ceiling for a 6-replicate test set
BIN_SCHEME="${BIN_SCHEME:-quantile}"
TEST_FRAC="${TEST_FRAC:-0.25}"
VAL_FRAC="${VAL_FRAC:-0}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/force_two_stage}"
PROBE_INPUT="${PROBE_INPUT:-bf}"
FREEZE="${FREEZE:-1}"
PROBE_L2="${PROBE_L2:-0.05}"   # L2 (AdamW weight decay) on the linear head
ONLY="${ONLY:-both}"

case "$ONLY" in both|2d|3d) ;; *) echo "ERROR: ONLY must be both|2d|3d" >&2; exit 1;; esac
case "$PROBE_INPUT" in bf|gfp) ;; *) echo "ERROR: PROBE_INPUT must be bf|gfp" >&2; exit 1;; esac
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════"
echo " Two-stage force: BF->GFP translation -> frozen-rep force probe"
echo "   data=$DATA_DIR  target=$TARGET_COL  groups=$GROUP_COLS"
echo "   n_bins=$N_BINS ($BIN_SCHEME) test=$TEST_FRAC seed=$SEED"
echo "   probe_input=$PROBE_INPUT  freeze=$FREEZE  out=$OUT_DIR"
echo "════════════════════════════════════════════════════════════"

for d in bf gfp; do
  [ -d "$DATA_DIR/$d" ] || { echo "ERROR: $DATA_DIR/$d not found." >&2; exit 1; }
done
[ -f "$METADATA" ] || { echo "ERROR: metadata '$METADATA' not found." >&2; exit 1; }

echo "Computing percentile stats (idempotent)…"
python compute_stats.py --data_dir "$DATA_DIR" $([ "${FORCE:-0}" = "1" ] && echo --force)

SPLIT="$OUT_DIR/force.split.json"
BFGFP_SPLIT="$OUT_DIR/force.bfgfp_split.json"

# ── Stage 0: emit the split manifest (arch-independent; uses the 2D cfg) ──
if [ ! -f "$SPLIT" ] || [ "${FORCE:-0}" = "1" ]; then
  echo ""; echo "# Stage 0: emit replicate split manifest"
  partial=(); [ "${ALLOW_PARTIAL_MATCH:-0}" = "1" ] && partial=(--allow_partial_match)
  python train_split_force_classifier.py \
    -c configs/gfp_classifier.yaml \
    --metadata "$METADATA" --data_dir "$DATA_DIR" --input gfp \
    --target_col "$TARGET_COL" --file_col "$FILE_COL" --group_cols "$GROUP_COLS" \
    --n_bins "$N_BINS" --bin_scheme "$BIN_SCHEME" \
    --test_frac "$TEST_FRAC" --val_frac "$VAL_FRAC" --seed "$SEED" \
    "${partial[@]+"${partial[@]}"}" --plan_only \
    --output "$OUT_DIR/force.json"
else
  echo "Stage 0: reusing $SPLIT (FORCE=1 to redo)"
fi
[ -f "$SPLIT" ] || { echo "ERROR: split manifest not emitted ($SPLIT)." >&2; exit 1; }
[ -f "$BFGFP_SPLIT" ] || { echo "ERROR: BF->GFP split not emitted ($BFGFP_SPLIT)." >&2; exit 1; }

# ── Stage 1: train BF->GFP on non-test stems ──
run_bfgfp() {
  local dims="$1" cfg="$2"
  local ck="$OUT_DIR/unet_new_${dims}"
  if [ -f "$ck/best.pth" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[bf->gfp $dims] exists, skipping: $ck/best.pth"; return 0
  fi
  echo ""; echo "# Stage 1 [$dims]: BF->GFP translation ($cfg)"
  python train.py -c "$cfg" \
    --data_dir "$DATA_DIR" --ckpt_dir "$ck" \
    --split_json "$BFGFP_SPLIT"
  [ -f "$ck/best.pth" ] || { echo "ERROR: $ck/best.pth not produced." >&2; exit 1; }
}

# ── Stage 2: frozen-representation force probe on the SAME split ──
run_probe() {
  local dims="$1" cfg="$2"
  local enc="$OUT_DIR/unet_new_${dims}/best.pth"
  local out="$OUT_DIR/probe_${dims}.json"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[probe $dims] exists, skipping: $out"; return 0
  fi
  local freeze_flag=(); [ "$FREEZE" = "1" ] && freeze_flag=(--freeze_encoder)
  echo ""; echo "# Stage 2 [$dims]: force probe (input=$PROBE_INPUT freeze=$FREEZE)"
  python train_split_force_classifier.py \
    -c "$cfg" --data_dir "$DATA_DIR" \
    --split_json "$SPLIT" --init_from "$enc" \
    --input "$PROBE_INPUT" "${freeze_flag[@]+"${freeze_flag[@]}"}" \
    --weight_decay "$PROBE_L2" \
    --save_ckpt "$OUT_DIR/probe_${dims}.pth" \
    --seed "$SEED" --output "$out"
}

if [ "$ONLY" = "both" ] || [ "$ONLY" = "2d" ]; then
  run_bfgfp 2d configs/unet_2d_imagenet_pearson.yaml
  run_probe 2d configs/gfp_classifier.yaml
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "3d" ]; then
  run_bfgfp 3d configs/unet_3d_imagenet_pearson.yaml
  run_probe 3d configs/gfp_classifier_3d.yaml
fi

# ── Stage 3: SmoothGrad saliency (XAI) of the frozen-rep probe(s) ──
# gradviz auto-detects the probe input modality from the saved ckpt (bf/gfp).
if [ "${SKIP_SALIENCY:-0}" != "1" ]; then
  echo ""; echo "# Stage 3: SmoothGrad saliency of the probe model(s)"
  OUT_DIR="$OUT_DIR" DATA_DIR="$DATA_DIR" ONLY="$ONLY" RESULT_PREFIX=probe \
    CKPT_2D="$OUT_DIR/probe_2d.pth" CKPT_3D="$OUT_DIR/probe_3d.pth" \
    bash scripts/gradviz_force.sh || echo "  (probe saliency skipped — see above)"
fi

# ── Compare: two-stage probe (2D/3D) and, if present, the direct GFP->force run ──
echo ""; echo "# Comparison"
python - "$OUT_DIR" "results/force_from_gfp_new" <<'PY'
import json, os, sys
two_dir, direct_dir = sys.argv[1], sys.argv[2]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
rows = []
for dims in ("2d", "3d"):
    pr = load(os.path.join(two_dir, f"probe_{dims}.json"))
    if pr:
        c = pr["correlation"]
        rows.append((f"two-stage probe {dims} (input={pr.get('input')}, "
                     f"freeze={pr.get('freeze_encoder')})",
                     pr.get("replicate_accuracy"), pr.get("chance"),
                     (pr.get("permutation_test") or {}).get("p_value_accuracy"),
                     c.get("spearman_expected_vs_force")))
    di = load(os.path.join(direct_dir, f"force_{dims}.json"))
    if di:
        c = di["correlation"]
        rows.append((f"direct GFP->force {dims} (fine-tuned, warm old enc)",
                     di.get("replicate_accuracy"), di.get("chance"),
                     (di.get("permutation_test") or {}).get("p_value_accuracy"),
                     c.get("spearman_expected_vs_force")))
if not rows:
    print("  (no result JSONs found to compare)"); sys.exit(0)
print(f"  {'model':52s} {'test_acc':>8} {'chance':>7} {'perm_p':>7} {'spearman':>9}")
for name, acc, ch, p, sp in rows:
    def f(x, d=3): return "  n/a" if x is None else f"{x:.{d}f}"
    print(f"  {name:52s} {f(acc):>8} {f(ch,2):>7} {f(p):>7} {f(sp):>9}")
PY

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Done. Outputs in $OUT_DIR/:"
echo "   force.split.json / force.bfgfp_split.json   the shared leak-free split"
echo "   unet_new_{2d,3d}/best.pth                    stage-1 BF->GFP models"
echo "   probe_{2d,3d}.json / .png / .pth             stage-2 force metrics + figures + weights"
echo "   saliency_{2d,3d}/                            stage-3 SmoothGrad (XAI) panels"
echo "════════════════════════════════════════════════════════════"
