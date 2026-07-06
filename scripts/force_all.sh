#!/bin/bash
# ONE script for the whole force-from-GFP investigation on the new data:
#
#   1. Direct GFP->force classifier (2D + 3D)      -> results/force_from_gfp_new/
#      warm-started encoder, replicate train/test split, accuracy + correlation
#      + confusion/scatter plots + 2D-vs-3D comparison.
#   2. SmoothGrad saliency of those models          -> results/force_from_gfp_new/saliency_{2d,3d}/
#      which GFP features drive each held-out force call.
#   3. Two-stage BF->GFP -> frozen-rep force probe  -> results/force_two_stage/
#      train BF->GFP on this data (same split), freeze it, linear-probe force;
#      also emits perf plots AND SmoothGrad (XAI) saliency for the probe models,
#      and ends with a direct-vs-two-stage comparison.
#
# All three share ONE set of knobs (data, force column, bins, split, seed), so the
# comparison is apples-to-apples. Sub-steps skip work that already exists unless
# FORCE=1, so re-running is cheap.
#
# Usage
# -----
#   bash scripts/force_all.sh                 # run everything, RETRAINING all models
#   FORCE=0 bash scripts/force_all.sh         # reuse existing outputs (skip retrain)
#   SKIP_TWOSTAGE=1 bash scripts/force_all.sh # just direct + saliency
#   ONLY=2d bash scripts/force_all.sh         # only the 2D arch throughout
#
# NOTE: retrains from scratch by default (FORCE=1) — redoes stats, split, the
# BF->GFP models, and every classifier. Set FORCE=0 to reuse what already exists.
#
# Key knobs (defaults in parens) — all forwarded to the sub-scripts:
#   DATA_DIR (data_phalloidin_mhc_051826_staged)  METADATA (phalloidin_..._SS edit.xlsx)
#   TARGET_COL (peak_amplitude_week3)  GROUP_COLS (plate,Tissue)
#   N_BINS (4)  BIN_SCHEME (quantile)  TEST_FRAC (0.25)  VAL_FRAC (0)  SEED (42)
#   ONLY (both|2d|3d)  ALLOW_PARTIAL_MATCH (1 here — the drop has 1 unmatched row)
#   PROBE_INPUT (bf|gfp, default bf)  FREEZE (1)  PROBE_L2 (0.05)  WEIGHT_DECAY (cfg)
#   SmoothGrad: N_SAMPLES (25)  NOISE (0.15)  TARGET (pred)  N_VOLS (6)
#   Stage gates: SKIP_DIRECT / SKIP_SALIENCY / SKIP_TWOSTAGE (0)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Shared knobs (export so the sub-scripts inherit identical settings).
export DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
export METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
export TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
export FILE_COL="${FILE_COL:-file}"
export GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
export N_BINS="${N_BINS:-4}"
export BIN_SCHEME="${BIN_SCHEME:-quantile}"
export TEST_FRAC="${TEST_FRAC:-0.25}"
export VAL_FRAC="${VAL_FRAC:-0.2}"   # held-out early stopping (0 = stop on train loss)
export SEED="${SEED:-42}"
export ONLY="${ONLY:-both}"
# TASK: classification | regression (selects trainers, output prefixes, metrics).
export TASK="${TASK:-classification}"
case "$TASK" in
  classification) PREFIX=force; PPREFIX=probe ;;
  regression)     PREFIX=reg;   PPREFIX=probe_reg ;;
  *) echo "ERROR: TASK must be classification|regression (got '$TASK')" >&2; exit 1 ;;
esac
# The 051826 drop has 1 force row with no staged volume; allow the matched subset.
export ALLOW_PARTIAL_MATCH="${ALLOW_PARTIAL_MATCH:-1}"
export PROBE_INPUT="${PROBE_INPUT:-bf}"
export FREEZE="${FREEZE:-1}"
export PROBE_L2="${PROBE_L2:-0.05}"
[ -n "${WEIGHT_DECAY:-}" ] && export WEIGHT_DECAY
# force_all.sh RETRAINS from scratch by default (redoes stats, split, BF->GFP, and
# every classifier). Set FORCE=0 to instead reuse any outputs that already exist.
export FORCE="${FORCE:-1}"
# SmoothGrad knobs (N_SAMPLES/NOISE/TARGET/N_VOLS) and any other VAR=value you pass
# on the command line are already in the environment, so the sub-scripts inherit
# them automatically — nothing to forward here.

echo "████████████████████████████████████████████████████████████"
echo " FORCE-FROM-GFP — full pipeline"
echo "   data=$DATA_DIR"
echo "   TASK=$TASK  target=$TARGET_COL  groups=$GROUP_COLS  seed=$SEED  ONLY=$ONLY"
echo "   two-stage: probe_input=$PROBE_INPUT freeze=$FREEZE L2=$PROBE_L2  val_frac=$VAL_FRAC"
echo "   partial_match=$ALLOW_PARTIAL_MATCH  retrain(FORCE)=$FORCE"
echo "████████████████████████████████████████████████████████████"

if [ "${SKIP_DIRECT:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 1/3  Direct GFP->force ($TASK)"
  bash scripts/force_from_gfp_new.sh
fi

if [ "${SKIP_SALIENCY:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 2/3  SmoothGrad saliency (direct model)"
  RESULT_PREFIX="$PREFIX" \
    CKPT_2D="results/force_from_gfp_new/${PREFIX}_ckpt_2d.pth" \
    CKPT_3D="results/force_from_gfp_new/${PREFIX}_ckpt_3d.pth" \
    bash scripts/gradviz_force.sh
fi

if [ "${SKIP_TWOSTAGE:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 3/3  Two-stage BF->GFP -> frozen-rep force probe"
  bash scripts/force_two_stage.sh
fi

# ── Unified summary across everything that ran ──
echo ""; echo "████████████████████████████████████████████████████████████"
echo " UNIFIED SUMMARY"
python - "$TASK" "$PREFIX" "$PPREFIX" <<'PY'
import json, os, sys
task, prefix, pprefix = sys.argv[1:4]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def fmt(x, d=3): return "  n/a" if x is None else f"{x:.{d}f}"
rows = []
for dims in ("2d", "3d"):
    for tag, d in [(f"direct {dims}", load(f"results/force_from_gfp_new/{prefix}_{dims}.json")),
                   (f"two-stage {dims}", load(f"results/force_two_stage/{pprefix}_{dims}.json"))]:
        if not d:
            continue
        if task == "regression":
            m = d["metrics"]
            rows.append((tag, m["mae"], m["r2"], m["spearman"],
                         m["baseline_mae_predict_mean"]))
        else:
            c = d.get("correlation", {})
            rows.append((tag, d.get("replicate_accuracy"), d.get("chance"),
                         c.get("spearman_expected_vs_force"),
                         (d.get("permutation_test") or {}).get("p_value_accuracy")))
if not rows:
    print("  (no result JSONs found)")
elif task == "regression":
    print(f"  {'model':22s} {'MAE':>7} {'R2':>7} {'spear':>7} {'base_MAE':>9}")
    for n, a, b, c, e in rows:
        print(f"  {n:22s} {fmt(a):>7} {fmt(b):>7} {fmt(c):>7} {fmt(e):>9}")
else:
    print(f"  {'model':22s} {'acc':>7} {'chance':>7} {'spear':>7} {'perm_p':>7}")
    for n, a, b, c, e in rows:
        print(f"  {n:22s} {fmt(a):>7} {fmt(b,2):>7} {fmt(c):>7} {fmt(e):>7}")
print()
P, PP = prefix, pprefix
for d, what in [("results/force_from_gfp_new", "direct: metrics + perf plots"),
                (f"results/force_from_gfp_new/saliency_{P}_2d", "direct 2D XAI"),
                (f"results/force_from_gfp_new/saliency_{P}_3d", "direct 3D XAI"),
                ("results/force_two_stage", "two-stage: probe + BF->GFP models"),
                (f"results/force_two_stage/saliency_{PP}_2d", "two-stage 2D XAI"),
                (f"results/force_two_stage/saliency_{PP}_3d", "two-stage 3D XAI")]:
    print(f"  {'[ok]' if os.path.isdir(d) else '[--]'} {d}  ({what})")
PY
echo "████████████████████████████████████████████████████████████"
