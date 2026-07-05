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
#      ends with a direct-vs-two-stage comparison.
#
# All three share ONE set of knobs (data, force column, bins, split, seed), so the
# comparison is apples-to-apples. Sub-steps skip work that already exists unless
# FORCE=1, so re-running is cheap.
#
# Usage
# -----
#   bash scripts/force_all.sh                 # run everything
#   FORCE=1 bash scripts/force_all.sh         # redo everything
#   SKIP_TWOSTAGE=1 bash scripts/force_all.sh # just direct + saliency
#   ONLY=2d bash scripts/force_all.sh         # only the 2D arch throughout
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
export VAL_FRAC="${VAL_FRAC:-0}"
export SEED="${SEED:-42}"
export ONLY="${ONLY:-both}"
# The 051826 drop has 1 force row with no staged volume; allow the matched subset.
export ALLOW_PARTIAL_MATCH="${ALLOW_PARTIAL_MATCH:-1}"
export PROBE_INPUT="${PROBE_INPUT:-bf}"
export FREEZE="${FREEZE:-1}"
export PROBE_L2="${PROBE_L2:-0.05}"
[ -n "${WEIGHT_DECAY:-}" ] && export WEIGHT_DECAY
[ -n "${FORCE:-}" ] && export FORCE
# SmoothGrad knobs (N_SAMPLES/NOISE/TARGET/N_VOLS) and any other VAR=value you pass
# on the command line are already in the environment, so the sub-scripts inherit
# them automatically — nothing to forward here.

echo "████████████████████████████████████████████████████████████"
echo " FORCE-FROM-GFP — full pipeline"
echo "   data=$DATA_DIR"
echo "   target=$TARGET_COL  groups=$GROUP_COLS  n_bins=$N_BINS  seed=$SEED  ONLY=$ONLY"
echo "   two-stage: probe_input=$PROBE_INPUT freeze=$FREEZE L2=$PROBE_L2"
echo "   partial_match=$ALLOW_PARTIAL_MATCH  force=${FORCE:-0}"
echo "████████████████████████████████████████████████████████████"

if [ "${SKIP_DIRECT:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 1/3  Direct GFP->force classifier"
  bash scripts/force_from_gfp_new.sh
fi

if [ "${SKIP_SALIENCY:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 2/3  SmoothGrad saliency"
  bash scripts/gradviz_force.sh
fi

if [ "${SKIP_TWOSTAGE:-0}" != "1" ]; then
  echo ""; echo "▶▶▶ STEP 3/3  Two-stage BF->GFP -> frozen-rep force probe"
  bash scripts/force_two_stage.sh
fi

# ── Unified summary across everything that ran ──
echo ""; echo "████████████████████████████████████████████████████████████"
echo " UNIFIED SUMMARY"
python - <<'PY'
import json, os
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def row(name, d):
    if not d: return None
    c = d.get("correlation", {})
    return (name, d.get("replicate_accuracy"), d.get("chance"),
            (d.get("permutation_test") or {}).get("p_value_accuracy"),
            c.get("spearman_expected_vs_force"),
            (d.get("config_flags") or {}).get("weight_decay_l2"))
rows = []
for dims in ("2d", "3d"):
    rows.append(row(f"direct GFP->force {dims} (fine-tuned)",
                    load(f"results/force_from_gfp_new/force_{dims}.json")))
    pr = load(f"results/force_two_stage/probe_{dims}.json")
    tag = (f"two-stage {dims} (input={pr.get('input')}, "
           f"freeze={pr.get('freeze_encoder')})") if pr else ""
    rows.append(row(tag, pr))
rows = [r for r in rows if r]
if rows:
    print(f"  {'model':46s} {'acc':>6} {'chance':>7} {'perm_p':>7} {'spear':>6} {'L2':>6}")
    for name, acc, ch, p, sp, l2 in rows:
        f = lambda x, d=3: "  n/a" if x is None else f"{x:.{d}f}"
        print(f"  {name:46s} {f(acc):>6} {f(ch,2):>7} {f(p):>7} {f(sp,2):>6} {f(l2,3):>6}")
else:
    print("  (no result JSONs found)")
print()
for d, what in [("results/force_from_gfp_new", "direct metrics + plots"),
                ("results/force_from_gfp_new/saliency_2d", "2D saliency panels"),
                ("results/force_from_gfp_new/saliency_3d", "3D saliency panels"),
                ("results/force_two_stage", "two-stage probe + BF->GFP models")]:
    print(f"  {'[ok]' if os.path.isdir(d) else '[--]'} {d}  ({what})")
PY
echo "████████████████████████████████████████████████████████████"
