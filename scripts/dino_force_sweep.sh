#!/bin/bash
# Frozen DINOv2 -> force, swept honestly.
#
# Context (from scripts/probe_intensity.sh, run first):
#   - peak_amplitude_week3 is 81% between-plate (eta^2=0.809) and NOT usable.
#     peak_amplitude_week1 is well posed (eta^2=0.165), has more labeled data
#     (59 volumes / 22 replicates vs 53/20), and its plate canary is at chance.
#   - Five intensity scalars carry no signal on week1. The target is sound and
#     unconfounded, so that is a FEATURE problem — which is what this sweeps.
#
# Stage A extracts frozen features once (minutes on a GPU, ~40 MB).
# Stage B fits linear probes over them — every config is sub-second, which is
# what makes the permutation machinery below affordable.
#
# Multiple comparisons, taken seriously
# ------------------------------------
# With ~20 replicates and a few dozen correlated configs, several WILL clear a
# nominal p<0.05 under the null. Every config here permutes the SAME replicate
# labels with the SAME seed, so their null distributions are matched. The
# summary therefore computes a MAX-STATISTIC family-wise p: for each
# permutation, take the best |spearman| across all configs, and rank the
# observed best against that. This accounts for both the number of configs and
# their correlation, and is the only claim from this sweep worth reporting.
#
# Usage
#   bash scripts/dino_force_sweep.sh
#   TARGET_COL=peak_amplitude_week1 DECONFOUND=plate bash scripts/dino_force_sweep.sh
#   SKIP_EXTRACT=1 bash scripts/dino_force_sweep.sh     # features already cached

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week1}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
FEAT_DIR="${FEAT_DIR:-results/dino_features}"
OUT_DIR="${OUT_DIR:-results/dino_sweep}"   # per-target subdir added below
MODEL="${MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}"
MODALITIES="${MODALITIES:-gfp}"
FRAMINGS="${FRAMINGS:-tiled}"
TOKENS="${TOKENS:-cls,patch_mean patch_mean,patch_std patch_std}"
NORM_SCOPE="${NORM_SCOPE:-volume}"
TASK="${TASK:-regression}"
N_BINS="${N_BINS:-4}"
AGGS="${AGGS:-mean fgmean mean+std}"
FG_MIN="${FG_MIN:-0.0}"
DECONFOUND="${DECONFOUND:-plate}"
N_PERM="${N_PERM:-1000}"
Z_STRIDE="${Z_STRIDE:-3}"
SEED="${SEED:-42}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FORCE="${FORCE:-0}"

[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR $DATA_DIR not found" >&2; exit 1; }
[ -f "$METADATA" ] || { echo "ERROR: METADATA '$METADATA' not found" >&2; exit 1; }

# Scope the results directory by everything that changes what is being
# predicted. Two targets must never share a directory: Stage C's max-statistic
# assumes every JSON in it permuted the SAME labels with the SAME seed.
RUN_KEY="$(printf '%s|%s|%s|%s|%s|%s' "$TARGET_COL" "$TASK" "$N_BINS" \
           "$MODEL" "$NORM_SCOPE" "$SEED" | cksum | cut -d' ' -f1)"
OUT_DIR="$OUT_DIR/${TARGET_COL}_${TASK}_b${N_BINS}_s${SEED}_${RUN_KEY}"
mkdir -p "$OUT_DIR"
echo "════════════════════════════════════════════════════════════════"
echo " Frozen DINOv2 -> force"
echo "   target=$TARGET_COL   deconfound=$DECONFOUND   task=$TASK"
echo "   results -> $OUT_DIR"
echo "   model=$MODEL"
echo "   modalities=[$MODALITIES] framings=$FRAMINGS"
echo "   tokens=[$TOKENS]"
echo "   aggs=[$AGGS]  fg_min=$FG_MIN"
echo "════════════════════════════════════════════════════════════════"

# ── Stage A: extract once ──
if [ "$SKIP_EXTRACT" != "1" ]; then
  for mod in $MODALITIES; do
    echo ""; echo "▶ A. extracting $mod features -> $FEAT_DIR/"
    xf=(); [ "$FORCE" = "1" ] && xf=(--force)
    python extract_dino_features.py --data_dir "$DATA_DIR" --input "$mod" \
      --output_dir "$FEAT_DIR" --model "$MODEL" --framing "$FRAMINGS" \
      --norm_scope "$NORM_SCOPE" --z_stride "$Z_STRIDE" \
      "${xf[@]+"${xf[@]}"}"
  done
else
  echo ""; echo "▶ A. skipped (SKIP_EXTRACT=1)"
fi

# ── Stage B: sweep linear probes ──
echo ""; echo "▶ B. probing"
n_cfg=0
for mod in $MODALITIES; do
  for framing in ${FRAMINGS//,/ }; do
    matches="$(ls -d "$FEAT_DIR/${mod}_${framing}_${NORM_SCOPE}"_* 2>/dev/null || true)"
    n_match="$(printf '%s' "$matches" | grep -c . || true)"
    if [ "$n_match" -eq 0 ]; then
      echo "  ERROR: no feature dir $FEAT_DIR/${mod}_${framing}_${NORM_SCOPE}_*" >&2
      echo "         (extraction did not run, or ran with other settings)" >&2
      exit 1
    fi
    if [ "$n_match" -gt 1 ]; then
      echo "  ERROR: $n_match feature dirs match ${mod}_${framing}_${NORM_SCOPE}_*:" >&2
      echo "$matches" >&2
      echo "         Config-hash suffixes differ; pass FEAT_DIR explicitly or" >&2
      echo "         delete the stale one. Refusing to guess." >&2
      exit 1
    fi
    fdir="$matches"
    for token in $TOKENS; do
     for agg in $AGGS; do
      dcs="none"
      [ "$DECONFOUND" != "none" ] && dcs="none $DECONFOUND"
      for dc in $dcs; do
        tag="${mod}_${framing}_${token//,/+}_${agg//+/-}_dc-${dc}"
        out="$OUT_DIR/${tag}.json"
        if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
          echo "  [$tag] cached"; n_cfg=$((n_cfg+1)); continue
        fi
        echo "  [$tag]"
        python probe_force_features.py --features dino --feature_dir "$fdir" \
          --token "$token" --data_dir "$DATA_DIR" --metadata "$METADATA" \
          --target_col "$TARGET_COL" --group_cols "$GROUP_COLS" \
          --modality "$mod" --task "$TASK" --n_bins "$N_BINS" \
          --agg "$agg" --fg_min "$FG_MIN" \
          --deconfound "$dc" --n_perm "$N_PERM" --seed "$SEED" --quiet \
          --output "$out" || echo "    (failed — continuing)"
        n_cfg=$((n_cfg+1))
      done
     done
    done
  done
done

# a shuffled-label control on one config: must land at chance
ctrl_dir="$(ls -d "$FEAT_DIR/$(echo $MODALITIES | cut -d' ' -f1)_${FRAMINGS%%,*}_${NORM_SCOPE}"_* 2>/dev/null | head -1 || true)"
if [ -n "$ctrl_dir" ] && [ -d "$ctrl_dir" ]; then
  echo "  [control: shuffled labels]"
  python probe_force_features.py --features dino --feature_dir "$ctrl_dir" \
    --token patch_mean --data_dir "$DATA_DIR" --metadata "$METADATA" \
    --target_col "$TARGET_COL" --group_cols "$GROUP_COLS" \
    --modality "$(echo $MODALITIES | cut -d' ' -f1)" \
    --task "$TASK" --n_bins "$N_BINS" --deconfound "$DECONFOUND" \
    --agg "$(echo $AGGS | cut -d' ' -f1)" --fg_min "$FG_MIN" \
    --shuffle --n_perm "$N_PERM" --seed "$SEED" --quiet \
    --output "$OUT_DIR/_control_shuffled.json" || true
fi

# ── Stage C: ranked table + family-wise max-statistic p ──
echo ""; echo "▶ C. summary"
python - "$OUT_DIR" <<'PY'
import json, os, glob, sys
import numpy as np
d = sys.argv[1]
files = sorted(f for f in glob.glob(os.path.join(d, "*.json"))
               if not os.path.basename(f).startswith("_"))
if not files:
    print("  (no results)"); raise SystemExit
rows, nulls = [], []
for f in files:
    r = json.load(open(f))
    rho = r.get("spearman_pred_vs_force")
    if rho is None:
        continue
    rows.append((os.path.basename(f)[:-5], r))
    ns = r.get("null_spearman")
    if ns:
        nulls.append(np.abs(np.asarray(ns)))

rows.sort(key=lambda t: -abs(t[1].get("spearman_pred_vs_force") or 0))
print(f"  {'config':38s} {'n':>3} {'acc':>6} {'spearman':>9} {'perm_p':>8} "
      f"{'null_mean':>10}")
for name, r in rows:
    f = lambda x, k=3: "     n/a" if x is None else f"{x:.{k}f}"
    print(f"  {name:38s} {r.get('n_replicates',0):>3} "
          f"{f(r.get('replicate_accuracy')):>6} "
          f"{f(r.get('spearman_pred_vs_force')):>9} "
          f"{f(r.get('permutation_p_spearman'),4):>8} "
          f"{f(r.get('null_spearman_mean')):>10}")

ctrl = os.path.join(d, "_control_shuffled.json")
if os.path.exists(ctrl):
    c = json.load(open(ctrl))
    cs = c.get("spearman_pred_vs_force")
    print("\n  control (shuffled labels): spearman="
          + ("n/a" if cs is None else f"{cs:.3f}")
          + f" perm_p={c.get('permutation_p_spearman')}")
    if (c.get("permutation_p_spearman") or 1) < 0.05:
        print("  *** the shuffled control is SIGNIFICANT — the folds leak.")
        print("      Every number above is void until that is fixed.")

if nulls and len(nulls) == len(rows):
    m = min(len(n) for n in nulls)
    stack = np.stack([n[:m] for n in nulls])       # (n_cfg, n_perm)
    maxnull = stack.max(axis=0)                    # best config per permutation
    obs = max(abs(r.get("spearman_pred_vs_force") or 0) for _, r in rows)
    fw = float((np.sum(maxnull >= obs) + 1) / (len(maxnull) + 1))
    print(f"\n  FAMILY-WISE (max-statistic over {len(rows)} configs, "
          f"{m} matched permutations)")
    print(f"    best observed |spearman| = {obs:.3f}")
    print(f"    null max |spearman|: mean {maxnull.mean():.3f}, "
          f"95th pct {np.percentile(maxnull,95):.3f}")
    print(f"    family-wise p = {fw:.4f}")
    if fw < 0.05:
        print("    -> survives correction for the whole sweep. Validate next by:")
        print("       leave-one-plate-out, the other force column, and a")
        print("       within-plate re-check before reporting it.")
    else:
        print("    -> does NOT survive. The best row is what a sweep of this")
        print("       size produces by chance; do not report it as a finding.")
else:
    print(f"\n  FAMILY-WISE p UNAVAILABLE: {len(nulls)} of {len(rows)} configs "
          f"carry a null distribution.")
    print("  Without it the best row is uncorrected and must NOT be reported as")
    print("  a finding. Rerun the missing configs with --n_perm > 0.")
PY
echo "════════════════════════════════════════════════════════════════"
