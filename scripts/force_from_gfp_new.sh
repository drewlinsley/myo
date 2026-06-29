#!/bin/bash
# Predict contraction FORCE from GFP volumes on the NEWLY COLLECTED data only
# (phalloidin/MHC 051826 drop), as a classification problem with a single
# replicate-level train/val/test split (NOT leave-one-out).
#
# Train and test live entirely within this one dataset — the model never sees the
# old perturbation tissues. Force is a per-tissue label, so the split is by
# REPLICATE (--group_cols, default plate,Tissue): every FOV of a tissue lands
# wholly in train, val, or test, so force can't leak between FOVs of one tissue.
#
# For both a 2D and a 3D model it:
#   - warm-starts the encoder from the matching BF->GFP U-Net,
#   - discretizes the per-tissue force column into ordinal bins (default terciles),
#   - trains GFP-volume -> force-bin on the TRAIN replicates (train/test split by
#     default; set VAL_FRAC>0 to carve a val split for early stopping),
#   - reports TEST accuracy, per-class accuracy, confusion, and Spearman/Pearson
#     correlation between true force and the model's expected force,
#   - saves a per-model figure. Then emits a 2D-vs-3D comparison.
#
# RUN THE PLAN FIRST:  PLAN_ONLY=1 bash scripts/force_from_gfp_new.sh
#   builds the metadata match + replicate groups + split and exits WITHOUT
#   training — confirm match coverage and that each split spans the force range.
#
# Usage
# -----
#   PLAN_ONLY=1 bash scripts/force_from_gfp_new.sh      # dry run, no GPU
#   bash scripts/force_from_gfp_new.sh                  # train 2D + 3D
#
# Env knobs
# ---------
#   DATA_DIR     staged dataset root with bf/ + gfp/    (default: data_phalloidin_mhc_051826_staged)
#   METADATA     mapping spreadsheet (.xlsx/.csv)        (default: phalloidin_mhc_mapping_051426_SS edit.xlsx)
#   TARGET_COL   force column to classify                (default: peak_amplitude_week3)
#   FILE_COL     spreadsheet column with the filename    (default: file)
#   GROUP_COLS   comma-sep columns = one replicate       (default: plate,Tissue)
#   N_BINS       number of force classes                 (default: 3)
#   BIN_SCHEME   quantile | uniform                      (default: quantile)
#   TEST_FRAC    fraction of replicates for TEST         (default: 0.25)
#   VAL_FRAC     fraction for VAL (0 -> train/test only) (default: 0; set >0 to
#                carve a val split for early stopping instead of train loss)
#   SEED         RNG seed                                 (default: 42)
#   OUT_DIR      results root                            (default: results/force_from_gfp_new)
#   ENC_CKPT_2D  override 2D BF->GFP encoder ckpt path
#   ENC_CKPT_3D  override 3D BF->GFP encoder ckpt path
#   SAVE_CKPTS=1 also persist best weights under <OUT_DIR>/ckpt_{2d,3d}.pth
#   ONLY=2d|3d   run just one arch                       (default: both)
#   FORCE=1      overwrite existing result JSONs
#   PLAN_ONLY=1  dry run: build match + split, then exit (no training)
#   ALLOW_PARTIAL_MATCH=1  train on the matched subset even if some force rows
#                fail to match a staged volume (default: hard error).
#   ALLOW_IMAGENET_FALLBACK=1  permit training from the ImageNet encoder when no
#                BF->GFP checkpoint is found (default: hard error).
#
# Note: cd's to the repo root, so RELATIVE env overrides resolve to the repo
# root, not your CWD. Use absolute paths to avoid surprises.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
FILE_COL="${FILE_COL:-file}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
N_BINS="${N_BINS:-3}"
BIN_SCHEME="${BIN_SCHEME:-quantile}"
TEST_FRAC="${TEST_FRAC:-0.25}"
VAL_FRAC="${VAL_FRAC:-0}"   # 0 -> train/test only (val collapsed into train)
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/force_from_gfp_new}"
ONLY="${ONLY:-both}"

case "$ONLY" in
  both|2d|3d) ;;
  *) echo "ERROR: ONLY must be one of both|2d|3d (got '$ONLY')" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════"
echo " Force-from-GFP (new data, replicate train/val/test split)"
echo "   data=$DATA_DIR"
echo "   meta=$METADATA  target=$TARGET_COL  groups=$GROUP_COLS"
echo "   n_bins=$N_BINS ($BIN_SCHEME)  test=$TEST_FRAC val=$VAL_FRAC seed=$SEED"
echo "   out=$OUT_DIR  ${PLAN_ONLY:+[PLAN_ONLY]}"
echo "════════════════════════════════════════════════════════════"

if [ ! -d "$DATA_DIR/gfp" ]; then
  echo "ERROR: $DATA_DIR/gfp not found. Stage GFP volumes first (stage_nd2.py)." >&2
  exit 1
fi
if [ ! -d "$DATA_DIR/bf" ]; then
  echo "ERROR: $DATA_DIR/bf not found (compute_stats.py needs bf/)." >&2
  exit 1
fi
if [ ! -f "$METADATA" ]; then
  echo "ERROR: metadata '$METADATA' not found. Set METADATA=/path/to/mapping.xlsx" >&2
  exit 1
fi

# ── Percentile stats — ALWAYS run (idempotent: skips stems that already have a
#    stats json, fills any missing ones). A pre-existing but PARTIAL stats/ dir
#    from an earlier eval would otherwise leave some staged volumes without
#    stats and crash dataset construction. gfp stats are added when gfp/ exists.
echo "Computing percentile stats (idempotent; fills any missing stems)…"
stats_flag=()
[ "${FORCE:-0}" = "1" ] && stats_flag=(--force)
python compute_stats.py --data_dir "$DATA_DIR" "${stats_flag[@]+"${stats_flag[@]}"}"

# ── Resolve BF->GFP encoder checkpoints (env override > auto-discover) ──
find_ckpt() {
  local d="$1"
  for cand in \
      "ckpts/unet_${d}_imagenet_pearson_frac100_holdPt/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson_frac100_holdEx/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson_frac100/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson/best.pth" \
      "ckpts/unet_${d}_imagenet/best.pth"; do
    if [ -f "$cand" ]; then echo "$cand"; return 0; fi
  done
  echo ""
}

ENC_CKPT_2D="${ENC_CKPT_2D:-$(find_ckpt 2d)}"
ENC_CKPT_3D="${ENC_CKPT_3D:-$(find_ckpt 3d)}"

# ── One training (or plan) run for a given arch ──
run_arch() {
  local dims="$1" cfg="$2" enc="$3"
  local out="$OUT_DIR/force_${dims}.json"

  if [ "${PLAN_ONLY:-0}" != "1" ] && [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$dims] exists, skipping: $out  (FORCE=1 to redo)"
    return 0
  fi

  local dims_up
  dims_up=$(printf '%s' "$dims" | tr '[:lower:]' '[:upper:]')

  # Warm-start encoder (required unless ALLOW_IMAGENET_FALLBACK); skip the check
  # in PLAN_ONLY (the plan needs no encoder).
  local init_flag=()
  if [ -n "$enc" ]; then
    [ "${PLAN_ONLY:-0}" != "1" ] && echo "[$dims] encoder warm-start: $enc"
    init_flag=(--init_from "$enc")
  elif [ "${PLAN_ONLY:-0}" = "1" ]; then
    : # plan doesn't train; encoder not needed
  elif [ "${ALLOW_IMAGENET_FALLBACK:-0}" = "1" ]; then
    echo "[$dims] WARNING: no BF->GFP $dims encoder found — ALLOW_IMAGENET_FALLBACK=1"
    echo "        set, so training from the ImageNet encoder (NOT the BF->GFP one)."
  else
    echo "[$dims] ERROR: no BF->GFP $dims encoder found. This task needs the" >&2
    echo "        pretrained BF->GFP encoder. Set ENC_CKPT_${dims_up}=<path/best.pth>," >&2
    echo "        or pass ALLOW_IMAGENET_FALLBACK=1 to train an ImageNet baseline." >&2
    echo "        searched: ckpts/unet_${dims}_imagenet_pearson*/best.pth" >&2
    exit 1
  fi

  local extra_flag=()
  [ "${PLAN_ONLY:-0}" = "1" ] && extra_flag+=(--plan_only)
  [ "${ALLOW_PARTIAL_MATCH:-0}" = "1" ] && extra_flag+=(--allow_partial_match)
  [ "${SAVE_CKPTS:-0}" = "1" ] && extra_flag+=(--save_ckpt "$OUT_DIR/ckpt_${dims}.pth")

  if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[$dims] planning (dry run, no GPU) — config=$cfg"
  else
    echo "[$dims] training force classifier — config=$cfg"
  fi
  python train_split_force_classifier.py \
    -c "$cfg" \
    --metadata "$METADATA" \
    --data_dir "$DATA_DIR" \
    --input gfp \
    --target_col "$TARGET_COL" \
    --file_col "$FILE_COL" \
    --group_cols "$GROUP_COLS" \
    --n_bins "$N_BINS" \
    --bin_scheme "$BIN_SCHEME" \
    --test_frac "$TEST_FRAC" \
    --val_frac "$VAL_FRAC" \
    --seed "$SEED" \
    "${init_flag[@]+"${init_flag[@]}"}" \
    "${extra_flag[@]+"${extra_flag[@]}"}" \
    --output "$out"
  if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[$dims] plan → ${out%.json}.plan.json"
  else
    echo "[$dims] done → $out (+ ${out%.json}.png)"
  fi
}

if [ "$ONLY" = "both" ] || [ "$ONLY" = "2d" ]; then
  run_arch 2d configs/gfp_classifier.yaml "$ENC_CKPT_2D"
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "3d" ]; then
  run_arch 3d configs/gfp_classifier_3d.yaml "$ENC_CKPT_3D"
fi

# The plan run for 2D and 3D is identical (same split); one is enough to inspect.
if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "════════════════════════════════════════════════════════════"
  echo " Plan written. Review match coverage + the train/val/test split above"
  echo " (and $OUT_DIR/force_*.plan.json), then rerun WITHOUT PLAN_ONLY to train."
  echo "════════════════════════════════════════════════════════════"
  exit 0
fi

# ── 2D-vs-3D comparison plot (only when both arches ran this invocation) ──
J2="$OUT_DIR/force_2d.json"
J3="$OUT_DIR/force_3d.json"
if [ "$ONLY" = "both" ] && [ -f "$J2" ] && [ -f "$J3" ]; then
  echo "Building 2D-vs-3D comparison plot…"
  python - "$J2" "$J3" "$OUT_DIR/force_2d_vs_3d.png" <<'PY'
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

j2, j3, out = sys.argv[1], sys.argv[2], sys.argv[3]
d2 = json.load(open(j2)); d3 = json.load(open(j3))

# Refuse to compare runs produced under different settings (stale JSON guard).
for k in ("n_bins", "target_col", "bin_scheme", "seed"):
    if d2.get(k) != d3.get(k):
        print(f"Skipping 2D-vs-3D comparison: '{k}' differs "
              f"({d2.get(k)} vs {d3.get(k)}). Rerun both with FORCE=1.")
        sys.exit(0)

models = [("2D", d2), ("3D", d3)]
def warm(d): return "warm" if d.get("warm_started") else "imagenet"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
labels = [f"{n}\n({warm(d)})" for n, d in models]
accs = [d["replicate_accuracy"] for _, d in models]
bars = ax.bar(labels, accs, color=["#4363d8", "#e6194b"])
for b, a, (_, d) in zip(bars, accs, models):
    p = (d.get("permutation_test") or {}).get("p_value_accuracy")
    txt = f"{a:.2f}" + (f"\np={p:.3f}" if p is not None else "")
    ax.text(b.get_x() + b.get_width()/2, a + 0.02, txt, ha="center", fontsize=9)
chance = 1.0 / d2["n_bins"]
ax.axhline(chance, color="gray", ls=":", label=f"chance={chance:.2f}")
ax.set_ylim(0, 1.18); ax.set_ylabel("Test replicate accuracy")
ax.set_title("Accuracy (held-out test)"); ax.legend(fontsize=8)

ax = axes[1]
metrics = [("spearman_expected_vs_force", "Spearman\n(primary)"),
           ("pearson_expected_vs_force", "Pearson")]
x = np.arange(len(metrics)); w = 0.35
for i, (name, d) in enumerate(models):
    vals = [d["correlation"].get(mk) for mk, _ in metrics]
    vals = [np.nan if v is None else v for v in vals]
    ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=["#4363d8", "#e6194b"][i])
ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
ax.axhline(0, color="k", lw=0.6)
ax.set_ylim(-1.05, 1.05); ax.set_ylabel("corr(E[force], true force)")
ax.set_title("Force correlation (test)"); ax.legend(fontsize=8)

tgt = d2.get("target_col", "force")
ntr = len(d2.get("split", {}).get("train", [])); nte = d2.get("n_test_replicates", "?")
fig.suptitle(f"Force-from-GFP (new data): 2D vs 3D | {tgt} | "
             f"train {ntr} / test {nte} reps, "
             f"{d2.get('n_bins','?')} bins ({d2.get('bin_scheme')})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out, dpi=150)
print("Saved", out)

print("\n=== summary (Spearman is the primary force correlation) ===")
for name, d in models:
    c = d["correlation"]
    pt = (d.get("permutation_test") or {}).get("p_value_accuracy")
    print(f"{name} [{warm(d)}]: test_acc={d['replicate_accuracy']:.3f} "
          f"(chance={d['chance']:.3f}, p={pt}) "
          f"spearman={c.get('spearman_expected_vs_force')} "
          f"pearson={c.get('pearson_expected_vs_force')} "
          f"vol_acc={d['volume_accuracy']:.3f}")
PY
fi

echo "════════════════════════════════════════════════════════════"
echo " Done. Outputs in $OUT_DIR/:"
echo "   force_2d.json / .png        per-model TEST metrics + figure"
echo "   force_3d.json / .png"
echo "   force_2d_vs_3d.png          2D vs 3D comparison"
echo "════════════════════════════════════════════════════════════"
