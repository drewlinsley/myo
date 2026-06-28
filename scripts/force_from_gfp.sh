#!/bin/bash
# Predict contraction FORCE from GFP volumes, as a classification problem.
#
# What it does
# ------------
#   For both a 2D and a 3D model:
#     - warm-starts the encoder from the matching BF->GFP U-Net (the encoder
#       that learned GFP-relevant features),
#     - discretizes the per-tissue force column into ordinal bins
#       (default terciles: low / mid / high),
#     - trains GFP-volume -> force-bin with leave-one-REPLICATE-out CV
#       (the only leak-free unit, since force is a per-tissue label),
#     - reports accuracy, per-class accuracy, a confusion matrix, and Pearson /
#       Spearman correlation between true force and the model's expected force,
#     - saves a per-model performance figure.
#   Then it emits a 2D-vs-3D comparison plot.
#
# Why leave-one-replicate-out: every field-of-view of one tissue shares one
# force value. A per-volume split would leak that value between FOVs of the same
# tissue. Splitting by replicate (tissue) holds out all of a tissue's FOVs at
# once, so the headline accuracy is honest.
#
# Usage
# -----
#   bash scripts/force_from_gfp.sh
#
# Env knobs
# ---------
#   DATA_DIR     dataset root with gfp/ + stats/         (default: data)
#   METADATA     metadata csv/xlsx                        (default: data_mapping_drew.csv)
#   TARGET_COL   numeric force column to classify         (default: peak_amplitude_week_5)
#   N_BINS       number of force classes                  (default: 3)
#   BIN_SCHEME   quantile | uniform                       (default: quantile)
#   CV_UNIT      replicate | volume                       (default: replicate)
#   SEED         RNG seed                                  (default: 42)
#   OUT_DIR      results root                             (default: results/force_from_gfp)
#   ENC_CKPT_2D  override 2D BF->GFP encoder ckpt path
#   ENC_CKPT_3D  override 3D BF->GFP encoder ckpt path
#   SAVE_CKPTS=1 also persist per-fold weights under <OUT_DIR>/ckpts_{2d,3d}
#   ONLY=2d|3d   run just one arch (default: both)
#   FORCE=1      overwrite existing result JSONs
#   ALLOW_IMAGENET_FALLBACK=1  permit training from the ImageNet encoder when no
#                BF->GFP checkpoint is found (default: hard error — the task
#                requires the pretrained BF->GFP encoder).
#
# Note: this script cd's to the repo root, so RELATIVE env overrides (DATA_DIR,
# METADATA, OUT_DIR, ENC_CKPT_*) are interpreted relative to the repo root, not
# your current directory. Use absolute paths to avoid surprises.

set -euo pipefail

# Run from the repo root regardless of the caller's CWD (all paths are relative).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
METADATA="${METADATA:-data_mapping_drew.csv}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week_5}"
N_BINS="${N_BINS:-3}"
BIN_SCHEME="${BIN_SCHEME:-quantile}"
CV_UNIT="${CV_UNIT:-replicate}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/force_from_gfp}"
ONLY="${ONLY:-both}"

case "$ONLY" in
  both|2d|3d) ;;
  *) echo "ERROR: ONLY must be one of both|2d|3d (got '$ONLY')" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════"
echo " Force-from-GFP classification"
echo "   data=$DATA_DIR  target=$TARGET_COL  n_bins=$N_BINS ($BIN_SCHEME)"
echo "   cv_unit=$CV_UNIT  seed=$SEED  out=$OUT_DIR"
echo "════════════════════════════════════════════════════════════"

if [ ! -d "$DATA_DIR/gfp" ]; then
  echo "ERROR: $DATA_DIR/gfp not found. Stage GFP volumes first (stage_nd2.py)." >&2
  exit 1
fi
if [ ! -d "$DATA_DIR/stats" ]; then
  echo "ERROR: $DATA_DIR/stats not found. Run compute_stats.py first." >&2
  exit 1
fi
if [ ! -f "$METADATA" ]; then
  echo "ERROR: metadata '$METADATA' not found. Set METADATA=/path/to/mapping.csv" >&2
  exit 1
fi

# ── Resolve BF->GFP encoder checkpoints (env override > auto-discover) ──
find_ckpt() {
  # $1 = "2d" | "3d"; echoes first existing candidate (or empty)
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

# ── One training run for a given arch ──
run_arch() {
  local dims="$1" cfg="$2" enc="$3"
  local out="$OUT_DIR/force_${dims}.json"

  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$dims] exists, skipping: $out  (FORCE=1 to redo)"
    return 0
  fi

  local dims_up
  dims_up=$(printf '%s' "$dims" | tr '[:lower:]' '[:upper:]')

  # Arrays keep flag-presence separate from content and survive paths with
  # spaces. The "${arr[@]+...}" form is safe under `set -u` on bash 3.2+.
  local init_flag=()
  if [ -n "$enc" ]; then
    echo "[$dims] encoder warm-start: $enc"
    init_flag=(--init_from "$enc")
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

  local save_flag=()
  if [ "${SAVE_CKPTS:-0}" = "1" ]; then
    save_flag=(--save_ckpt_dir "$OUT_DIR/ckpts_${dims}")
  fi

  echo "[$dims] training force classifier (config=$cfg)…"
  python train_loo_force_classifier.py \
    -c "$cfg" \
    --metadata "$METADATA" \
    --data_dir "$DATA_DIR" \
    --input gfp \
    --target_col "$TARGET_COL" \
    --n_bins "$N_BINS" \
    --bin_scheme "$BIN_SCHEME" \
    --cv_unit "$CV_UNIT" \
    --seed "$SEED" \
    "${init_flag[@]+"${init_flag[@]}"}" "${save_flag[@]+"${save_flag[@]}"}" \
    --output "$out"
  echo "[$dims] done → $out (+ ${out%.json}.png)"
}

# DATA_DIR is passed explicitly via --data_dir (overrides the config default).
if [ "$ONLY" = "both" ] || [ "$ONLY" = "2d" ]; then
  run_arch 2d configs/gfp_classifier.yaml "$ENC_CKPT_2D"
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "3d" ]; then
  run_arch 3d configs/gfp_classifier_3d.yaml "$ENC_CKPT_3D"
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
for k in ("n_bins", "target_col", "bin_scheme", "seed", "cv_unit"):
    if d2.get(k) != d3.get(k):
        print(f"Skipping 2D-vs-3D comparison: '{k}' differs "
              f"({d2.get(k)} vs {d3.get(k)}). Rerun both with FORCE=1.")
        sys.exit(0)

models = [("2D", d2), ("3D", d3)]

def warm(d):
    return "warm" if d.get("warm_started") else "imagenet"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (A) accuracy + per-model chance line
ax = axes[0]
labels = [f"{n}\n({warm(d)})" for n, d in models]
accs = [d["replicate_accuracy"] for _, d in models]
bars = ax.bar(labels, accs, color=["#4363d8", "#e6194b"])
for b, a, (_, d) in zip(bars, accs, models):
    p = (d.get("permutation_test") or {}).get("p_value_accuracy")
    txt = f"{a:.2f}" + (f"\np={p:.3f}" if p is not None else "")
    ax.text(b.get_x() + b.get_width()/2, a + 0.02, txt, ha="center", fontsize=9)
chance = 1.0 / d2["n_bins"]  # identical across models (guarded above)
ax.axhline(chance, color="gray", ls=":", label=f"chance={chance:.2f}")
ax.set_ylim(0, 1.18); ax.set_ylabel("Replicate-LOO accuracy")
ax.set_title("Accuracy"); ax.legend(fontsize=8)

# (B) force correlation — Spearman primary (Pearson shown but de-emphasized)
ax = axes[1]
metrics = [("spearman_expected_vs_force", "Spearman\n(primary)"),
           ("pearson_expected_vs_force", "Pearson")]
x = np.arange(len(metrics)); w = 0.35
for i, (name, d) in enumerate(models):
    vals = [d["correlation"].get(mk) for mk, _ in metrics]
    vals = [np.nan if v is None else v for v in vals]
    ax.bar(x + (i - 0.5) * w, vals, w, label=name,
           color=["#4363d8", "#e6194b"][i])
ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
ax.axhline(0, color="k", lw=0.6)
ax.set_ylim(-1.05, 1.05); ax.set_ylabel("corr(E[force], true force)")
ax.set_title("Force correlation"); ax.legend(fontsize=8)

tgt = d2.get("target_col", "force")
fig.suptitle(f"Force-from-GFP: 2D vs 3D | {tgt} | "
             f"{d2.get('n_replicates','?')} replicates, "
             f"{d2.get('n_bins','?')} bins ({d2.get('bin_scheme')})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out, dpi=150)
print("Saved", out)

print("\n=== summary (Spearman is the primary force correlation) ===")
for name, d in models:
    c = d["correlation"]
    pt = (d.get("permutation_test") or {}).get("p_value_accuracy")
    print(f"{name} [{warm(d)}]: acc={d['replicate_accuracy']:.3f} "
          f"(chance={d['chance']:.3f}, p={pt}) "
          f"spearman={c.get('spearman_expected_vs_force')} "
          f"pearson={c.get('pearson_expected_vs_force')} "
          f"vol_acc={d['volume_accuracy']:.3f}")
PY
fi

echo "════════════════════════════════════════════════════════════"
echo " Done. Outputs in $OUT_DIR/:"
echo "   force_2d.json / .png        per-model metrics + figure (confusion, per-class, scatter)"
echo "   force_3d.json / .png"
echo "   force_2d_vs_3d.png          2D vs 3D comparison"
echo "════════════════════════════════════════════════════════════"
