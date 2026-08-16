#!/bin/bash
# Shortcut-decorrelated force training, end to end:
#
#   A. attr_maps.py    — full-volume attribution maps of the REFERENCE
#                        (shortcut) force model, for every non-test volume.
#   B. train_decorr_force.py — train a NEW 2D and/or 3D classifier whose own
#                        input-attribution (double backprop) is penalized for
#                        correlating with those maps.
#   C. summary         — decorrelated vs baseline test metrics.
#
# Uses the SAME split manifest as the baseline force run (leak-free, identical
# test set), so run scripts/force_all.sh (or force_from_gfp_new.sh) first.
#
# Usage
# -----
#   bash scripts/decorr_force.sh                    # ref = direct 3D model
#   LAMBDA=0.5 ONLY=3d bash scripts/decorr_force.sh
#   REF_CKPT=... REF_CFG=configs/gfp_classifier.yaml bash scripts/decorr_force.sh
#
# Env knobs (defaults in parens)
#   DATA_DIR    (data_phalloidin_mhc_051826_staged)
#   BASE_DIR    (results/force_from_gfp_new)   baseline run root
#   OUT_DIR     (results/force_decorr)
#   REF_CKPT    ($BASE_DIR/force_ckpt_3d.pth)  the shortcut model
#   REF_CFG     (configs/gfp_classifier_3d.yaml)  its architecture config
#   SPLIT_JSON  ($BASE_DIR/force_3d.split.json, falls back to force_2d)
#   LAMBDA      (1.0)   decorrelation weight
#   ATTR_SCORE  (true)  true|pred — which logit's gradient to decorrelate
#   N_SAMPLES   (8)     SmoothGrad samples per tile for the reference maps
#   NOISE       (0.1)
#   ONLY        (both)  both|2d|3d — student arch(es)
#   BATCH_2D/BATCH_3D   override batch size (double backprop ~2x memory)
#   ENC_CKPT_2D/3D      BF->GFP warm-start ckpts (auto-discovered)
#   FORCE       (0)     1 = recompute maps + retrain

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
BASE_DIR="${BASE_DIR:-results/force_from_gfp_new}"
OUT_DIR="${OUT_DIR:-results/force_decorr}"
REF_CKPT="${REF_CKPT:-$BASE_DIR/force_ckpt_3d.pth}"
REF_CFG="${REF_CFG:-configs/gfp_classifier_3d.yaml}"
LAMBDA="${LAMBDA:-1.0}"
ATTR_SCORE="${ATTR_SCORE:-true}"
N_SAMPLES="${N_SAMPLES:-8}"
NOISE="${NOISE:-0.1}"
ONLY="${ONLY:-both}"
ATTR_DIR="${ATTR_DIR:-$OUT_DIR/attr_ref}"
case "$ONLY" in both|2d|3d) ;; *) echo "ERROR: ONLY must be both|2d|3d" >&2; exit 1;; esac

SPLIT_JSON="${SPLIT_JSON:-}"
if [ -z "$SPLIT_JSON" ]; then
  for cand in "$BASE_DIR/force_3d.split.json" "$BASE_DIR/force_2d.split.json"; do
    [ -f "$cand" ] && SPLIT_JSON="$cand" && break
  done
fi
[ -n "$SPLIT_JSON" ] && [ -f "$SPLIT_JSON" ] || {
  echo "ERROR: no split manifest found (looked in $BASE_DIR/force_{3d,2d}.split.json)." >&2
  echo "       Run scripts/force_all.sh (or force_from_gfp_new.sh) first." >&2; exit 1; }
[ -f "$REF_CKPT" ] || {
  echo "ERROR: reference ckpt $REF_CKPT not found (the shortcut model to decorrelate from)." >&2
  exit 1; }

# BF->GFP warm-start ckpts (same search order as force_from_gfp_new.sh)
find_ckpt() {
  local d="$1"
  for cand in \
      "ckpts/unet_${d}_imagenet_pearson_frac100_holdPt/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson_frac100_holdEx/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson_frac100/best.pth" \
      "ckpts/unet_${d}_imagenet_pearson/best.pth" \
      "ckpts/unet_${d}_imagenet/best.pth" \
      "results/force_two_stage/unet_new_${d}/best.pth"; do
    if [ -f "$cand" ]; then echo "$cand"; return 0; fi
  done
  echo ""
}
ENC_CKPT_2D="${ENC_CKPT_2D:-$(find_ckpt 2d)}"
ENC_CKPT_3D="${ENC_CKPT_3D:-$(find_ckpt 3d)}"

mkdir -p "$OUT_DIR"
echo "════════════════════════════════════════════════════════════"
echo " Shortcut decorrelation"
echo "   reference: $REF_CKPT ($REF_CFG)"
echo "   split:     $SPLIT_JSON   lambda=$LAMBDA score=$ATTR_SCORE  ONLY=$ONLY"
echo "════════════════════════════════════════════════════════════"

# ── A. reference attribution maps for every NON-TEST volume ──
echo ""; echo "▶ A. reference attribution maps -> $ATTR_DIR/"
NONTEST_STEMS="$(python - "$SPLIT_JSON" <<'PY'
import json, sys
sp = json.load(open(sys.argv[1]))
gs = sp["train_groups"] + sp["val_groups"]
print(" ".join(s for g in gs for s in sp["groups"][g]["stems"]))
PY
)"
[ -n "$NONTEST_STEMS" ] || { echo "ERROR: split manifest yielded no train/val stems" >&2; exit 1; }
force_flag=(); [ "${FORCE:-0}" = "1" ] && force_flag=(--force)
python attr_maps.py -c "$REF_CFG" --ckpt "$REF_CKPT" --data_dir "$DATA_DIR" \
  --stems $NONTEST_STEMS --n_samples "$N_SAMPLES" --noise_level "$NOISE" \
  --output_dir "$ATTR_DIR" "${force_flag[@]+"${force_flag[@]}"}"

# ── B. decorrelated students ──
run_student() {
  local dims="$1" cfg="$2" enc="$3" bs="$4"
  local out="$OUT_DIR/decorr_${dims}.json"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$dims] exists, skipping: $out  (FORCE=1 to redo)"; return 0
  fi
  local flags=()
  [ -n "$enc" ] && flags+=(--init_from "$enc") \
    || echo "[$dims] WARNING: no BF->GFP encoder found — training from ImageNet init"
  [ -n "$bs" ] && flags+=(--batch_size "$bs")
  echo "[$dims] training decorrelated student (lambda=$LAMBDA)"
  python train_decorr_force.py -c "$cfg" \
    --split_json "$SPLIT_JSON" --data_dir "$DATA_DIR" \
    --attr_dir "$ATTR_DIR" --attr_lambda "$LAMBDA" --attr_score "$ATTR_SCORE" \
    --save_ckpt "$OUT_DIR/decorr_ckpt_${dims}.pth" \
    "${flags[@]+"${flags[@]}"}" --output "$out"
  echo "[$dims] done -> $out"
}
if [ "$ONLY" = "both" ] || [ "$ONLY" = "2d" ]; then
  run_student 2d configs/gfp_classifier.yaml "$ENC_CKPT_2D" "${BATCH_2D:-}"
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "3d" ]; then
  run_student 3d configs/gfp_classifier_3d.yaml "$ENC_CKPT_3D" "${BATCH_3D:-}"
fi

# ── C. decorrelated vs baseline ──
echo ""; echo "▶ C. summary (decorrelated vs baseline, same test replicates)"
python - "$BASE_DIR" "$OUT_DIR" <<'PY'
import json, os, sys
base, out = sys.argv[1], sys.argv[2]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
print(f"  {'model':22s} {'acc':>6} {'chance':>7} {'spear':>7} {'perm_p':>7} {'attr_corr':>10}")
for dims in ("2d", "3d"):
    for tag, d in [(f"baseline {dims}", load(f"{base}/force_{dims}.json")),
                   (f"decorr   {dims}", load(f"{out}/decorr_{dims}.json"))]:
        if not d:
            continue
        c = d.get("correlation", {})
        pp = (d.get("permutation_test") or {}).get("p_value_accuracy")
        ac = d.get("history", [{}])[-1].get("attr_corr") if "history" in d else None
        f = lambda x, k=3: "   n/a" if x is None else f"{x:.{k}f}"
        print(f"  {tag:22s} {f(d.get('replicate_accuracy')):>6} "
              f"{f(d.get('chance'),2):>7} "
              f"{f(c.get('spearman_expected_vs_force')):>7} {f(pp):>7} "
              f"{f(ac,4):>10}")
print("\n  saliency of a decorrelated model (reuses the baseline test-vol pick):")
print("    OUT_DIR=results/force_decorr RESULT_PREFIX=decorr \\")
print("      CKPT_2D=results/force_decorr/decorr_ckpt_2d.pth \\")
print("      CKPT_3D=results/force_decorr/decorr_ckpt_3d.pth \\")
print("      DATA_DIR=<staged root> bash scripts/gradviz_force.sh")
PY
echo "════════════════════════════════════════════════════════════"
