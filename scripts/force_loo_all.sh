#!/bin/bash
# Full force rerun on the LOO design — replaces the single 12/4/4 split.
#
# Why rerun everything
# --------------------
# The 12/4/4 split left 4 test replicates against 4 classes, so the best
# attainable p-value was 0.0039 and only from a flawless model; every reported
# number was one replicate's worth of noise. Worse, eval_force_ckpt.py showed
# the 3D reference had collapsed to a confident constant predictor (train CE
# 2.27 vs chance 1.39). Leave-one-replicate-out fixes both: 20 held-out
# predictions instead of 4, and 19 training replicates per fold instead of 12.
#
# Arms (4 by default)
#   gfp:2d  gfp:3d   GFP volume -> force
#   bf:2d   bf:3d    brightfield -> force through a BF->GFP-pretrained encoder
#
# Modes
#   probe  --freeze_encoder, trains only the head. A ResNeXt-50 fine-tuned on
#          ~20 replicate labels is what collapsed; this is the right-sized model.
#   ft     full fine-tune. Off by default — add it with MODES="probe ft" once
#          the probe tells you whether there is any signal to chase.
#
# Cost: 20 folds per arm. 2D probe folds are minutes; 3D fine-tune folds are
# not. Arms run cheapest-first so you get 2D answers before the 3D runs finish,
# and every fold is cached (--fold_cache), so a crash resumes at the fold it
# died on rather than at fold 0. Safe to re-run this script at any time.
#
# Usage
#   bash scripts/force_loo_all.sh                     # 4 arms, probe only
#   MODES="probe ft" bash scripts/force_loo_all.sh    # add fine-tune arms
#   ARMS="gfp:2d" bash scripts/force_loo_all.sh       # one arm
#   PLAN_ONLY=1 bash scripts/force_loo_all.sh         # print the plan, run nothing
#
# Env knobs (defaults in parens)
#   DATA_DIR      (data_phalloidin_mhc_051826_staged)
#   OUT_DIR       (results/force_loo)
#   METADATA      (phalloidin_mhc_mapping_051426_SS edit.xlsx)
#   TARGET_COL    (peak_amplitude_week3)
#   GROUP_COLS    (plate,Tissue)  spreadsheet columns = one replicate
#   N_BINS        (4)   keep 4: under LOO, 4 bins reach p<0.05 at 45% accuracy
#                       while 2 bins need 75% — lower chance buys more power
#                       than the easier task costs
#   ARMS          ("gfp:2d gfp:3d bf:2d bf:3d")
#   MODES         ("probe")            probe | ft | "probe ft"
#   EPOCHS_PROBE  (200)  PATIENCE_PROBE (30)
#   EPOCHS_FT     (150)  PATIENCE_FT    (25)
#   MIN_EPOCHS    (20)   early stopping cannot fire before this
#   INNER_VAL     (0.2)  fraction of TRAIN replicates held out for early stopping
#   ALLOW_PARTIAL (1)   tolerate force-labeled spreadsheet rows whose volume
#                       was never staged. This drop has exactly one
#                       (261805_20X_12L_R003), so strict mode blocks every
#                       run for a file that simply is not on disk. Set 0 to
#                       fail instead — worth doing on a NEW drop, where many
#                       unmatched rows would mean broken filename matching.
#   SEED          (42)
#   FORCE         (0)    1 = ignore cached folds and recompute
#   PLAN_ONLY     (0)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
OUT_DIR="${OUT_DIR:-results/force_loo}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
N_BINS="${N_BINS:-4}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
ARMS="${ARMS:-gfp:2d gfp:3d bf:2d bf:3d}"
MODES="${MODES:-probe}"
EPOCHS_PROBE="${EPOCHS_PROBE:-200}"
PATIENCE_PROBE="${PATIENCE_PROBE:-30}"
EPOCHS_FT="${EPOCHS_FT:-150}"
PATIENCE_FT="${PATIENCE_FT:-25}"
MIN_EPOCHS="${MIN_EPOCHS:-20}"
INNER_VAL="${INNER_VAL:-0.2}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR $DATA_DIR not found" >&2; exit 1; }
[ -f "$METADATA" ] || { echo "ERROR: METADATA $METADATA not found — force labels" \
  "live here; set METADATA=<path>" >&2; exit 1; }
for m in $MODES; do
  case "$m" in probe|ft) ;; *) echo "ERROR: MODES must be probe|ft" >&2; exit 1;; esac
done

# BF->GFP encoder to warm-start from (same search order as decorr_force.sh).
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
ENC_2D="${ENC_2D:-$(find_ckpt 2d)}"
ENC_3D="${ENC_3D:-$(find_ckpt 3d)}"

mkdir -p "$OUT_DIR"
echo "════════════════════════════════════════════════════════════════"
echo " Force, leave-one-replicate-out"
echo "   data=$DATA_DIR  metadata=$METADATA"
echo "   target=$TARGET_COL  groups=$GROUP_COLS"
echo "   n_bins=$N_BINS (chance $(python -c "print(f'{1/$N_BINS:.3f}')"))"
echo "   arms=[$ARMS]  modes=[$MODES]  seed=$SEED  allow_partial=$ALLOW_PARTIAL"
[ "$ALLOW_PARTIAL" = "1" ] && echo "   (check the match report below: expect ~53 matched / 20 replicates)"
echo "   encoder 2d: ${ENC_2D:-<none found>}"
echo "   encoder 3d: ${ENC_3D:-<none found>}"
echo "════════════════════════════════════════════════════════════════"

run_arm() {
  local input="$1" dims="$2" mode="$3"
  local cfg enc epochs patience
  case "$dims" in
    2d) cfg=configs/gfp_classifier.yaml;    enc="$ENC_2D" ;;
    3d) cfg=configs/gfp_classifier_3d.yaml; enc="$ENC_3D" ;;
    *)  echo "ERROR: bad dims '$dims' (want 2d|3d)" >&2; return 1 ;;
  esac
  if [ "$mode" = "probe" ]; then
    epochs="$EPOCHS_PROBE"; patience="$PATIENCE_PROBE"
  else
    epochs="$EPOCHS_FT";    patience="$PATIENCE_FT"
  fi

  local tag="${input}_${dims}_${mode}"
  local out="$OUT_DIR/loo_${tag}.json"
  local cache="$OUT_DIR/folds_${tag}"

  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
    echo ""; echo "▶ [$tag] complete, skipping: $out  (FORCE=1 to redo)"; return 0
  fi

  local flags=()
  if [ -n "$enc" ]; then
    flags+=(--init_from "$enc")
  elif [ "$mode" = "probe" ]; then
    local envname="ENC_2D"; [ "$dims" = "3d" ] && envname="ENC_3D"
    echo "▶ [$tag] SKIPPED: a linear probe needs a BF->GFP encoder and none was" \
         "found. Set $envname=<path> or train BF->GFP first." >&2
    return 0
  else
    echo "▶ [$tag] WARNING: no BF->GFP encoder — training from ImageNet init"
  fi
  [ "$mode" = "probe" ] && flags+=(--freeze_encoder)
  [ "$FORCE" = "1" ] && flags+=(--force_folds)
  [ "$ALLOW_PARTIAL" = "1" ] && flags+=(--allow_partial_match)

  echo ""
  echo "──────────────────────────────────────────────────────────────"
  echo "▶ [$tag] input=$input dims=$dims mode=$mode"
  echo "    epochs<=$epochs patience=$patience min_epochs=$MIN_EPOCHS"
  echo "    folds cached in $cache/"
  echo "──────────────────────────────────────────────────────────────"
  if [ "$PLAN_ONLY" = "1" ]; then echo "    (PLAN_ONLY — not running)"; return 0; fi

  python train_loo_force_classifier.py -c "$cfg" \
    --metadata "$METADATA" --target_col "$TARGET_COL" \
    --group_cols "$GROUP_COLS" \
    --data_dir "$DATA_DIR" --input "$input" \
    --cv_unit replicate --n_bins "$N_BINS" \
    --epochs "$epochs" --patience "$patience" --min_epochs "$MIN_EPOCHS" \
    --inner_val_frac "$INNER_VAL" --seed "$SEED" \
    --fold_cache "$cache" \
    --save_ckpt_dir "$OUT_DIR/ckpts_${tag}" \
    "${flags[@]+"${flags[@]}"}" \
    --output "$out"
  echo "▶ [$tag] done -> $out"
}

# Cheapest first: 2D probes finish while the 3D runs are still going.
for mode in $MODES; do
  for want_dims in 2d 3d; do
    for arm in $ARMS; do
      input="${arm%%:*}"; dims="${arm##*:}"
      [ "$dims" = "$want_dims" ] || continue
      run_arm "$input" "$dims" "$mode"
    done
  done
done

echo ""
echo "▶ summary"
python - "$OUT_DIR" "$N_BINS" <<'PY'
import json, os, sys, glob
from math import comb
out_dir, n_bins = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(out_dir, "loo_*.json")))
if not files:
    print("  (no results yet)"); raise SystemExit
chance = 1.0 / n_bins
print(f"  {'arm':22s} {'n':>3} {'acc':>6} {'chance':>7} {'spear':>7} "
      f"{'perm_p':>7} {'binom_p':>8}")
for f in files:
    d = json.load(open(f))
    n = d.get("n_replicates") or 0
    acc = d.get("replicate_accuracy")
    k = round(acc * n) if (acc is not None and n) else None
    # exact one-sided binomial against chance, alongside the permutation test
    bp = (sum(comb(n, i) * chance**i * (1-chance)**(n-i) for i in range(k, n+1))
          if k is not None else None)
    c = d.get("correlation", {})
    pp = (d.get("permutation_test") or {}).get("p_value_accuracy")
    fmt = lambda x, w=3: "    n/a" if x is None else f"{x:.{w}f}"
    name = os.path.basename(f)[4:-5]
    print(f"  {name:22s} {n:>3} {fmt(acc):>6} {fmt(chance):>7} "
          f"{fmt(c.get('spearman_expected_vs_force')):>7} {fmt(pp):>7} "
          f"{fmt(bp,4):>8}")
print(f"\n  With n=20 folds at chance {chance:.2f}, p<0.05 needs "
      f"{next(k for k in range(21) if sum(comb(20,i)*chance**i*(1-chance)**(20-i) for i in range(k,21)) < 0.05)}"
      f"/20 correct.")
print("  An arm at chance across 20 folds is a real result: force is not")
print("  decodable from these representations at this sample size.")
PY
echo "════════════════════════════════════════════════════════════════"
