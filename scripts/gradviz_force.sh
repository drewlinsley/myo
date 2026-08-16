#!/bin/bash
# SmoothGrad saliency for the trained GFP->force model(s): which parts of a GFP
# volume drive the force-class prediction. Runs on the HELD-OUT TEST volumes by
# default (most meaningful — the model never trained on them).
#
# Requires the force run to have saved checkpoints (force_from_gfp_new.sh does so
# by default at <OUT_DIR>/ckpt_{2d,3d}.pth).
#
# Usage
# -----
#   bash scripts/gradviz_force.sh
#   N_SAMPLES=50 NOISE=0.2 bash scripts/gradviz_force.sh
#   STEMS="stemA stemB" bash scripts/gradviz_force.sh          # explicit volumes
#
# Env knobs
#   OUT_DIR     force results root         (default: results/force_from_gfp_new)
#   DATA_DIR    staged root                (default: data_phalloidin_mhc_051826_staged)
#   CKPT_2D/3D  checkpoint paths           (default: <OUT_DIR>/ckpt_{2d,3d}.pth)
#   RESULT_PREFIX result-json prefix for test-vol pick (default: force; use "probe"
#               for the two-stage models, i.e. reads <OUT_DIR>/probe_<dims>.json)
#   N_SAMPLES   SmoothGrad noisy samples   (default: 25)
#   NOISE       sigma as frac of range     (default: 0.15)
#   TARGET      pred|high|low class logit  (default: pred)
#   VIEW        full|crop                  (default: full = whole H×W plane)
#   MAX_HW      full-view H/W cap (OOM)    (default: 1024 2D / 512 3D; 0=no cap)
#   SAL_SMOOTH  Gaussian sigma (px) on the final map (default: 1.0; 0=off)
#   N_VOLS      #test vols to visualize    (default: 6)
#   STEMS       explicit stems (overrides test-vol auto-pick)
#   ONLY        both|2d|3d                 (default: both)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/force_from_gfp_new}"
DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
N_SAMPLES="${N_SAMPLES:-25}"
NOISE="${NOISE:-0.15}"
TARGET="${TARGET:-pred}"
N_VOLS="${N_VOLS:-6}"
ONLY="${ONLY:-both}"
case "$ONLY" in both|2d|3d) ;; *) echo "ERROR: ONLY must be both|2d|3d" >&2; exit 1;; esac

# Pull held-out TEST volume stems from a result JSON (per_test_replicate[].per_volume[].stem).
test_stems() {
  local js="$1" n="$2"
  python - "$js" "$n" <<'PY'
import json, sys
js, n = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(js))
except Exception:
    sys.exit(0)
out = []
for r in d.get("per_test_replicate", []):
    for v in r.get("per_volume", []):      # classification schema
        out.append(v["stem"])
    out.extend(r.get("stems", []))         # regression schema
print(" ".join(out[:n]))
PY
}

run_arch() {
  local dims="$1" cfg="$2" ckpt="$3"
  [ -f "$ckpt" ] || { echo "[$dims] no checkpoint at $ckpt — run force_from_gfp_new.sh (SAVE_CKPTS=1) first; skipping" >&2; return 0; }
  local stems="${STEMS:-}"
  if [ -z "$stems" ]; then
    stems="$(test_stems "$OUT_DIR/${RESULT_PREFIX:-force}_${dims}.json" "$N_VOLS")"
  fi
  local sal_dir="$OUT_DIR/saliency_${RESULT_PREFIX:-force}_${dims}"
  local stem_flag=(); [ -n "$stems" ] && stem_flag=(--stems $stems)
  local view_flag=(--view "${VIEW:-full}")
  [ -n "${MAX_HW:-}" ] && view_flag+=(--max_hw "$MAX_HW")
  [ -n "${SAL_SMOOTH:-}" ] && view_flag+=(--sal_smooth "$SAL_SMOOTH")
  echo "[$dims] SmoothGrad on ${stems:-<first $N_VOLS vols>}"
  python gradviz_force.py \
    -c "$cfg" --ckpt "$ckpt" --data_dir "$DATA_DIR" \
    --n_samples "$N_SAMPLES" --noise_level "$NOISE" --target "$TARGET" \
    "${view_flag[@]}" --limit "$N_VOLS" "${stem_flag[@]+"${stem_flag[@]}"}" \
    --output_dir "$sal_dir"
  echo "[$dims] -> $sal_dir/"
}

DEF_PFX="${RESULT_PREFIX:-force}"
if [ "$ONLY" = "both" ] || [ "$ONLY" = "2d" ]; then
  run_arch 2d configs/gfp_classifier.yaml "${CKPT_2D:-$OUT_DIR/${DEF_PFX}_ckpt_2d.pth}"
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "3d" ]; then
  run_arch 3d configs/gfp_classifier_3d.yaml "${CKPT_3D:-$OUT_DIR/${DEF_PFX}_ckpt_3d.pth}"
fi
echo "Done. Saliency panels in $OUT_DIR/saliency_${DEF_PFX}_{2d,3d}/"
