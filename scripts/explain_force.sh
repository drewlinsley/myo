#!/bin/bash
# Figures + explanation for the frozen-DINOv2 -> force probe.
#
# Run scripts/dino_force_sweep.sh first (or let stage 1 below re-run it).
#
# Stage 1 re-runs the sweep with FORCE=1. That is not optional the first time:
# results written before the readout change carry neither the per-replicate
# plate labels the scatter colors by nor the `.readout.npz` sidecar the
# explanation reads. Set SKIP_SWEEP=1 once you have re-run it.
#
# Stage 3/4 attribute the prediction back to image regions. Read the caveat
# printed at the end: the best config in this sweep is INSIDE its permutation
# null, and an attribution map looks equally convincing either way. That is why
# stage 4 always renders the shuffled-label control beside the real readout --
# the comparison, not the map, is the evidence.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week1}"
FEAT_DIR="${FEAT_DIR:-results/dino_features}"
MODALITY="${MODALITY:-gfp}"
FRAMING="${FRAMING:-tiled}"
NORM_SCOPE="${NORM_SCOPE:-volume}"
FIG_DIR="${FIG_DIR:-results/figures}"
XAI_DIR="${XAI_DIR:-results/xai}"
N_VOLUMES="${N_VOLUMES:-6}"
DENSE="${DENSE:-1}"          # 0 to skip the GPU pass
# Which foreground mask weights the patch tokens in the dense pass.
#   bf   rebuild the brightfield mask the features were extracted with. This
#        is the only mode faithful to the model that was actually fit, and its
#        map integrates to the prediction.
#   dino foreground from PC1 of the patch tokens, thresholded by Otsu with the
#        sign fixed by mean pixel intensity. Same representation the probe
#        reads, so it cannot be misaligned to the tiles -- but it is NOT the
#        mask the cached features were pooled with, so it answers "what would
#        it draw from", not "what did it draw from".
#   none no token weighting at all. Paints background the model never pooled;
#        useful only for showing what that artifact looks like.
MASK_MODE="${MASK_MODE:-bf}"

# ── 1. results ──
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  echo "▶ 1. re-running the sweep so every config writes a readout sidecar"
  FORCE=1 SKIP_EXTRACT="${SKIP_EXTRACT:-1}" TARGET_COL="$TARGET_COL" \
    bash scripts/dino_force_sweep.sh
else
  echo "▶ 1. skipped (SKIP_SWEEP=1)"
fi

# ── 2. locate the results dir the sweep just wrote ──
RES_DIR="$(ls -dt results/dino_sweep/${TARGET_COL}_* 2>/dev/null | head -1 || true)"
if [ -z "$RES_DIR" ]; then
  echo "  ERROR: no results/dino_sweep/${TARGET_COL}_* directory." >&2
  echo "         Run scripts/dino_force_sweep.sh first." >&2
  exit 1
fi
echo ""; echo "▶ 2. figures from $RES_DIR"
python plot_force_probe.py --results_dir "$RES_DIR" --out "$FIG_DIR"

# ── 3. pick the config to explain ──
# The top-ranked config is not automatically explainable: attribution
# decomposes over a MEAN pool, so a config whose leading token block is a
# dispersion (std) term has no per-patch decomposition. Pick the best config
# that does decompose, and say so when that is not the best config overall.
CFG="$(python - "$RES_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
best = (None, -1e9)
best_ok = (None, -1e9)
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    if os.path.basename(f).startswith("_"):
        continue
    try:
        r = json.load(open(f))
    except Exception:
        continue
    rho = r.get("spearman_pred_vs_force")
    if rho is None or rho != rho:
        continue
    name = os.path.basename(f)[:-5]
    if rho > best[1]:
        best = (name, rho)
    tok = str(r.get("feature_names", [""])[0]).split("[")[0]
    lead = tok.split("+")[0]
    if lead in ("patch_mean", "patch_mean_fg", "cls") and \
       "-std" not in name and rho > best_ok[1]:
        best_ok = (name, rho)
if best_ok[0] is None:
    print("")
else:
    if best_ok[0] != best[0]:
        print(f"NOTE {best[0]} scored higher ({best[1]:+.3f}) but pools a "
              f"dispersion term, which has no per-patch decomposition; "
              f"explaining {best_ok[0]} ({best_ok[1]:+.3f}) instead.",
              file=sys.stderr)
    print(best_ok[0])
PY
)"
if [ -z "$CFG" ]; then
  echo "  no config with a mean-pooled leading token; nothing to explain" >&2
  exit 0
fi
READOUT="$RES_DIR/$CFG.readout.npz"
CTRL="$RES_DIR/_control_shuffled.readout.npz"
if [ ! -f "$READOUT" ]; then
  echo "  ERROR: $READOUT missing. Re-run with SKIP_SWEEP unset so the" >&2
  echo "         probe writes the readout sidecar." >&2
  exit 1
fi
ctrl_arg=()
if [ -f "$CTRL" ]; then
  ctrl_arg=(--permuted_readout "$CTRL")
else
  echo "  WARNING: no shuffled-label readout at $CTRL — the control panels"
  echo "           will be missing, and they are the part that tells you"
  echo "           whether the maps mean anything."
fi

# ── 4. explain ──
fmatches="$(ls -d "$FEAT_DIR/${MODALITY}_${FRAMING}_${NORM_SCOPE}"_* 2>/dev/null || true)"
fcount="$(printf '%s' "$fmatches" | grep -c . || true)"
if [ "$fcount" -eq 0 ]; then
  echo "  ERROR: no feature dir $FEAT_DIR/${MODALITY}_${FRAMING}_${NORM_SCOPE}_*" >&2
  exit 1
fi
if [ "$fcount" -gt 1 ]; then
  # After a re-extraction (e.g. MASK_POLARITY=dark) the old cache is still on
  # disk under its old hash. Explaining features from the wrong one would
  # silently attribute a model that was never fit on them.
  echo "  ERROR: $fcount feature dirs match:" >&2
  echo "$fmatches" >&2
  echo "         Delete the stale one (rm -rf <dir>) or pass FEAT_DIR." >&2
  exit 1
fi
fdir="$fmatches"
echo ""; echo "▶ 3. view-level attribution (no GPU)  cfg=$CFG"
python explain_dino_probe.py --readout "$READOUT" \
  --feature_dir "$fdir" --data_dir "$DATA_DIR" --modality "$MODALITY" \
  --level view --n_volumes "$N_VOLUMES" --out "$XAI_DIR" \
  "${ctrl_arg[@]+"${ctrl_arg[@]}"}"

if [ "$DENSE" = "1" ]; then
  echo ""; echo "▶ 4. patch-level attribution (re-runs DINOv2 on $N_VOLUMES volumes)"
  python explain_dino_probe.py --readout "$READOUT" \
    --feature_dir "$fdir" --data_dir "$DATA_DIR" --modality "$MODALITY" \
    --level patch --n_volumes "$N_VOLUMES" --mask_mode "$MASK_MODE" \
    --out "$XAI_DIR" \
    "${ctrl_arg[@]+"${ctrl_arg[@]}"}"
else
  echo ""; echo "▶ 4. skipped (DENSE=0)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " figures  -> $FIG_DIR/"
echo " xai      -> $XAI_DIR/"
echo ""
echo " Before showing these: panel B/C of the summary figure is the"
echo " context for every attribution map here. If the observed statistic"
echo " sits inside its permutation null, the maps show what the readout"
echo " direction responds to — which is a real property of the features —"
echo " but they are NOT evidence that the readout tracks force. The"
echo " shuffled-label control is rendered for exactly that comparison."
echo "════════════════════════════════════════════════════════════════"
