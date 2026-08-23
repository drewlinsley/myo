#!/bin/bash
# Stage 0 — the zero-GPU test of the intensity hypothesis.
#
# The claim under test: per-volume percentile normalization
# (src/data/normalization.py) rescales every tissue to [0,1] and therefore
# erases absolute GFP brightness — a plausible proxy for myotube density and
# hence contraction force.
#
# That claim is testable with NO GPU and NO features, because compute_stats.py
# already wrote each volume's p_low/p_high to disk. If absolute brightness
# predicts force, then those scalars alone should show leave-one-replicate-out
# signal. If they don't, per-volume normalization probably wasn't what was
# holding the models back, and the DINOv2 work should be aimed elsewhere.
#
# Runs in seconds. Run this BEFORE spending GPU time on feature extraction.
#
# It runs five arms, and the last four exist to stop you believing the first:
#   1. real        GFP intensity -> force, leave-one-replicate-out
#   2. shuffled    same, with permuted labels. MUST be at chance, or the
#                  fold logic leaks and arm 1 means nothing.
#   3. canary      one-hot PLATE identity as the only feature. Absolute
#                  intensity carries exposure/gain/staining batch effects, and
#                  replicates share plates, so leave-one-replicate-out does NOT
#                  control for plate. This measures what a purely batch-
#                  confounded feature could earn for free.
#   4. plate-CV    leave-one-PLATE-out, which does control for it. If arm 1 is
#                  strong and arm 4 collapses, arm 1 was a batch effect.
#   5. within-plate features AND force centered within each plate (train rows
#                  only), so only within-plate variation is modelled. With just
#                  4 plates, leave-one-plate-out has 4 folds and almost no
#                  power; this keeps all 20 replicates while remaining immune to
#                  batch by construction. THIS IS THE ARM THAT MATTERS: it is
#                  the strongest confound-free question the dataset can answer.
#
# Usage:  bash scripts/probe_intensity.sh
#         TASK=classification N_BINS=4 bash scripts/probe_intensity.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
METADATA="${METADATA:-phalloidin_mhc_mapping_051426_SS edit.xlsx}"
TARGET_COL="${TARGET_COL:-peak_amplitude_week3}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
MODALITY="${MODALITY:-gfp}"
TASK="${TASK:-regression}"
N_BINS="${N_BINS:-4}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-results/probe_intensity}"

[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR $DATA_DIR not found" >&2; exit 1; }
[ -f "$METADATA" ] || { echo "ERROR: METADATA '$METADATA' not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "════════════════════════════════════════════════════════════════"
echo " Stage 0: does absolute intensity predict force?  (no GPU)"
echo "   data=$DATA_DIR  modality=$MODALITY  target=$TARGET_COL"
echo "   task=$TASK  n_bins=$N_BINS  perms=$N_PERM"
echo "════════════════════════════════════════════════════════════════"

common=(--features stats --data_dir "$DATA_DIR" --metadata "$METADATA"
        --target_col "$TARGET_COL" --group_cols "$GROUP_COLS"
        --modality "$MODALITY" --task "$TASK" --n_bins "$N_BINS"
        --n_perm "$N_PERM" --seed "$SEED")

echo ""; echo "▶ 1/5  real labels, leave-one-replicate-out"
python probe_force_features.py "${common[@]}" \
  --output "$OUT_DIR/intensity_real.json"

echo ""; echo "▶ 2/5  SHUFFLED labels — leak check, must be at chance"
python probe_force_features.py "${common[@]}" --shuffle --quiet \
  --output "$OUT_DIR/intensity_shuffled.json"

echo ""; echo "▶ 3/5  plate canary — what plate identity alone earns"
python probe_force_features.py "${common[@]}" --canary plate --quiet \
  --output "$OUT_DIR/intensity_canary_plate.json"

echo ""; echo "▶ 4/5  leave-one-PLATE-out — controls for acquisition batch"
python probe_force_features.py "${common[@]}" --cv_group plate --quiet \
  --output "$OUT_DIR/intensity_plateCV.json"

echo ""; echo "▶ 5/5  WITHIN-PLATE — batch removed, all 20 replicates kept"
python probe_force_features.py "${common[@]}" --deconfound plate --quiet \
  --output "$OUT_DIR/intensity_withinplate.json"

echo ""; echo "▶ verdict"
python - "$OUT_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
def load(n):
    p = os.path.join(d, f"intensity_{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None
rows = [("real", load("real")), ("shuffled", load("shuffled")),
        ("canary plate", load("canary_plate")), ("plate-CV", load("plateCV")),
        ("within-plate", load("withinplate"))]
print(f"  {'arm':14s} {'n':>3} {'acc':>6} {'chance':>7} {'binom_p':>8} "
      f"{'spearman':>9} {'perm_p':>7}")
for name, r in rows:
    if not r:
        print(f"  {name:14s}  (missing)"); continue
    f = lambda x, k=3: "    n/a" if x is None else f"{x:.{k}f}"
    print(f"  {name:14s} {r['n_replicates']:>3} {f(r['replicate_accuracy']):>6} "
          f"{f(r['chance']):>7} {f(r.get('binomial_p'),4):>8} "
          f"{f(r.get('spearman_pred_vs_force')):>9} "
          f"{f(r.get('permutation_p_spearman'),4):>7}")

real, sh = rows[0][1], rows[1][1]
can, pcv, wip = rows[2][1], rows[3][1], rows[4][1]
print("")
if sh and sh.get("permutation_p_spearman", 1) < 0.05:
    print("  *** SHUFFLED labels beat chance — the folds LEAK. Every other")
    print("      number here is void. Fix that before reading anything else.")
elif real:
    p = real.get("permutation_p_spearman", 1.0)
    if p < 0.05:
        print(f"  Intensity alone predicts force (perm p={p:.4f}).")
        print("  Per-volume normalization WAS discarding real signal —")
        print("  --norm_scope global is justified. Before believing it, check:")
        if can:
            print(f"    - plate canary spearman = "
                  f"{can.get('spearman_pred_vs_force')}: if that is comparable,")
            print("      this is an acquisition batch effect, not biology.")
        if pcv:
            print(f"    - leave-one-plate-out spearman = "
                  f"{pcv.get('spearman_pred_vs_force')}: if this collapses,")
            print("      the replicate-level result was carried by plate.")
    else:
        print(f"  Intensity alone does NOT predict force (perm p={p:.4f}).")
        print("  Per-volume normalization is probably not the bottleneck, so")
        print("  --norm_scope global is unlikely to rescue the models on its")
        print("  own. Weight the DINO work toward richer features (framing,")
        print("  pooling, foreground) rather than toward intensity recovery.")

e2 = (real or {}).get("eta2_plate")
npl = (real or {}).get("n_plates")
if e2 is not None:
    print("")
    print(f"  between-plate share of force variance: eta^2 = {e2:.3f} "
          f"over {npl} plates")
    if e2 > 0.5:
        print("  PLATE CONFOUND (by EFFECT SIZE, not p-value — with so few")
        print("  plates no p-value can settle it): most of this target's")
        print("  variance is between plates, so leave-one-REPLICATE-out cannot")
        print("  validate anything. Any feature encoding acquisition batch")
        print("  scores above chance with no biology in it. Read the")
        print("  within-plate arm, and check inventory_metadata.py for a")
        print("  target column with lower eta^2.")
    else:
        print("  Most variance is WITHIN plates — this target is well posed.")

if wip:
    wp = wip.get("permutation_p_spearman")
    print("")
    wr = wip.get("spearman_pred_vs_force")
    print("  WITHIN-PLATE (the confound-free question): spearman="
          + ("n/a" if wr is None else f"{wr:.3f}")
          + (f"  perm p={wp:.4f}" if wp is not None else ""))
    if wp is not None and wp < 0.05:
        print("  -> Inside a single batch, this feature ranks which tissues pull")
        print("     harder. That is a real result and cannot be a plate effect.")
    elif e2 is not None and e2 > 0.5:
        # confounded target: no within-plate signal is uninterpretable, because
        # there is barely any within-plate variation to predict in the first place
        print("  -> No within-plate signal, AND this target is mostly a plate")
        print("     property (eta^2 = %.3f). There is little within-plate" % e2)
        print("     variation to predict, so this tells you nothing about the")
        print("     images. Switch to a lower-eta^2 target column before")
        print("     concluding anything; more plates is the structural fix.")
    else:
        # well-posed target, uninformative features: this IS interpretable
        print("  -> No within-plate signal from THESE features. Note the target")
        print("     itself is well posed (eta^2 = %.3f), and the plate canary is"
              % (e2 if e2 is not None else float("nan")))
        print("     at chance, so this is not a confound problem — it is a")
        print("     feature problem. A handful of intensity scalars simply do")
        print("     not carry it. That is exactly the case where richer features")
        print("     (DINOv2 framing / pooling / foreground) are worth the GPU")
        print("     time: the experiment is well posed and only the")
        print("     representation is in question.")
PY
echo "════════════════════════════════════════════════════════════════"
