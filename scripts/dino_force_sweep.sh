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
# 'categorical' for a string label (treated/untreated, perturbed/control).
# Run inventory_metadata.py first: it lists the categorical columns and, for
# each, how well PLATE ALONE predicts it. A column plate determines is not a
# usable target -- any batch-encoding feature scores on it.
TARGET_TYPE="${TARGET_TYPE:-numeric}"
GROUP_COLS="${GROUP_COLS:-plate,Tissue}"
FEAT_DIR="${FEAT_DIR:-results/dino_features}"
OUT_DIR="${OUT_DIR:-results/dino_sweep}"   # per-target subdir added below
MODEL="${MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}"
MODALITIES="${MODALITIES:-gfp}"
FRAMINGS="${FRAMINGS:-tiled}"
# Foreground-only by default: the fields are sparse tissue on
# background, so background tokens are not evidence about force.
# patch_mean_fg / patch_std_fg are mask-weighted WITHIN each view.
TOKENS="${TOKENS:-patch_mean_fg patch_mean_fg,patch_std_fg}"
NORM_SCOPE="${NORM_SCOPE:-volume}"
TASK="${TASK:-regression}"
N_BINS="${N_BINS:-4}"
# fgmean weights whole views by their foreground fraction, on top of the
# token-level weighting above.
AGGS="${AGGS:-fgmean fgmean+std}"
# Tiles must be mostly TISSUE. 0.75 = at most 25% background. The old 0.02
# let a 98%-background tile into the average; fgmean discounted it, but it was
# still there, and the fgmean+std arm folded it into the dispersion term at
# full weight.
FG_MIN="${FG_MIN:-0.75}"
# 'label' averages the encodings of every FOV sharing a force value into one
# row BEFORE fitting. Those FOVs are one tissue (build_force_groups hard-fails
# on a group with two force values), so this is one row per level of the
# dependent variable. Fitting per-FOV instead treats imaging noise as
# between-tissue variation, and that noise is in the PREDICTORS -- it attenuates
# ridge coefficients in a way averaging predictions afterwards cannot undo.
AGGREGATE="${AGGREGATE:-label}"
# Foreground acts in three independent places. Know which one you are testing:
#   1. view SELECTION  (--fg_min)      drop tiles below a foreground fraction
#   2. view WEIGHTING  (--agg fgmean)  weight whole tiles by their fg fraction
#   3. token WEIGHTING (patch_mean_fg) weight individual patch tokens by the
#                                      mask, inside each view
# All three are ON by default. Hard selection (1) is only safe because
# extraction now uses --mask_scope global: ONE dataset-wide intensity threshold,
# so view_fg_frac means the same thing in every volume. Under the old
# per-volume thresholding a fixed cutoff would have dropped unequal fractions
# across volumes, a bias that could correlate with tissue density and hence
# with the label.
# Set FG_MIN=0 AGGS=mean TOKENS=patch_mean for a no-foreground comparison.
DECONFOUND="${DECONFOUND:-plate}"
N_PERM="${N_PERM:-1000}"
# z_auto is a 35-slice band, so stride 3 sampled only 12 of them. Stride 1
# takes all 35. This does NOT add label information -- force is still one
# number per tissue -- it reduces noise in the volume mean by averaging ~3x
# more views. Extraction cost scales with it; the feature-dir hash includes it,
# so stride-3 features stay cached and untouched.
Z_STRIDE="${Z_STRIDE:-1}"
# Grid-aware terms appended to the pooled vector, space-separated list of
# comma-lists. Each extra value is another config and RAISES the family-wise
# bar, so the default is a single value (no inflation).
#   STRUCTS="none grad_z,centroid"   compares structured pooling to the mean
STRUCTS="${STRUCTS:-none}"
MODEL_CLASS="${MODEL_CLASS:-ridge}"   # ridge|lasso|elasticnet|xgboost
SEED="${SEED:-42}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FORCE="${FORCE:-0}"

[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR $DATA_DIR not found" >&2; exit 1; }
[ -f "$METADATA" ] || { echo "ERROR: METADATA '$METADATA' not found" >&2; exit 1; }

# Scope the results directory by everything that changes what is being
# predicted. Two targets must never share a directory: Stage C's max-statistic
# assumes every JSON in it permuted the SAME labels with the SAME seed.
# Everything that changes the FEATURES or the FIT belongs in this key. It
# used to cover only target/task/bins/model/norm/seed, so changing FG_MIN or
# AGGREGATE reused the previous run's directory -- and Stage C globs the whole
# directory, so a stale JSON from the old settings was silently pooled into the
# ranking AND into the max-statistic null. DECONFOUND is deliberately absent:
# it is swept within a run (dc-none and dc-plate are both configs).
RUN_KEY="$(printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
           "$TARGET_COL" "$TASK" "$N_BINS" "$MODEL" "$NORM_SCOPE" "$SEED" \
           "$TARGET_TYPE" "$FG_MIN" "$AGGREGATE" "$TOKENS" "$AGGS" \
           "$FRAMINGS" "$STRUCTS" "$MODEL_CLASS" "$Z_STRIDE" \
           | cksum | cut -d' ' -f1)"
OUT_DIR="$OUT_DIR/${TARGET_COL}_${TASK}_b${N_BINS}_s${SEED}_${RUN_KEY}"
mkdir -p "$OUT_DIR"
# Belt and braces: even inside a correctly-keyed directory, refuse to pool
# JSONs written by settings that differ from this run.
python - "$OUT_DIR" "$FG_MIN" "$AGGREGATE" "$TARGET_TYPE" <<'PYCHK'
import json, glob, os, sys
d, fg, ag, tt = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
stale = []
for f in glob.glob(os.path.join(d, "*.json")):
    try:
        r = json.load(open(f))
    except Exception:
        stale.append((os.path.basename(f), "unreadable")); continue
    if r.get("fg_min") is None and r.get("aggregate") is None:
        stale.append((os.path.basename(f), "written before these were recorded"))
    elif (abs(float(r.get("fg_min", -1)) - fg) > 1e-9
          or r.get("aggregate") != ag or r.get("target_type", "numeric") != tt):
        stale.append((os.path.basename(f),
                      f"fg_min={r.get('fg_min')} aggregate={r.get('aggregate')} "
                      f"target_type={r.get('target_type')}"))
if stale:
    print(f"  {len(stale)} stale result(s) in {d}:")
    for n, why in stale[:8]:
        print(f"    {n}  [{why}]")
    print("  These came from different settings. Stage C would rank them")
    print("  alongside this run's and fold them into the max-statistic null.")
    print(f"  Delete them first:  rm -rf '{d}'")
    raise SystemExit(1)
PYCHK
echo "════════════════════════════════════════════════════════════════"
echo " Frozen DINOv2 -> force"
echo "   target=$TARGET_COL ($TARGET_TYPE)   deconfound=$DECONFOUND   task=$TASK"
echo "   results -> $OUT_DIR"
echo "   model=$MODEL"
echo "   modalities=[$MODALITIES] framings=$FRAMINGS"
echo "   tokens=[$TOKENS]"
echo "   aggs=[$AGGS]  fg_min=$FG_MIN  aggregate=$AGGREGATE"
echo "   z_stride=$Z_STRIDE  structs=[$STRUCTS]  model=$MODEL_CLASS"
echo "   (fg acts at 3 levels: --fg_min selects views, --agg fgmean weights views,"
echo "    patch_mean_fg weights tokens within a view)"
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
    have_tokens="$(python -c "
import numpy as np, glob, sys
f = sorted(glob.glob(sys.argv[1] + '/*.npz'))
print(' '.join(np.load(f[0]).files) if f else '')
" "$fdir" 2>/dev/null || true)"
    for token in $TOKENS; do
     missing=0
     for t in ${token//,/ }; do
       case " $have_tokens " in *" $t "*) ;; *) missing=1 ;; esac
     done
     if [ "$missing" = "1" ]; then
       echo "  [skip $token] not in the cached features (have: $have_tokens)"
       continue
     fi
     for agg in $AGGS; do
      for st in $STRUCTS; do
      dcs="none"
      [ "$DECONFOUND" != "none" ] && dcs="none $DECONFOUND"
      for dc in $dcs; do
        tag="${mod}_${framing}_${token//,/+}_${agg//+/-}_dc-${dc}"
        [ "$st" != "none" ] && tag="${tag}_st-${st//,/+}"
        [ "$MODEL_CLASS" != "ridge" ] && tag="${tag}_${MODEL_CLASS}"
        out="$OUT_DIR/${tag}.json"
        if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
          echo "  [$tag] cached"; n_cfg=$((n_cfg+1)); continue
        fi
        echo "  [$tag]"
        python probe_force_features.py --features dino --feature_dir "$fdir" \
          --token "$token" --data_dir "$DATA_DIR" --metadata "$METADATA" \
          --target_col "$TARGET_COL" --group_cols "$GROUP_COLS" \
          --modality "$mod" --task "$TASK" --n_bins "$N_BINS" \
          --agg "$agg" --fg_min "$FG_MIN" --aggregate "$AGGREGATE" \
          --target_type "$TARGET_TYPE" --struct "$st" \
          --model_class "$MODEL_CLASS" \
          --deconfound "$dc" --n_perm "$N_PERM" --seed "$SEED" --quiet \
          --output "$out" || echo "    (failed — continuing)"
        n_cfg=$((n_cfg+1))
       done
      done
     done
    done
  done
done

# a shuffled-label control on one config: must land at chance
ctrl_dir="$(ls -d "$FEAT_DIR/$(echo $MODALITIES | cut -d' ' -f1)_${FRAMINGS%%,*}_${NORM_SCOPE}"_* 2>/dev/null | head -1 || true)"
if [ -n "$ctrl_dir" ] && [ -d "$ctrl_dir" ]; then
  echo "  [control: shuffled labels]"
  # Use the SAME token set the sweep is testing. This was hardcoded to
  # patch_mean while the configs used patch_mean_fg, so the leak canary
  # validated a representation nobody was reporting on.
  python probe_force_features.py --features dino --feature_dir "$ctrl_dir" \
    --token "$(echo $TOKENS | cut -d' ' -f1)" \
    --data_dir "$DATA_DIR" --metadata "$METADATA" \
    --target_col "$TARGET_COL" --group_cols "$GROUP_COLS" \
    --modality "$(echo $MODALITIES | cut -d' ' -f1)" \
    --task "$TASK" --n_bins "$N_BINS" --deconfound "$DECONFOUND" \
    --agg "$(echo $AGGS | cut -d' ' -f1)" --fg_min "$FG_MIN" \
    --aggregate "$AGGREGATE" --target_type "$TARGET_TYPE" \
    --struct "$(echo $STRUCTS | cut -d' ' -f1)" --model_class "$MODEL_CLASS" \
    --shuffle --n_perm "$N_PERM" --seed "$SEED" --quiet \
    --output "$OUT_DIR/_control_shuffled.json" || ctrl_failed=1
  if [ "${ctrl_failed:-0}" = "1" ]; then
    echo "  *** THE SHUFFLED CONTROL FAILED TO RUN."
    echo "      It was previously swallowed by '|| true', so a run could"
    echo "      finish with no leak check and still print a ranked table."
    echo "      Nothing below is validated until this runs clean."
  fi
else
  echo "  *** NO SHUFFLED CONTROL: no feature dir matched"
  echo "      $FEAT_DIR/<mod>_${FRAMINGS%%,*}_${NORM_SCOPE}_*"
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
if not os.path.exists(os.path.join(d, "_control_shuffled.json")):
    print("  *** NO SHUFFLED-LABEL CONTROL IN THIS RUN.")
    print("      The control is what rules out fold leakage. Without it a")
    print("      significant row cannot be distinguished from a broken split,")
    print("      so treat everything below as unvalidated.")
rows, nulls = [], []
# A nominal label has no ordering, so rank correlation is not a result for it.
# Rank and family-wise-correct on ACCURACY instead, against the accuracy null.
CAT = False
for f in files:
    if json.load(open(f)).get("target_type") == "categorical":
        CAT = True
        break
STAT = "replicate_accuracy" if CAT else "spearman_pred_vs_force"
NULLK = "null_accuracy" if CAT else "null_spearman"
PK = "permutation_p_accuracy" if CAT else "permutation_p_spearman"
MK = "null_accuracy_mean" if CAT else "null_spearman_mean"
for f in files:
    r = json.load(open(f))
    rho = r.get(STAT)
    if rho is None:
        continue
    rows.append((os.path.basename(f)[:-5], r))
    ns = r.get(NULLK)
    if ns:
        # SIGNED, not |.|: the leave-one-out null sits well below zero, so an
        # anti-correlated model would otherwise be selected as the family best.
        nulls.append(np.asarray(ns))

# S4/S11: matched nulls are the premise of the max-statistic. Verify it.
def _key(r):
    return (r.get("target_col"), r.get("task"), r.get("cv_group"),
            r.get("deconfound"), r.get("n_replicates"),
            r.get("n_permutations"), r.get("perm_scope"))
keys = {}
for name, r in rows:
    keys.setdefault(tuple(_key(r)[:3] + _key(r)[4:]), []).append(name)
if len(keys) > 1:
    print("  *** REFUSING to combine these results: they did not come from one")
    print("      experiment, so their permutations are NOT matched and a")
    print("      max-statistic over them is invalid. Groups found:")
    for k, names in keys.items():
        print(f"        target/task/cv/n_reps/n_perm/scope = {k}")
        for n in names[:4]:
            print(f"          - {n}")
    print("      Delete the stale jsons in this directory and re-run.")
    raise SystemExit(1)

# non-finite rho cannot be ranked; drop with a notice rather than letting
# max() return NaN depending on list order
bad = [n for n, r in rows
       if not isinstance(r.get(STAT), (int, float))
       or r.get(STAT) != r.get(STAT)]
if bad:
    print(f"  NOTE: {len(bad)} config(s) produced a non-finite spearman "
          f"(degenerate features?) and are excluded: {bad[:3]}")
    rows = [(n, r) for n, r in rows if n not in bad]
    nulls = nulls[:len(rows)]

rows.sort(key=lambda t: -(t[1].get(STAT) if t[1].get(STAT) is not None else -9))
print(f"  {'config':38s} {'n':>3} {'acc':>6} {('accuracy' if CAT else 'spearman'):>9} {'perm_p':>8} "
      f"{'null_mean':>10}")
for name, r in rows:
    f = lambda x, k=3: "     n/a" if x is None else f"{x:.{k}f}"
    print(f"  {name:38s} {r.get('n_replicates',0):>3} "
          f"{f(r.get('replicate_accuracy')):>6} "
          f"{f(r.get(STAT)):>9} "
          f"{f(r.get(PK),4):>8} "
          f"{f(r.get(MK)):>10}")

ctrl = os.path.join(d, "_control_shuffled.json")
if os.path.exists(ctrl):
    c = json.load(open(ctrl))
    cs = c.get(STAT)
    print("\n  control (shuffled labels): spearman="
          + ("n/a" if cs is None else f"{cs:.3f}")
          + f" perm_p={c.get(PK)}")
    if (c.get(PK) or 1) < 0.05:
        print("  *** the shuffled control is SIGNIFICANT — the folds leak.")
        print("      Every number above is void until that is fixed.")

if nulls and len(nulls) == len(rows):
    m = min(len(n) for n in nulls)
    stack = np.stack([n[:m] for n in nulls])       # (n_cfg, n_perm)
    maxnull = stack.max(axis=0)                    # best config per permutation
    obs = max((r.get(STAT) if r.get(STAT) is not None else -9)
              for _, r in rows)
    fw = float((np.sum(maxnull >= obs) + 1) / (len(maxnull) + 1))
    print(f"\n  FAMILY-WISE (max-statistic over {len(rows)} configs, "
          f"{m} matched permutations)")
    print(f"    best observed {'accuracy' if CAT else 'spearman'} = "
          f"{obs:+.3f}  (one-sided; signed)")
    print(f"    null max {'accuracy' if CAT else 'spearman'}: "
          f"mean {maxnull.mean():+.3f}, "
          f"95th pct {np.percentile(maxnull,95):+.3f}")
    # Do NOT hardcode the direction of this claim. Plain leave-one-out IS
    # structurally anti-correlated (holding a point out pulls the training
    # mean away from it), but ridge shrinkage and --deconfound plate can pull
    # the null back to ~0. Read it off the permutations instead of asserting
    # it, or the caveat contradicts the numbers printed beside it.
    _nm = float(np.mean(np.concatenate(list(stack))))
    if CAT:
        # The anti-correlation caveat is about a RANK statistic. Accuracy's
        # null centres on the majority-class rate, and calling that "near
        # zero" is nonsense.
        print(f"    null accuracy centres at {_nm:.3f} (majority-class rate);")
        print(f"    only accuracy above the bar above means anything.")
    elif _nm < -0.1:
        print(f"    NOTE the null is centered near {_nm:+.3f} — "
              f"leave-one-out is")
        print("    structurally anti-correlated here, so a NEGATIVE observed")
        print("    value is the null, never a finding.")
    else:
        print(f"    NOTE the null is centered near {_nm:+.3f} — near zero, so")
        print("    this config's null does NOT carry the usual leave-one-out")
        print("    anti-correlation; rho may be read in the ordinary direction.")
    print(f"    family-wise p = {fw:.4f}")
    if fw < 0.05:
        print("    -> survives correction for the whole sweep. Validate next by:")
        print("       leave-one-plate-out, the other force column, and a")
        print("       within-plate re-check before reporting it.")
    else:
        print("    -> does NOT survive. The best row is what a sweep of this")
        print("       size produces by chance; do not report it as a finding.")
        # A negative result is only interpretable next to what the design
        # COULD have seen. Without this, "no signal" and "no power" read the
        # same, and at n=22 replicates they are very different conclusions.
        thr = float(np.percentile(maxnull, 95))
        print(f"\n    POWER: to clear alpha=0.05 after correcting for these")
        print(f"    {len(rows)} configs, a config needed "
              f"{'accuracy' if CAT else 'spearman'} > {thr:+.3f}.")
        print(f"    The best here reached {obs:+.3f}. So this run rules out an")
        print(f"    effect large enough to survive that bar — it does NOT rule")
        print(f"    out a real but weaker association. Distinguishing those")
        print(f"    needs more replicates, not more configs: the threshold is")
        print(f"    set by n={rows[0][1].get('n_replicates')}, not by the model.")
else:
    if any(r.get("degenerate_permutation") for _, r in rows):
        # Distinguish "you forgot to permute" from "permutation is impossible
        # here". Telling someone to rerun with --n_perm > 0 when the label is
        # constant within every stratum sends them in a circle.
        print(f"\n  NO FAMILY-WISE p — AND NONE IS POSSIBLE.")
        print(f"  '{rows[0][1].get('target_col')}' is constant within every "
              f"plate, so the restricted")
        print("  permutation cannot move it. This target is fully confounded")
        print("  with acquisition batch: no image feature can be shown to")
        print("  predict it BEYOND plate, because plate already determines it.")
        print("  This is a property of the experimental design, NOT a result.")
    else:
        print(f"\n  FAMILY-WISE p UNAVAILABLE: {len(nulls)} of {len(rows)} "
              f"configs carry a null distribution.")
        print("  Without it the best row is uncorrected and must NOT be "
              "reported as")
        print("  a finding. Rerun the missing configs with --n_perm > 0.")
    if nulls == [] and any(r.get(PK) is not None
                           for _, r in rows):
        print("  (Configs DO report perm_p but carry no 'null_spearman' array.")
        print("   That combination means stale JSONs written before the")
        print("   null-serialization fix — delete this results dir and rerun.)")
PY
echo "════════════════════════════════════════════════════════════════"
