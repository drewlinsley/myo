"""Leave-one-replicate-out force probe over CACHED per-volume features.

Why this exists
---------------
Every GPU force model so far sits at chance, and each LOO arm costs 20 model
trainings — so nothing can be swept. This probe decouples feature extraction
from fitting: given features already on disk, a full 20-fold LOO takes well
under a second, which makes label-permutation nulls and honest multiple-
comparison control affordable for the first time.

It ships with a zero-GPU feature mode that tests the central hypothesis
directly. The per-volume intensity percentiles written by compute_stats.py are
already on disk. If "absolute GFP brightness proxies myotube density proxies
contraction force" is true, then p_high ALONE should already carry LOO signal:

    python probe_force_features.py --features stats \
        --data_dir data_phalloidin_mhc_051826_staged \
        --metadata "phalloidin_mhc_mapping_051426_SS edit.xlsx" \
        --target_col peak_amplitude_week3 --task regression

If that is at chance, per-volume normalization was probably not the thing
holding the models back. If it predicts force, then normalizing each volume by
its own percentiles was discarding the signal — and any richer feature set
should beat this baseline to justify itself.

Leak rules (mirroring train_loo_force_classifier.py)
----------------------------------------------------
Force is a per-tissue label, so the CV unit is the replicate. Within every
fold, EVERY data-driven quantity is fit on training replicates only: bin
edges, feature standardization, PCA, and the inner hyperparameter search.
Held-out volumes are averaged to one prediction per replicate before scoring.

Confound checks (run these before believing any positive result)
----------------------------------------------------------------
  --shuffle        permute replicate force labels; MUST land at chance.
                   Anything above chance means the fold logic leaks.
  --canary plate   predict force from one-hot plate identity alone. Absolute
                   intensity carries exposure/gain/staining batch effects, and
                   replicates share plates, so leave-one-REPLICATE-out does not
                   control for plate. This is the ceiling on what a purely
                   batch-confounded feature could "earn".
  --cv_group plate leave-one-PLATE-out instead, which does control for it.
"""

import os
import json
import glob
import argparse
import importlib.util
from math import comb

import numpy as np

# Import force_metadata BY PATH: src/data/__init__.py pulls in torch, and this
# probe is deliberately torch-free so it runs anywhere the features do.
_fm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "data", "force_metadata.py")
_spec = importlib.util.spec_from_file_location("_force_metadata", _fm_path)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)
build_force_groups = _fm.build_force_groups


# --------------------------------------------------------------------------
# binning (train-only edges, same semantics as train_loo_force_classifier)
# --------------------------------------------------------------------------
def compute_bin_edges(values, n_bins, scheme="quantile"):
    v = np.asarray(sorted(float(x) for x in values), dtype=np.float64)
    if scheme == "uniform":
        return np.linspace(v.min(), v.max(), n_bins + 1)[1:-1]
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(v, qs)


def assign_bin(value, edges):
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64),
                               float(value), side="right"))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def binom_p(n_correct, n, chance):
    """Exact one-sided binomial p for >= n_correct successes."""
    return float(sum(comb(n, i) * chance**i * (1 - chance)**(n - i)
                     for i in range(int(n_correct), n + 1)))


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def load_stats_features(stats_dir, stems, modality):
    """Zero-GPU baseline: the per-volume intensity percentiles already on disk.

    compute_stats.py writes {"bf": {"p_low","p_high"}, "gfp": {...}, "z_auto"}.
    Absolute brightness is exactly what per-volume normalization destroys, so
    this is the most direct possible test of that hypothesis.
    """
    feats, kept = [], []
    for s in stems:
        p = os.path.join(stats_dir, f"{s}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            st = json.load(f)
        if modality not in st:
            continue
        lo = float(st[modality]["p_low"])
        hi = float(st[modality]["p_high"])
        z = st.get("z_auto") or [0, 0]
        feats.append([lo, hi, hi - lo,
                      np.log10(max(hi, 1.0)), float(z[1] - z[0])])
        kept.append(s)
    names = ["p_low", "p_high", "p_range", "log10_p_high", "z_width"]
    return np.asarray(feats, dtype=np.float64), kept, names


def load_dino_features(feature_dir, stems, token, agg, fg_min):
    """Per-view DINOv2 features -> one vector per volume."""
    feats, kept = [], []
    for s in stems:
        p = os.path.join(feature_dir, f"{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if token not in z:
            raise SystemExit(f"{p}: no '{token}' array (has {list(z.keys())})")
        v = np.asarray(z[token], dtype=np.float64)      # (n_views, D)
        if fg_min > 0 and "view_fg_frac" in z:
            keep = np.asarray(z["view_fg_frac"], float) >= fg_min
            if keep.any():
                v = v[keep]
        if agg == "median":
            vec = np.median(v, axis=0)
        elif agg == "mean+std":
            vec = np.concatenate([v.mean(axis=0), v.std(axis=0)])
        else:
            vec = v.mean(axis=0)
        feats.append(vec)
        kept.append(s)
    if not feats:
        raise SystemExit(f"no .npz features found in {feature_dir}")
    names = [f"{token}[{i}]" for i in range(len(feats[0]))]
    return np.asarray(feats, dtype=np.float64), kept, names


# --------------------------------------------------------------------------
# the LOO probe
# --------------------------------------------------------------------------
def run_loo(X, vol_group, vol_force, cv_groups, task, n_bins, pca_dim,
            seed=0, fixed_alpha=None, deconfound=None):
    """Leave-one-group-out; returns per-replicate predictions.

    X            (n_vol, D)   volume-level features
    vol_group    (n_vol,)     replicate id per volume (the AGGREGATION unit)
    vol_force    (n_vol,)     force per volume (constant within a replicate)
    cv_groups    (n_vol,)     the HELD-OUT unit (replicate, or plate)
    fixed_alpha  skip the nested search and use this (for permutation nulls,
                 where selecting on real data would bias the null)
    deconfound   (n_vol,) confound id per volume (e.g. plate), or None.
                 When given, features AND force are centered within each
                 confound level before fitting, so only WITHIN-level variation
                 is modelled. This asks the confound-free question — "inside one
                 acquisition batch, does the image rank which tissues pull
                 harder?" — and keeps all 20 replicates of power, instead of
                 dropping to the 4 folds that leave-one-plate-out allows.
                 Centering constants come from TRAINING rows only; using the
                 held-out replicate's own value to build its plate mean would
                 leak the label.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold

    reps = sorted(set(vol_group))
    rep_force = {g: float(vol_force[list(vol_group).index(g)]) for g in reps}
    folds = sorted(set(cv_groups))
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    pred_force, true_force, pred_bin, true_bin, held_reps = [], [], [], [], []

    for held in folds:
        te = np.array([g == held for g in cv_groups])
        tr = ~te
        if tr.sum() < 3 or te.sum() == 0:
            continue
        tr_groups = np.asarray(vol_group)[tr]

        # Within-confound centering, fit on TRAINING rows only.
        Xf = np.array(X, dtype=np.float64, copy=True)
        yf = np.array(vol_force, dtype=np.float64, copy=True)
        if deconfound is not None:
            dc = np.asarray(deconfound, dtype=object)
            gmu_X = Xf[tr].mean(axis=0)
            gmu_y = float(yf[tr].mean())
            for lvl in set(dc):
                m_tr = tr & (dc == lvl)
                mu_X = Xf[m_tr].mean(axis=0) if m_tr.any() else gmu_X
                mu_y = float(yf[m_tr].mean()) if m_tr.any() else gmu_y
                m_all = dc == lvl
                Xf[m_all] -= mu_X
                yf[m_all] -= mu_y

        # every data-driven quantity is fit on TRAINING rows only
        sc = StandardScaler().fit(Xf[tr])
        Xtr, Xte = sc.transform(Xf[tr]), sc.transform(Xf[te])
        n_comp = int(min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
        if n_comp >= 1 and Xtr.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

        ytr_force = yf[tr]
        fold_rep_force = {}
        for g, v in zip(np.asarray(vol_group)[tr], ytr_force):
            fold_rep_force.setdefault(g, float(v))
        for g, v in zip(np.asarray(vol_group)[te], yf[te]):
            fold_rep_force.setdefault(g, float(v))
        edges = compute_bin_edges(
            [fold_rep_force[g] for g in sorted(set(tr_groups))], n_bins)
        ytr_bin = np.array([assign_bin(v, edges) for v in ytr_force])

        # sample weights: a 4-FOV tissue must not outvote a 1-FOV one
        cnt = {g: int((tr_groups == g).sum()) for g in set(tr_groups)}
        w = np.array([1.0 / cnt[g] for g in tr_groups])

        if task == "regression":
            # centered force is signed, so log10 only applies to the raw target
            ytr = (ytr_force if deconfound is not None
                   else np.log10(np.clip(ytr_force, 1e-9, None)))
            alpha = fixed_alpha
            if alpha is None:
                alpha = _inner_select(
                    Xtr, ytr, tr_groups, w, grid,
                    lambda a: Ridge(alpha=a), "r2", seed)
            m = Ridge(alpha=alpha).fit(Xtr, ytr, sample_weight=w)
            p = m.predict(Xte)
        else:
            if len(set(ytr_bin)) < 2:
                continue
            C = fixed_alpha
            if C is None:
                C = _inner_select(
                    Xtr, ytr_bin, tr_groups, w, grid,
                    lambda c: LogisticRegression(C=c, max_iter=2000,
                                                 class_weight="balanced"),
                    "acc", seed)
            m = LogisticRegression(C=C, max_iter=2000,
                                   class_weight="balanced").fit(
                Xtr, ytr_bin, sample_weight=w)
            p = m.predict_proba(Xte)

        # collapse held-out VOLUMES to one prediction per replicate
        te_groups = np.asarray(vol_group)[te]
        for g in sorted(set(te_groups)):
            sel = te_groups == g
            y_true_g = fold_rep_force[g]
            tb = assign_bin(y_true_g, edges)
            if task == "regression":
                mp = float(np.mean(p[sel]))
                pred_force.append(mp)
                pb = assign_bin(mp if deconfound is not None else 10 ** mp, edges)
            else:
                pr = p[sel].mean(axis=0)
                pb = int(np.argmax(pr))
                pred_force.append(float(np.dot(pr, _bin_reps(
                    edges, n_bins,
                    [fold_rep_force[q] for q in sorted(set(tr_groups))]))))
            held_reps.append(g)
            true_force.append(y_true_g)
            pred_bin.append(pb)
            true_bin.append(tb)

    return {"replicates": held_reps,
            "true_force": np.asarray(true_force),
            "pred_score": np.asarray(pred_force),
            "true_bin": np.asarray(true_bin),
            "pred_bin": np.asarray(pred_bin)}


def _bin_reps(edges, n_bins, train_forces):
    """Representative force per bin, from TRAIN replicates only."""
    tf = np.asarray(train_forces, float)
    out = np.zeros(n_bins)
    for b in range(n_bins):
        m = np.array([assign_bin(v, edges) == b for v in tf])
        out[b] = tf[m].mean() if m.any() else tf.mean()
    return out


def _inner_select(X, y, groups, w, grid, make, metric, seed):
    """Nested hyperparameter search over TRAINING replicates only."""
    from sklearn.model_selection import GroupKFold
    uniq = sorted(set(groups))
    n_splits = min(5, len(uniq))
    if n_splits < 2:
        return grid[len(grid) // 2]
    gkf = GroupKFold(n_splits=n_splits)
    best, best_s = grid[len(grid) // 2], -np.inf
    for hp in grid:
        scores = []
        for itr, iva in gkf.split(X, y, groups=groups):
            if len(set(y[itr])) < 2 and metric == "acc":
                continue
            try:
                m = make(hp).fit(X[itr], y[itr], sample_weight=w[itr])
                pr = m.predict(X[iva])
            except Exception:
                continue
            if metric == "acc":
                scores.append(float(np.mean(pr == y[iva])))
            else:
                ss = ((y[iva] - pr) ** 2).sum()
                st = ((y[iva] - y[iva].mean()) ** 2).sum()
                scores.append(1.0 - ss / st if st > 0 else 0.0)
        if scores and np.mean(scores) > best_s:
            best_s, best = float(np.mean(scores)), hp
    return best


def _permute_replicate_force(vol_group, vol_force, rng):
    """Permute force across REPLICATES (every volume of a tissue keeps one
    shared value). Permuting per-volume would break the group structure and
    make the null easier than the real problem."""
    reps = sorted(set(vol_group))
    rf = {}
    for g, f in zip(vol_group, vol_force):
        rf.setdefault(g, float(f))
    shuffled = dict(zip(reps, rng.permutation([rf[g] for g in reps])))
    return [shuffled[g] for g in vol_group]


def score(res, n_bins, task):
    n = len(res["true_bin"])
    chance = 1.0 / n_bins
    acc = float(np.mean(res["true_bin"] == res["pred_bin"])) if n else float("nan")
    rho = spearman(res["true_force"], res["pred_score"])
    return {"n_replicates": n, "chance": chance,
            "replicate_accuracy": acc,
            "n_correct": int(np.sum(res["true_bin"] == res["pred_bin"])),
            "binomial_p": binom_p(round(acc * n), n, chance) if n else None,
            "spearman_pred_vs_force": rho,
            "task": task}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["stats", "dino"], default="stats",
                   help="'stats' = zero-GPU intensity baseline from the stats "
                        "JSONs; 'dino' = cached DINOv2 .npz features.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--feature_dir", default=None, help="dino mode")
    p.add_argument("--token", default="patch_mean", help="dino mode")
    p.add_argument("--agg", default="mean", choices=["mean", "median", "mean+std"])
    p.add_argument("--fg_min", type=float, default=0.0,
                   help="dino mode: drop views whose foreground fraction is "
                        "below this (removes pure-background tiles).")
    p.add_argument("--metadata", required=True)
    p.add_argument("--target_col", default="peak_amplitude_week3")
    p.add_argument("--file_col", default="file")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--modality", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--allow_partial_match", action="store_true", default=True)
    p.add_argument("--task", choices=["regression", "classification"],
                   default="regression")
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--pca_dim", type=int, default=20)
    p.add_argument("--cv_group", choices=["replicate", "plate"],
                   default="replicate",
                   help="'plate' = leave-one-plate-out, which DOES control for "
                        "acquisition batch effects that leave-one-replicate-out "
                        "does not.")
    p.add_argument("--deconfound", choices=["none", "plate"], default="none",
                   help="'plate' centers features AND force within each plate "
                        "(train rows only) before fitting, so ONLY within-plate "
                        "variation is modelled. Any result is then immune to "
                        "acquisition batch by construction, and you keep all 20 "
                        "replicates instead of the 4 folds leave-one-plate-out "
                        "allows. This is the strongest confound-free question "
                        "this dataset can answer.")
    p.add_argument("--canary", choices=["none", "plate"], default="none",
                   help="'plate' replaces the features with one-hot plate "
                        "identity — the ceiling on what a purely batch-"
                        "confounded feature could earn.")
    p.add_argument("--shuffle", action="store_true",
                   help="Permute replicate force labels once. MUST land at "
                        "chance; anything else means the folds leak.")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="Label permutations for the Spearman/accuracy null.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    group_cols = tuple(c.strip() for c in args.group_cols.split(",") if c.strip())
    stats_dir = os.path.join(args.data_dir, "stats")

    # which stems do we actually have features for?
    if args.features == "dino":
        if not args.feature_dir:
            raise SystemExit("--features dino requires --feature_dir")
        avail = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(args.feature_dir, "*.npz")))
    else:
        avail = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(stats_dir, "*.json")))
    if not avail:
        raise SystemExit("no feature files found")

    data = build_force_groups(args.metadata, args.data_dir, args.target_col,
                              file_col=args.file_col, group_cols=group_cols,
                              modality=args.modality, staged_stems=avail)
    if not args.quiet:
        print("\n".join(data["report"]))
    if data["n_matched"] == 0:
        raise SystemExit("no metadata force rows matched any volume")
    if data["unmatched_meta"] and not args.allow_partial_match:
        raise SystemExit(f"{len(data['unmatched_meta'])} force row(s) matched "
                         "no volume; pass --allow_partial_match")

    forces = data["forces"]
    groups = data["groups"]
    stems = sorted(forces)

    if args.features == "dino":
        X, stems, names = load_dino_features(
            args.feature_dir, stems, args.token, args.agg, args.fg_min)
    else:
        X, stems, names = load_stats_features(stats_dir, stems, args.modality)

    stem_group = {s: g for g, ss in groups.items() for s in ss}
    keep = [i for i, s in enumerate(stems) if s in stem_group]
    X = X[keep]
    stems = [stems[i] for i in keep]
    vol_group = [stem_group[s] for s in stems]
    vol_force = [float(forces[s]) for s in stems]

    def plate_of(g):
        for part in str(g).split("_"):
            if part.lower().startswith("plate="):
                return part.split("=", 1)[1]
        return "NA"

    if args.canary == "plate":
        plates = sorted({plate_of(g) for g in vol_group})
        X = np.array([[1.0 if plate_of(g) == pl else 0.0 for pl in plates]
                      for g in vol_group])
        names = [f"plate={p}" for p in plates]

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        reps = sorted(set(vol_group))
        rf = {g: float(vol_force[list(vol_group).index(g)]) for g in reps}
        shuffled = dict(zip(reps, rng.permutation([rf[g] for g in reps])))
        vol_force = [shuffled[g] for g in vol_group]

    cv_groups = ([plate_of(g) for g in vol_group]
                 if args.cv_group == "plate" else list(vol_group))

    # Plate composition. Force clustering by plate is a confound that
    # leave-one-REPLICATE-out cannot control: replicates share plates, so any
    # feature encoding acquisition batch can score above chance with no biology.
    if not args.quiet:
        by_plate = {}
        for g, f in zip(vol_group, vol_force):
            by_plate.setdefault(plate_of(g), {}).setdefault(g, float(f))
        print("\n  force by plate (the confound check reads this):")
        print(f"    {'plate':10s} {'reps':>5} {'min':>9} {'median':>9} {'max':>9}")
        for pl in sorted(by_plate):
            v = np.array(sorted(by_plate[pl].values()))
            print(f"    {pl:10s} {len(v):>5} {v.min():>9.2f} "
                  f"{np.median(v):>9.2f} {v.max():>9.2f}")
        allf = np.array([f for d in by_plate.values() for f in d.values()])
        # between-plate share of total variance: 1.0 means force IS plate
        gm = allf.mean()
        ssb = sum(len(d) * (np.mean(list(d.values())) - gm) ** 2
                  for d in by_plate.values())
        sst = ((allf - gm) ** 2).sum()
        eta2 = float(ssb / sst) if sst > 0 else float("nan")
        print(f"    between-plate share of force variance (eta^2) = {eta2:.3f}"
              + ("   <-- force is largely a plate property" if eta2 > 0.5 else ""))

    if not args.quiet:
        print(f"\n  features={args.features} ({X.shape[1]}-d) "
              f"volumes={len(stems)} replicates={len(set(vol_group))} "
              f"plates={len(set(plate_of(g) for g in vol_group))}")
        print(f"  task={args.task} n_bins={args.n_bins} "
              f"cv_group={args.cv_group} folds={len(set(cv_groups))}"
              + ("  [SHUFFLED LABELS]" if args.shuffle else "")
              + (f"  [CANARY {args.canary}]" if args.canary != "none" else ""))

    deconf = ([plate_of(g) for g in vol_group]
              if args.deconfound == "plate" else None)
    if deconf is not None and not args.quiet:
        print("  deconfound=plate: modelling WITHIN-plate variation only; a "
              "positive result here\n    cannot be an acquisition batch effect.")

    res = run_loo(X, vol_group, vol_force, cv_groups, args.task,
                  args.n_bins, args.pca_dim, args.seed, deconfound=deconf)
    out = score(res, args.n_bins, args.task)

    # ---- permutation null: RE-RUN the whole LOO under permuted labels ----
    #
    # It is not enough to permute (true, pred) pairs after the fact. An
    # uninformative LOO model is STRUCTURALLY anti-correlated with the truth:
    # leave-one-out excludes the held-out replicate from the training mean, so
    # a high-force holdout sees a mean pulled down and a low-force holdout sees
    # one pulled up. Predicting exactly the training mean gives spearman = -1.0
    # deterministically. Scoring against a null of rho = 0 therefore treats that
    # artifact as evidence. Re-running the entire fold loop on permuted labels
    # absorbs the bias, because the null is generated by the same procedure.
    #
    # The hyperparameter is FIXED across observed and permuted runs — selecting
    # it on real data but not under permutation would bias the null the other
    # way.
    if args.n_perm > 0 and out["n_replicates"] >= 3:
        rng = np.random.default_rng(args.seed)
        obs_rho, obs_acc = out["spearman_pred_vs_force"], out["replicate_accuracy"]
        fixed = 1.0          # fixed alpha (Ridge) / C (LogisticRegression)
        null_rho, null_acc = [], []
        for _ in range(args.n_perm):
            vf = _permute_replicate_force(vol_group, vol_force, rng)
            r = run_loo(X, vol_group, vf, cv_groups, args.task, args.n_bins,
                        args.pca_dim, args.seed, fixed_alpha=fixed,
                        deconfound=deconf)
            if not len(r["true_bin"]):
                continue
            null_rho.append(spearman(r["true_force"], r["pred_score"]))
            null_acc.append(float(np.mean(r["true_bin"] == r["pred_bin"])))
        null_rho = np.asarray([x for x in null_rho if not np.isnan(x)])
        null_acc = np.asarray(null_acc)
        if len(null_rho):
            out["permutation_p_spearman"] = float(
                (np.sum(np.abs(null_rho) >= abs(obs_rho)) + 1) / (len(null_rho) + 1))
            out["null_spearman_mean"] = float(null_rho.mean())
            out["null_spearman_ci"] = [float(np.percentile(null_rho, 2.5)),
                                       float(np.percentile(null_rho, 97.5))]
            # Save the null itself. Every config in a sweep permutes the SAME
            # labels with the SAME seed, so these arrays are matched across
            # configs — which lets the sweep compute a max-statistic
            # family-wise p that accounts for both the number of configs AND
            # their correlation. Nothing weaker is honest at n=20.
            out["null_spearman"] = [float(x) for x in null_rho]
        if len(null_acc):
            out["permutation_p_accuracy"] = float(
                (np.sum(null_acc >= obs_acc) + 1) / (len(null_acc) + 1))
            out["null_accuracy_mean"] = float(null_acc.mean())
            out["null_accuracy"] = [float(x) for x in null_acc]
        out["n_permutations"] = int(len(null_acc))

    _bp = {}
    for g, f in zip(vol_group, vol_force):
        _bp.setdefault(plate_of(g), {})[g] = float(f)
    _allf = np.array([v for d in _bp.values() for v in d.values()])
    _gm = _allf.mean()
    _sst = ((_allf - _gm) ** 2).sum()
    _ssb = sum(len(d) * (np.mean(list(d.values())) - _gm) ** 2
               for d in _bp.values())
    out = dict(out)
    out["eta2_plate"] = float(_ssb / _sst) if _sst > 0 else None
    out["n_plates"] = len(_bp)
    out.update({"deconfound": args.deconfound,
                "features": args.features, "modality": args.modality,
                "target_col": args.target_col, "cv_group": args.cv_group,
                "canary": args.canary, "shuffled": bool(args.shuffle),
                "n_volumes": len(stems), "feature_dim": int(X.shape[1]),
                "feature_names": names[:32],
                "per_replicate": [
                    {"group": g, "true_force": float(t), "pred_score": float(s),
                     "true_bin": int(a), "pred_bin": int(b)}
                    for g, t, s, a, b in zip(
                        res["replicates"], res["true_force"], res["pred_score"],
                        res["true_bin"], res["pred_bin"])]})

    print(f"\n  LOO n={out['n_replicates']}  "
          f"acc={out['replicate_accuracy']:.3f} (chance {out['chance']:.3f}, "
          f"{out['n_correct']}/{out['n_replicates']})  "
          f"binom_p={out['binomial_p']:.4f} [anti-conservative: assumes 20 "
          f"independent trials; trust perm_p]")
    print(f"  spearman(pred, true force) = {out['spearman_pred_vs_force']:.3f}"
          + (f"   perm_p={out['permutation_p_spearman']:.4f}"
             if "permutation_p_spearman" in out else ""))
    if args.shuffle:
        print("  ^ labels were SHUFFLED: this must be at chance. If it is not,"
              "\n    the fold logic leaks and every other number is void.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
