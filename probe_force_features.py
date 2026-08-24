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


def _rankdata(x):
    """Average ranks, so tied values share a rank.

    Ordinal ranks (argsort of argsort) break ties by input order, which makes
    the correlation depend on row ordering whenever values repeat — and force
    measurements have limited precision, so ties are expected.
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average the ranks within each run of equal values
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
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


def _wcov_axis(v, coord, w):
    """Weighted covariance of each embedding dim with a standardized axis.

    This is the ordered-structure term a plain mean throws away: "does the
    embedding drift as you go deeper / across the field", as opposed to "how
    much does it vary", which is what a std captures.
    """
    c = np.asarray(coord, dtype=np.float64)
    mu_c = np.average(c, weights=w)
    sd_c = np.sqrt(np.average((c - mu_c) ** 2, weights=w))
    if sd_c < 1e-12:
        return np.zeros(v.shape[1])
    cz = (c - mu_c) / sd_c
    return (v * (w * cz)[:, None]).sum(0) / w.sum()


def _within_group_std(v, keys, w):
    """Mean weighted std WITHIN groups (e.g. across z at a fixed tile).

    Separates the two dispersions the single `+std` term conflates: variation
    through depth at one place, versus variation between places at one depth.
    """
    out = np.zeros(v.shape[1])
    tw = 0.0
    keys = [tuple(np.atleast_1d(k).tolist()) for k in keys]
    for k in set(keys):
        m = np.array([q == k for q in keys])
        if m.sum() < 2:
            continue
        wk = w[m]
        s = wk.sum()
        if s <= 0:
            continue
        wn = wk / s
        mu = (v[m] * wn[:, None]).sum(0)
        var = (wn[:, None] * (v[m] - mu) ** 2).sum(0)
        out += s * np.sqrt(np.clip(var, 0, None))
        tw += s
    return out / tw if tw > 0 else out


def structured_terms(v, vz, vy, vx, w, terms):
    """Grid-aware summaries appended to the pooled volume vector.

    NOTE a raw position embedding concatenated before a MEAN is a no-op: every
    volume has the same grid, so the position half averages to the same
    constant and the standardizer drops it. Only `centroid` survives pooling,
    and only because fg_min keeps a different subset of views per volume --
    what it encodes is where the tissue is, not positional structure. Genuine
    positional conditioning needs a learned aggregator.
    """
    outs = []
    for t in terms:
        if t == "grad_z":
            outs.append(_wcov_axis(v, vz, w))
        elif t == "grad_y":
            outs.append(_wcov_axis(v, vy, w))
        elif t == "grad_x":
            outs.append(_wcov_axis(v, vx, w))
        elif t == "std_z":
            outs.append(_within_group_std(v, list(zip(vy, vx)), w))
        elif t == "std_xy":
            outs.append(_within_group_std(v, list(vz), w))
        elif t == "centroid":
            def _c(a):
                a = np.asarray(a, float)
                rng_ = a.max() - a.min()
                return (np.average(a, weights=w) - a.min()) / (rng_ + 1e-12)
            outs.append(np.array([_c(vz), _c(vy), _c(vx)]))
        else:
            raise SystemExit(f"unknown --struct term {t!r}")
    return np.concatenate(outs) if outs else np.zeros(0)


def load_dino_features(feature_dir, stems, token, agg, fg_min, struct=()):
    """Per-view DINOv2 features -> one vector per volume.

    `token` may be a comma list ("patch_mean,patch_std"), in which case the
    arrays are concatenated per view. patch_std carries the spatial
    heterogeneity of the token grid that patch_mean discards, so the pair is
    usually more informative than either alone.

    `agg` collapses views to a volume. With tiled framing the views tile the
    whole field, so agg="mean" IS the whole-FOV representation — reached by
    averaging embeddings of native-resolution crops rather than by resizing the
    field down into one view.
    """
    toks = [t.strip() for t in str(token).split(",") if t.strip()]
    feats, kept = [], []
    fg_seen, dropped, n_kept, n_total = [], [], [], []
    for s in stems:
        p = os.path.join(feature_dir, f"{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        missing = [t for t in toks if t not in z]
        if missing:
            raise SystemExit(f"{p}: no {missing} array(s) (has {list(z.keys())})")
        arrs = []
        for t in toks:
            a = np.asarray(z[t], dtype=np.float64)
            if a.ndim > 2:          # e.g. patch_grid (n_views, g, g, D)
                a = a.reshape(a.shape[0], -1)
            arrs.append(a)
        v = np.concatenate(arrs, axis=-1)                # (n_views, sum_D)
        fg_all = (np.asarray(z["view_fg_frac"], float) if "view_fg_frac" in z
                  else np.ones(len(v)))
        fg_seen.append(fg_all)
        keep = np.ones(len(v), dtype=bool)
        if fg_min > 0 and "view_fg_frac" in z:
            keep = fg_all >= fg_min
            if not keep.any():
                # NEVER fall back to "keep everything" here. That was the old
                # behaviour and it silently inverted the request: a volume with
                # no tissue-rich tile contributed a pure-background vector,
                # while tissue-rich volumes contributed tissue -- so the
                # feature that best separated volumes was how much background
                # they had. Drop the volume and say so.
                dropped.append((s, float(fg_all.max())))
                continue
        v = v[keep]
        fg = fg_all[keep]
        n_kept.append(int(keep.sum()))
        n_total.append(int(len(keep)))
        if agg == "median":
            vec = np.median(v, axis=0)
        elif agg == "mean+std":
            # std ACROSS views = across-field heterogeneity. Distinct from the
            # patch_std token, which is within-view heterogeneity.
            vec = np.concatenate([v.mean(axis=0), v.std(axis=0)])
        elif agg == "fgmean":
            # Weight each view by its foreground fraction. The fields are
            # sparse tissue on background, so an unweighted mean over 144 tiles
            # is mostly background texture with a small tissue perturbation —
            # and the between-volume variance a probe needs lives in that
            # perturbation.
            w = np.clip(fg, 0, None)
            vec = ((v * w[:, None]).sum(axis=0) / w.sum()
                   if w.sum() > 1e-8 else v.mean(axis=0))
        elif agg == "fgmean+std":
            # BOTH terms foreground-weighted. Pairing a tissue-weighted mean
            # with an unweighted std let background-heavy tiles back in through
            # the dispersion half of the vector.
            w = np.clip(fg, 0, None)
            if w.sum() > 1e-8:
                mu = (v * w[:, None]).sum(axis=0) / w.sum()
                sd = np.sqrt(np.clip(
                    ((v - mu) ** 2 * w[:, None]).sum(axis=0) / w.sum(), 0, None))
            else:
                mu, sd = v.mean(axis=0), v.std(axis=0)
            vec = np.concatenate([mu, sd])
        else:
            vec = v.mean(axis=0)
        if struct:
            if "view_z" not in z or "view_yx" not in z:
                raise SystemExit(
                    f"{p}: --struct needs view_z/view_yx, which these features "
                    f"predate. Re-extract.")
            vzk = np.asarray(z["view_z"]).astype(float)[keep]
            vyk = np.asarray(z["view_yx"]).astype(float)[keep, 0]
            vxk = np.asarray(z["view_yx"]).astype(float)[keep, 1]
            ws = np.clip(fg, 1e-12, None)
            vec = np.concatenate(
                [vec, structured_terms(v, vzk, vyk, vxk, ws, struct)])
        feats.append(vec)
        kept.append(s)
    if fg_seen:
        allfg = np.concatenate(fg_seen)
        qs = np.percentile(allfg, [50, 75, 90, 95, 100])
        print(f"  views: foreground fraction median {qs[0]:.3f}, "
              f"p75 {qs[1]:.3f}, p90 {qs[2]:.3f}, p95 {qs[3]:.3f}, "
              f"max {qs[4]:.3f}")
        if n_total:
            print(f"  fg_min={fg_min}: kept {sum(n_kept)}/{sum(n_total)} views "
                  f"({sum(n_kept)/max(1,sum(n_total)):.1%}), "
                  f"{len(kept)} volume(s) usable")
        if fg_min > 0 and float(allfg.max()) < fg_min:
            raise SystemExit(
                f"--fg_min {fg_min} exceeds the best tile in the WHOLE dataset "
                f"({allfg.max():.3f}). No volume can pass. Either lower it, or "
                f"re-extract with foreground-guided tile placement "
                f"(--framing tiled_fg) so tiles are positioned on tissue "
                f"instead of on a fixed grid.")
    if dropped:
        print(f"  DROPPED {len(dropped)} volume(s) with no tile at "
              f"fg>={fg_min} (best tile shown): "
              + ", ".join(f"{s}({m:.2f})" for s, m in dropped[:6])
              + (" ..." if len(dropped) > 6 else ""))
        print(f"  NOTE dropping volumes is not neutral: if tissue coverage "
              f"correlates with force, this drops a biased subset.")
    if not feats:
        raise SystemExit(f"no .npz features found in {feature_dir}")
    names = [f"{'+'.join(toks)}[{i}]" for i in range(len(feats[0]))]
    return np.asarray(feats, dtype=np.float64), kept, names


# --------------------------------------------------------------------------
# the readout direction (what the model actually reads off the features)
# --------------------------------------------------------------------------
def _readout_direction(m, comp, sd):
    """Map a fitted linear readout back into ORIGINAL feature space.

    The probe is a chain of affine maps: standardize by (mu, sd), rotate into
    the top-`n_comp` weighted-PCA subspace, then a linear model. So the whole
    thing is linear in the raw feature vector and collapses to a single
    direction `u` with

        prediction = u . x  +  const

    That matters for explanation: because pooling over views is itself a
    weighted mean, `u . x` distributes EXACTLY over views (and, one level
    down, over patch tokens). No gradient approximation is involved -- the
    per-view contributions literally sum to the prediction. See
    explain_dino_probe.py, which consumes this.

    Returns None for a model with no `coef_` (xgboost), and for multiclass
    logistic, where "the" direction is not one vector.
    """
    coef = getattr(m, "coef_", None)
    if coef is None:
        return None
    cf = np.asarray(coef, dtype=np.float64)
    if cf.ndim > 1:
        if cf.shape[0] != 1:
            return None            # multiclass: no single direction
        cf = cf[0]
    cf = cf.ravel()
    if comp is not None:
        cf = comp.T @ cf           # PCA space -> standardized feature space
    return cf / sd                 # -> raw feature space


def direction_stability(dirs):
    """Mean pairwise cosine between per-fold readout directions.

    This is the honesty check on any "here is what the model looks at" claim.
    Each LOO fold refits on 21 of 22 replicates, so the folds SHOULD agree
    almost perfectly if the direction is driven by the data. If they do not,
    the attribution maps are a property of the particular resample, not of the
    biology, and no feature story should be told from them.
    """
    if len(dirs) < 2:
        return None
    D = np.asarray(dirs, dtype=np.float64)
    n = np.linalg.norm(D, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    D = D / n
    G = D @ D.T
    iu = np.triu_indices(len(D), k=1)
    return float(G[iu].mean())


# --------------------------------------------------------------------------
# the LOO probe
# --------------------------------------------------------------------------
def run_loo(X, vol_group, vol_force, cv_groups, task, n_bins, pca_dim,
            seed=0, fixed_alpha=None, deconfound=None, categorical=False,
            model_class="ridge"):
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
    from sklearn.linear_model import Ridge, LogisticRegression
    folds = sorted(set(cv_groups))
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    pred_force, true_force, pred_bin, true_bin, held_reps = [], [], [], [], []
    fold_dirs = []

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
                if not categorical:
                    # Centring a 0/1 class code turns it into a continuous
                    # residual and the classes stop existing. For a categorical
                    # target the confound control belongs on the FEATURES only:
                    # remove each plate's feature offset, keep the labels.
                    yf[m_all] -= mu_y

        # Sample weights: a 4-FOV tissue must not outvote a 1-FOV one. These
        # have to apply to the STANDARDIZATION and the PCA too, not just the
        # regression. PCA to pca_dim is the real regularizer here (768 -> 20),
        # so if those directions are chosen with a high-FOV tissue counting 4x,
        # the subspace is tilted toward it and the later down-weighting cannot
        # recover what fell outside.
        tr_groups_w = np.asarray(vol_group)[tr]
        cnt_w = {g: int((tr_groups_w == g).sum()) for g in set(tr_groups_w)}
        w = np.array([1.0 / cnt_w[g] for g in tr_groups_w])
        w = w / w.sum() * len(w)          # mean 1, so alpha keeps its scale

        # weighted standardization (fit on TRAINING rows only)
        mu = np.average(Xf[tr], axis=0, weights=w)
        var = np.average((Xf[tr] - mu) ** 2, axis=0, weights=w)
        sd = np.sqrt(var)
        sd[sd < 1e-12] = 1.0
        Xtr, Xte = (Xf[tr] - mu) / sd, (Xf[te] - mu) / sd

        # PCA is the regularizer for ridge. Sparse and tree readouts do their
        # own selection, and putting them behind a 20-component bottleneck
        # would test the bottleneck rather than the model class.
        n_comp = int(min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
        if model_class != "ridge":
            n_comp = 0
        comp = None
        if n_comp >= 1 and Xtr.shape[1] > n_comp:
            # weighted PCA == SVD of the sqrt(w)-scaled, weighted-centered data
            Z = Xtr * np.sqrt(w)[:, None]
            _u, _s, vt = np.linalg.svd(Z, full_matrices=False)
            comp = vt[:n_comp]
            Xtr, Xte = Xtr @ comp.T, Xte @ comp.T

        ytr_force = yf[tr]
        fold_rep_force = {}
        for g, v in zip(np.asarray(vol_group)[tr], ytr_force):
            fold_rep_force.setdefault(g, float(v))
        for g, v in zip(np.asarray(vol_group)[te], yf[te]):
            fold_rep_force.setdefault(g, float(v))
        if categorical:
            # The class code IS the bin. Quantile edges on a 0/1 column are
            # degenerate: percentile([0,1], 50) lands ON a class, and
            # searchsorted(..., side="right") then puts BOTH classes in bin 1,
            # so every fold would train on a single-class target.
            edges = None
            _bin = lambda v: int(round(float(v)))
        else:
            edges = compute_bin_edges(
                [fold_rep_force[g] for g in sorted(set(tr_groups))], n_bins)
            _bin = lambda v: assign_bin(v, edges)
        ytr_bin = np.array([_bin(v) for v in ytr_force])

        if task == "regression":
            # centered force is signed, so log10 only applies to the raw target
            if deconfound is not None:
                ytr = ytr_force          # centered force is signed
            else:
                if np.any(ytr_force <= 0):
                    raise SystemExit(
                        f"non-positive force value(s) "
                        f"{ytr_force[ytr_force <= 0][:3]} cannot be log-scaled. "
                        f"Clipping would map them to -9, a catastrophic outlier "
                        f"for Ridge. Fix the metadata or use --deconfound.")
                ytr = np.log10(ytr_force)
            make = _make_regressor(model_class, seed)
            alpha = fixed_alpha
            if alpha is None:
                alpha = _inner_select(Xtr, ytr, tr_groups, w,
                                      _hp_grid(model_class), make, "r2", seed)
            m = make(alpha).fit(Xtr, ytr, sample_weight=w)
            p = m.predict(Xte)
            fold_dirs.append(_readout_direction(m, comp, sd))
        else:
            if len(set(ytr_bin)) < 2:
                continue
            C = fixed_alpha
            if C is None:
                C = _inner_select(
                    Xtr, ytr_bin, tr_groups, w, grid,
                    lambda c: LogisticRegression(C=c, max_iter=2000),
                    "acc", seed)
            # Class balancing must be computed on the WEIGHTED class totals,
            # not on raw volume counts; class_weight="balanced" would ignore w
            # and then multiply on top of it.
            wb = w.copy()
            tot = w.sum()
            for c in np.unique(ytr_bin):
                m_c = ytr_bin == c
                wc = w[m_c].sum()
                if wc > 0:
                    wb[m_c] *= tot / (len(np.unique(ytr_bin)) * wc)
            m = LogisticRegression(C=C, max_iter=2000).fit(
                Xtr, ytr_bin, sample_weight=wb)
            p = m.predict_proba(Xte)
            fold_dirs.append(_readout_direction(m, comp, sd))

        # collapse held-out VOLUMES to one prediction per replicate
        te_groups = np.asarray(vol_group)[te]
        for g in sorted(set(te_groups)):
            sel = te_groups == g
            y_true_g = fold_rep_force[g]
            tb = _bin(y_true_g)
            if task == "regression":
                mp = float(np.mean(p[sel]))
                pred_force.append(mp)
                pb = _bin(mp if deconfound is not None else 10 ** mp)
            else:
                pr = p[sel].mean(axis=0)
                pb = int(m.classes_[int(np.argmax(pr))])
                if categorical:
                    # No numeric "bin representative" exists for a nominal
                    # class -- averaging class codes would invent an ordering.
                    # Use P(positive class) for binary (a usable continuous
                    # score), else the predicted code. Either way `categorical`
                    # runs are scored by ACCURACY, not by rank correlation.
                    cls = list(m.classes_)
                    pred_force.append(float(pr[cls.index(1)])
                                      if len(cls) == 2 and 1 in cls
                                      else float(pb))
                else:
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
            "pred_bin": np.asarray(pred_bin),
            "fold_dirs": [d for d in fold_dirs if d is not None]}


def _hp_grid(model_class):
    if model_class == "ridge":
        return [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    if model_class in ("lasso", "elasticnet"):
        return [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    return [2, 3, 4, 6]          # xgboost: n_estimators tier


def _make_regressor(model_class, seed):
    """Return f(hp) -> unfitted estimator.

    Depth-1 stumps and a handful of trees is not timidity: with ~21 training
    rows and 768+ columns, a deeper tree picks the best of 768 splits on 21
    points, which is selecting noise. This is the most generous tree model the
    sample size supports.
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    if model_class == "ridge":
        return lambda a: Ridge(alpha=a)
    if model_class == "lasso":
        return lambda a: Lasso(alpha=a, max_iter=20000)
    if model_class == "elasticnet":
        return lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000)
    if model_class == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise SystemExit(
                "--model_class xgboost needs the xgboost package "
                "(pip install xgboost)")
        return lambda n: XGBRegressor(
            n_estimators=int(n), max_depth=1, learning_rate=0.3,
            reg_lambda=10.0, subsample=1.0, colsample_bytree=0.3,
            random_state=seed, verbosity=0)
    raise SystemExit(f"unknown model_class {model_class!r}")


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


def _permute_replicate_force(vol_group, vol_force, rng, strata=None):
    """Permute force across REPLICATES (every volume of a tissue keeps one
    shared value). Permuting per-volume would break the group structure and
    make the null easier than the real problem.

    strata: optional per-volume confound level (e.g. plate). When given, force
        is shuffled only WITHIN a stratum. This matters for a nested design:
        free permutation destroys the plate-force association, so the null
        becomes "no feature-force association at all" and any feature encoding
        acquisition batch beats it. The restricted null asks the question we
        actually care about — is there association BEYOND plate.
    """
    reps = sorted(set(vol_group))
    rf, rs = {}, {}
    for i, (g, f) in enumerate(zip(vol_group, vol_force)):
        rf.setdefault(g, float(f))
        if strata is not None:
            rs.setdefault(g, strata[i])

    shuffled = {}
    if strata is None:
        shuffled = dict(zip(reps, rng.permutation([rf[g] for g in reps])))
    else:
        by_stratum = {}
        for g in reps:
            by_stratum.setdefault(rs[g], []).append(g)
        for _lvl, gs in sorted(by_stratum.items(), key=lambda kv: str(kv[0])):
            gs = sorted(gs)
            shuffled.update(zip(gs, rng.permutation([rf[g] for g in gs])))
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
    p.add_argument("--token", default="patch_mean,patch_std",
                   help="dino mode: which per-view array(s) to use. Comma list "
                        "concatenates. patch_std is the spatial heterogeneity "
                        "of the token grid, which patch_mean discards.")
    p.add_argument("--agg", default="mean",
                   choices=["mean", "median", "mean+std", "fgmean",
                            "fgmean+std"],
                   help="How to collapse views to one volume vector. 'fgmean' "
                        "weights each view by its foreground fraction — the "
                        "fields are sparse tissue on background, so a plain "
                        "mean is dominated by background. '*+std' appends the "
                        "std ACROSS views (across-field heterogeneity), which "
                        "is different from the patch_std token (within-view).")
    p.add_argument("--aggregate", choices=["volume", "label"], default="volume",
                   help="'volume': one row per FOV, weighted 1/n_FOV, and "
                        "predictions averaged per tissue afterwards (default, "
                        "the historical behaviour). 'label': average the "
                        "encodings of every FOV sharing a force value into one "
                        "row BEFORE fitting -- less predictor noise, so less "
                        "errors-in-variables attenuation of the ridge fit.")
    p.add_argument("--struct", default="none",
                   help="Comma list of grid-aware terms appended to the pooled "
                        "vector: grad_z,grad_y,grad_x,std_z,std_xy,centroid. "
                        "These are what a mean over views discards. A position "
                        "embedding concatenated before a mean is a NO-OP "
                        "(same grid every volume -> constant), which is why "
                        "these are covariances and within-group spreads "
                        "instead.")
    p.add_argument("--model_class",
                   choices=["ridge", "lasso", "elasticnet", "xgboost"],
                   default="ridge",
                   help="Readout. 'ridge' runs on PCA(pca_dim); the sparse and "
                        "tree options skip PCA and see all features, which is "
                        "the point of trying them. NOTE each class is a "
                        "separate config and so raises the family-wise bar.")
    p.add_argument("--fg_min", type=float, default=0.0,
                   help="dino mode: drop views whose foreground fraction is "
                        "below this (removes pure-background tiles).")
    p.add_argument("--metadata", required=True)
    p.add_argument("--target_col", default="peak_amplitude_week3")
    p.add_argument("--target_type", choices=["numeric", "categorical"],
                   default="numeric",
                   help="'categorical' for a string label column such as "
                        "treated/untreated or perturbed/unperturbed. Forces "
                        "--task classification, sets n_bins to the number of "
                        "classes, and scores by accuracy rather than by rank "
                        "correlation (a nominal label has no ordering).")
    p.add_argument("--file_col", default="file")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--modality", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--strict_match", action="store_true",
                   help="Fail if any force-labeled metadata row matched no "
                        "staged volume. Off by default: this drop has one "
                        "known-unstaged file.")
    p.add_argument("--task", choices=["regression", "classification"],
                   default="regression")
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--pca_dim", type=int, default=20)
    p.add_argument("--perm_scope", choices=["within_plate", "free"],
                   default="within_plate",
                   help="'within_plate' (default) shuffles force only among "
                        "replicates sharing a plate. Free permutation destroys "
                        "the plate-force link, so any batch-encoding feature "
                        "beats the null — the very confound this file guards "
                        "against.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Fixed Ridge alpha / LogisticRegression C, used for the "
                        "observed run AND every permutation so the two are "
                        "produced by an identical procedure.")
    p.add_argument("--tune_alpha", action="store_true",
                   help="Nested-CV select the hyperparameter on the observed "
                        "run. NOT recommended: the permutations stay fixed, so "
                        "the observed statistic would be tuned against untuned "
                        "nulls, and on null data the inner R^2 criterion "
                        "systematically picks maximum shrinkage.")
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
                              modality=args.modality, staged_stems=avail,
                              target_type=args.target_type)
    _cat = args.target_type == "categorical"
    if _cat:
        # A nominal target has no ordering, so a rank correlation between
        # predicted and "true" class codes is meaningless -- 0 vs 1 vs 2 is not
        # a scale. Score it as classification and let the permutation test run
        # on ACCURACY.
        if args.task != "classification":
            print(f"  target_type=categorical -> forcing --task classification "
                  f"(was {args.task})")
            args.task = "classification"
        n_cls = len(data["classes"])
        if args.n_bins != n_cls:
            print(f"  target_type=categorical -> n_bins={n_cls} "
                  f"(classes: {', '.join(data['classes'])})")
            args.n_bins = n_cls
    if not args.quiet:
        print("\n".join(data["report"]))
    if data["n_matched"] == 0:
        raise SystemExit("no metadata force rows matched any volume")
    if data["unmatched_meta"] and args.strict_match:
        raise SystemExit(f"{len(data['unmatched_meta'])} force row(s) matched "
                         "no volume (this is expected for the one known "
                         "unstaged file); rerun without --strict_match to "
                         "proceed")

    forces = data["forces"]
    groups = data["groups"]
    stems = sorted(forces)

    if args.features == "dino":
        X, stems, names = load_dino_features(
            args.feature_dir, stems, args.token, args.agg, args.fg_min,
            struct=[t.strip() for t in args.struct.split(",")
                    if t.strip() and t.strip() != "none"])
    else:
        X, stems, names = load_stats_features(stats_dir, stems, args.modality)

    stem_group = {s: g for g, ss in groups.items() for s in ss}
    keep = [i for i, s in enumerate(stems) if s in stem_group]
    X = X[keep]
    stems = [stems[i] for i in keep]
    vol_group = [stem_group[s] for s in stems]
    vol_force = [float(forces[s]) for s in stems]

    if args.aggregate == "label":
        # Collapse every volume that shares a force value into ONE row before
        # fitting. Those are exactly the FOVs of one tissue: build_force_groups
        # hard-fails on a group carrying two force values, so a group IS a
        # level of the dependent variable.
        #
        # This is not cosmetic. Fitting on individual FOVs treats per-field
        # imaging noise as if it were between-tissue variation, and that noise
        # sits in the PREDICTORS -- classic errors-in-variables, which biases
        # ridge coefficients toward zero. Averaging first cuts that noise by
        # ~sqrt(n_FOV) before the fit ever sees it. Averaging predictions
        # afterwards (what the default does) cannot recover it: the attenuation
        # already happened during fitting.
        _order = sorted(set(vol_group))
        _vf = np.asarray(vol_force, dtype=np.float64)
        _vg = np.asarray(vol_group, dtype=object)
        _Xa, _ga, _fa, _nn = [], [], [], []
        for g in _order:
            m = _vg == g
            _Xa.append(X[m].mean(axis=0))
            _ga.append(g)
            _fa.append(float(_vf[m][0]))
            _nn.append(int(m.sum()))
        X = np.asarray(_Xa, dtype=np.float64)
        vol_group, vol_force = _ga, _fa
        stems = [f"{g}(mean of {n})" for g, n in zip(_ga, _nn)]
        print(f"  aggregate=label: {len(_vg)} volumes -> {len(_ga)} rows, "
              f"one per force value ({min(_nn)}-{max(_nn)} FOVs each)")

    def plate_of(g):
        for part in str(g).split("_"):
            if part.lower().startswith("plate="):
                return part.split("=", 1)[1]
        return "NA"

    _plates = sorted({plate_of(g) for g in vol_group})
    if (args.deconfound == "plate" or args.canary == "plate"
            or args.cv_group == "plate" or args.perm_scope == "within_plate"):
        if len(_plates) < 2 or _plates == ["NA"]:
            raise SystemExit(
                f"plate identity could not be parsed from the group ids "
                f"(got {_plates[:5]}). Every confound control here depends on "
                f"it: --deconfound would become a global no-op, --canary a "
                f"constant column, and the restricted permutation would "
                f"degenerate to a free one — all silently. Make sure "
                f"--group_cols includes 'plate' (currently "
                f"'{args.group_cols}').")

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

    if args.deconfound == "plate" and args.cv_group == "plate":
        raise SystemExit(
            "--deconfound plate with --cv_group plate is degenerate: the "
            "held-out plate has no training rows, so its centering falls back "
            "to the global mean while every training plate had its own offset "
            "removed — a systematic test-time shift. Use --cv_group replicate "
            "with --deconfound plate (within-plate question, all replicates), "
            "or --cv_group plate with --deconfound none (across-plate "
            "generalization).")

    deconf = ([plate_of(g) for g in vol_group]
              if args.deconfound == "plate" else None)
    if deconf is not None and not args.quiet:
        print("  deconfound=plate: modelling WITHIN-plate variation only; a "
              "positive result here\n    cannot be an acquisition batch effect.")

    # The observed statistic MUST be produced by the same procedure as the
    # null, hyperparameter included. Nested selection on the observed run only
    # would compare a tuned model against untuned nulls.
    res = run_loo(X, vol_group, vol_force, cv_groups, args.task,
                  args.n_bins, args.pca_dim, args.seed, deconfound=deconf,
                  categorical=_cat, model_class=args.model_class,
                  fixed_alpha=(None if args.tune_alpha else args.alpha))
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
    # Restricted permutation for the nested design unless explicitly freed.
    perm_strata = (None if args.perm_scope == "free"
                   else [plate_of(g) for g in vol_group])
    if args.n_perm > 0 and perm_strata is not None:
        # A restricted permutation can only shuffle labels that VARY inside a
        # stratum. If every plate carries one label value (a treatment applied
        # plate-wise, say), within-plate permutation is the identity: the null
        # equals the observed exactly and p is 1.0 by construction. That is not
        # a negative result, it is an untestable design, and it must not be
        # reported as "does not survive correction".
        _by = {}
        for _g, _f, _s in zip(vol_group, vol_force, perm_strata):
            _by.setdefault(_s, set()).add(round(float(_f), 9))
        _varies = [s for s, v in _by.items() if len(v) > 1]
        if not _varies:
            out["degenerate_permutation"] = True
            print(f"\n  *** '{args.target_col}' is CONSTANT within every "
                  f"{args.perm_scope.replace('within_', '')}.")
            print(f"      Within-stratum permutation cannot move it, so the "
                  f"null is identical to")
            print(f"      the observed value and any p it reports is an "
                  f"artifact. This label is")
            print(f"      fully confounded with the stratum: nothing in the "
                  f"images can be shown")
            print(f"      to predict it BEYOND the batch. Use "
                  f"--perm_scope free to ask the")
            print(f"      weaker question (does anything predict it at all), "
                  f"knowing that plate")
            print(f"      alone would answer yes.")
            args.n_perm = 0

    if args.n_perm > 0 and out["n_replicates"] >= 3:
        rng = np.random.default_rng(args.seed)
        obs_rho, obs_acc = out["spearman_pred_vs_force"], out["replicate_accuracy"]
        fixed = args.alpha          # same value for observed AND permuted
        if perm_strata is not None and not args.quiet:
            n_lv = len(set(perm_strata))
            print(f"  permutation null: RESTRICTED within {n_lv} plate(s) — "
                  f"tests association BEYOND plate, not merely any association")
        null_rho, null_acc = [], []
        for _ in range(args.n_perm):
            vf = _permute_replicate_force(vol_group, vol_force, rng,
                                          strata=perm_strata)
            r = run_loo(X, vol_group, vf, cv_groups, args.task, args.n_bins,
                        args.pca_dim, args.seed, fixed_alpha=fixed,
                        deconfound=deconf, categorical=_cat,
                        model_class=args.model_class)
            if not len(r["true_bin"]):
                continue
            null_rho.append(spearman(r["true_force"], r["pred_score"]))
            null_acc.append(float(np.mean(r["true_bin"] == r["pred_bin"])))
        null_rho = np.asarray([x for x in null_rho if not np.isnan(x)])
        null_acc = np.asarray(null_acc)
        if len(null_rho) and np.isfinite(obs_rho):
            # ONE-SIDED on signed rho. Using |rho| would let an anti-correlated
            # model win: the leave-one-out null is centered well below zero, so
            # "far from 0" is achieved by over-shrinkage as easily as by signal.
            out["permutation_p_spearman"] = float(
                (np.sum(null_rho >= obs_rho) + 1) / (len(null_rho) + 1))
            out["p_is_one_sided"] = True
        elif len(null_rho):
            # S3: abs(nan) comparisons are all-False, which would have produced
            # the SMALLEST attainable p (1/(B+1)) for a degenerate model whose
            # predictions collapsed to a constant.
            out["permutation_p_spearman"] = 1.0
            out["degenerate"] = ("observed spearman is not finite (constant "
                                 "predictions?) — p forced to 1.0")
        # UNCONDITIONAL on purpose. These lines used to live inside the `elif`
        # above, so the null was serialized ONLY when the observed rho was NaN
        # — i.e. never, on any healthy run. The sweep's max-statistic stage
        # then reported "0 of N configs carry a null distribution" and silently
        # fell back to uncorrected p-values, which is exactly the failure mode
        # the max-statistic exists to prevent.
        if len(null_rho):
            out["null_spearman_mean"] = float(null_rho.mean())
            out["null_spearman_ci"] = [float(np.percentile(null_rho, 2.5)),
                                       float(np.percentile(null_rho, 97.5))]
            # The smallest observed rho this design could have called
            # significant at alpha=0.05. Report it with every null result: it
            # separates "no signal" from "no power to see one".
            out["mde_spearman_95"] = float(np.percentile(null_rho, 95))
            # Save the null itself. Every config in a sweep permutes the SAME
            # labels with the SAME seed, so these arrays are matched across
            # configs — which lets the sweep compute a max-statistic
            # family-wise p that accounts for both the number of configs AND
            # their correlation. Nothing weaker is honest at n=20.
            out["null_spearman"] = [float(x) for x in null_rho]
        if len(null_acc) and np.isfinite(obs_acc):
            out["permutation_p_accuracy"] = float(
                (np.sum(null_acc >= obs_acc) + 1) / (len(null_acc) + 1))
        elif len(null_acc):
            out["permutation_p_accuracy"] = 1.0
        if len(null_acc):
            out["null_accuracy_mean"] = float(null_acc.mean())
            out["mde_accuracy_95"] = float(np.percentile(null_acc, 95))
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
                "target_type": args.target_type,
                "struct": args.struct, "model_class": args.model_class,
                "aggregate": args.aggregate, "fg_min": args.fg_min,
                "canary": args.canary, "shuffled": bool(args.shuffle),
                "n_volumes": len(stems), "feature_dim": int(X.shape[1]),
                "feature_names": names[:32],
                "per_replicate": [
                    {"group": g, "plate": plate_of(g),
                     "true_force": float(t), "pred_score": float(s),
                     "true_bin": int(a), "pred_bin": int(b)}
                    for g, t, s, a, b in zip(
                        res["replicates"], res["true_force"], res["pred_score"],
                        res["true_bin"], res["pred_bin"])]})

    # ---- the readout direction, for explanation ----
    #
    # Saved as a sidecar .npz rather than inline, because it is one float per
    # feature dimension (768-1536) and would swamp the JSON. `mean_direction`
    # is the average of the per-fold directions; `fold_cosine` says whether
    # averaging them was meaningful in the first place.
    fdirs = res.get("fold_dirs") or []
    out["readout_fold_cosine"] = direction_stability(fdirs)
    out["readout_n_folds"] = len(fdirs)
    if fdirs and args.output:
        _dir_path = os.path.splitext(args.output)[0] + ".readout.npz"
        os.makedirs(os.path.dirname(_dir_path) or ".", exist_ok=True)
        _D = np.asarray(fdirs, dtype=np.float64)
        np.savez_compressed(
            _dir_path, mean_direction=_D.mean(axis=0), fold_directions=_D,
            feature_names=np.asarray(names, dtype=object),
            token=args.token, agg=args.agg, fg_min=float(args.fg_min),
            struct=args.struct, deconfound=args.deconfound,
            target_col=args.target_col, task=args.task,
            model_class=args.model_class,
            fold_cosine=(np.nan if out["readout_fold_cosine"] is None
                         else out["readout_fold_cosine"]))
        out["readout_path"] = _dir_path
    if out["readout_fold_cosine"] is not None:
        _fc = out["readout_fold_cosine"]
        print(f"  readout direction: mean pairwise cosine across "
              f"{len(fdirs)} folds = {_fc:+.3f}"
              + ("" if _fc >= 0.9 else
                 "  <-- UNSTABLE: the folds disagree about which feature "
                 "direction predicts force, so any saliency map from this "
                 "model describes one resample, not the biology"))

    print(f"\n  LOO n={out['n_replicates']}  "
          f"acc={out['replicate_accuracy']:.3f} (chance {out['chance']:.3f}, "
          f"{out['n_correct']}/{out['n_replicates']})  "
          f"binom_p={out['binomial_p']:.4f} [anti-conservative: assumes 20 "
          f"independent trials; trust perm_p]")
    if args.target_type == "categorical":
        # Rank correlation on a nominal label is not a result. Lead with
        # accuracy and put the permutation test on accuracy too.
        if "permutation_p_accuracy" in out:
            print(f"  accuracy perm_p={out['permutation_p_accuracy']:.4f} "
                  f"({out.get('n_permutations', 0)} permutations)")
        if "mde_accuracy_95" in out:
            print(f"  null accuracy mean {out['null_accuracy_mean']:.3f}; "
                  f"detectable only if accuracy > "
                  f"{out['mde_accuracy_95']:.3f} (alpha=0.05, this config)")
        if args.shuffle:
            print("  ^ labels were SHUFFLED: this must be at chance. If it is "
                  "not,\n    the fold logic leaks and every other number is "
                  "void.")
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  saved {args.output}")
        return
    print(f"  spearman(pred, true force) = {out['spearman_pred_vs_force']:.3f}"
          + (f"   perm_p={out['permutation_p_spearman']:.4f}"
             if "permutation_p_spearman" in out else ""))
    if "mde_spearman_95" in out:
        # A null result means nothing without this number. It is the smallest
        # rho this design could have distinguished from chance — read it as
        # "anything weaker than X was invisible here regardless of the model".
        print(f"  null: mean {out['null_spearman_mean']:+.3f}, "
              f"95% CI [{out['null_spearman_ci'][0]:+.3f}, "
              f"{out['null_spearman_ci'][1]:+.3f}]  "
              f"({out['n_permutations']} permutations)")
        print(f"  detectable only if spearman > {out['mde_spearman_95']:+.3f} "
              f"(alpha=0.05, this config alone)")
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
