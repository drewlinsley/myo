"""What labels and grouping structure does a mapping spreadsheet actually hold?

Answers "is there other validation/test data, or just these 4 plates?" without
guessing: enumerates every column, finds every usable numeric label column, and
— for each one — reports how much of its variance is explained by plate.

That last number is the one that matters. Leave-one-replicate-out cannot control
for acquisition batch, because replicates share plates. A label whose variance is
mostly between-plate (eta^2 near 1) cannot be validated on this dataset no matter
what model you fit: any feature encoding batch will predict it, and holding out
plates leaves too few independent units to test. A label with low eta^2 has
within-plate spread, which IS testable under leave-one-plate-out.

Usage:
    python inventory_metadata.py "phalloidin_mhc_mapping_051426_SS edit.xlsx"
    python inventory_metadata.py data_mapping_drew.csv --group_cols plate,Tissue
    python inventory_metadata.py <meta> --data_dir <staged> --modality gfp
        (adds how many rows actually match staged volumes)
"""

import os
import sys
import json
import argparse
import importlib.util

import numpy as np

_fm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "data", "force_metadata.py")
_spec = importlib.util.spec_from_file_location("_force_metadata", _fm_path)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)


def as_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("na", "nan", "none", "-", "#n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def eta_squared(values, groups):
    """Between-group share of total variance. 1.0 = the label IS the group.

    NOTE: eta^2 is biased upward. With k groups and n observations its expected
    value under NO group effect is (k-1)/(n-1) — 0.143 for 4 plates and 22
    replicates. Always compare against `eta2_null` below rather than against 0.
    """
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if len(v) < 3:
        return float("nan")
    gm = v.mean()
    sst = ((v - gm) ** 2).sum()
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for u in set(g):
        m = g == u
        ssb += m.sum() * (v[m].mean() - gm) ** 2
    return float(ssb / sst)


def eta2_null(k, n):
    """E[eta^2] under no group effect. Verified by simulation."""
    return (k - 1) / (n - 1) if n > 1 else float("nan")


def omega_squared(values, groups):
    """Bias-corrected effect size: eta^2 minus its null expectation, rescaled.
    ~0 means 'no more between-group structure than chance'."""
    v = np.asarray(values, dtype=np.float64)
    k = len(set(map(str, groups)))
    n = len(v)
    e2 = eta_squared(values, groups)
    if not np.isfinite(e2) or n <= k:
        return float("nan")
    return float(max(0.0, (e2 - eta2_null(k, n)) / (1 - eta2_null(k, n))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metadata")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--plate_col", default="plate")
    p.add_argument("--file_col", default="file")
    p.add_argument("--data_dir", default=None,
                   help="Staged root; if given, reports how many rows match a "
                        "staged volume.")
    p.add_argument("--modality", default="gfp")
    p.add_argument("--min_frac", type=float, default=0.3,
                   help="A column counts as a label if this fraction of rows "
                        "parse as numbers.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    header, rows = _fm._read_rows(args.metadata)
    if not header:
        raise SystemExit(f"{args.metadata}: no header row")

    print("=" * 70)
    print(f" {args.metadata}")
    print("=" * 70)
    print(f"  {len(rows)} non-empty rows, {len(header)} columns")
    print(f"  columns: {', '.join(str(h) for h in header)}")

    plate_h = _fm._ci_resolve(header, args.plate_col)
    group_hs = [_fm._ci_resolve(header, c.strip())
                for c in args.group_cols.split(",") if c.strip()]
    group_hs = [h for h in group_hs if h]

    # ---- grouping structure ----
    print("\n  grouping structure")
    for h in header:
        vals = [str(r.get(h)).strip() for r in rows
                if str(r.get(h) or "").strip()]
        n_uniq = len(set(vals))
        if 1 < n_uniq <= max(30, len(rows) // 2) and len(vals) > len(rows) * 0.5:
            marker = ""
            if plate_h and h == plate_h:
                marker = "   <-- plate (the batch unit)"
            elif h in group_hs:
                marker = "   <-- part of --group_cols"
            print(f"    {str(h):28s} {n_uniq:>4} distinct{marker}")

    if plate_h:
        plates = {}
        for r in rows:
            pv = str(r.get(plate_h) or "").strip()
            if pv:
                plates.setdefault(pv, 0)
                plates[pv] += 1
        print(f"\n  plates: {len(plates)} -> "
              + ", ".join(f"{k}({v} rows)" for k, v in sorted(plates.items())))
        if len(plates) < 6:
            print(f"    NOTE: {len(plates)} plates means leave-one-plate-out has "
                  f"only {len(plates)} folds — that is the number of genuinely "
                  f"independent\n          acquisition batches, and it caps what "
                  f"any model on this data can be shown to generalize across.")

    # ---- candidate label columns ----
    print("\n  numeric label columns (candidate prediction targets)")
    hdr = (f"    {'column':30s} {'reps':>4} {'min':>9} {'median':>9} {'max':>9}"
           + (f" {'eta2':>7} {'null':>7} {'omega2':>8}" if plate_h else ""))
    print(hdr)
    out = {"metadata": args.metadata, "n_rows": len(rows),
           "columns": [str(h) for h in header], "labels": {}}
    rep_hs = [h for h in group_hs] or ([plate_h] if plate_h else [])
    for h in header:
        vals, gs = [], []
        # Aggregate to the REPLICATE (the modelling unit). Iterating raw rows
        # weights a 4-FOV tissue four times, so the eta^2 would be FOV-weighted
        # while the models are replicate-weighted — and this table is what you
        # use to CHOOSE a target.
        seen = {}
        for r in rows:
            f = as_float(r.get(h))
            if f is None:
                continue
            rk = tuple(str(r.get(g) or "NA") for g in rep_hs) if rep_hs else None
            pl = str(r.get(plate_h) or "NA") if plate_h else "NA"
            if rk is None:
                vals.append(f); gs.append(pl)
            elif rk not in seen:
                seen[rk] = True
                vals.append(f); gs.append(pl)
        n_reps = len(seen) if rep_hs else len(rows)
        if len(vals) < max(3, args.min_frac * max(n_reps, 1)):
            continue
        if len(set(vals)) < 3:
            continue          # an index or a flag, not a measurement
        v = np.asarray(vals)
        e2 = eta_squared(vals, gs) if plate_h and len(set(gs)) > 1 else float("nan")
        w2 = omega_squared(vals, gs) if plate_h and len(set(gs)) > 1 else float("nan")
        null = eta2_null(len(set(gs)), len(v)) if len(set(gs)) > 1 else float("nan")
        flag = ""
        if not np.isnan(w2):
            # Judge against the bias-corrected value, not raw eta^2: with 4
            # plates and ~22 replicates, eta^2 = 0.143 is what pure noise gives.
            if w2 > 0.4:
                flag = "  CONFOUNDED w/ plate"
            elif w2 < 0.1:
                flag = "  <-- no more plate structure than chance: testable"
        line = (f"    {str(h):30s} {len(v):>4} {v.min():>9.2f} "
                f"{np.median(v):>9.2f} {v.max():>9.2f}")
        if plate_h:
            line += (f" {e2:>7.3f} {null:>7.3f} {w2:>8.3f}"
                     if not np.isnan(e2) else f" {'n/a':>7} {'n/a':>7} {'n/a':>8}")
        print(line + flag)
        out["labels"][str(h)] = {
            "n": int(len(v)), "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v)),
            "eta2_plate": None if np.isnan(e2) else float(e2),
            "eta2_null": None if np.isnan(null) else float(null),
            "omega2_plate": None if np.isnan(w2) else float(w2)}

    # ---- categorical label columns (treated/not, perturbed/not, ...) ----
    print("\n  categorical label columns (candidate classification targets)")
    print(f"    {'column':30s} {'reps':>4} {'classes':>7} {'balance':>9} "
          f"{'plate->label':>12}")
    out["categorical"] = {}
    for h in header:
        if str(h) in out["labels"]:
            continue                       # already reported as numeric
        vals, pls, seen = [], [], {}
        for r in rows:
            s = str(r.get(h) or "").strip()
            if not s or s.lower() in ("na", "nan", "none", "-", "#n/a"):
                continue
            rk = tuple(str(r.get(g) or "NA") for g in rep_hs) if rep_hs else None
            if rk is not None and rk in seen:
                continue
            if rk is not None:
                seen[rk] = True
            vals.append(s)
            pls.append(str(r.get(plate_h) or "NA") if plate_h else "NA")
        n_cls = len(set(vals))
        if not (2 <= n_cls <= 6) or len(vals) < 6:
            continue
        cnt = {c: vals.count(c) for c in set(vals)}
        bal = max(cnt.values()) / len(vals)
        # Can plate ALONE predict the label? If it can, a model that only
        # learns acquisition batch scores perfectly and means nothing.
        hit = 0
        for pl in set(pls):
            sub = [v for v, q in zip(vals, pls) if q == pl]
            hit += max(sub.count(c) for c in set(sub))
        pacc = hit / len(vals)
        flag = ""
        if pacc > 0.95:
            flag = "  CONFOUNDED: plate determines this label"
        elif pacc - bal < 0.1:
            flag = "  <-- plate adds little: testable"
        print(f"    {str(h):30s} {len(vals):>4} {n_cls:>7} {bal:>8.2f} "
              f"{pacc:>11.2f}{flag}")
        out["categorical"][str(h)] = {
            "n": len(vals), "n_classes": n_cls, "classes": sorted(set(vals)),
            "majority_frac": bal, "plate_predicts_label": pacc}
    print("    balance      = fraction in the largest class (chance for a")
    print("                   majority-class guesser)")
    print("    plate->label = accuracy of predicting the label from plate")
    print("                   ALONE. Near 1.0 means the label IS the plate:")
    print("                   any batch-encoding feature 'predicts' it, and")
    print("                   --deconfound plate will remove the whole signal")
    print("                   because there is nothing else there.")

    # ---- how many rows correspond to real volumes ----
    if args.data_dir:
        mod_dir = os.path.join(args.data_dir, args.modality)
        if os.path.isdir(mod_dir):
            from glob import glob
            staged = sorted(os.path.splitext(os.path.basename(f))[0]
                            for f in glob(os.path.join(mod_dir, "*.npy")))
            print(f"\n  staged '{args.modality}' volumes on disk: {len(staged)}")
            out["n_staged"] = len(staged)
        else:
            print(f"\n  (no {mod_dir}/ — skipping the staged-volume check)")

    print("\n  how to read these")
    print("    eta2   raw between-plate share of variance. BIASED UPWARD: with")
    print("           k plates and n replicates its expected value under NO")
    print("           plate effect is (k-1)/(n-1), shown as 'null'. Never")
    print("           compare eta2 to zero.")
    print("    omega2 eta2 corrected for that bias. ~0 means the label has no")
    print("           more plate structure than chance -> testable. Large means")
    print("           the label is largely a plate property, so any batch-")
    print("           encoding feature predicts it and holding plates out")
    print("           leaves too few independent units.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  saved {args.output}")


if __name__ == "__main__":
    main()
