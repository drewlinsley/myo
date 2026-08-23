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
    """Between-group share of total variance. 1.0 = the label IS the group."""
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
    hdr = (f"    {'column':30s} {'n':>4} {'min':>9} {'median':>9} {'max':>9}"
           + (f" {'eta2_plate':>11}" if plate_h else ""))
    print(hdr)
    out = {"metadata": args.metadata, "n_rows": len(rows),
           "columns": [str(h) for h in header], "labels": {}}
    for h in header:
        vals, gs = [], []
        for r in rows:
            f = as_float(r.get(h))
            if f is not None:
                vals.append(f)
                gs.append(str(r.get(plate_h) or "NA") if plate_h else "NA")
        if len(vals) < max(3, args.min_frac * len(rows)):
            continue
        if len(set(vals)) < 3:
            continue          # an index or a flag, not a measurement
        v = np.asarray(vals)
        e2 = eta_squared(vals, gs) if plate_h and len(set(gs)) > 1 else float("nan")
        flag = ""
        if not np.isnan(e2):
            if e2 > 0.6:
                flag = "  CONFOUNDED w/ plate"
            elif e2 < 0.3:
                flag = "  <-- within-plate spread: testable"
        line = (f"    {str(h):30s} {len(v):>4} {v.min():>9.2f} "
                f"{np.median(v):>9.2f} {v.max():>9.2f}")
        if plate_h:
            line += f" {e2:>11.3f}" if not np.isnan(e2) else f" {'n/a':>11}"
        print(line + flag)
        out["labels"][str(h)] = {
            "n": int(len(v)), "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v)),
            "eta2_plate": None if np.isnan(e2) else float(e2)}

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

    print("\n  how to read eta2_plate")
    print("    ~1.0  the label is essentially a plate property. Not testable")
    print("          here: any batch-encoding feature predicts it, and holding")
    print("          plates out leaves too few independent units.")
    print("    <0.3  real within-plate spread. This is the column worth")
    print("          modelling, and it survives leave-one-plate-out as a test.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  saved {args.output}")


if __name__ == "__main__":
    main()
