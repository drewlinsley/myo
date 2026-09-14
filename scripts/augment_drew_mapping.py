#!/usr/bin/env python
"""data_mapping_drew.csv -> data_mapping_drew_aug.csv, with two derived columns.

plate      The acquisition batch, read out of the filename's 6-digit date
           token (093025_fixed_* and day12_093025_* -> 093025;
           T26_101525_* -> 101525). This sheet has no plate column, but the
           imaging day IS the batch unit, and every honest split/deconfound in
           this pipeline is keyed on one. Note the batch structure it reveals:
           all Dexamethasone rows are 093025 and all Activin rows are 101525
           -- drug FAMILY is day-determined -- but each day carries its own
           control (DMSO on 093025, BSA on 101525), so perturbed-vs-control
           is NOT batch-determined, unlike Exercise on the 051826 drop.

perturbed  'control' for the vehicle rows (DMSO, BSA), 'perturbed' for any
           drug, empty for NA (exercise rows; empty targets are excluded by
           the loader). The raw Perturbation column has six levels with 1-3
           tissues each, which no classifier can be validated on; the binary
           collapse is the trainable question.

Deterministic; rerun after any edit to the source CSV.
"""
import csv
import os
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "data_mapping_drew.csv"
dst = os.path.splitext(src)[0] + "_aug.csv"

rows = list(csv.DictReader(open(src)))
out_fields = list(rows[0].keys()) + ["plate", "perturbed"]
n_date = 0
for r in rows:
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", r["File"])
    if not m:
        raise SystemExit(f"no 6-digit date token in File={r['File']!r}; "
                         f"the plate/batch column cannot be derived")
    r["plate"] = m.group(1)
    n_date += 1
    p = r.get("Perturbation", "").strip()
    if not p or p.upper() == "NA":
        r["perturbed"] = ""
    elif "control" in p.lower():
        r["perturbed"] = "control"
    else:
        r["perturbed"] = "perturbed"

with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(rows)

from collections import Counter
print(f"{dst}: {len(rows)} rows")
print("  plate:", dict(Counter(r["plate"] for r in rows)))
print("  perturbed:", dict(Counter(r["perturbed"] or "(empty)" for r in rows)))
print("  perturbed x plate:",
      dict(Counter((r["perturbed"] or "-", r["plate"]) for r in rows)))
