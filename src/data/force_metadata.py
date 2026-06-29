"""Load per-replicate force labels from a mapping spreadsheet and match them to
staged GFP volumes (for the phalloidin/MHC 051826 drop and similar new datasets).

Why this is separate from ``extract_features.load_metadata``
-----------------------------------------------------------
The new drop ships a different schema with case-mixed headers, e.g.::

    file, peak_amplitude_week1, peak_amplitude_week3, slideinfo, plate, Tissue

``extract_features.load_metadata`` hard-codes a capital ``File`` column and the
old perturbation columns. This loader is case-insensitive, drops rows whose
chosen force column is empty/NA ("some files are empty -- exclude those"), and
matches the spreadsheet ``file`` values to the *staged* ``gfp/<stem>.npy`` files,
which were renamed by ``stage_nd2.safe_stem`` (path flattened, odd chars stripped).

Force semantics
---------------
``peak_amplitude_*`` is a **per-replicate (per-tissue) label**: every FOV / z-stack
of one physical tissue shares one force value. The replicate group id is built
from ``--group_cols`` (default ``plate,Tissue``) and is validated to be
force-homogeneous, so a single replicate never straddles a train/test split.
"""

import os
import re
from glob import glob


# ----------------------------------------------------------------------------
# Canonicalization — mirror stage_nd2.safe_stem so spreadsheet `file` values and
# staged `<stem>.npy` names land in the same namespace.
# ----------------------------------------------------------------------------
def canon_stem(name):
    """Canonical key for matching: basename, no extension, safe_stem char rules,
    lowercased. ``a/b c.nd2`` and ``b_c`` both canon to ``b_c``."""
    base = os.path.basename(str(name).strip())
    base = os.path.splitext(base)[0]
    base = re.sub(r"[\\/\s]+", "_", base)
    base = re.sub(r"[^A-Za-z0-9._-]", "", base)
    return base.lower()


def _read_rows(path):
    """Return (header_list, list_of_dicts) for .xlsx/.xls/.csv/.tsv."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        try:
            raw_header = next(it)
        except StopIteration:
            wb.close()
            return [], []
        header = ["" if c is None else str(c).strip() for c in raw_header]
        rows = []
        for r in it:
            # Skip fully-empty rows (trailing blanks are common in xlsx).
            if r is None or all(c is None or str(c).strip() == "" for c in r):
                continue
            row = {}
            for i, h in enumerate(header):
                if h == "":
                    continue
                row[h] = r[i] if i < len(r) else None
            rows.append(row)
        wb.close()
        return [h for h in header if h != ""], rows
    delim = "," if ext == ".csv" else "\t"
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def _ci_resolve(header, name):
    """Case-insensitive column resolver. Returns the actual header or None."""
    want = str(name).strip().lower()
    for h in header:
        if h is not None and str(h).strip().lower() == want:
            return h
    return None


def _to_float(v):
    """Parse a numeric force cell; '', 'NA', 'nan', None -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "NA" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _cell_str(v):
    if v is None:
        return ""
    return str(v).strip()


def list_staged_stems(staged_dir, modality="gfp"):
    """Stems of staged volumes: basenames of {staged_dir}/{modality}/*.npy."""
    mod_dir = os.path.join(staged_dir, modality)
    return sorted(os.path.splitext(os.path.basename(f))[0]
                  for f in glob(os.path.join(mod_dir, "*.npy")))


def build_force_groups(metadata_path, staged_dir, force_col,
                       file_col="file", group_cols=("plate", "Tissue"),
                       modality="gfp", staged_stems=None):
    """Match spreadsheet force labels to staged volumes and group by replicate.

    Returns a dict with:
        forces        {stem: float}                  matched, force-labeled vols
        groups        {group_id: [stem, ...]}        replicate -> its FOV stems
        rep_force     {group_id: float}              one force per replicate
        report        list[str]                      human-readable diagnostics
        unmatched_meta  [(file_value, canon, force)] force rows with no staged vol
        unmatched_staged [stem, ...]                 staged vols with no force row
        n_rows_total, n_rows_force, n_matched, n_fallback
        columns       {"file":..., "force":..., "groups":[...]}

    Raises ValueError on hard schema errors (missing columns) or on a
    force-inhomogeneous replicate group (force is per-tissue, so a group that
    mixes force values means group_cols is too coarse).
    """
    header, rows = _read_rows(metadata_path)
    if not header:
        raise ValueError(f"{metadata_path}: no header row found")

    file_h = _ci_resolve(header, file_col)
    if file_h is None:
        raise ValueError(
            f"{metadata_path}: no '{file_col}' column (case-insensitive). "
            f"Available: {header}")
    force_h = _ci_resolve(header, force_col)
    if force_h is None:
        raise ValueError(
            f"{metadata_path}: no force column '{force_col}' (case-insensitive). "
            f"Available: {header}")
    group_hs = []
    missing_group = []
    for gc in group_cols:
        h = _ci_resolve(header, gc)
        if h is None:
            missing_group.append(gc)
        else:
            group_hs.append((gc, h))
    if missing_group:
        raise ValueError(
            f"{metadata_path}: group column(s) {missing_group} not found "
            f"(case-insensitive). Available: {header}")

    if staged_stems is None:
        staged_stems = list_staged_stems(staged_dir, modality)
    # Canonical index of staged stems (detect collisions so matching stays 1:1).
    staged_index = {}
    staged_collisions = {}
    for s in staged_stems:
        c = canon_stem(s)
        if c in staged_index:
            staged_collisions.setdefault(c, [staged_index[c]]).append(s)
        else:
            staged_index[c] = s
    staged_canons = list(staged_index.keys())

    forces = {}
    groups = {}
    group_force = {}          # group_id -> set of rounded forces (homogeneity)
    matched_stems = set()
    unmatched_meta = []
    n_rows_force = 0
    n_fallback = 0
    fallback_examples = []
    dup_stem_conflicts = []

    for row in rows:
        fval = _cell_str(row.get(file_h))
        if fval == "":
            continue
        force = _to_float(row.get(force_h))
        if force is None:
            continue                       # "some files are empty -- exclude those"
        n_rows_force += 1
        c = canon_stem(fval)

        # 1) exact canonical match; 2) unique suffix fallback (staged canon ends
        #    with "_"+c or == c) for the case where staging added a path prefix.
        stem = staged_index.get(c)
        via_fallback = False
        if stem is None:
            cands = [sc for sc in staged_canons
                     if sc == c or sc.endswith("_" + c)]
            if len(cands) == 1:
                stem = staged_index[cands[0]]
                via_fallback = True
            # 0 or >1 candidates -> leave unmatched (ambiguous is not matched)
        if stem is None:
            unmatched_meta.append((fval, c, force))
            continue

        if via_fallback:
            n_fallback += 1
            if len(fallback_examples) < 5:
                fallback_examples.append((fval, stem))

        if stem in matched_stems and abs(forces.get(stem, force) - force) > 1e-9:
            dup_stem_conflicts.append((stem, forces.get(stem), force))
            continue
        matched_stems.add(stem)
        forces[stem] = force

        gid = "_".join(f"{name}={_cell_str(row.get(h)) or 'NA'}"
                       for name, h in group_hs)
        groups.setdefault(gid, [])
        if stem not in groups[gid]:
            groups[gid].append(stem)
        group_force.setdefault(gid, set()).add(round(force, 6))

    # Force-homogeneity: a replicate group must carry exactly one force value.
    inhomogeneous = {g: sorted(v) for g, v in group_force.items() if len(v) > 1}
    if inhomogeneous:
        sample = list(inhomogeneous.items())[:5]
        raise ValueError(
            "Replicate group(s) mix force values (group_cols too coarse for a "
            "per-tissue label). Add a finer column to --group_cols. Examples: "
            + "; ".join(f"{g} -> {vs}" for g, vs in sample))

    rep_force = {g: float(next(iter(group_force[g]))) for g in groups}

    unmatched_staged = [s for s in staged_stems if s not in matched_stems]

    # ---- diagnostics report ----
    report = []
    report.append(f"metadata: {metadata_path}")
    report.append(f"  columns -> file='{file_h}' force='{force_h}' "
                  f"groups={[h for _, h in group_hs]}")
    report.append(f"  rows with numeric '{force_h}': {n_rows_force} "
                  f"(of {len(rows)} non-empty rows)")
    report.append(f"staged '{modality}' volumes: {len(staged_stems)}")
    report.append(f"matched force<->volume: {len(matched_stems)} "
                  f"({n_fallback} via suffix fallback)")
    report.append(f"replicate groups (group_cols={list(group_cols)}): "
                  f"{len(groups)}")
    if fallback_examples:
        report.append("  suffix-fallback matches (file -> staged stem):")
        for fv, st in fallback_examples:
            report.append(f"    {fv}  ->  {st}")
    if staged_collisions:
        report.append(f"  WARNING: {len(staged_collisions)} canonical "
                      f"collision(s) among staged stems (ambiguous names):")
        for c, names in list(staged_collisions.items())[:5]:
            report.append(f"    {c} <- {names}")
    if dup_stem_conflicts:
        report.append(f"  WARNING: {len(dup_stem_conflicts)} staged vol(s) "
                      f"got conflicting forces from >1 metadata row (kept first):")
        for st, a, b in dup_stem_conflicts[:5]:
            report.append(f"    {st}: {a} vs {b}")
    if unmatched_meta:
        report.append(f"  {len(unmatched_meta)} force-labeled metadata row(s) "
                      f"matched NO staged volume, e.g.:")
        for fv, c, fo in unmatched_meta[:8]:
            report.append(f"    file='{fv}' (canon='{c}', force={fo})")
    if unmatched_staged:
        report.append(f"  {len(unmatched_staged)} staged volume(s) had NO force "
                      f"row (excluded from training), e.g.:")
        for s in unmatched_staged[:8]:
            report.append(f"    {s}")

    return {
        "forces": forces,
        "groups": groups,
        "rep_force": rep_force,
        "report": report,
        "unmatched_meta": unmatched_meta,
        "unmatched_staged": unmatched_staged,
        "n_rows_total": len(rows),
        "n_rows_force": n_rows_force,
        "n_matched": len(matched_stems),
        "n_fallback": n_fallback,
        "staged_collisions": staged_collisions,
        "dup_stem_conflicts": dup_stem_conflicts,
        "columns": {"file": file_h, "force": force_h,
                    "groups": [h for _, h in group_hs]},
    }
