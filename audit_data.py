"""Data audit: is every collected volume actually reaching training?

Traces the full data funnel and reports exactly where volumes drop out:

    raw .nd2 files            (--nd2_dir, optional)
      -> staged bf/ + gfp/ .npy pairs        (missing pair / missing stats)
      -> usable after z_range                (volumes with too few Z are
                                              SILENTLY skipped by the datasets)
      -> matched to a force label            (--metadata; unmatched rows and
                                              unlabeled staged volumes)
      -> assigned to a train/val/test split  (--splits manifests; stems in no
                                              split, split stems missing on disk)

Torch-free (numpy + yaml + openpyxl) so it runs in any env on the VM.

Usage
-----
    python audit_data.py --data_dir data_phalloidin_mhc_051826_staged \
        --metadata "phalloidin_mhc_mapping_051426_SS edit.xlsx" \
        --nd2_dir /path/to/raw_nd2_drop \
        --out results/audit/data_audit.json
    # or just:  bash scripts/audit_data.sh
"""

import os
import sys
import json
import argparse
import importlib.util
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import load_config  # torch-free  # noqa: E402


def _load_force_metadata_mod():
    """Import src/data/force_metadata.py by path — importing it as
    src.data.force_metadata would pull src/data/__init__.py -> torch."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "data", "force_metadata.py")
    spec = importlib.util.spec_from_file_location("force_metadata_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hdr(title):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def npy_shape(path):
    """Shape + dtype from the .npy header only (no data read)."""
    try:
        a = np.load(path, mmap_mode="r")
        return tuple(int(x) for x in a.shape), str(a.dtype)
    except Exception as e:  # corrupt / truncated file
        return None, f"ERROR: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Staged root with bf/ gfp/ stats/")
    p.add_argument("--nd2_dir", default=None,
                   help="Raw .nd2 drop root (recursive scan; optional)")
    p.add_argument("--metadata", default=None,
                   help="Force mapping spreadsheet (.xlsx/.csv; optional)")
    p.add_argument("--target_col", default="peak_amplitude_week3")
    p.add_argument("--file_col", default="file")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--config", default="configs/base.yaml",
                   help="Config supplying z_range / patch_depth / crop_size")
    p.add_argument("--splits", nargs="*", default=None,
                   help="Split manifest JSONs (*.split.json). Default: glob "
                        "results/*/*.split.json")
    p.add_argument("--out", default=None, help="Write the full report JSON here")
    args = p.parse_args()

    report = {"data_dir": args.data_dir}
    funnel = []

    # ------------------------------------------------------------------
    # 1. staged inventory
    # ------------------------------------------------------------------
    hdr(f"1. Staged volumes — {args.data_dir}")
    bf_dir = os.path.join(args.data_dir, "bf")
    gfp_dir = os.path.join(args.data_dir, "gfp")
    stats_dir = os.path.join(args.data_dir, "stats")
    bf_stems = {os.path.splitext(os.path.basename(f))[0]
                for f in glob(os.path.join(bf_dir, "*.npy"))}
    gfp_stems = {os.path.splitext(os.path.basename(f))[0]
                 for f in glob(os.path.join(gfp_dir, "*.npy"))}
    stat_stems = {os.path.splitext(os.path.basename(f))[0]
                  for f in glob(os.path.join(stats_dir, "*.json"))}
    paired = sorted(bf_stems & gfp_stems)
    bf_only = sorted(bf_stems - gfp_stems)
    gfp_only = sorted(gfp_stems - bf_stems)
    no_stats = sorted(s for s in paired if s not in stat_stems)

    print(f"bf: {len(bf_stems)}   gfp: {len(gfp_stems)}   "
          f"paired: {len(paired)}   stats jsons: {len(stat_stems)}")
    for name, lst in [("bf WITHOUT gfp (unusable for BF->GFP)", bf_only),
                      ("gfp WITHOUT bf", gfp_only),
                      ("paired but MISSING stats json", no_stats)]:
        if lst:
            print(f"  !! {len(lst)} {name}: {lst[:6]}"
                  + (" ..." if len(lst) > 6 else ""))
    if not (bf_only or gfp_only or no_stats):
        print("  all volumes paired and stats-covered ✓")

    shapes = {}
    bad_files = {}
    for s in paired:
        for mod, d in (("bf", bf_dir), ("gfp", gfp_dir)):
            shp, dt = npy_shape(os.path.join(d, f"{s}.npy"))
            if shp is None:
                bad_files[f"{mod}/{s}"] = dt
            else:
                shapes.setdefault(s, {})[mod] = {"shape": shp, "dtype": dt}
    mismatched = sorted(s for s, v in shapes.items()
                        if "bf" in v and "gfp" in v
                        and v["bf"]["shape"] != v["gfp"]["shape"])
    if bad_files:
        print(f"  !! {len(bad_files)} UNREADABLE npy file(s): "
              f"{list(bad_files)[:4]}")
    if mismatched:
        print(f"  !! {len(mismatched)} stem(s) with bf/gfp SHAPE MISMATCH: "
              f"{mismatched[:4]}")
    shape_counts = {}
    for s, v in shapes.items():
        if "bf" in v:
            shape_counts[str(v["bf"]["shape"])] = \
                shape_counts.get(str(v["bf"]["shape"]), 0) + 1
    print("  volume shapes (Z, H, W):")
    for shp, n in sorted(shape_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {shp}: {n} vols")
    report["staged"] = {
        "n_bf": len(bf_stems), "n_gfp": len(gfp_stems), "n_paired": len(paired),
        "bf_only": bf_only, "gfp_only": gfp_only, "missing_stats": no_stats,
        "shape_mismatch": mismatched, "unreadable": bad_files,
        "shape_counts": shape_counts,
    }
    funnel.append(("staged paired bf+gfp", len(paired)))

    # ------------------------------------------------------------------
    # 2. raw nd2 coverage (optional)
    # ------------------------------------------------------------------
    fm = _load_force_metadata_mod()
    if args.nd2_dir:
        hdr(f"2. Raw .nd2 coverage — {args.nd2_dir}")
        nd2s = sorted(glob(os.path.join(args.nd2_dir, "**", "*.nd2"),
                           recursive=True))
        staged_canons = {fm.canon_stem(s): s for s in (bf_stems | gfp_stems)}
        unstaged = []
        for f in nd2s:
            c = fm.canon_stem(f)
            hit = staged_canons.get(c) or next(
                (v for k, v in staged_canons.items()
                 if k == c or k.endswith("_" + c)), None)
            if hit is None:
                unstaged.append(os.path.relpath(f, args.nd2_dir))
        print(f"raw .nd2 files: {len(nd2s)}   matched to a staged stem: "
              f"{len(nd2s) - len(unstaged)}")
        if unstaged:
            print(f"  !! {len(unstaged)} nd2 file(s) NOT staged "
                  f"(rerun stage_nd2.py?):")
            for f in unstaged[:10]:
                print(f"    {f}")
            if len(unstaged) > 10:
                print(f"    ... and {len(unstaged) - 10} more")
        report["nd2"] = {"n_raw": len(nd2s), "unstaged": unstaged}
        funnel.insert(0, ("raw .nd2 files", len(nd2s)))
    else:
        print("\n(2. raw nd2 scan skipped — pass --nd2_dir to enable)")

    # ------------------------------------------------------------------
    # 3. z_range usability
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    dcfg = cfg.get("data", {})
    z_range = dcfg.get("z_range", None)
    patch_depth = dcfg.get("patch_depth", 32)
    crop_size = dcfg.get("crop_size", 256)
    hdr(f"3. z_range funnel — config {args.config}: z_range={z_range}, "
        f"patch_depth={patch_depth}, crop_size={crop_size}")
    auto = isinstance(z_range, str) and z_range == "auto"
    dropped_z, shallow_z, small_hw, missing_band, bands = [], [], [], [], []
    for s in paired:
        v = shapes.get(s, {}).get("bf")
        if not v:
            continue
        z, h, w = v["shape"][:3]
        if auto:
            try:
                with open(os.path.join(stats_dir, f"{s}.json")) as f:
                    band = json.load(f).get("z_auto")
            except Exception:
                band = None
            if band is None:
                missing_band.append(s)
                usable = 0
            else:
                usable = min(z, band[1]) - max(0, band[0])
                bands.append((s, band, usable))
        else:
            usable = (min(z, z_range[1]) - max(0, z_range[0])) if z_range else z
        if usable <= 0 and s not in missing_band:
            dropped_z.append((s, z))
        elif 0 < usable < patch_depth:
            shallow_z.append((s, usable))
        if h < crop_size or w < crop_size:
            small_hw.append((s, (h, w)))
    n_usable = len(paired) - len(dropped_z) - len(missing_band)
    print(f"usable after z_range: {n_usable}/{len(paired)}")
    if auto and missing_band:
        print(f"  !! {len(missing_band)} volume(s) with NO z_auto band in "
              f"stats — rerun compute_stats.py (it upgrades stats in place): "
              f"{missing_band[:6]}")
    if auto and bands:
        los = [b[1][0] for b in bands]
        print(f"  adaptive bands: start ranges {min(los)}..{max(los)}, "
              f"window lengths {sorted({b[2] for b in bands})}")
    if dropped_z:
        print(f"  !! {len(dropped_z)} volume(s) SILENTLY DROPPED — stack has "
              f"fewer Z-planes than z_range needs (0 slices survive the crop):")
        for s, z in dropped_z[:8]:
            print(f"    {s}: Z={z}")
        print("  -> fix: adjust z_range in the config, or re-acquire/restage.")
    if shallow_z:
        print(f"  note: {len(shallow_z)} volume(s) usable-Z < patch_depth "
              f"({patch_depth}) — reflect-padded in depth, e.g. "
              f"{shallow_z[:4]}")
    if small_hw:
        print(f"  note: {len(small_hw)} volume(s) smaller than crop_size in "
              f"H/W — reflect-padded, e.g. {small_hw[:4]}")
    if not dropped_z:
        print("  no volumes lost to z_range ✓")
    report["z_range"] = {"z_range": z_range, "dropped": dropped_z,
                         "shallow": shallow_z, "small_hw": small_hw,
                         "missing_z_auto": missing_band,
                         "bands": [(s, b) for s, b, _ in bands]}
    funnel.append(("usable after z_range", n_usable))

    # ------------------------------------------------------------------
    # 4. force-label match (optional)
    # ------------------------------------------------------------------
    matched_stems = None
    if args.metadata:
        hdr(f"4. Force-label match — {args.metadata} "
            f"(target={args.target_col})")
        try:
            data = fm.build_force_groups(
                args.metadata, args.data_dir, args.target_col,
                file_col=args.file_col,
                group_cols=tuple(c.strip() for c in args.group_cols.split(",")
                                 if c.strip()),
                modality="gfp")
            print("\n".join(data["report"]))
            matched_stems = set(data["forces"].keys())
            reps = sorted(data["groups"].items(),
                          key=lambda kv: data["rep_force"][kv[0]])
            print(f"\n  replicates ({len(reps)}), by force:")
            for g, stems in reps:
                print(f"    {g}: {len(stems)} vols  "
                      f"force={data['rep_force'][g]:.3f}")
            report["metadata"] = {
                "n_rows_force": data["n_rows_force"],
                "n_matched": data["n_matched"],
                "unmatched_meta": [list(u) for u in data["unmatched_meta"]],
                "unmatched_staged": data["unmatched_staged"],
                "replicates": {g: {"n_vols": len(v),
                                   "force": data["rep_force"][g]}
                               for g, v in data["groups"].items()},
            }
            funnel.append(("matched to a force label", data["n_matched"]))
        except Exception as e:
            print(f"  metadata match FAILED: {e}")
            report["metadata"] = {"error": str(e)}
    else:
        print("\n(4. force-label match skipped — pass --metadata to enable)")

    # ------------------------------------------------------------------
    # 5. split coverage
    # ------------------------------------------------------------------
    split_files = args.splits if args.splits else \
        sorted(glob("results/*/*.split.json"))
    report["splits"] = {}
    if split_files:
        hdr("5. Split coverage")
        for sf in split_files:
            try:
                with open(sf) as f:
                    sp = json.load(f)
            except Exception as e:
                print(f"  {sf}: unreadable ({e})")
                continue
            groups = sp.get("groups", {})
            in_split = {st for g in groups.values() for st in g["stems"]}
            tr = [g for g in sp.get("train_groups", [])]
            va = [g for g in sp.get("val_groups", [])]
            te = [g for g in sp.get("test_groups", [])]
            nv = lambda gs: sum(len(groups[g]["stems"]) for g in gs if g in groups)
            print(f"\n  {sf}")
            print(f"    train {len(tr)} reps/{nv(tr)} vols | "
                  f"val {len(va)} reps/{nv(va)} vols | "
                  f"test {len(te)} reps/{nv(te)} vols")
            missing_disk = sorted(s for s in in_split if s not in gfp_stems)
            if missing_disk:
                print(f"    !! {len(missing_disk)} split stem(s) NOT on disk: "
                      f"{missing_disk[:5]}")
            if matched_stems is not None:
                not_in_split = sorted(matched_stems - in_split)
                if not_in_split:
                    print(f"    !! {len(not_in_split)} force-matched stem(s) "
                          f"in NO split (stale manifest? rerun with FORCE=1): "
                          f"{not_in_split[:5]}")
                else:
                    print("    every force-matched stem is in the split ✓")
            # paired BF->GFP split, if present
            bfsf = sf.replace(".split.json", ".bfgfp_split.json")
            if os.path.exists(bfsf):
                with open(bfsf) as f:
                    bsp = json.load(f)
                btr, bva = set(bsp.get("train", [])), set(bsp.get("val", []))
                test_stems = {st for g in te for st in groups[g]["stems"]}
                leak = sorted((btr | bva) & test_stems)
                print(f"    bfgfp split: train {len(btr)} / val {len(bva)}"
                      + (f"  !! LEAK: {len(leak)} test stem(s) in BF->GFP "
                         f"train/val: {leak[:5]}" if leak
                         else "  (test excluded ✓)"))
            report["splits"][sf] = {
                "train_reps": len(tr), "val_reps": len(va), "test_reps": len(te),
                "train_vols": nv(tr), "val_vols": nv(va), "test_vols": nv(te),
                "missing_on_disk": missing_disk,
            }
        n_split_vols = max((v["train_vols"] + v["val_vols"] + v["test_vols"]
                            for v in report["splits"].values()), default=0)
        if n_split_vols:
            funnel.append(("in a train/val/test split", n_split_vols))
    else:
        print("\n(5. no *.split.json manifests found under results/*/)")

    # ------------------------------------------------------------------
    # bottom line
    # ------------------------------------------------------------------
    hdr("FUNNEL")
    prev = None
    for name, n in funnel:
        drop = f"   (-{prev - n})" if prev is not None and prev > n else ""
        print(f"  {name:34s} {n:5d}{drop}")
        prev = n
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
