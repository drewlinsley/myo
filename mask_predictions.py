"""Post-process: re-mask already-saved per-slice predictions using the BF
foreground. Lets you iterate on mask params without re-running model inference.

Input layout (what predict_per_slice.py writes when --mask_background is OFF or
when you want to redo masking with different params):
    pred_dir/<stem>/z{idx:04d}.npy   (H, W) float32

For each stem, reads matching BF volume from {data_dir}/bf/<stem>.npy, computes
foreground mask (same logic as src/data/foreground_mask.py), applies it to every
slice, and writes masked slices to out_dir/<stem>/z{idx:04d}.npy.

Usage:
    python mask_predictions.py \\
        --pred_dir predictions/new_dataset/unet_3d_imagenet_pearson_frac100 \\
        --data_dir data_phalloidin_mhc_051826_staged \\
        --out_dir predictions/new_dataset/unet_3d_imagenet_pearson_frac100_otsu_d5 \\
        --mask_method otsu --mask_dilate 5 --mask_min_frac 0.02
"""

import argparse
import glob
import os

import numpy as np

from src.data.foreground_mask import compute_bf_foreground_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True,
                   help="Directory of <stem>/z????.npy prediction slices")
    p.add_argument("--data_dir", required=True,
                   help="Dataset root containing bf/<stem>.npy")
    p.add_argument("--out_dir", required=True,
                   help="Where to write masked predictions (one folder per stem)")
    p.add_argument("--mask_method", default="minimum",
                   choices=["minimum", "otsu", "li", "triangle"])
    p.add_argument("--mask_dilate", type=int, default=3)
    p.add_argument("--mask_min_frac", type=float, default=0.01)
    p.add_argument("--stems", nargs="*", default=None,
                   help="Subset of stems to process; default = all subdirs")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    bf_dir = os.path.join(args.data_dir, "bf")
    if not os.path.isdir(bf_dir):
        raise SystemExit(f"Missing {bf_dir}/")

    if args.stems:
        stems = list(args.stems)
    else:
        stems = sorted(
            d for d in os.listdir(args.pred_dir)
            if os.path.isdir(os.path.join(args.pred_dir, d)))
    if not stems:
        raise SystemExit(f"No stem folders found under {args.pred_dir}")

    print(f"Re-masking {len(stems)} volume(s) "
          f"method={args.mask_method} dilate={args.mask_dilate} "
          f"min_frac={args.mask_min_frac}")

    n_done = n_skip = n_err = 0
    for stem in stems:
        stem_in = os.path.join(args.pred_dir, stem)
        stem_out = os.path.join(args.out_dir, stem)
        if os.path.isdir(stem_out) and not args.force:
            existing = sorted(glob.glob(os.path.join(stem_out, "z*.npy")))
            expected = sorted(glob.glob(os.path.join(stem_in, "z*.npy")))
            if len(existing) == len(expected) and existing:
                n_skip += 1
                continue
        os.makedirs(stem_out, exist_ok=True)

        bf_path = os.path.join(bf_dir, f"{stem}.npy")
        if not os.path.exists(bf_path):
            print(f"  ERR {stem}: missing BF at {bf_path}")
            n_err += 1
            continue
        bf_raw = np.load(bf_path)

        slice_files = sorted(glob.glob(os.path.join(stem_in, "z*.npy")))
        if not slice_files:
            print(f"  ERR {stem}: no prediction slices in {stem_in}")
            n_err += 1
            continue
        if bf_raw.shape[0] != len(slice_files):
            print(f"  WARN {stem}: BF has {bf_raw.shape[0]} Z but pred dir has "
                  f"{len(slice_files)} slices — clipping to the smaller of the two")
        n_z = min(bf_raw.shape[0], len(slice_files))

        # Compute mask once per volume (foreground_mask handles per-slice cleanup)
        fg = compute_bf_foreground_mask(
            bf_raw[:n_z], method=args.mask_method,
            dilate=args.mask_dilate,
            min_component_frac=args.mask_min_frac)

        for z in range(n_z):
            pred = np.load(slice_files[z])
            if pred.shape != fg[z].shape:
                print(f"  ERR {stem}: shape mismatch pred={pred.shape} mask={fg[z].shape}")
                n_err += 1
                break
            masked = (pred * fg[z].astype(pred.dtype)).astype(np.float32)
            np.save(os.path.join(stem_out, f"z{z:04d}.npy"), masked)
        else:
            n_done += 1
            if n_done <= 3 or n_done % 20 == 0:
                kept = float(fg.mean())
                print(f"  [{n_done}/{len(stems)}] {stem}: "
                      f"kept {kept:.2%} of pixels, wrote {n_z} slices")
            continue
        # break landed here — already counted as err

    print(f"Done. wrote={n_done} skipped={n_skip} errors={n_err}")
    print(f"  out: {args.out_dir}")


if __name__ == "__main__":
    main()
