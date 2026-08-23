"""Score a trained force classifier on TRAIN / VAL / TEST from its split manifest.

Why this exists
---------------
The decorrelation experiment assumes the reference model learned a *shortcut*:
something that fits the training replicates but does not generalize. Its saved
JSON only reports TEST metrics, so a reference at chance on test is ambiguous
between two very different situations:

  memorizing  train acc high, test acc ~chance -> a real shortcut. Its
              attribution maps mean something and decorrelating from them is
              a well-posed experiment.
  dead        train acc ~chance too -> nothing was learned at all. The
              attribution maps are gradients of a constant-ish function, and
              decorrelating from them is decorrelating from noise, which is a
              perfectly good explanation for a noisy student saliency map.

This prints both, so you can tell which one you have before spending more GPU
hours on lambda sweeps.

Usage
-----
    python eval_force_ckpt.py -c configs/gfp_classifier_3d.yaml \
        --ckpt results/force_from_gfp_new/force_ckpt_3d.pth \
        --split_json results/force_from_gfp_new/force_3d.split.json \
        --data_dir data_phalloidin_mhc_051826_staged
"""

import os
import json
import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, validate_config
from src.utils import set_seed, tune_cudnn
from src.data.regression_dataset import VolumeRegressionDataset
from src.models.gfp_classifier import build_gfp_classifier
from train_loo_force_classifier import assign_bin, build_transforms, _eval_det


def score_split(model, loader, n_stems, n_bins, device, criterion):
    """Returns (per-volume probs, patch counts, mean patch CE)."""
    model.eval()
    sums = np.zeros((n_stems, n_bins))
    counts = np.zeros(n_stems, dtype=int)
    tot_ce, n_seen = 0.0, 0
    with torch.no_grad():
        for img, tgt, fidx in loader:
            lex, _ = model(img.to(device))
            tot_ce += criterion(lex, tgt.long().to(device)).item() * img.shape[0]
            n_seen += img.shape[0]
            sm = lex.softmax(dim=1).cpu().numpy()
            for row, i in zip(sm, fidx.numpy().reshape(-1)):
                sums[int(i)] += row
                counts[int(i)] += 1
    probs = np.zeros_like(sums)
    valid = counts > 0
    probs[valid] = sums[valid] / counts[valid, None]
    return probs, counts, tot_ce / max(n_seen, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--patches_per_volume", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = validate_config(load_config(args.config))
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    tune_cudnn(tcfg.get("cudnn_benchmark", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = cfg["model"].get("dims", "2d")

    with open(args.split_json) as f:
        sp = json.load(f)
    n_bins = int(sp["n_bins"])
    classes = sp["classes"]
    edges = np.array(sp["scoring_bin_edges"], dtype=np.float64)
    groups = {g: list(v["stems"]) for g, v in sp["groups"].items()}
    group_bin = {g: int(v["bin"]) for g, v in sp["groups"].items()}
    forces_per_stem = {s: float(v) for s, v in sp["stem_force"].items()}
    targets = {s: float(assign_bin(forces_per_stem[s], edges))
               for s in forces_per_stem}

    data_dir = args.data_dir
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)
    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    pd_, cs_ = dcfg.get("patch_depth", 32), dcfg.get("crop_size", 256)
    bs = args.batch_size or tcfg["batch_size"]

    # Honour the normalization the checkpoint was TRAINED with. Defaulting to
    # "volume" for a globally-normalized model would score it on inputs it
    # never saw, with nothing detecting the mismatch.
    _peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_norm = _peek.get("norm_scope", "volume")
    ck_gpct = _peek.get("global_pct")
    if ck_norm != "volume":
        print(f"  checkpoint trained with norm_scope={ck_norm} "
              f"(global_pct={ck_gpct}) — matching it for evaluation")

    model = build_gfp_classifier(cfg, n_bins, 2).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"loaded {args.ckpt}"
          + (f"  (saved at epoch {ck['epoch']})" if "epoch" in ck else ""))
    criterion = nn.CrossEntropyLoss()
    chance = 1.0 / n_bins

    print(f"\n  n_bins={n_bins}  chance={chance:.3f}  "
          f"CE at chance = ln({n_bins}) = {math.log(n_bins):.4f}")
    print(f"  {'split':6s} {'reps':>5} {'vols':>5} {'vol_acc':>8} "
          f"{'rep_acc':>8} {'patch_CE':>9}   pred-dist / confidence")

    out = {"ckpt": args.ckpt, "split_json": args.split_json,
           "n_bins": n_bins, "chance": chance, "splits": {}}
    for name, key in (("train", "train_groups"), ("val", "val_groups"),
                      ("test", "test_groups")):
        gs = sp.get(key) or []
        if not gs:
            print(f"  {name:6s} {'-':>5} {'-':>5} {'(empty)':>8}")
            continue
        stems = [s for g in gs for s in groups[g]]
        ds = VolumeRegressionDataset(
            [os.path.join(mod_dir, f"{s}.npy") for s in stems],
            stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, False), z_range=z_range,
            apply_timm=apply_timm, mode=dims, patch_depth=pd_,
            patches_per_volume=args.patches_per_volume, crop_size=cs_,
            modality=args.input, norm_scope=ck_norm,
            global_pct=tuple(ck_gpct) if ck_gpct else None)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=False, num_workers=0)
        probs, counts, ce = _eval_det(
            lambda: score_split(model, loader, len(stems), n_bins, device,
                                criterion), seed + 9973)

        pred_vol = probs.argmax(axis=1)
        true_vol = np.array([int(targets[s]) for s in stems])
        seen = counts > 0
        vol_acc = float(np.mean(pred_vol[seen] == true_vol[seen])) if seen.any() else float("nan")

        # replicate level: average the per-volume probabilities within a rep
        rep_ok = []
        for g in gs:
            idx = [stems.index(s) for s in groups[g] if s in stems]
            idx = [i for i in idx if counts[i] > 0]
            if not idx:
                continue
            rep_ok.append(int(probs[idx].mean(axis=0).argmax()) == group_bin[g])
        rep_acc = float(np.mean(rep_ok)) if rep_ok else float("nan")

        # Collapse check. A model that emits one class regardless of input
        # scores at chance but has a CE far ABOVE chance, because it is
        # confidently wrong on every other class. That looks nothing like a
        # model that simply failed to learn, and it needs a different fix.
        hist = np.bincount(pred_vol[seen], minlength=n_bins) if seen.any() \
            else np.zeros(n_bins, dtype=int)
        conf = float(probs[seen].max(axis=1).mean()) if seen.any() else float("nan")
        top_frac = float(hist.max() / max(hist.sum(), 1))

        print(f"  {name:6s} {len(gs):>5} {int(seen.sum()):>5} {vol_acc:>8.3f} "
              f"{rep_acc:>8.3f} {ce:>9.4f}   "
              f"pred={list(hist)} conf={conf:.2f}")
        out["splits"][name] = {"n_replicates": len(gs),
                               "n_volumes": int(seen.sum()),
                               "volume_accuracy": vol_acc,
                               "replicate_accuracy": rep_acc,
                               "patch_ce": float(ce),
                               "pred_histogram": [int(h) for h in hist],
                               "mean_confidence": conf,
                               "top_class_fraction": top_frac}

    tr = out["splits"].get("train", {})
    tr_acc = tr.get("volume_accuracy")
    print("")
    tr_ce, tr_top = tr.get("patch_ce"), tr.get("top_class_fraction")
    if tr_ce is not None and tr_ce > math.log(n_bins) * 1.15:
        print(f"  NOTE: train CE {tr_ce:.3f} is well ABOVE chance "
              f"({math.log(n_bins):.3f}) — the model is confidently WRONG on "
              f"data it was allowed to memorize.")
        if tr_top is not None and tr_top > 0.6:
            print(f"        {tr_top*100:.0f}% of train volumes get the same "
                  f"predicted class: it has collapsed to a constant predictor.")
    if tr_acc is None or (isinstance(tr_acc, float) and math.isnan(tr_acc)):
        verdict = "could not score the training split"
    elif tr_acc > chance + 0.25:
        verdict = ("MEMORIZING — it fits train well above chance. A real "
                   "shortcut exists; its attribution maps are meaningful and "
                   "decorrelating from them is a well-posed experiment.")
    else:
        verdict = ("DEAD — train accuracy is near chance, so this model never "
                   "learned anything, on shortcuts or otherwise. Its "
                   "attribution maps are gradients of a near-constant "
                   "function. Decorrelating from them cannot help, and it "
                   "explains a noisy student saliency map. Fix the base "
                   "classifier (fewer bins / more data / LOO) before "
                   "spending more time on decorrelation.")
    print(f"  verdict: {verdict}")
    out["verdict"] = verdict

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
