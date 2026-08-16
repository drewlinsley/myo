"""Train a GFP->force classifier that is DECORRELATED from another model's
attribution maps (shortcut suppression via double backprop).

Motivation
----------
The direct 3D force classifier appears to latch onto shortcut features. Given
that model's per-volume attribution maps (attr_maps.py), this trainer fits a
NEW 2D or 3D classifier with an extra loss term:

    L = CE(logits, force_bin) + lambda * mean_i | corr( |dS/dx|_i , A_i ) |

where |dS/dx| is the student's own input-attribution (gradient of the target
class logit w.r.t. the input patch, computed with create_graph=True so the
penalty backpropagates through the gradient itself — double backprop), and A is
the reference (shortcut) model's attribution over the SAME patch. Minimizing
the absolute Pearson correlation pushes the student to justify its predictions
with pixels the shortcut model did NOT rely on. Runs in fp32 (double backprop
and AMP grad scaling don't mix).

Split hygiene: this trainer only CONSUMES a split manifest emitted by
train_split_force_classifier.py (--split_json <...>.split.json), so the
replicate train/val/test split — and therefore the leak-free test set — is
identical to the baseline it is compared against.

Usage
-----
    python attr_maps.py -c configs/gfp_classifier_3d.yaml \
        --ckpt results/force_from_gfp_new/force_ckpt_3d.pth \
        --data_dir data_phalloidin_mhc_051826_staged \
        --output_dir results/force_decorr/attr_ref
    python train_decorr_force.py -c configs/gfp_classifier_3d.yaml \
        --split_json results/force_from_gfp_new/force_3d.split.json \
        --data_dir data_phalloidin_mhc_051826_staged \
        --attr_dir results/force_decorr/attr_ref --attr_lambda 1.0 \
        --init_from ckpts/unet_3d_imagenet_pearson/best.pth \
        --save_ckpt results/force_decorr/decorr_ckpt_3d.pth \
        --output results/force_decorr/decorr_3d.json
    # (or: bash scripts/decorr_force.sh)
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.config import load_config, validate_config
from src.utils import set_seed, tune_cudnn
from src.data.normalization import normalize
from src.data.zband import resolve_z_range
from src.data.regression_dataset import VolumeRegressionDataset
from src.models.gfp_classifier import build_gfp_classifier
from train_loo_force_classifier import (
    assign_bin, edges_to_ranges, pearson, spearman, _seed_worker, _eval_det,
    load_encoder_from_unet, build_transforms)
from train_split_force_classifier import make_plot


# ---------------------------------------------------------------------------
# Dataset: modality patch + the SAME crop of the reference attribution map,
# with shared geometric augmentation (jitter applies to the image only).
# ---------------------------------------------------------------------------
class PairedAttrDataset(Dataset):
    def __init__(self, files, attr_dir, stats_dir, targets, z_range=None,
                 apply_timm=True, mode="2d", patch_depth=32,
                 patches_per_volume=32, crop_size=256, modality="gfp",
                 jitter=0.05):
        self.files = files
        self.attr_dir = attr_dir
        self.z_range = z_range
        self.apply_timm = apply_timm
        self.mode = mode
        self.patch_depth = patch_depth
        self.crop_size = crop_size
        self.modality = modality
        self.jitter = jitter

        self.stats, self.target_vals, self.attr_files = [], [], []
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(stats_dir, f"{stem}.json")) as f:
                self.stats.append(json.load(f))
            self.target_vals.append(float(targets[stem]))
            ap = os.path.join(attr_dir, f"{stem}.npy")
            if not os.path.exists(ap):
                raise FileNotFoundError(
                    f"no reference attribution map for {stem} — run "
                    f"attr_maps.py first (looked at {ap})")
            self.attr_files.append(ap)

        self.index_map = []
        for i, path in enumerate(files):
            n_z = np.load(path, mmap_mode="r").shape[0]
            if z_range is not None:
                z_lo, z_hi = resolve_z_range(z_range, self.stats[i], n_z)
                n_z = z_hi - z_lo
            if n_z < 1:
                continue
            if mode == "2d":
                for z in range(n_z):
                    self.index_map.append((i, z))
            else:
                for p in range(patches_per_volume):
                    self.index_map.append((i, p))
        self._cache = {}

    def __len__(self):
        return len(self.index_map)

    def _load_raw(self, i):
        if i in self._cache:
            return self._cache[i]
        raw = np.load(self.files[i], mmap_mode="r")
        if self.z_range is not None:
            z_lo, z_hi = resolve_z_range(self.z_range, self.stats[i],
                                         raw.shape[0])
            raw = raw[z_lo:z_hi]
        attr = np.load(self.attr_files[i], mmap_mode="r")
        if tuple(attr.shape) != tuple(raw.shape):
            raise ValueError(
                f"attr map shape {attr.shape} != z-cropped volume shape "
                f"{raw.shape} for {self.files[i]} — the maps were computed in "
                f"a different z_range frame; rerun attr_maps.py with the same "
                f"config.")
        self._cache[i] = (raw, attr)
        return raw, attr

    def _augment(self, img, attr):
        """Shared geometric augs on (..., H, W[, extra]) numpy pairs. 2D arrays
        are (H, W); 3D are (D, H, W). Rot90 acts on the last two axes."""
        hw = (img.ndim - 2, img.ndim - 1)
        if np.random.rand() < 0.5:
            img, attr = img[..., ::-1], attr[..., ::-1]
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=hw[0])
            attr = np.flip(attr, axis=hw[0])
        if img.ndim == 3 and np.random.rand() < 0.5:
            img, attr = img[::-1], attr[::-1]
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k, axes=hw)
            attr = np.rot90(attr, k, axes=hw)
        # intensity jitter on the IMAGE only (attribution is a fixed target)
        if self.jitter:
            alpha = 1.0 + np.random.uniform(-self.jitter, self.jitter)
            beta = np.random.uniform(-self.jitter, self.jitter)
            img = img * alpha + beta
        return np.ascontiguousarray(img), np.ascontiguousarray(attr)

    def __getitem__(self, idx):
        file_idx, slot = self.index_map[idx]
        raw, attr = self._load_raw(file_idx)
        st = self.stats[file_idx]
        cs = self.crop_size

        if self.mode == "2d":
            img = np.asarray(raw[slot], dtype=np.float32)
            amp = np.asarray(attr[slot], dtype=np.float32)
            ph, pw = max(0, cs - img.shape[0]), max(0, cs - img.shape[1])
            if ph or pw:
                img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
                amp = np.pad(amp, ((0, ph), (0, pw)), mode="reflect")
            y = np.random.randint(0, img.shape[0] - cs + 1)
            x = np.random.randint(0, img.shape[1] - cs + 1)
            img, amp = img[y:y + cs, x:x + cs], amp[y:y + cs, x:x + cs]
            img = normalize(img, st[self.modality]["p_low"],
                            st[self.modality]["p_high"],
                            apply_timm=self.apply_timm)
            img, amp = self._augment(img, amp)
            t = torch.from_numpy(img[None]).float()          # (1, H, W)
            a = torch.from_numpy(amp[None]).float()
        else:
            z, h, w = raw.shape
            pd = self.patch_depth
            zd = np.random.randint(0, max(z - pd, 0) + 1)
            yh = np.random.randint(0, max(h - cs, 0) + 1)
            xw = np.random.randint(0, max(w - cs, 0) + 1)
            img = np.asarray(raw[zd:zd + pd, yh:yh + cs, xw:xw + cs],
                             dtype=np.float32)
            amp = np.asarray(attr[zd:zd + pd, yh:yh + cs, xw:xw + cs],
                             dtype=np.float32)
            pad = ((0, pd - img.shape[0]), (0, cs - img.shape[1]),
                   (0, cs - img.shape[2]))
            if any(p[1] > 0 for p in pad):
                img = np.pad(img, pad, mode="reflect")
                amp = np.pad(amp, pad, mode="reflect")
            img = normalize(img, st[self.modality]["p_low"],
                            st[self.modality]["p_high"],
                            apply_timm=self.apply_timm)
            img, amp = self._augment(img, amp)
            # (D, H, W) -> (1, H, W, D), matching ToTensor3D's layout
            t = torch.from_numpy(img.transpose(1, 2, 0)[None].copy()).float()
            a = torch.from_numpy(amp.transpose(1, 2, 0)[None].copy()).float()
        return t, a, float(self.target_vals[file_idx]), int(file_idx)


# ---------------------------------------------------------------------------
# The decorrelation penalty
# ---------------------------------------------------------------------------
def attr_corr_penalty(model, x, target, ref_attr, score_mode):
    """Returns (logits, mean |pearson(|dS/dx|, ref_attr)| over valid samples).

    S is the chosen class logit per sample; the gradient is taken with
    create_graph=True so the penalty trains the model (double backprop).
    Samples whose reference patch is (near-)constant contribute no penalty.
    """
    x = x.requires_grad_(True)
    logits, _ = model(x)
    if score_mode == "pred":
        idx = logits.argmax(dim=1)
    else:                      # 'true'
        idx = target
    score = logits.gather(1, idx.view(-1, 1)).sum()
    g = torch.autograd.grad(score, x, create_graph=True)[0].abs()

    b = x.shape[0]
    gf = g.reshape(b, -1)
    af = ref_attr.reshape(b, -1)
    gz = gf - gf.mean(dim=1, keepdim=True)
    az = af - af.mean(dim=1, keepdim=True)
    denom = gz.norm(dim=1) * az.norm(dim=1)
    corr = (gz * az).sum(dim=1) / (denom + 1e-12)
    valid = (az.norm(dim=1) > 1e-8) & (gz.norm(dim=1) > 1e-8)
    if valid.any():
        pen = corr[valid].abs().mean()
    else:
        pen = torch.zeros((), device=x.device)
    return logits, pen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True,
                   help="STUDENT architecture config (gfp_classifier[_3d].yaml)")
    p.add_argument("--split_json", required=True,
                   help="<force_run>.split.json from train_split_force_classifier "
                        "(identical leak-free replicate split as the baseline)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--attr_dir", required=True,
                   help="Reference attribution maps from attr_maps.py")
    p.add_argument("--attr_lambda", type=float, default=1.0,
                   help="Weight of the decorrelation penalty (0 = plain CE)")
    p.add_argument("--attr_score", choices=["true", "pred"], default="true",
                   help="Which logit's input-gradient to decorrelate")
    p.add_argument("--input", choices=["bf", "gfp"], default="gfp")
    p.add_argument("--init_from", default=None,
                   help="BF->GFP U-Net ckpt to warm-start the encoder")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override cfg (double backprop ~2x memory; halve if OOM)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n_permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_ckpt", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = validate_config(load_config(args.config))
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    tune_cudnn(tcfg.get("cudnn_benchmark", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = cfg["model"].get("dims", "2d")
    eval_seed = seed + 9973

    # ---- consume the split manifest (identical to the baseline run) ----
    with open(args.split_json) as f:
        sp = json.load(f)
    n_bins = int(sp["n_bins"])
    classes = sp["classes"]
    edges = np.array(sp["scoring_bin_edges"], dtype=np.float64)
    class_rep = np.array(sp["class_rep_force"], dtype=np.float64)
    groups = {g: list(v["stems"]) for g, v in sp["groups"].items()}
    rep_force = {g: float(v["force"]) for g, v in sp["groups"].items()}
    group_bin = {g: int(v["bin"]) for g, v in sp["groups"].items()}
    train_g, val_g, test_g = (sp["train_groups"], sp["val_groups"],
                              sp["test_groups"])
    forces_per_stem = {s: float(v) for s, v in sp["stem_force"].items()}
    targets = {s: float(assign_bin(forces_per_stem[s], edges))
               for s in forces_per_stem}
    train_stems = [s for g in train_g for s in groups[g]]
    val_stems = [s for g in val_g for s in groups[g]]
    test_stems = [s for g in test_g for s in groups[g]]
    print(f"split: train {len(train_g)} reps/{len(train_stems)} vols | "
          f"val {len(val_g)}/{len(val_stems)} | test {len(test_g)}/{len(test_stems)}"
          f"  (n_bins={n_bins}, from {args.split_json})")

    data_dir = args.data_dir
    stats_dir = os.path.join(data_dir, "stats")
    mod_dir = os.path.join(data_dir, args.input)
    apply_timm = cfg["model"].get("encoder_weights") is not None
    z_range = dcfg.get("z_range", None)
    pd_, cs_ = dcfg.get("patch_depth", 32), dcfg.get("crop_size", 256)

    train_ds = PairedAttrDataset(
        [os.path.join(mod_dir, f"{s}.npy") for s in train_stems],
        attr_dir=args.attr_dir, stats_dir=stats_dir, targets=targets,
        z_range=z_range, apply_timm=apply_timm, mode=dims, patch_depth=pd_,
        patches_per_volume=dcfg.get("patches_per_volume", 32),
        crop_size=cs_, modality=args.input)

    def make_eval_ds(stem_list):
        return VolumeRegressionDataset(
            [os.path.join(mod_dir, f"{s}.npy") for s in stem_list],
            stats_dir=stats_dir, targets=targets,
            transform=build_transforms(cfg, False), z_range=z_range,
            apply_timm=apply_timm, mode=dims, patch_depth=pd_,
            patches_per_volume=8, crop_size=cs_, modality=args.input)

    bs = args.batch_size or tcfg["batch_size"]
    gen = torch.Generator(); gen.manual_seed(seed * 1000 + 7)
    nw = tcfg.get("num_workers", 4)
    wkw = ({"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {})
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True,
        num_workers=nw, worker_init_fn=_seed_worker, generator=gen, **wkw)
    test_loader = torch.utils.data.DataLoader(
        make_eval_ds(test_stems), batch_size=bs, shuffle=False, num_workers=0)
    val_loader = (torch.utils.data.DataLoader(
        make_eval_ds(val_stems), batch_size=bs, shuffle=False, num_workers=0)
        if val_stems else None)

    model = build_gfp_classifier(cfg, n_bins, 2).to(device)
    warm_started = bool(args.init_from)
    if args.init_from:
        n_match, n_keys = load_encoder_from_unet(model, args.init_from, "cpu")
        print(f"warm-started encoder: matched {n_match}/{n_keys} tensors")
        if n_match == 0:
            raise SystemExit("--init_from matched 0 encoder tensors — "
                             "architecture mismatch?")
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg.get("weight_decay", 0.01))
    epochs = args.epochs or tcfg.get("epochs", 100)
    patience = tcfg.get("patience", 15)
    min_delta = tcfg.get("min_delta", 1e-3)
    print(f"training {dims} student: lambda={args.attr_lambda} "
          f"score={args.attr_score} bs={bs} epochs<={epochs} fp32 "
          f"(double backprop)")

    def eval_loss_on(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for img, tgt, _ in loader:
                lex, _ = model(img.to(device))
                tot += criterion(lex, tgt.long().to(device)).item() * img.shape[0]
                n += img.shape[0]
        return tot / max(n, 1)

    def predict_probs(loader, n_stems):
        model.eval()
        sums = np.zeros((n_stems, n_bins)); counts = np.zeros(n_stems, dtype=int)
        with torch.no_grad():
            for img, _tgt, fidx in loader:
                lex, _ = model(img.to(device))
                sm = lex.softmax(dim=1).cpu().numpy()
                for row, i in zip(sm, fidx.numpy().reshape(-1)):
                    sums[int(i)] += row; counts[int(i)] += 1
        probs = np.zeros_like(sums); valid = counts > 0
        probs[valid] = sums[valid] / counts[valid, None]
        return probs, counts

    history = []
    best_sig, best_probs, best_counts, best_epoch, no_improve = \
        float("inf"), None, None, 0, 0
    for ep in range(epochs):
        model.train()
        ces, pens = [], []
        for img, ref, tgt, _ in train_loader:
            img = img.to(device, non_blocking=True)
            ref = ref.to(device, non_blocking=True)
            target = tgt.long().to(device)
            if args.attr_lambda > 0:
                logits, pen = attr_corr_penalty(model, img, target, ref,
                                                args.attr_score)
            else:
                logits, _ = model(img)
                pen = torch.zeros((), device=device)
            ce = criterion(logits, target)
            loss = ce + args.attr_lambda * pen
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ces.append(ce.item()); pens.append(float(pen.item()))
        tr_ce = float(np.mean(ces)) if ces else float("inf")
        tr_pen = float(np.mean(pens)) if pens else 0.0

        if val_loader is not None:
            sig = _eval_det(lambda: eval_loss_on(val_loader), eval_seed)
            sig_name = "val_ce"
        else:
            sig, sig_name = tr_ce, "train_ce"
        probs, counts = _eval_det(
            lambda: predict_probs(test_loader, len(test_stems)), eval_seed)
        history.append({"epoch": ep + 1, "train_ce": tr_ce,
                        "attr_corr": tr_pen, sig_name: sig})
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep+1}/{epochs} train_ce={tr_ce:.4f} "
                  f"attr_corr={tr_pen:.4f} {sig_name}={sig:.4f}")
        if sig < best_sig - min_delta:
            best_sig, best_probs, best_counts, best_epoch = sig, probs, counts, ep + 1
            no_improve = 0
            if args.save_ckpt:
                os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
                tmp = args.save_ckpt + ".tmp"
                torch.save({
                    "epoch": best_epoch, "val_loss": float(sig),
                    "model_state_dict": model.state_dict(), "head": "exercise",
                    "target_col": sp.get("target_col"), "n_bins": n_bins,
                    "bin_edges": [float(e) for e in edges], "classes": classes,
                    "class_rep_force": class_rep.tolist(), "input": args.input,
                    "split": "train_val_test",
                    "decorr": {"attr_lambda": args.attr_lambda,
                               "attr_dir": args.attr_dir,
                               "attr_score": args.attr_score},
                }, tmp)
                os.replace(tmp, args.save_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop ep{ep+1} ({sig_name}={best_sig:.4f} "
                      f"@ ep{best_epoch})")
                break

    if best_probs is None:
        best_probs, best_counts = probs, counts

    # ---- per-volume -> per-replicate aggregation on TEST ----
    valid = best_counts > 0
    stem_to_probs = {s: best_probs[i] for i, s in enumerate(test_stems)
                     if valid[i]}
    results = []
    for g in test_g:
        member = [(s, stem_to_probs[s]) for s in groups[g] if s in stem_to_probs]
        if not member:
            continue
        vp = np.array([m[1] for m in member])
        rep_prob = vp.mean(axis=0)
        rep_pred = int(rep_prob.argmax())
        results.append({
            "group": g, "true_bin": int(group_bin[g]),
            "true_class": classes[group_bin[g]],
            "true_force": float(rep_force[g]), "pred_bin": rep_pred,
            "pred_class": classes[rep_pred],
            "expected_force": float(np.dot(rep_prob, class_rep)),
            "correct": int(rep_pred == group_bin[g]),
            "rep_probs": [float(x) for x in rep_prob],
            "per_volume": [{"stem": s, "pred_bin": int(pr.argmax()),
                            "expected_force": float(np.dot(pr, class_rep)),
                            "probs": [float(x) for x in pr]}
                           for s, pr in member],
        })
    if not results:
        raise SystemExit("no test replicate produced a prediction")

    true_bins = np.array([r["true_bin"] for r in results])
    pred_bins = np.array([r["pred_bin"] for r in results])
    true_force = np.array([r["true_force"] for r in results])
    exp_force = np.array([r["expected_force"] for r in results])
    rep_acc = float(np.mean(true_bins == pred_bins))
    vol_true = [r["true_bin"] for r in results for _ in r["per_volume"]]
    vol_pred = [v["pred_bin"] for r in results for v in r["per_volume"]]
    vol_acc = float(np.mean(np.array(vol_true) == np.array(vol_pred)))
    confusion = np.zeros((n_bins, n_bins), dtype=int)
    for t, q in zip(true_bins, pred_bins):
        confusion[t, q] += 1
    ranges = edges_to_ranges(edges, n_bins)
    per_class = {classes[b]: {
        "total": int((true_bins == b).sum()),
        "correct": int(((true_bins == b) & (pred_bins == b)).sum()),
        "accuracy": (float(((true_bins == b) & (pred_bins == b)).sum()
                           / (true_bins == b).sum())
                     if (true_bins == b).any() else float("nan")),
        "force_range_display": ranges[b]} for b in range(n_bins)}
    corr = {
        "spearman_expected_vs_force": spearman(true_force, exp_force),
        "pearson_expected_vs_force": pearson(true_force, exp_force),
        "spearman_predbin_vs_force": spearman(true_force,
                                              pred_bins.astype(float)),
    }
    perm_info = None
    if args.n_permutations and len(results) > 2:
        rng = np.random.default_rng(0)
        pa = np.array([np.mean(rng.permutation(true_bins) == pred_bins)
                       for _ in range(args.n_permutations)])
        perm_info = {"n_permutations": args.n_permutations,
                     "p_value_accuracy": (int((pa >= rep_acc).sum()) + 1)
                     / (args.n_permutations + 1),
                     "perm_mean": float(pa.mean())}

    chance = 1.0 / n_bins
    print(f"\nTEST replicate accuracy: {rep_acc:.3f} (chance {chance:.3f}) | "
          f"per-volume {vol_acc:.3f} | spearman(E[force], true)="
          f"{corr['spearman_expected_vs_force']:.3f} | "
          f"final attr_corr={history[-1]['attr_corr']:.4f}")

    summary = {
        "task": "force_classification_decorr",
        "target_col": sp.get("target_col"), "input": args.input, "dims": dims,
        "attr_lambda": args.attr_lambda, "attr_score": args.attr_score,
        "attr_dir": args.attr_dir, "consumed_split_json": args.split_json,
        "init_from": args.init_from, "warm_started": warm_started,
        "seed": int(seed), "n_bins": n_bins, "classes": classes,
        "bin_scheme": sp.get("bin_scheme"),
        "scoring_bin_edges": [float(e) for e in edges],
        "class_rep_force": class_rep.tolist(), "chance": chance,
        "n_test_replicates": len(results),
        "replicate_accuracy": rep_acc, "volume_accuracy": vol_acc,
        "per_class": per_class, "confusion_matrix": confusion.tolist(),
        "confusion_axes": {"rows": "true", "cols": "pred", "order": classes},
        "correlation": corr, "permutation_test": perm_info,
        "best_epoch": best_epoch, "history": history,
        "split": {k: [{"group": g, "force": rep_force[g],
                       "bin": group_bin[g], "n_vols": len(groups[g])}
                      for g in gl]
                  for k, gl in [("train", train_g), ("val", val_g),
                                ("test", test_g)]},
        "per_test_replicate": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.output}")
    try:
        make_plot(summary, os.path.splitext(args.output)[0] + ".png")
        print(f"Saved {os.path.splitext(args.output)[0]}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
