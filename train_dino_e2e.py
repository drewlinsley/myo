#!/usr/bin/env python
"""End-to-end: finetune the DINOv2 backbone on force, leave-one-replicate-out.

The frozen probe asks "do the pretrained features carry force?". This asks the
stronger question: can the backbone LEARN force-relevant features from these
images? The honest protocol is unchanged -- the unit of validation is the
replicate, so every one of the ~22 folds finetunes its own backbone with the
held-out tissue never seen. That is what makes this expensive: one condition
is ~22 trainings, and the shuffled-label control doubles it.

What does NOT change with more capacity: the detection floor. At n=22
replicates the permutation null spans roughly +/-0.55 whatever the model is,
so end-to-end adds expressiveness, not statistical power. Expect this to be
worth running only as a comparison against the corrected-mask frozen
baseline, and read both against the same yardstick.

Guardrails built in, in order of importance:
  * fixed hyperparameters for every fold AND every permutation -- selecting
    them on the observed data but not under the null would bias the null;
  * --shuffle trains the full LOO on permuted labels (the leak canary);
  * --n_perm N builds a real permutation null by re-running the ENTIRE
    LOO N times on within-plate-permuted labels. Default 0 because each
    permutation costs a full condition; 19 permutations gives p-resolution
    0.05 and is an overnight job;
  * adaptation is low-rank (LoRA on the last N blocks) when peft is
    available, falling back to unfreezing the last N blocks -- 86M free
    parameters against 21 training tissues is a memorization machine, so the
    trainable set is kept small;
  * --final_fit trains one model on ALL volumes for XAI. That checkpoint has
    seen every label: it exists to be LOOKED AT, and any statistic computed
    downstream of it (including probing features re-extracted with it) is
    LEAKY and must not be reported.

Pooling matches the frozen probe exactly -- token-level foreground weighting
(the mask), view-level fg weighting (fgmean), --fg_min floor -- so the
prediction is linear in the pooled embedding and the whole XAI machinery
(readout direction, exact attribution) carries over to the finetuned space.

Output JSON is schema-compatible with probe_force_features.py, so
plot_force_probe.py renders it unchanged.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import time

import numpy as np

from probe_force_features import (build_force_groups, compute_bin_edges,
                                  assign_bin, score, spearman,
                                  _permute_replicate_force)
from src.data.dino_views import (plan_views, make_view, area_resize,
                                 view_foreground)
from src.data.zband import resolve_z_range
from src.data.normalization import normalize, global_percentiles
from extract_dino_features import build_dino, global_mask_threshold

TILE = 518


def plate_of(g):
    for part in str(g).split("_"):
        if part.lower().startswith("plate="):
            return part.split("=", 1)[1]
    return "NA"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def prepare_volumes(args):
    """One record per volume: mmap, z-band, eligible views, token weights."""
    mod_dir = os.path.join(args.data_dir, args.modality)
    stats_dir = os.path.join(args.data_dir, "stats")
    avail = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(mod_dir, "*.npy")))
    data = build_force_groups(args.metadata, args.data_dir, args.target_col,
                              file_col=args.file_col,
                              group_cols=tuple(args.group_cols.split(",")),
                              modality=args.modality, staged_stems=avail,
                              target_type=args.target_type)
    if data["n_matched"] == 0:
        raise SystemExit("no metadata force rows matched any volume")
    forces, groups = data["forces"], data["groups"]
    stem_group = {s: g for g, ss in groups.items() for s in ss}

    gpct = (global_percentiles(stats_dir, args.modality)
            if args.norm_scope == "global" else None)
    msk_dir = os.path.join(args.data_dir, args.mask_source)
    g_thresh = global_mask_threshold(
        sorted(forces), msk_dir, stats_dir, "auto", args.z_stride,
        args.mask_method, projection=args.mask_projection)
    print(f"mask: {args.mask_source}/{args.mask_method} "
          f"projection={args.mask_projection} polarity={args.mask_polarity} "
          f"threshold={g_thresh}")

    gx, gy = (int(v) for v in args.tile_grid.split(","))
    vols, dropped = [], []
    for stem in sorted(forces):
        if stem not in stem_group:
            continue
        st = json.load(open(os.path.join(stats_dir, f"{stem}.json")))
        vol = np.load(os.path.join(mod_dir, f"{stem}.npy"), mmap_mode="r")
        n_z, H, W = vol.shape
        z_lo, z_hi = resolve_z_range("auto", st, n_z)
        zs = list(range(z_lo, z_hi, max(1, args.z_stride)))
        p_low, p_high = (gpct if gpct else
                         (float(st[args.modality]["p_low"]),
                          float(st[args.modality]["p_high"])))

        mv = np.load(os.path.join(msk_dir, f"{stem}.npy"), mmap_mode="r")
        band = np.asarray(mv[z_lo:z_hi:max(1, args.z_stride)],
                          dtype=np.float32)
        proj = band.max(axis=0) if args.mask_projection == "max" else band[0]
        mask2d = (proj > g_thresh if args.mask_polarity == "bright"
                  else proj < g_thresh)

        specs = plan_views(H, W, "tiled", tile_size=TILE, tile_grid=(gx, gy))
        elig, fgs, tws = [], [], []
        g = TILE // 14
        for sp in specs:
            fg = view_foreground(mask2d, sp)
            if fg is None or fg < args.fg_min:
                continue
            mt = mask2d[sp["y"]:sp["y"] + TILE,
                        sp["x"]:sp["x"] + TILE].astype(np.float32)
            tw = np.clip(area_resize(mt, g, g), 0, None)
            tw = (np.full((g, g), 1.0 / (g * g), np.float32)
                  if tw.sum() <= 1e-6 else (tw / tw.sum()).astype(np.float32))
            elig.append(sp)
            fgs.append(float(fg))
            tws.append(tw)
        if not elig:
            dropped.append(stem)
            continue
        fgs = np.asarray(fgs, np.float64)
        vols.append({"stem": stem, "group": stem_group[stem],
                     "force": float(forces[stem]),
                     "plate": plate_of(stem_group[stem]),
                     "vol": vol, "zs": zs, "p": (p_low, p_high),
                     "specs": elig, "fg": fgs, "fg_p": fgs / fgs.sum(),
                     "tw": tws})
    if dropped:
        print(f"  DROPPED {len(dropped)} volume(s) with no tile at "
              f"fg>={args.fg_min}: {', '.join(dropped[:5])}")
    n_views = [len(v["zs"]) * len(v["specs"]) for v in vols]
    print(f"{len(vols)} volumes; eligible views/volume "
          f"median {int(np.median(n_views))} "
          f"(tiles at fg>={args.fg_min}: "
          f"median {int(np.median([len(v['specs']) for v in vols]))})")
    return vols, data.get("classes")


def sample_views(v, K, rng, ctx, augment):
    """K (z, spec) draws: spec ~ fg, z uniform. Flips move mask AND image."""
    si = rng.choice(len(v["specs"]), size=K, p=v["fg_p"])
    zi = rng.integers(0, len(v["zs"]), size=K)
    views, tws, fg = [], [], []
    for s, z in zip(si, zi):
        sp = v["specs"][s]
        sl = np.asarray(v["vol"][v["zs"][z]])
        img = make_view(sl, sp, v["p"][0], v["p"][1],
                        ctx["mean"], ctx["std"], normalize)
        tw = v["tw"][s]
        if augment:
            if rng.random() < 0.5:
                img = img[:, ::-1, :].copy()
                tw = tw[::-1, :].copy()
            if rng.random() < 0.5:
                img = img[:, :, ::-1].copy()
                tw = tw[:, ::-1].copy()
        views.append(img)
        tws.append(tw)
        fg.append(v["fg"][s])
    return (np.stack(views), np.stack(tws),
            np.asarray(fg, np.float32))


def all_views(v, ctx):
    for s, sp in enumerate(v["specs"]):
        for z in v["zs"]:
            sl = np.asarray(v["vol"][z])
            yield (make_view(sl, sp, v["p"][0], v["p"][1],
                             ctx["mean"], ctx["std"], normalize),
                   v["tw"][s], v["fg"][s])


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_trainable(args, device):
    """Fresh backbone + head, with a small trainable set."""
    import torch
    import torch.nn as nn

    ctx = build_dino(args.model, device)
    backbone = ctx["model"]
    backbone.requires_grad_(False)
    if args.grad_checkpoint and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)

    n_blocks = len(backbone.blocks)
    tune_ids = list(range(n_blocks - args.tune_blocks, n_blocks))
    mode = args.tune
    if mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
            pat = "|".join(str(i) for i in tune_ids)
            cfg = LoraConfig(
                r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.1,
                bias="none",
                target_modules=rf".*blocks\.({pat})\."
                               rf"(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$")
            backbone = get_peft_model(backbone, cfg)
        except Exception as e:
            print(f"  peft unavailable/failed ({type(e).__name__}: {e}); "
                  f"falling back to unfreezing the last {args.tune_blocks} "
                  f"block(s)")
            mode = "last"
    if mode == "last":
        for i in tune_ids:
            backbone.blocks[i].requires_grad_(True)
        backbone.norm.requires_grad_(True)

    head = nn.Linear(ctx["dim"], getattr(args, "n_out", 1)).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    adapt_params = [p for p in backbone.parameters() if p.requires_grad]
    n_adapt = sum(p.numel() for p in adapt_params)
    return ctx, backbone, head, adapt_params, mode, n_adapt


def pooled_embedding(backbone, ctx, x, tw, device, use_amp):
    """Token-fg-weighted mean per view: the e2e twin of patch_mean_fg."""
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        tok = backbone.forward_features(x)
    tok = tok.float()[:, ctx["n_prefix"]:]                # (V, g*g, D)
    w = tw.reshape(tw.shape[0], -1, 1)
    return (tok * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)


def predict_volume(backbone, head, ctx, v, device, batch_size, use_amp):
    import torch
    es, ws = [], []
    buf_x, buf_t, buf_w = [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to(device)
        t = torch.from_numpy(np.stack(buf_t)).to(device)
        with torch.no_grad():
            es.append(pooled_embedding(backbone, ctx, x, t, device, use_amp))
        ws.extend(buf_w)
        buf_x.clear(); buf_t.clear(); buf_w.clear()

    for img, tw, fg in all_views(v, ctx):
        buf_x.append(img); buf_t.append(tw); buf_w.append(fg)
        if len(buf_x) >= batch_size:
            flush()
    flush()
    e = torch.cat(es)
    w = torch.tensor(ws, dtype=e.dtype, device=e.device).unsqueeze(1)
    vol_e = (e * w).sum(dim=0) / w.sum()
    return head(vol_e).detach().cpu().numpy()


# --------------------------------------------------------------------------
# one fold = one finetuning
# --------------------------------------------------------------------------
def train_fold(vols_tr, args, device, seed):
    import torch

    ctx, backbone, head, adapt_params, mode, n_adapt = build_trainable(
        args, device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # sample weights: a 3-FOV tissue must not outvote a 1-FOV one
    cnt = {}
    for v in vols_tr:
        cnt[v["group"]] = cnt.get(v["group"], 0) + 1
    for v in vols_tr:
        v["_w"] = 1.0 / cnt[v["group"]]
    if getattr(args, "n_out", 1) > 1:
        # Class balance computed on the WEIGHTED totals, mirroring the probe:
        # balancing on raw volume counts would ignore the FOV weighting and
        # then multiply on top of it.
        tot = sum(v["_w"] for v in vols_tr)
        by_c = {}
        for v in vols_tr:
            by_c[int(v["_target"])] = by_c.get(int(v["_target"]), 0.0) + v["_w"]
        for v in vols_tr:
            v["_w"] *= tot / (len(by_c) * by_c[int(v["_target"])])

    opt = torch.optim.AdamW(
        [{"params": adapt_params, "lr": args.lr_backbone},
         {"params": head.parameters(), "lr": args.lr_head}],
        weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(vols_tr) // args.batch_vols)
    total = steps_per_epoch * args.epochs
    warm = max(1, total // 10)

    def lr_scale(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    backbone.train()
    step = 0
    for _ep in range(args.epochs):
        order = rng.permutation(len(vols_tr))
        for b0 in range(0, steps_per_epoch * args.batch_vols,
                        args.batch_vols):
            batch = [vols_tr[i] for i in order[b0:b0 + args.batch_vols]]
            if not batch:
                continue
            s = lr_scale(step)
            for gparam, base in zip(opt.param_groups,
                                    (args.lr_backbone, args.lr_head)):
                gparam["lr"] = base * s
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for v in batch:
                imgs, tws, fg = sample_views(v, args.views_per_step, rng,
                                             ctx, augment=True)
                x = torch.from_numpy(imgs).to(device)
                t = torch.from_numpy(tws).to(device)
                e = pooled_embedding(backbone, ctx, x, t, device, use_amp)
                w = torch.from_numpy(fg).to(e.device,
                                            e.dtype).unsqueeze(1)
                vol_e = (e * w).sum(dim=0) / w.sum()
                out_v = head(vol_e)
                if getattr(args, "n_out", 1) > 1:
                    tgt = torch.tensor(int(v["_target"]), device=out_v.device)
                    loss = loss + v["_w"] * torch.nn.functional.cross_entropy(
                        out_v[None], tgt[None])
                else:
                    loss = loss + v["_w"] * (out_v[0] - v["_target"]) ** 2
            loss = loss / len(batch)
            loss.backward()
            if step == 0:
                # Reentrant gradient checkpointing (old timm/torch) can
                # silently produce NO grads for params inside checkpointed
                # blocks when no input requires grad. Catch it on step 0
                # instead of training 22 folds of a frozen model.
                gsum = sum(float(p.grad.abs().sum())
                           for p in adapt_params if p.grad is not None)
                if not (np.isfinite(gsum) and gsum > 0):
                    raise SystemExit(
                        "no gradient reached the adapted backbone params "
                        "(grad sum 0). Likely reentrant gradient "
                        "checkpointing; rerun with --no_grad_checkpoint.")
            torch.nn.utils.clip_grad_norm_(
                adapt_params + list(head.parameters()), 1.0)
            opt.step()
            step += 1
    backbone.eval()
    return ctx, backbone, head, mode, n_adapt, use_amp


def run_loo(vols, vol_force, args, device, seed, quiet=False):
    """The full LOO. `vol_force` is passed in so permutations reuse this.

    The FOLD unit is args.cv_group (replicate, or plate for a
    plate-determined label); the SCORING unit is always the replicate --
    every held-out replicate contributes one prediction, averaged over its
    FOVs, exactly as in the probe.
    """
    import torch
    categorical = args.target_type == "categorical"
    cvkey = (lambda v: v["group"]) if args.cv_group == "replicate" \
        else (lambda v: v["plate"])
    by_cv = {}
    for v, f in zip(vols, vol_force):
        by_cv.setdefault(cvkey(v), []).append((v, float(f)))
    folds = sorted(by_cv)

    pred_force, true_force, pred_bin, true_bin, held = [], [], [], [], []
    t0 = time.time()
    for fi, k in enumerate(folds):
        te = by_cv[k]
        tr = [(v, f) for kk in folds if kk != k for v, f in by_cv[kk]]
        tr_vols = [v for v, _f in tr]
        tr_f = np.asarray([f for _v, f in tr], np.float64)

        # target construction, train stats only (mirrors the probe)
        if categorical:
            if len({int(f) for _v, f in tr}) < 2:
                print(f"  fold [{k}]: training set has ONE class -- the "
                      f"held-out {args.cv_group} carries the only other "
                      f"class. Skipped; this design cannot test it.")
                continue
            for v, f in tr:
                v["_target"] = int(f)
        elif args.deconfound == "plate":
            mu_all = float(tr_f.mean())
            mus = {}
            for (v, f) in tr:
                mus.setdefault(v["plate"], []).append(f)
            mus = {kk: float(np.mean(x)) for kk, x in mus.items()}
            for v, f in tr:
                v["_target"] = f - mus.get(v["plate"], mu_all)
        else:
            if np.any(tr_f <= 0):
                raise SystemExit("non-positive force cannot be log-scaled; "
                                 "use --deconfound plate")
            for v, f in tr:
                v["_target"] = math.log10(f)

        ctx, backbone, head, mode, n_adapt, use_amp = train_fold(
            tr_vols, args, device, seed + 7919 * fi)

        # bins from TRAIN replicate values, same scale as the binned quantity
        if not categorical:
            rep_f = {}
            for v, f in tr:
                rep_f.setdefault(v["group"],
                                 v["_target"] if args.deconfound == "plate"
                                 else f)
            edges = compute_bin_edges(sorted(rep_f.values()), args.n_bins)
            if args.deconfound == "plate":
                mus_te = mus  # defined above in the dc-plate branch
                mu_all_te = mu_all

        # score per held-out REPLICATE
        by_rep = {}
        for v, f in te:
            by_rep.setdefault(v["group"], []).append((v, f))
        for g in sorted(by_rep):
            outs = np.stack([predict_volume(backbone, head, ctx, v, device,
                                            args.batch_size, use_amp)
                             for v, _f in by_rep[g]])
            f_raw = float(by_rep[g][0][1])
            if categorical:
                z = outs - outs.max(axis=1, keepdims=True)
                pr = (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
                      ).mean(axis=0)
                pb, tb = int(np.argmax(pr)), int(f_raw)
                # binary: P(class 1) is a usable continuous score; else code
                sc = float(pr[1]) if len(pr) == 2 else float(pb)
                pred_force.append(sc); true_force.append(float(tb))
            else:
                mp = float(outs[:, 0].mean())
                if args.deconfound == "plate":
                    y_te = f_raw - mus_te.get(by_rep[g][0][0]["plate"],
                                              mu_all_te)
                    pb, tb = assign_bin(mp, edges), assign_bin(y_te, edges)
                else:
                    y_te = f_raw
                    pb = assign_bin(10 ** mp, edges)
                    tb = assign_bin(y_te, edges)
                pred_force.append(mp); true_force.append(y_te)
            pred_bin.append(pb); true_bin.append(tb); held.append(g)

        del backbone, head
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not quiet:
            el = time.time() - t0
            print(f"  fold {fi+1}/{len(folds)} [{k}] "
                  f"{len(by_rep)} replicate(s)  last pred={pred_force[-1]:+.3f} "
                  f"true={true_force[-1]:+.3f}  ({mode}, {n_adapt/1e6:.2f}M "
                  f"trainable; {el/(fi+1):.0f}s/fold, "
                  f"~{el/(fi+1)*(len(folds)-fi-1)/60:.0f} min left)")

    return {"replicates": held,
            "true_force": np.asarray(true_force),
            "pred_score": np.asarray(pred_force),
            "true_bin": np.asarray(true_bin),
            "pred_bin": np.asarray(pred_bin)}


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--target_col", default="peak_amplitude_week1")
    p.add_argument("--file_col", default="file")
    p.add_argument("--group_cols", default="plate,Tissue")
    p.add_argument("--modality", default="gfp")
    p.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--deconfound", choices=["none", "plate"], default="plate")
    p.add_argument("--target_type", choices=["numeric", "categorical"],
                   default="numeric")
    p.add_argument("--cv_group", choices=["replicate", "plate"],
                   default="replicate",
                   help="the held-out unit. For a PLATE-DETERMINED label "
                        "(Exercise) use 'plate': batch artifacts cannot "
                        "transfer to an unseen plate, which is the only "
                        "defensible test for such a label -- but 4 plates "
                        "caps the certainty regardless.")
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--norm_scope", choices=["volume", "global"],
                   default="volume")
    p.add_argument("--z_stride", type=int, default=1)
    p.add_argument("--tile_grid", default="4,3")
    p.add_argument("--mask_source", default="gfp")
    p.add_argument("--mask_method", default="li")
    p.add_argument("--mask_projection", choices=["none", "max"],
                   default="max")
    p.add_argument("--mask_polarity", choices=["bright", "dark"],
                   default="bright")
    p.add_argument("--fg_min", type=float, default=0.2)
    p.add_argument("--tune", choices=["lora", "last"], default="lora")
    p.add_argument("--tune_blocks", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_vols", type=int, default=2)
    p.add_argument("--views_per_step", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16,
                   help="eval view batch")
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_checkpoint", action="store_true", default=True)
    p.add_argument("--no_grad_checkpoint", dest="grad_checkpoint",
                   action="store_false")
    p.add_argument("--shuffle", action="store_true",
                   help="permute labels once and run the FULL LOO: the leak "
                        "canary. Must land at chance.")
    p.add_argument("--n_perm", type=int, default=0,
                   help="permutation null = this many additional FULL LOO "
                        "runs on within-plate-permuted labels. 19 gives "
                        "p-resolution 0.05 and is an overnight job.")
    p.add_argument("--final_fit", default=None,
                   help="after the LOO, train on ALL volumes and save this "
                        "checkpoint (+ .readout.npz sidecar) for XAI. LEAKY "
                        "for statistics -- it saw every label.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA. One fold on CPU is hours; this is not "
              "realistically runnable without a GPU.")

    vols, classes = prepare_volumes(args)
    categorical = args.target_type == "categorical"
    if categorical:
        if args.deconfound != "none":
            raise SystemExit(
                "--deconfound plate with a categorical target: centering a "
                "class code destroys the classes, and for a plate-determined "
                "label it would subtract the very variation the label lives "
                "in. Use --deconfound none --cv_group plate.")
        args.n_bins = len(classes)
        args.n_out = len(classes)
        print(f"classes: {', '.join(classes)}  (chance "
              f"{1.0/len(classes):.2f})")
        if args.cv_group == "replicate":
            print("WARNING: cv_group=replicate with a plate-determined label "
                  "lets plate batch artifacts predict the class perfectly. "
                  "Use --cv_group plate for the defensible test.")
    else:
        args.n_out = 1
    vol_group = [v["group"] for v in vols]
    vol_force = [v["force"] for v in vols]
    rng = np.random.default_rng(args.seed)

    # Permutation strata: within-plate for a numeric target (the nested
    # design's null), but FREE for a categorical one -- a label constant
    # within every plate makes within-plate permutation the identity.
    strata = (None if categorical else [v["plate"] for v in vols])

    if args.shuffle:
        vol_force = _permute_replicate_force(vol_group, vol_force, rng,
                                             strata=strata)
        print("labels SHUFFLED: this run must land at chance")

    print(f"\nLOO over {len(set(cvu for cvu in (v['group'] if args.cv_group == 'replicate' else v['plate'] for v in vols)))} "
          f"{args.cv_group} fold(s), {len(set(vol_group))} replicates "
          f"({len(vols)} volumes)  tune={args.tune} "
          f"blocks={args.tune_blocks} epochs={args.epochs}")
    task = "classification" if categorical else "regression"
    res = run_loo(vols, vol_force, args, device, args.seed)
    out = score(res, args.n_bins, task=task)

    null_rho, null_acc = [], []
    for pi in range(args.n_perm):
        pf = _permute_replicate_force(vol_group, [v["force"] for v in vols],
                                      rng, strata=strata)
        print(f"\npermutation {pi+1}/{args.n_perm}")
        r = run_loo(vols, pf, args, device, args.seed + 1000 * (pi + 1),
                    quiet=True)
        null_rho.append(spearman(r["pred_score"], r["true_force"]))
        null_acc.append(float(np.mean(r["true_bin"] == r["pred_bin"]))
                        if len(r["true_bin"]) else float("nan"))
        print(f"  null spearman {null_rho[-1]:+.3f}  "
              f"null acc {null_acc[-1]:.3f}")

    obs = out["spearman_pred_vs_force"]
    if null_rho:
        nr = np.asarray(null_rho)
        out["permutation_p_spearman"] = float(
            (1 + (nr >= obs).sum()) / (1 + len(nr)))
        out["null_spearman"] = [float(x) for x in nr]
        out["null_spearman_mean"] = float(nr.mean())
        out["null_spearman_ci"] = [float(np.percentile(nr, 2.5)),
                                   float(np.percentile(nr, 97.5))]
        out["mde_spearman_95"] = float(np.percentile(nr, 95))
        na = np.asarray([a for a in null_acc if np.isfinite(a)])
        if len(na):
            out["permutation_p_accuracy"] = float(
                (1 + (na >= out["replicate_accuracy"]).sum()) / (1 + len(na)))
            out["null_accuracy"] = [float(x) for x in na]
            out["null_accuracy_mean"] = float(na.mean())
            out["mde_accuracy_95"] = float(np.percentile(na, 95))
        out["n_permutations"] = len(nr)

    _fov = {}
    for g in vol_group:
        _fov[g] = _fov.get(g, 0) + 1
    _bp = {}
    for v in vols:
        _bp.setdefault(v["plate"], {})[v["group"]] = v["force"]
    _allf = np.array([f for d in _bp.values() for f in d.values()])
    _sst = ((_allf - _allf.mean()) ** 2).sum()
    _ssb = sum(len(d) * (np.mean(list(d.values())) - _allf.mean()) ** 2
               for d in _bp.values())
    out.update({
        "features": "dino_e2e", "modality": args.modality,
        "model_class": f"e2e_{args.tune}{args.tune_blocks}",
        "target_col": args.target_col, "target_type": args.target_type,
        "classes": classes, "cv_group": args.cv_group,
        "deconfound": args.deconfound,
        "aggregate": "volume", "fg_min": args.fg_min, "struct": "none",
        "canary": "none", "shuffled": bool(args.shuffle),
        "n_volumes": len(vols), "feature_dim": 768,
        "eta2_plate": float(_ssb / _sst) if _sst > 0 else None,
        "n_plates": len(_bp),
        "epochs": args.epochs, "seed": args.seed,
        "per_replicate": [
            {"group": g, "plate": plate_of(g), "n_fov": _fov.get(g, 0),
             "true_force": float(t), "pred_score": float(s),
             "true_bin": int(a), "pred_bin": int(b)}
            for g, t, s, a, b in zip(res["replicates"], res["true_force"],
                                     res["pred_score"], res["true_bin"],
                                     res["pred_bin"])]})

    if categorical:
        print(f"\nLOO n={out['n_replicates']}  "
              f"acc={out['replicate_accuracy']:.3f} "
              f"(chance {out['chance']:.3f}, "
              f"{out['n_correct']}/{out['n_replicates']})"
              + (f"  perm_p={out.get('permutation_p_accuracy', float('nan')):.3f} "
                 f"({len(null_acc)} permutations -- resolution "
                 f"{1.0/(1+len(null_acc)):.2f})" if null_acc else
                 "  (no permutation null: run with --n_perm for a p-value)"))
    else:
        print(f"\nLOO n={out['n_replicates']}  "
              f"spearman(pred, true) = {obs:+.3f}"
              + (f"  perm_p={out['permutation_p_spearman']:.3f} "
                 f"({len(null_rho)} permutations -- resolution "
                 f"{1.0/(1+len(null_rho)):.2f})" if null_rho else
                 "  (no permutation null: run with --n_perm for a p-value; "
                 "the frozen probe's null is NOT transferable)"))
    if args.shuffle:
        print("^ labels were SHUFFLED: this must be at chance. If not, the "
              "fold logic leaks and every other number is void.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"saved {args.output}")

    # ---- final fit for XAI (LEAKY for statistics, by construction) ----
    if args.final_fit and not args.shuffle:
        print("\nfinal fit on ALL volumes (XAI ONLY -- this model saw every "
              "label; do not report statistics computed from it)")
        tr = list(zip(vols, [v["force"] for v in vols]))
        tr_f = np.asarray([f for _v, f in tr])
        if categorical:
            for v, f in tr:
                v["_target"] = int(f)
        elif args.deconfound == "plate":
            mus = {}
            for v, f in tr:
                mus.setdefault(v["plate"], []).append(f)
            mus = {k: float(np.mean(x)) for k, x in mus.items()}
            for v, f in tr:
                v["_target"] = f - mus[v["plate"]]
        else:
            for v, f in tr:
                v["_target"] = math.log10(f)
        ctx, backbone, head, mode, n_adapt, use_amp = train_fold(
            vols, args, device, args.seed + 424242)
        bb = backbone
        if hasattr(bb, "merge_and_unload"):
            try:
                bb = bb.merge_and_unload()
            except Exception as e:
                print(f"  merge_and_unload failed ({e}); saving wrapped "
                      f"state dict")
        os.makedirs(os.path.dirname(args.final_fit) or ".", exist_ok=True)
        torch.save({"backbone": bb.state_dict(),
                    "head_weight": head.weight.detach().cpu(),
                    "head_bias": head.bias.detach().cpu(),
                    "meta": {"model": args.model, "tune": mode,
                             "tune_blocks": args.tune_blocks,
                             "target_col": args.target_col,
                             "deconfound": args.deconfound,
                             "leaky_for_statistics": True}},
                   args.final_fit)
        hw = head.weight.detach().cpu().numpy().astype(np.float64)
        if categorical and len(hw) == 2:
            # Binary classifier: the decision is the LOGIT MARGIN, a single
            # direction w1 - w0 -- exactly the linear readout the attribution
            # machinery decomposes. Red = toward class 1.
            w = (hw[1] - hw[0]).ravel()
        elif categorical:
            print(f"  {len(hw)}-class head has no single readout direction; "
                  f"skipping the sidecar (dense XAI unavailable)")
            print(f"saved {args.final_fit}")
            return
        else:
            w = hw.ravel()
        sidecar = os.path.splitext(args.final_fit)[0] + ".readout.npz"
        np.savez_compressed(
            sidecar, mean_direction=w, fold_directions=w[None],
            feature_names=np.asarray(
                [f"e2e[{i}]" for i in range(len(w))], dtype=object),
            token="patch_mean_fg", agg="fgmean", fg_min=float(args.fg_min),
            struct="none", deconfound=args.deconfound,
            target_col=args.target_col,
            task=("classification" if categorical else "regression"),
            classes=("|".join(classes) if categorical else ""),
            model_class=f"e2e_{mode}", fold_cosine=np.nan)
        print(f"saved {args.final_fit}\nsaved {sidecar}")


if __name__ == "__main__":
    main()
