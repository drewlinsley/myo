"""Profile the training pipeline: is it data-bound or GPU-bound?

Times three things independently and prints a verdict:
  1. dataset[i] latency (single process — the raw cost of one sample)
  2. DataLoader throughput with the configured workers (what training sees)
  3. model forward+backward step time on GPU (the compute floor)

If (2) delivers batches slower than (3) consumes them, training is data-bound:
fix storage (scripts/localize_data.sh), workers, or the dataset. Otherwise the
GPU is the bottleneck and only model/batch changes help.

Usage (on the VM)
-----------------
    python scripts/profile_pipeline.py -c configs/unet_3d_imagenet_pearson.yaml \
        --data_dir data_phalloidin_mhc_051826_staged
    python scripts/profile_pipeline.py -c configs/gfp_classifier_3d.yaml \
        --data_dir data_phalloidin_mhc_051826_staged --task classifier
"""

import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, validate_config  # noqa: E402
from src.utils import set_seed, tune_cudnn           # noqa: E402


def time_getitem(ds, n):
    n = min(n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n, dtype=int)
    t0 = time.perf_counter()
    for i in idxs:
        ds[int(i)]
    dt = (time.perf_counter() - t0) / n
    return dt


def time_loader(ds, batch_size, num_workers, n_batches):
    kw = dict(batch_size=batch_size, shuffle=True, drop_last=True,
              pin_memory=True, num_workers=num_workers)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    loader = torch.utils.data.DataLoader(ds, **kw)
    it = iter(loader)
    next(it)                       # absorb worker startup
    t0 = time.perf_counter()
    got = 0
    for _ in range(n_batches):
        try:
            next(it)
            got += 1
        except StopIteration:
            break
    dt = (time.perf_counter() - t0) / max(got, 1)
    del it, loader
    return dt


def time_step(model, batch, criterion_fn, device, n_steps, amp):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion_fn(batch)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):             # warmup + cudnn.benchmark autotune
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--task", choices=["bfgfp", "classifier"], default="bfgfp",
                   help="bfgfp = train.py datasets/model; classifier = "
                        "GFP->force encoder+head on VolumeRegressionDataset")
    p.add_argument("--n_items", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=None,
                   help="override cfg training.num_workers")
    args = p.parse_args()

    cfg = validate_config(load_config(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    tcfg = cfg["training"]
    set_seed(cfg.get("seed", 42))
    tune_cudnn(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = cfg["model"].get("dims", "2d")
    bs = tcfg["batch_size"]
    nw = args.workers if args.workers is not None else tcfg.get("num_workers", 4)

    print(f"config={args.config} dims={dims} batch_size={bs} workers={nw} "
          f"device={device} data_dir={cfg['data']['data_dir']}")

    # ---- build dataset + model + a loss closure for the chosen task ----
    if args.task == "bfgfp":
        from train import build_datasets
        from src.models import build_model
        from src.losses import build_loss
        ds, _, _, _ = build_datasets(cfg)
        model = build_model(cfg)
        criterion = build_loss(cfg)

        def make_batch(items):
            bf = torch.stack([it[0] for it in items]).to(device)
            fl = torch.stack([it[1] for it in items]).to(device)
            mk = torch.stack([it[2] for it in items]).to(device)
            return bf, fl, mk

        def crit(batch):
            bf, fl, mk = batch
            return criterion(model(bf), fl, mask=mk)
    else:
        from glob import glob
        from src.data.regression_dataset import VolumeRegressionDataset
        from src.models.gfp_classifier import build_gfp_classifier
        from train_loo_force_classifier import build_transforms
        dcfg = cfg["data"]
        data_dir = dcfg["data_dir"]
        files = sorted(glob(os.path.join(data_dir, "gfp", "*.npy")))
        if not files:
            raise SystemExit(f"no volumes under {data_dir}/gfp/")
        targets = {os.path.splitext(os.path.basename(f))[0]: 0.0 for f in files}
        ds = VolumeRegressionDataset(
            files, stats_dir=os.path.join(data_dir, "stats"), targets=targets,
            transform=build_transforms(cfg, True),
            z_range=dcfg.get("z_range", None),
            apply_timm=cfg["model"].get("encoder_weights") is not None,
            percentile_clip=tuple(dcfg.get("percentile_clip", [0.5, 99.5])),
            mode=dims, patch_depth=dcfg.get("patch_depth", 32),
            patches_per_volume=dcfg.get("patches_per_volume", 32),
            crop_size=dcfg.get("crop_size", 256), modality="gfp")
        model = build_gfp_classifier(cfg, 4, 2)
        ce = torch.nn.CrossEntropyLoss()

        def make_batch(items):
            img = torch.stack([it[0] for it in items]).to(device)
            tgt = torch.tensor([int(it[1]) for it in items]).to(device)
            return img, tgt

        def crit(batch):
            img, tgt = batch
            lex, _ = model(img)
            return ce(lex, tgt)

    print(f"dataset: {len(ds)} samples")

    # 1. raw per-sample cost
    dt_item = time_getitem(ds, args.n_items)
    print(f"[1] dataset[i]      : {dt_item*1e3:8.1f} ms/sample "
          f"({1.0/dt_item:6.1f} samples/s single-process)")

    # 2. loader throughput
    dt_batch = time_loader(ds, bs, nw, args.n_batches)
    loader_sps = bs / dt_batch
    print(f"[2] DataLoader      : {dt_batch*1e3:8.1f} ms/batch  "
          f"({loader_sps:6.1f} samples/s with {nw} workers)")

    # 3. GPU step time
    batch = make_batch([ds[i] for i in range(bs)])
    amp = bool(tcfg.get("mixed_precision", False)) and device.type == "cuda"
    dt_step = time_step(model, batch, crit, device, args.n_steps, amp)
    gpu_sps = bs / dt_step
    print(f"[3] fwd+bwd step    : {dt_step*1e3:8.1f} ms/batch  "
          f"({gpu_sps:6.1f} samples/s, amp={amp})")

    print()
    if loader_sps < gpu_sps * 0.9:
        print(f"VERDICT: DATA-BOUND — the loader ({loader_sps:.1f} samples/s) "
              f"can't keep up with the GPU ({gpu_sps:.1f} samples/s).")
        print("  -> localize data to fast disk (scripts/localize_data.sh), "
              "raise num_workers, or check [1] for a slow __getitem__.")
    else:
        print(f"VERDICT: GPU-BOUND — compute ({gpu_sps:.1f} samples/s) is the "
              f"limit; the loader ({loader_sps:.1f} samples/s) keeps up.")
        print("  -> only model/batch-size/precision changes will speed this up.")


if __name__ == "__main__":
    main()
