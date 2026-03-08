"""Training script for pix2pix-turbo (Experiment 5).

GAN-based training loop with L2 + LPIPS + vision-aided discriminator.
Separate from train.py because the generator/discriminator alternation
and the diffusion-based architecture are fundamentally different.

Usage:
    python train_pix2pix.py -c configs/pix2pix_turbo.yaml
    python train_pix2pix.py -c configs/pix2pix_turbo.yaml --resume ckpts/pix2pix_turbo/latest.pkl
"""

import os
import csv
import json
import shutil
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from src.config import load_config
from src.utils import set_seed, prepare_env, make_train_val_split, get_git_hash
from src.models.pix2pix_turbo import Pix2Pix_Turbo
from src.data.datasets_pix2pix import SliceDatasetPix2Pix


def build_datasets(cfg, tokenizer):
    """Build train and val datasets for pix2pix-turbo."""
    dcfg = cfg["data"]
    data_dir = dcfg["data_dir"]
    stats_dir = os.path.join(data_dir, "stats")

    bf_dir = os.path.join(data_dir, "bf")
    gfp_dir = os.path.join(data_dir, "gfp")
    bf_files = sorted(glob(os.path.join(bf_dir, "*.npy")))
    assert len(bf_files) > 0, f"No .npy files found in {bf_dir}"

    stems = [os.path.splitext(os.path.basename(f))[0] for f in bf_files]
    gfp_files = [os.path.join(gfp_dir, f"{s}.npy") for s in stems]
    for gf in gfp_files:
        assert os.path.exists(gf), f"Missing GFP file: {gf}"

    train_stems, val_stems = make_train_val_split(
        stems, val_fraction=dcfg.get("val_fraction", 0.15), seed=cfg.get("seed", 42)
    )

    def stems_to_paths(stem_list):
        bf = [os.path.join(bf_dir, f"{s}.npy") for s in stem_list]
        gfp = [os.path.join(gfp_dir, f"{s}.npy") for s in stem_list]
        return bf, gfp

    train_bf, train_gfp = stems_to_paths(train_stems)
    val_bf, val_gfp = stems_to_paths(val_stems)

    crop_size = dcfg.get("crop_size", 512)
    cache = dcfg.get("cache_volumes", False)

    train_ds = SliceDatasetPix2Pix(
        train_bf, train_gfp, stats_dir, tokenizer,
        crop_size=crop_size, train=True, cache_volumes=cache,
    )
    val_ds = SliceDatasetPix2Pix(
        val_bf, val_gfp, stats_dir, tokenizer,
        crop_size=crop_size, train=False, cache_volumes=cache,
    )
    return train_ds, val_ds, train_stems, val_stems


def main(config_path, resume_from=None):
    cfg = load_config(config_path)
    # Skip standard validation (pix2pix_turbo doesn't use encoder/in_channels etc.)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    experiment_name = cfg.get("experiment_name", "pix2pix_turbo")

    set_seed(cfg.get("seed", 42))
    accelerator, device, tqdm = prepare_env(
        mixed_precision=tcfg.get("mixed_precision", False)
    )
    accelerator.print(f"Experiment: {experiment_name}")

    ckpt_dir = tcfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(ckpt_dir, "config.yaml"))

    # Build model
    pretrained_model = mcfg.get("pretrained_model", "stabilityai/sd-turbo")
    lora_rank_unet = mcfg.get("lora_rank_unet", 8)
    lora_rank_vae = mcfg.get("lora_rank_vae", 4)

    pretrained_path = resume_from if resume_from and resume_from.endswith(".pkl") else None
    accelerator.print(f"Loading SD-Turbo backbone from {pretrained_model}...")
    net = Pix2Pix_Turbo(
        pretrained_path=pretrained_path,
        pretrained_model=pretrained_model,
        lora_rank_unet=lora_rank_unet,
        lora_rank_vae=lora_rank_vae,
    )
    net.to(device)
    net.set_train()

    # Enable gradient checkpointing if requested
    if tcfg.get("gradient_checkpointing", False):
        net.unet.enable_gradient_checkpointing()

    # Enable xformers if requested
    if tcfg.get("enable_xformers", False):
        try:
            net.unet.enable_xformers_memory_efficient_attention()
            net.vae.enable_xformers_memory_efficient_attention()
            accelerator.print("xformers memory-efficient attention enabled")
        except Exception as e:
            accelerator.print(f"xformers not available: {e}")

    # Count trainable params
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in net.parameters())
    accelerator.print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Datasets
    train_ds, val_ds, train_stems, val_stems = build_datasets(cfg, net.tokenizer)
    accelerator.print(
        f"Train: {len(train_ds)} slices ({len(train_stems)} volumes), "
        f"Val: {len(val_ds)} slices ({len(val_stems)} volumes)"
    )

    with open(os.path.join(ckpt_dir, "split.json"), "w") as f:
        json.dump({"train": train_stems, "val": val_stems}, f, indent=2)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=tcfg.get("num_workers", 4),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=tcfg.get("num_workers", 4),
    )

    # Losses
    loss_weights = tcfg.get("losses", {"l2": 1.0, "lpips": 5.0, "gan": 0.5})
    w_l2 = loss_weights.get("l2", 1.0)
    w_lpips = loss_weights.get("lpips", 5.0)
    w_gan = loss_weights.get("gan", 0.5)

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.requires_grad_(False)

    # Vision-aided discriminator
    import vision_aided_loss
    disc = vision_aided_loss.Discriminator(
        cv_type="clip", loss_type="multilevel_sigmoid_s", device=device
    )
    disc.cv_ensemble.requires_grad_(False)  # freeze CLIP backbone
    disc.train()

    # Optimizers
    gen_params = [p for p in net.parameters() if p.requires_grad]
    optimizer_gen = torch.optim.AdamW(gen_params, lr=tcfg["lr"], weight_decay=0.01)
    optimizer_disc = torch.optim.AdamW(
        disc.parameters(), lr=tcfg["lr"], weight_decay=0.01
    )

    # Scheduler
    total_epochs = tcfg["epochs"]
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen, T_max=total_epochs, eta_min=tcfg["lr"] * 0.01
    )
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc, T_max=total_epochs, eta_min=tcfg["lr"] * 0.01
    )

    # Prepare with accelerator
    net, disc, optimizer_gen, optimizer_disc, train_loader, val_loader = (
        accelerator.prepare(
            net, disc, optimizer_gen, optimizer_disc, train_loader, val_loader
        )
    )
    scheduler_gen = accelerator.prepare(scheduler_gen)
    scheduler_disc = accelerator.prepare(scheduler_disc)

    grad_accum = tcfg.get("grad_accumulation_steps", 1)

    # Resume epoch counter
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from and resume_from.endswith(".json"):
        with open(resume_from) as f:
            meta = json.load(f)
        start_epoch = meta.get("epoch", 0) + 1
        best_val_loss = meta.get("val_loss", float("inf"))
        accelerator.print(f"Resuming from epoch {start_epoch}")

    # CSV logger
    csv_path = os.path.join(ckpt_dir, "log.csv")
    csv_exists = os.path.exists(csv_path) and resume_from
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "epoch", "train_loss_gen", "train_loss_disc",
            "val_loss_l2", "val_loss_lpips", "lr",
        ])

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        # === Train ===
        net.set_train()
        disc.train()
        gen_losses, disc_losses = [], []
        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        progress = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{total_epochs} [train]",
        )

        for step, batch in enumerate(train_loader):
            cond = batch["conditioning_pixel_values"]  # (B, 3, H, W) [-1, 1]
            target = batch["output_pixel_values"]       # (B, 3, H, W) [-1, 1]
            tokens = batch["input_ids"]                 # (B, seq_len)

            # --- Generator step ---
            with accelerator.accumulate(net):
                pred = net(cond, prompt_tokens=tokens, deterministic=True)

                loss_l2 = F.mse_loss(pred, target)
                loss_lp = lpips_fn(pred, target).mean()
                loss_gan_g = disc(pred, for_real=True).mean() * (-1)  # fool disc
                loss_gen = w_l2 * loss_l2 + w_lpips * loss_lp + w_gan * loss_gan_g

                accelerator.backward(loss_gen / grad_accum)
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer_gen.step()
                    optimizer_gen.zero_grad()

            gen_losses.append(loss_gen.item())

            # --- Discriminator step ---
            with accelerator.accumulate(disc):
                loss_real = disc(target, for_real=True)
                loss_fake = disc(pred.detach(), for_real=False)
                loss_disc = (loss_real + loss_fake).mean()

                accelerator.backward(loss_disc / grad_accum)
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer_disc.step()
                    optimizer_disc.zero_grad()

            disc_losses.append(loss_disc.item())

            progress.set_postfix(
                g=f"{gen_losses[-1]:.4f}", d=f"{disc_losses[-1]:.4f}"
            )
            progress.update()
        progress.close()

        scheduler_gen.step()
        scheduler_disc.step()

        # === Validate ===
        net.set_eval()
        val_l2s, val_lpipss = [], []
        with torch.no_grad():
            progress = tqdm(
                total=len(val_loader),
                desc=f"Epoch {epoch + 1}/{total_epochs} [val]",
            )
            for batch in val_loader:
                cond = batch["conditioning_pixel_values"]
                target = batch["output_pixel_values"]
                tokens = batch["input_ids"]

                pred = net(cond, prompt_tokens=tokens, deterministic=True)
                l2 = F.mse_loss(pred, target).item()
                lp = lpips_fn(pred, target).mean().item()
                val_l2s.append(l2)
                val_lpipss.append(lp)
                progress.set_postfix(l2=f"{l2:.4f}", lpips=f"{lp:.4f}")
                progress.update()
            progress.close()

        mean_gen = np.mean(gen_losses)
        mean_disc = np.mean(disc_losses)
        mean_val_l2 = np.mean(val_l2s)
        mean_val_lpips = np.mean(val_lpipss)
        # Combined val metric (same weighting as training)
        mean_val = mean_val_l2 * w_l2 + mean_val_lpips * w_lpips
        current_lr = optimizer_gen.param_groups[0]["lr"]

        accelerator.print(
            f"Epoch {epoch + 1}: gen_loss={mean_gen:.4f}  disc_loss={mean_disc:.4f}  "
            f"val_l2={mean_val_l2:.4f}  val_lpips={mean_val_lpips:.4f}  lr={current_lr:.2e}"
        )

        csv_writer.writerow([
            epoch + 1, f"{mean_gen:.6f}", f"{mean_disc:.6f}",
            f"{mean_val_l2:.6f}", f"{mean_val_lpips:.6f}", f"{current_lr:.2e}",
        ])
        csv_file.flush()

        # Save best
        unwrapped_net = accelerator.unwrap_model(net)
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            accelerator.print(f"  -> New best val loss: {best_val_loss:.4f}")
            unwrapped_net.save_model(os.path.join(ckpt_dir, "best.pkl"))
            with open(os.path.join(ckpt_dir, "best_meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "val_loss": best_val_loss,
                    "val_l2": mean_val_l2, "val_lpips": mean_val_lpips,
                }, f, indent=2)

        # Periodic save
        save_every = tcfg.get("save_every", 10)
        if (epoch + 1) % save_every == 0:
            unwrapped_net.save_model(
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pkl")
            )

        # Always save latest
        unwrapped_net.save_model(os.path.join(ckpt_dir, "latest.pkl"))
        with open(os.path.join(ckpt_dir, "latest_meta.json"), "w") as f:
            json.dump({
                "epoch": epoch, "val_loss": mean_val,
                "val_l2": mean_val_l2, "val_lpips": mean_val_lpips,
            }, f, indent=2)

        # Save discriminator state
        torch.save(
            accelerator.unwrap_model(disc).state_dict(),
            os.path.join(ckpt_dir, "disc_latest.pth"),
        )

    csv_file.close()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pix2pix-turbo BF -> GFP")
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to .pkl checkpoint or latest_meta.json to resume from",
    )
    args = parser.parse_args()
    main(args.config, args.resume)
