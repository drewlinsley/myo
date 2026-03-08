"""Training script for brightfield -> fluorescence regression."""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from torchvision.transforms import v2 as transforms

import segmentation_models_pytorch_3d as smp
from src import transforms as video_transforms
from src.dataloaders import BrightfieldFluorescence
from src import utils


def compute_psnr(pred, target, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)


def main(config):
    cfg = utils.read_config(config)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)

    # Prepare environment
    accelerator, device, tqdm, TIMM = utils.prepare_env(mcfg["encoder"])
    mu = np.float32(TIMM["mean"][0])
    std = np.float32(TIMM["std"][0])

    # Discover paired files
    bf_files = sorted(glob(os.path.join(dcfg["brightfield_dir"], "**", dcfg["file_pattern"]), recursive=True))
    fl_files = sorted(glob(os.path.join(dcfg["fluorescence_dir"], "**", dcfg["file_pattern"]), recursive=True))
    assert len(bf_files) == len(fl_files), \
        f"Found {len(bf_files)} brightfield but {len(fl_files)} fluorescence files"
    assert len(bf_files) > 0, f"No files found in {dcfg['brightfield_dir']}"

    # Train/val split
    n_val = max(1, int(len(bf_files) * dcfg["val_fraction"]))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(bf_files))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_bf = [bf_files[i] for i in train_idx]
    train_fl = [fl_files[i] for i in train_idx]
    val_bf = [bf_files[i] for i in val_idx]
    val_fl = [fl_files[i] for i in val_idx]

    accelerator.print(f"Train: {len(train_bf)} pairs, Val: {len(val_bf)} pairs")

    # Transforms
    pre_hw = dcfg.get("pre_hw", dcfg["hw"] + 32)
    train_trans = transforms.Compose([
        video_transforms.RandomCrop(dcfg["hw"]),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomVerticalFlip(),
        video_transforms.ToTensor(),
    ])
    val_trans = transforms.Compose([
        video_transforms.CenterCrop(dcfg["hw"]),
        video_transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = BrightfieldFluorescence(
        train_bf, train_fl, mu, std,
        transform=train_trans, time=dcfg["timesteps"], pre_hw=pre_hw,
    )
    val_dataset = BrightfieldFluorescence(
        val_bf, val_fl, mu, std,
        transform=val_trans, time=dcfg["timesteps"], pre_hw=pre_hw,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=tcfg["batch_size"],
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=tcfg.get("num_workers", 4),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=tcfg["batch_size"],
        shuffle=False, drop_last=False,
        num_workers=tcfg.get("num_workers", 4),
    )

    # Model
    model = smp.create_model(
        mcfg["arch"],
        encoder_name=mcfg["encoder"],
        encoder_weights=mcfg.get("encoder_weights", "imagenet"),
        in_channels=mcfg["in_channels"],
        classes=mcfg["out_channels"],
        activation=None,  # raw regression output
        time_kernel=mcfg.get("time_kernel", [32, 1, 1]),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.01),
    )

    # Scheduler
    scheduler = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"], eta_min=tcfg["lr"] * 0.01,
        )

    # Prepare for distributed
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    if scheduler:
        scheduler = accelerator.prepare(scheduler)

    # Loss weights
    mse_w = tcfg.get("loss_mse_weight", 1.0)
    l1_w = tcfg.get("loss_l1_weight", 0.1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(tcfg["epochs"]):
        # === Train ===
        model.train()
        train_losses = []
        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{tcfg['epochs']} [train]")
        for bf, fl in train_loader:
            optimizer.zero_grad()
            pred = model(bf)
            loss = mse_w * F.mse_loss(pred, fl) + l1_w * F.l1_loss(pred, fl)
            accelerator.backward(loss)
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
            progress.update()
        progress.close()

        if scheduler:
            scheduler.step()

        # === Validate ===
        model.eval()
        val_losses, val_psnrs = [], []
        with torch.no_grad():
            progress = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{tcfg['epochs']} [val]")
            for bf, fl in val_loader:
                pred = model(bf)
                loss = mse_w * F.mse_loss(pred, fl) + l1_w * F.l1_loss(pred, fl)
                psnr = compute_psnr(pred, fl)
                val_losses.append(loss.item())
                val_psnrs.append(psnr.item() if psnr != float('inf') else 50.0)
                progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{val_psnrs[-1]:.1f}")
                progress.update()
            progress.close()

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        mean_psnr = np.mean(val_psnrs)
        accelerator.print(
            f"Epoch {epoch+1}: train_loss={mean_train:.4f}  val_loss={mean_val:.4f}  val_psnr={mean_psnr:.1f}dB"
        )

        # Save best
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            accelerator.print(f"  -> New best val loss: {best_val_loss:.4f}")
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(tcfg["checkpoint_dir"], "best.pth"))

        # Periodic save
        save_every = tcfg.get("save_every", 50)
        if (epoch + 1) % save_every == 0:
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': mean_val,
            }, os.path.join(tcfg["checkpoint_dir"], f"epoch_{epoch+1}.pth"))

    accelerator.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brightfield -> fluorescence model")
    parser.add_argument('-c', '--config', required=True, type=str, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
