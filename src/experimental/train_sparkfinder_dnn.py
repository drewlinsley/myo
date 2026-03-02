import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transforms as video_transforms
from torchvision.transforms import v2 as transforms
from segmentation_models_pytorch_3d.decoders.classifier import Unet
from dataloaders import Classification_train
import utils

def main(spark_file="spark_classifier_file.npz", pretrain_ckpt="ca2_classification_cytoplasm_far_split_rois/epoch_122.pth", hold_out=0.1):
    """
    Train a spark classification model using pre-processed cell data.

    This script loads pre-processed spark data, splits it into training and testing sets,
    and trains a 3D Unet model for spark classification. The model is initialized with
    pre-trained weights and fine-tuned on the spark data.

    Args:
        spark_file (str): Path to the NPZ file containing spark data and labels.
        pretrain_ckpt (str): Path to the pre-trained model checkpoint.
        hold_out (float): Fraction of data to use for testing (default: 0.1).

    The script saves the best model checkpoint based on the lowest test loss.
    """
    # Constants
    EPOCHS = 1000
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    TIMESTEPS = 32
    TRAIN_WORKERS = 16
    TEST_WORKERS = 4
    TIMM_MODEL = "resnext50_32x4d"
    HW = 128
    PRE_HW = HW + 32
    TIME_KERNEL = [TIMESTEPS, 1, 1]
    DIMS = "BCDHW"

    # Setup
    ckpts = "ca2_spark_classification"
    os.makedirs(ckpts, exist_ok=True)
    accelerator, device, tqdm, TIMM = utils.prepare_env(TIMM_MODEL)

    # Data transforms
    train_trans = transforms.Compose([
        video_transforms.RandomCrop(HW),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomVerticalFlip(),
        video_transforms.ToTensor(),
    ])
    test_trans = transforms.Compose([
        video_transforms.CenterCrop(HW),
        video_transforms.ToTensor(),
    ])

    # Load and split data
    with accelerator.main_process_first():
        spark_zip = np.load(spark_file)
        files, labels = spark_zip["files"], spark_zip["labels"]
        
        np.random.seed(0)
        file_idx = np.random.permutation(len(files))
        test_idx, train_idx = file_idx[:int(len(file_idx) * hold_out)], file_idx[int(len(file_idx) * hold_out):]
        train_files, test_files = files[train_idx], files[test_idx]
        train_y, test_y = labels[train_idx], labels[test_idx]

        train_dataset = Classification_train(train_files, train_y, mu=TIMM["mean"], std=TIMM["std"],
                                             pre_HW=PRE_HW, transform=train_trans, time=TIMESTEPS, dims=DIMS)
        test_dataset = Classification_train(test_files, test_y, mu=TIMM["mean"], std=TIMM["std"],
                                            pre_HW=PRE_HW, transform=test_trans, time=TIMESTEPS, dims=DIMS)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=TRAIN_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=TEST_WORKERS)

    # Initialize model
    model = Unet(encoder_name=TIMM_MODEL, in_channels=1, classes=2, time_kernel=TIME_KERNEL)
    ckpt_weights = torch.load(pretrain_ckpt, map_location="cuda:0")
    ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items() if "segmentation_head" not in k}
    model.load_state_dict(ckpt_weights, strict=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # Prepare for distributed training
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for image, label in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            pred, _, _ = model(image, return_features=True)
            loss = F.cross_entropy(pred.squeeze(), label.long())
            accelerator.backward(loss)
            optimizer.step()

        model.eval()
        test_losses = []
        with torch.no_grad():
            for image, label in tqdm(test_loader, desc=f"Testing Epoch {epoch}"):
                pred, _, _ = model(image, return_features=True)
                loss = F.cross_entropy(pred.squeeze(), label.long())
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        if avg_test_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(ckpts, f'epoch_{epoch}.pth'))
            best_loss = avg_test_loss
        print(f"Epoch {epoch}, Test Loss: {avg_test_loss:.4f}, Best Test Loss: {best_loss:.4f}")
