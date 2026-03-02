import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from src import transforms as video_transforms
from torchvision.transforms import v2 as transforms
from segmentation_models_pytorch_3d.decoders.classifier import Unet
from src.dataloaders import Classification_train
from src import utils

# Configuration parameters
epochs = 1000
tr_bs = 32  # Training batch size
te_bs = 32  # Testing batch size
timesteps = 32
train_workers = 16
test_workers = 4
timm_model = "resnext50_32x4d"
HW = 128  # Height and width of input images
pre_HW = HW + 32  # Size to random crop from
time_kernel = [timesteps, 1, 1]
dims = "BCDHW"

# Experiments to use for testing
test_exps = [
    "20230412 RA306 Experiment 10uM",
    "20230428 AIP Experiment 585nM Plate2",
    "20230906 EL20 Experiment 1uM",
]

def main(model_type=None):
    """
    Train a classification model using specified cell data type.
    
    Args:
        model_type (str): Type of cell data to use for training. Must be one of
                          {nucleus_rois, cytoplasm_rois, cytoplasm_far_rois, 
                           cytoplasm_split_rois, cytoplasm_far_split_rois}
    """
    assert model_type is not None, "Pass a model_type in {nucleus_rois, cytoplasm_rois, cytoplasm_far_rois, cytoplasm_split_rois, cytoplasm_far_split_rois}"
    ckpts = f"ca2_classification_{model_type}"
    os.makedirs(ckpts, exist_ok=True)

    # Prepare environment and data transforms
    accelerator, device, tqdm, TIMM = utils.prepare_env(timm_model)
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

    # Select files for training and testing
    with accelerator.main_process_first():
        files = glob(os.path.join(model_type, "*", "*", "*.npy"))
        files = [x for x in files if "_ca" not in x]
        train_files, test_files = [], []
        for f in files:
            if any(filt in f for filt in test_exps):
                test_files.append(f)
            else:
                train_files.append(f)

        # Prepare labels for training and testing data
        train_u_conds, train_y = np.unique([x.split(os.path.sep)[2].split("-")[0].lower() for x in train_files], return_inverse=True)
        train_u_exps, train_y_exps = np.unique([True if "post" in x else False for x in train_files], return_inverse=True)
        train_y_exps[train_y_exps == 1] = train_y.max() + 1
        train_y = train_y + train_y_exps

        test_u_conds, test_y = np.unique([x.split(os.path.sep)[2].split("-")[0].lower() for x in test_files], return_inverse=True)
        test_u_exps, test_y_exps = np.unique([True if "post" in x else False for x in test_files], return_inverse=True)
        test_y_exps[test_y_exps == 1] = test_y.max() + 1
        test_y = test_y + test_y_exps

        train_y = torch.from_numpy(train_y)
        test_y = torch.from_numpy(test_y)

        # Create datasets
        train_dataset = Classification_train(
            train_files, train_y, mu=TIMM["mean"], std=TIMM["std"],
            pre_HW=pre_HW, transform=train_trans, time=timesteps, dims=dims)
        test_dataset = Classification_train(
            test_files, test_y, mu=TIMM["mean"], std=TIMM["std"],
            pre_HW=pre_HW, transform=test_trans, time=timesteps, dims=dims)
        print("Finished loading data")

    # Create data loaders
    print("Building dataloaders")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tr_bs, shuffle=True, drop_last=True, pin_memory=True, num_workers=train_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=te_bs, shuffle=True, drop_last=True, num_workers=test_workers)

    # Prepare model and optimizer
    print("Preparing models")
    model = Unet(
        encoder_name="resnext50_32x4d",
        in_channels=1,
        classes=train_y.max() + 1,
        time_kernel=time_kernel,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    # Training loop
    best_loss = 10000
    for epoch in range(epochs):
        model.train()
        progress = tqdm(total=len(train_loader), desc="Training")
        for source in train_loader:
            optimizer.zero_grad()
            image, label = source
            pred  = model(image, return_features=False)
            pred = pred.squeeze()
            loss = F.cross_entropy(pred, label)
            if torch.isnan(loss):
                import pdb;pdb.set_trace()

            accelerator.backward(loss)
            optimizer.step()
            progress.set_postfix({"Epoch": epoch, "Train_loss": loss.item()})
            progress.update()

        model.eval()
        progress = tqdm(total=len(test_loader), desc="Testing")
        with torch.no_grad():
            losses = []
            for source in test_loader:
                image, label = source
                pred  = model(image, return_features=False)
                pred = pred.squeeze()
                loss = F.cross_entropy(pred, label)
                loss = loss.item()
                losses.append(loss)
                progress.set_postfix({"Epoch": epoch, "Test_loss": loss})
                progress.update()
            loss = np.mean(losses)
            if loss < best_loss:
                torch.save(model.state_dict(), os.path.join(ckpts, 'epoch_{}.pth'.format(epoch)))
                best_loss = loss
            progress.set_postfix({"Current_test_loss": loss, "Best_test_loss": best_loss})
            progress.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', default=None, type=str, help="Pass a model_type in {nucleus_rois, cytoplasm_rois, cytoplasm_far_rois, cytoplasm_split_rois, cytoplasm_far_split_rois}")
    args = vars(parser.parse_args())
    main(**args)
