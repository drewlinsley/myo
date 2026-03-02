import os
import numpy as np
import torch
import torch.nn.functional as F
import transforms as video_transforms
from torchvision.transforms import v2 as transforms
from segmentation_models_pytorch_3d.decoders.classifier import Unet
from dataloaders import Classification_train
import utils


epochs = 1000
tr_bs = 32  # 3
te_bs = 32  # 3
timesteps = 32
train_workers = 16
test_workers = 4
timm_model = "resnext50_32x4d"
HW = 128  # 384
pre_HW = HW + 32  # size to random crop from
time_kernel = [timesteps, 1, 1]
dims = "BCDHW"
np.random.seed(0)


def main(
        spark_file="spark_classifier_file.npz",
        ckpts="ca2_spark_classification",
        pretrain_ckpt="ca2_classification_cytoplasm_far_split_rois/epoch_122.pth",
        test_hold_out=0.1):
    """
    Train a model to detect sparks in calcium imaging data.

    Args:
        spark_file (str): Path to the .npz file containing spark data and labels.
        ckpts (str): Directory to save model checkpoints.
        pretrain_ckpt (str): Path to the pretrained model checkpoint for transfer learning.
        test_hold_out (float): Fraction of data to hold out for testing (0-1).

    Inputs:
        - spark_file: A .npz file containing 'files' (image data) and 'labels' arrays.
        - pretrain_ckpt: A .pth file with pretrained model weights.

    Outputs:
        - Saves model checkpoints to the 'ckpts' directory.
        - Best model is saved as 'epoch_X.pth' where X is the epoch number.

    The function trains a U-Net model for binary classification of sparks,
    using transfer learning from a pretrained model. It performs data loading,
    preprocessing, model training, and evaluation.
    """
    os.makedirs(ckpts, exist_ok=True)

    # Here we additionally normalize them to the pretrained weights of our model
    accelerator, _, tqdm, TIMM = utils.prepare_env(timm_model)
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
        spark_zip = np.load(spark_file)
        files = spark_zip["files"]
        labels = spark_zip["labels"]
        file_idx = np.random.permutation(len(files))
        test_idx = file_idx[:int(len(file_idx) * test_hold_out)]
        train_idx = file_idx[int(len(file_idx) * test_hold_out):]
        train_files = files[train_idx]
        test_files = files[test_idx]
        train_y = labels[train_idx]
        test_y = labels[test_idx]
        train_dataset = Classification_train(
            train_files,
            train_y,
            mu=TIMM["mean"],
            std=TIMM["std"],
            pre_HW=pre_HW,
            transform=train_trans,
            time=timesteps,
            dims=dims)
        test_dataset = Classification_train(
            test_files,
            test_y,
            mu=TIMM["mean"],
            std=TIMM["std"],
            pre_HW=pre_HW,
            transform=test_trans,
            time=timesteps,
            dims=dims)
        print("Finished loading data")

    print("Building dataloaders")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tr_bs, shuffle=True, drop_last=True, pin_memory=True, num_workers=train_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=te_bs, shuffle=True, drop_last=True, num_workers=test_workers)

    print("Preparing models")
    model = Unet(
        encoder_name="resnext50_32x4d",
        in_channels=1,
        classes=2,
        time_kernel=time_kernel,
    )
    print("Using ckpt: {}".format(pretrain_ckpt))
    ckpt_weights = torch.load(pretrain_ckpt, map_location="cuda:0")
    key_check = [x for x in ckpt_weights.keys()][0]
    if key_check.split(".")[0] == "module":
        ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items()}
    ckpt_weights = {k: v for k, v in ckpt_weights.items() if "segmentation_head" not in k}
    model.load_state_dict(ckpt_weights, strict=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    best_loss = 10000
    for epoch in range(epochs):

        model.train()
        progress = tqdm(total=len(train_loader), desc="Training")
        for source in train_loader:

            optimizer.zero_grad()
            image, label = source
            label = label.long()
            pred, maps, latents = model(image, return_features=True)
            pred = pred.squeeze()
            import pdb;pdb.set_trace()
            loss = F.cross_entropy(pred, label)
            if torch.isnan(loss):
                import pdb;pdb.set_trace()

            accelerator.backward(loss)
            optimizer.step()
            progress.set_postfix({"Epoch": epoch, "Train_loss": loss.item()})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
            progress.update()

        model.eval()
        progress = tqdm(total=len(test_loader), desc="Testing")
        with torch.no_grad():
            losses = []
            for source in test_loader:

                image, label = source
                label = label.long()
                pred, maps, latents  = model(image, return_features=True)
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
            progress.set_postfix({"Current_test_loss": loss, "Best_test_loss": best_loss})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
            progress.update()


if __name__ == "__main__":
    main()
