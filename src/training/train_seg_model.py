import os
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch_3d as smp
from src import transforms as video_transforms
from torchvision.transforms import v2 as transforms
from src.dataloaders import IntraData_train
from src import utils


# Configuration
ddir = "ca2"
version = 1
epochs = 1000
tr_bs = 4  # Training batch size
te_bs = 4  # Testing batch size
val_frames = 64
timesteps = 32
freeze_encoder = False
normalize_input = True
mask_questionable_sources = False
discretization = 2  # Max value counting from 1
ckpts = "ca2_whole_cell_no_norm"
os.makedirs(ckpts, exist_ok=True)
timm_model = "resnext50_32x4d"
HW = 384
time_kernel = [32, 1, 1]
dims = "BCDHW"

# Prepare environment and model
accelerator, device, tqdm, TIMM = utils.prepare_env(timm_model)

# Define data transformations
train_trans = transforms.Compose([
    video_transforms.RandomCrop(HW),
    video_transforms.RandomHorizontalFlip(),
    video_transforms.RandomVerticalFlip(),
    video_transforms.ToTensor(),
])

val_trans = transforms.Compose([
    video_transforms.CenterCrop(HW),
    video_transforms.ToTensor(),
])

# Load and preprocess data
with accelerator.main_process_first():
    print("Started loading data")
    train_x = np.load(os.path.join(ddir, "2023-9-8_control_baseline.npy"))
    train_y = np.load(os.path.join(ddir, "inner.npy"))
    test_x = train_x[-val_frames:]
    train_x = train_x[:-val_frames]
    test_y = train_y

    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)

    # Create datasets
    train_dataset = IntraData_train(train_x, train_y, mu=TIMM["mean"], std=TIMM["std"], transform=train_trans, time=timesteps, dims=dims)
    test_dataset = IntraData_train(test_x, test_y, mu=TIMM["mean"], std=TIMM["std"], transform=val_trans, time=timesteps, dims=dims)

    # Calculate class weights
    classes, class_counts = np.unique(train_y, return_counts=True)
    num_classes = float(len(classes))
    num_samples = float(len(train_y.ravel()))
    class_weights = [num_samples / (num_classes * count) for count in class_counts]
    class_weights = torch.Tensor(class_weights).float()
    print("Finished loading data")

# Create data loaders
print("Building dataloaders")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tr_bs, shuffle=True, drop_last=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=te_bs, shuffle=False, drop_last=False)

# Initialize model
print("Preparing models")
model = smp.Unet(
    encoder_name=timm_model,
    in_channels=1,
    classes=2,
    time_kernel=time_kernel,
)

# Set up optimizer and prepare for distributed training
optimizer = torch.optim.AdamW(model.parameters())
model, optimizer, train_loader, test_loader, class_weights = accelerator.prepare(model, optimizer, train_loader, test_loader, class_weights)
class_weights = class_weights.to(device)

# Training loop
best_loss = 10000
for epoch in range(epochs):
    # Training phase
    model.train()
    progress = tqdm(total=len(train_loader), desc="Training")
    for source in train_loader:
        optimizer.zero_grad()
        image, label = source

        pred = model(image)
        loss = F.cross_entropy(pred, label, weight=class_weights)
        if torch.isnan(loss):
            import pdb;pdb.set_trace()

        accelerator.backward(loss)
        optimizer.step()
        progress.set_postfix({"Epoch": epoch, "Train_loss": loss.item()})
        progress.update()

    # Evaluation phase
    model.eval()
    progress = tqdm(total=len(test_loader), desc="Testing")
    with torch.no_grad():
        losses = []
        for source in test_loader:
            image, label = source
            pred = model(image)
            loss = F.cross_entropy(pred, label)
            loss = loss.item()
            losses.append(loss)
            progress.set_postfix({"Epoch": epoch, "Test_loss": loss})
            progress.update()
        
        # Calculate average loss and save best model
        loss = np.mean(losses)
        if loss < best_loss:
            torch.save(model.state_dict(), os.path.join(ckpts, f'epoch_{epoch}.pth'))
            best_loss = loss
        progress.set_postfix({"Current_test_loss": loss, "Best_test_loss": best_loss})
        progress.update()
