import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.model_unet import UNet
from src.utils import (
    SegmentationDataset,
    get_segmentation_transforms,
    dice_coefficient,
    BCEDiceLoss 
)

def train_unet(model, train_loader, val_loader, device, epochs, save_path):
    criterion = BCEDiceLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_dice = 0.0
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_dice = evaluate_unet(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with Val Dice: {val_dice:.4f}")

def evaluate_unet(model, loader, device):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            dice = dice_coefficient(outputs, masks)
            total_dice += dice
    return total_dice / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for abdomen segmentation")
    parser.add_argument("--train_images", type=str, required=True, help="Path to train images folder")
    parser.add_argument("--train_masks", type=str, required=True, help="Path to train masks folder")
    parser.add_argument("--val_images", type=str, required=True, help="Path to validation images folder")
    parser.add_argument("--val_masks", type=str, required=True, help="Path to validation masks folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="models/unet_best.pth")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    image_transform, mask_transform = get_segmentation_transforms()

    # Datasets & DataLoaders
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, image_transform, mask_transform)
    val_dataset = SegmentationDataset(args.val_images, args.val_masks, image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bat_
