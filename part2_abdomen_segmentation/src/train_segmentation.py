import torch
from torch import optim
from tqdm import tqdm

def train_unet(model, train_loader, val_loader, device, epochs=10):
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_dice = evaluate_unet(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Dice = {val_dice:.4f}")

def evaluate_unet(model, val_loader, device):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            total_dice += dice
    return total_dice / len(val_loader)
