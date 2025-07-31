import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model_resnet import get_finetune_resnet_model
from src.utils import FocalLoss, mixup_data, mixup_criterion, evaluate
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(base_path, batch_size):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_loader = DataLoader(
        ImageFolder(os.path.join(base_path, 'train'), transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        ImageFolder(os.path.join(base_path, 'val'), transform=val_test_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        ImageFolder(os.path.join(base_path, 'test'), transform=val_test_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def score_frames(model, frames_tensor):
    model.eval()
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs


def train(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cuda", model_save_path="best_model.pt"):
    scaler = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True)
    best_val_acc = 0
    patience = 6
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f" Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(" Early stopping triggered")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classification model for Acouslic-AI")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to balanced dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_save_path", type=str, default="models/resnet50_3class.pt")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    # Create output folder
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Initialize model and optimizer
    model = get_finetune_resnet_model(num_classes=3)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f" Training started with config: {args}")

    # Train model
    train(model, train_loader, val_loader, optimizer, criterion, epochs=args.epochs, model_save_path=args.model_save_path)

    # Evaluate on test set
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f" Test Accuracy: {test_acc:.4f}")
