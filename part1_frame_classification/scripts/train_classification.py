import math
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from part1_frame_classification.src.model_resnet import get_finetune_resnet_model
from part1_frame_classification.src.utils import (FocalLoss, mixup_data, mixup_criterion, evaluate,
                                                  get_create_model_dir, ParamsReadWrite)
from torch.amp import autocast, GradScaler
from PIL import Image, ImageFile
import pandas as pd


def safe_pil_loader(path: str):
    """Open image as RGB; return None if it can't be read."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as e:
        print(f"[WARN] bad image skipped: {path} ({e})")
        return None

class SafeImageFolder(ImageFolder):
    """Wraps __getitem__ to skip files that failed to load/transform."""
    def __getitem__(self, index):
        # try a few indices until one works
        tries = 0
        while tries < 5:
            path, target = self.samples[index]
            try:
                sample = self.loader(path)              # uses our safe_pil_loader
                if sample is None:                      # unreadable image
                    raise OSError("loader returned None")
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                print(f"[WARN] skipping {path}: {e}")
                # move to next sample
                index = (index + 1) % len(self.samples)
                tries += 1
        # if many consecutive failures, surface an error instead of looping forever
        raise RuntimeError("Too many consecutive bad images.")


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


def score_frames(model, frames_tensor, device=None, batch_size=256):
    """
    batch_size:    number of frames per forward pass
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    rng = range(0, len(frames_tensor), batch_size)

    probs_chunks = []

    # inference_mode is a bit lighter than no_grad for eval
    with torch.inference_mode():
        for i in rng:
            batch = frames_tensor[i:i+batch_size].to(device, non_blocking=True)
            outputs = model(batch)                    # [B, num_classes]
            probs = torch.softmax(outputs, dim=1)     # softmax over class dim
            probs_chunks.append(probs.cpu())          # keep on CPU to free VRAM

            # (Optional) free references quickly; usually not needed but harmless
            del batch, outputs, probs

    return torch.cat(probs_chunks, dim=0).numpy()     # [N, num_classes]


def score_frames_all_at_ones(model, frames_tensor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs


def train(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cuda", model_save_path = 'model.pt',
          metrics_path='train_metics.csv', patience = 6, min_epoch = 10):
    scaler = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
    best_val_acc = 0
    counter = 0
    df = pd.DataFrame({
        "epoch": pd.Series(dtype="int"),
        "train_loss": pd.Series(dtype="float32"),
        "train_acc": pd.Series(dtype="float32"),
        "val_acc": pd.Series(dtype="float32"),
    })
    df.to_csv(metrics_path)
    #print('number of training images is: ' + str(len(train_loader.sampler)))
    for epoch in range(epochs):
        print('Training epoch ' + str(epoch))
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

        if val_acc > best_val_acc and epoch>=min_epoch:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f" Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
        elif epoch < min_epoch:
            continue
        else:
            counter += 1
            if counter >= patience:
                print(" Early stopping triggered")
                break

        #save epoch data
        metrics_df = pd.read_csv(metrics_path)
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc}
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    import argparse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train classification model for Acouslic-AI")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to balanced dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_dir", type=str, default="/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--min_epoch", type=int, default=10, help="minimum epoch from which to save model")
    args = parser.parse_args()

    model_dir = get_create_model_dir(args.log_dir)
    model_save_path = os.path.join(model_dir, 'resnet50_3class.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    out_cfg_path = os.path.join(model_dir, 'config.json')
    ParamsReadWrite.write_config(out_cfg_path, args.data_dir, args.epochs, args.batch_size, args.lr, args.weight_decay,
                                 args.patience, args.min_epoch)
    metrics_path = os.path.join(model_dir, 'train_metrics.csv')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Train model
    train(model, train_loader, val_loader, optimizer, criterion, epochs=args.epochs, model_save_path=model_save_path,
          metrics_path=metrics_path, patience=args.patience, min_epoch=args.min_epoch)

    # Evaluate on test set
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f" Test Accuracy: {test_acc:.4f}")
