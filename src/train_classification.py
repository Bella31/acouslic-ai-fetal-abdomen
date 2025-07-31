
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model_resnet import get_finetune_resnet_model
from src.utils import FocalLoss
import os

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_finetune_resnet_model(num_classes=3)
criterion = FocalLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

def score_frames(model, frames_tensor):
    model.eval()
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs
