import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from PIL import Image

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def load_mha_frames(mha_path, transform):
    try:
        itk_image = sitk.ReadImage(mha_path)
        volume = sitk.GetArrayFromImage(itk_image)  # shape: [N, H, W]
    except Exception as e:
        raise RuntimeError(f"Error reading {mha_path}: {e}")

    processed_frames = []
    for frame in volume:
        img = Image.fromarray(frame).convert('L')
        processed_frames.append(transform(img))

    return torch.stack(processed_frames)  # [N, 1, H, W]


def score_frames(model, frames_tensor, device="cuda"):
    model.eval()
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs


def get_best_frame_index(mha_path, model, transform, optimal_threshold=0.3, suboptimal_threshold=0.3):
    frames_tensor = load_mha_frames(mha_path, transform)
    probs = score_frames(model, frames_tensor)  # shape: [N, 3]
    preds = np.argmax(probs, axis=1)

    optimal_scores = probs[:, 1]
    suboptimal_scores = probs[:, 2]
    irrelevant_scores = probs[:, 0]

    # Check optimal frames
    optimal_candidates = np.where(preds == 1)[0]
    if len(optimal_candidates) > 0:
        best_idx = optimal_candidates[np.argmax(optimal_scores[optimal_candidates])]
        if optimal_scores[best_idx] >= optimal_threshold:
            return best_idx, optimal_scores[best_idx]

    # Check suboptimal frames
    suboptimal_candidates = np.where(preds == 2)[0]
    if len(suboptimal_candidates) > 0:
        best_idx = suboptimal_candidates[np.argmax(suboptimal_scores[suboptimal_candidates])]
        if suboptimal_scores[best_idx] >= suboptimal_threshold:
            return best_idx, suboptimal_scores[best_idx]

    # Fallback: pick highest irrelevant score
    best_idx = np.argmax(irrelevant_scores)
    return best_idx, irrelevant_scores[best_idx]
