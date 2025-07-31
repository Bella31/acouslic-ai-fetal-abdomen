import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
import numpy as np
import torch

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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def load_mha_frames(mha_path, transform):
    itk_image = sitk.ReadImage(mha_path)
    volume = sitk.GetArrayFromImage(itk_image)  
    processed_frames = []
    for frame in volume:
        img = Image.fromarray(frame).convert('L')
        transformed = transform(img)
        processed_frames.append(transformed)
    return torch.stack(processed_frames) 

def get_best_frame_index(mha_path, model, transform, optimal_threshold=0.3, suboptimal_threshold=0.3):
    frames_tensor = load_mha_frames(mha_path, transform)
    probs = score_frames(model, frames_tensor)  # shape: [N, 3]
    preds = np.argmax(probs, axis=1)

    optimal_scores = probs[:, 1]
    suboptimal_scores = probs[:, 2]
    irrelevant_scores = probs[:, 0]

    optimal_candidates = np.where(preds == 1)[0]
    if len(optimal_candidates) > 0:
        best_idx = optimal_candidates[np.argmax(optimal_scores[optimal_candidates])]
        if optimal_scores[best_idx] >= optimal_threshold:
            return best_idx, optimal_scores[best_idx]

    suboptimal_candidates = np.where(preds == 2)[0]
    if len(suboptimal_candidates) > 0:
        best_idx = suboptimal_candidates[np.argmax(suboptimal_scores[suboptimal_candidates])]
        if suboptimal_scores[best_idx] >= suboptimal_threshold:
            return best_idx, suboptimal_scores[best_idx]

    best_idx = np.argmax(irrelevant_scores)
    return best_idx, irrelevant_scores[best_idx]

