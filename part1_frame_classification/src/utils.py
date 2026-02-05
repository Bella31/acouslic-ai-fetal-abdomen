import argparse
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from PIL import Image
import os
import json


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

class WFSSFrameSurrogateLoss(nn.Module):
# For each sample i:
#         reward_i = w[y_i] * p(y_i | x_i)
#     where w[irrelevant]=0, w[sub]=subopt_score, w[opt]=1.
#     We minimize 1 - mean(reward), i.e. maximize weighted correct probability.
    def __init__(self, idx_opt: int, idx_sub: int, subopt_score: float = 0.6):
        super().__init__()
        self.idx_opt = int(idx_opt)
        self.idx_sub = int(idx_sub)
        self.subopt_score = float(subopt_score)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        C = probs.size(1)

        if C <= max(self.idx_opt, self.idx_sub):
            raise ValueError("WFSSFrameSurrogateLoss: class indices out of range.")

        # class weights: irrelevant=0, sub=subopt_score, opt=1
        weights = torch.zeros(C, device=probs.device, dtype=probs.dtype)
        weights[self.idx_sub] = self.subopt_score
        weights[self.idx_opt] = 1.0

        # p(correct class)
        idx = torch.arange(logits.size(0), device=probs.device)
        p_correct = probs[idx, targets]

        # reward for each sample
        w_correct = weights[targets]
        reward = w_correct * p_correct  # in [0, 1]

        # we want to maximize reward -> minimize (1 - reward)
        return 1.0 - reward.mean()

class CEWFSSLoss(nn.Module):
    #L = ce_weight * CE + wfss_weight * WFSS_surrogate
    def __init__(self,
                 idx_opt: int,
                 idx_sub: int,
                 subopt_score: float = 0.6,
                 ce_weight: float = 1.0,
                 wfss_weight: float = 0.3,
                 label_smoothing: float = 0.0):
        super().__init__()
        ls = max(0.0, min(0.99, float(label_smoothing)))
        self.ce = nn.CrossEntropyLoss(label_smoothing=ls)
        self.wfss = WFSSFrameSurrogateLoss(idx_opt=idx_opt,
                                           idx_sub=idx_sub,
                                           subopt_score=subopt_score)
        self.ce_weight = float(ce_weight)
        self.wfss_weight = float(wfss_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_wfss = self.wfss(logits, targets)
        return self.ce_weight * loss_ce + self.wfss_weight * loss_wfss

def mixup_data(x, y, alpha=0.1):
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


class ParamsReadWrite:
    @staticmethod
    def list_dump(lst, out_file):
        np.savetxt(out_file, lst, fmt='%s')

    @staticmethod
    def list_load(in_file):
        return list(np.loadtxt(in_file, dtype=str, ndmin=1))

    @staticmethod
    def save_split_data(model_dir, train_lst, valid_lst, test_lst):
        """
        Save training, validation andtest data
        """
        split_path = os.path.join(model_dir, 'data_split')
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        ParamsReadWrite.list_dump(train_lst, os.path.join(split_path, 'training_ids.txt'))
        ParamsReadWrite.list_dump(valid_lst, os.path.join(split_path, 'validation_ids.txt'))
        ParamsReadWrite.list_dump(test_lst, os.path.join(split_path, 'test_ids.txt'))
    @staticmethod
    def write_config(out_path, data_dir, epochs, batch_size, lr, weight_decay, patience, min_epoch, apply_mixup,
                     num_classes, loss, pos_only, opt_only, model_name):
        config = {
            "data_dir": data_dir,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "patience": patience,
            "min_epoch": min_epoch,
            "apply_mixup": str(apply_mixup),
            "num_classes": num_classes,
            "loss": loss,
            "pos_only": pos_only,
            "opt_only": opt_only,
            "model_name": model_name
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

def get_create_model_dir(experiments_dir):
    """
    List directories in experiments_dir and find the latest index
    Create a new index for the experiment with id=latest_id+1
    """
    dir_ids = os.listdir(experiments_dir)
    max_id = 0
    for dir_id in dir_ids:
        try:
            val = int(dir_id)  # handles '001', '42', '-3', etc.
        except ValueError:
            continue
        if val > max_id:
            max_id = val

    dir_path= os.path.join(experiments_dir, str(max_id + 1))
    if os.path.exists(dir_path) is False:
        print('creating experiment directory: ' + dir_path)
        os.mkdir(dir_path)

    return dir_path


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
