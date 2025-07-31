from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

class SegmentationDataset(Dataset):
    """
    Custom Dataset for segmentation: loads grayscale images and corresponding masks.
    """
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("L")  # Grayscale
        mask = Image.open(mask_path).convert("L")  # Binary mask

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Ensure mask is binary
        mask = (mask > 0).float()

        return image, mask


def get_segmentation_transforms():
    """
    Returns separate transforms for image and mask.
    """
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return image_transform, mask_transform


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dice_coefficient(preds, targets, threshold=0.5, smooth=1.0):
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()


def hausdorff_distance(pred_mask, true_mask):
    # find nonzero (y,x) points on each mask
    p_pts = np.argwhere(pred_mask>0)
    t_pts = np.argwhere(true_mask>0)
    if len(p_pts)==0 or len(t_pts)==0:
        return np.nan
    d1 = directed_hausdorff(p_pts, t_pts)[0]
    d2 = directed_hausdorff(t_pts, p_pts)[0]
    return max(d1, d2)

def ellipse_perimeter(mask):
    # find largest contour and return its arc length in pixels
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    return cv2.arcLength(c, True)

def normalized_ac_error(pred_mask, true_mask, spacing_mm):
    # perimeter in px → mm
    p_pred = ellipse_perimeter(pred_mask) * spacing_mm
    p_true = ellipse_perimeter(true_mask) * spacing_mm
    if max(p_pred, p_true) == 0:
        return np.nan
    return abs(p_true - p_pred) / max(p_pred, p_true)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return bce_loss + (1 - dice)
