from src.utils import SegmentationDataset, get_segmentation_transforms
from torch.utils.data import DataLoader

train_img_dir = "./part2_abdomen_segmentation/data/split/train/images"
train_mask_dir = "./part2_abdomen_segmentation/data/split/train/masks"
val_img_dir = "./part2_abdomen_segmentation/data/split/val/images"
val_mask_dir = "./part2_abdomen_segmentation/data/split/val/masks"
test_img_dir = "./part2_abdomen_segmentation/data/split/test/images"
test_mask_dir = "./part2_abdomen_segmentation/data/split/test/masks"

image_transform, mask_transform = get_segmentation_transforms()

train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, image_transform, mask_transform)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, image_transform, mask_transform)
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, image_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
