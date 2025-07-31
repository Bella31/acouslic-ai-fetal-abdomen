import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(image_dir, mask_dir, output_base, train_ratio=0.7, val_ratio=0.15):
    all_files = os.listdir(image_dir)
    scan_ids = sorted({f.rsplit("_", 1)[0] + ".mha" for f in all_files if f.endswith(".jpg")})

    # 1. Split into train and temp
    train_scans, temp_scans = train_test_split(scan_ids, test_size=(1 - train_ratio), random_state=42)

    # 2. Split temp into val and test
    val_size = val_ratio / (1 - train_ratio)
    val_scans, test_scans = train_test_split(temp_scans, test_size=(1 - val_size), random_state=42)

    print(f"Train: {len(train_scans)}, Val: {len(val_scans)}, Test: {len(test_scans)}")

    def copy_split(split_name, scan_list):
        img_out = os.path.join(output_base, split_name, "images")
        msk_out = os.path.join(output_base, split_name, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(msk_out, exist_ok=True)
        for fname in all_files:
            sid = fname.split("_")[0] + ".mha"
            if sid in scan_list:
                shutil.copy(os.path.join(image_dir, fname), os.path.join(img_out, fname))
                shutil.copy(os.path.join(mask_dir, fname), os.path.join(msk_out, fname))

    # Copy for train, val, test
    copy_split("train", train_scans)
    copy_split("val", val_scans)
    copy_split("test", test_scans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split segmentation dataset into train/val/test")
    parser.add_argument("--images", type=str, required=True, help="Path to images folder")
    parser.add_argument("--masks", type=str, required=True, help="Path to masks folder")
    parser.add_argument("--output", type=str, required=True, help="Base output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.images, args.masks, args.output, args.train_ratio, args.val_ratio)
