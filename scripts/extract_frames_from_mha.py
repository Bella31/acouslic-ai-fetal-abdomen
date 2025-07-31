import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import argparse

def extract_frames(csv_path, mha_dir, output_base):
    df = pd.read_csv(csv_path)

    # Create train/val/test split
    unique_scans = df["Filename"].unique()
    train_scans, val_scans, test_scans = np.split(
        unique_scans, [int(.7 * len(unique_scans)), int(.85 * len(unique_scans))]
    )

    split_map = {fn: 'train' for fn in train_scans}
    split_map.update({fn: 'val' for fn in val_scans})
    split_map.update({fn: 'test' for fn in test_scans})

    label_map = {0: "irrelevant", 1: "optimal", 2: "suboptimal"}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["Filename"]
        frame_idx = int(row["Frame"])
        label = row["Label"]
        split = split_map.get(filename)

        if not split:
            continue

        save_dir = os.path.join(output_base, split, label_map[label])
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, f"{filename.replace('.mha', '')}_{frame_idx}.jpg")
        if os.path.exists(output_path):
            continue  # Skip already processed frame

        mha_path = os.path.join(mha_dir, filename)
        if not os.path.exists(mha_path):
            print(f"Missing scan: {mha_path}")
            continue

        try:
            volume = sitk.GetArrayFromImage(sitk.ReadImage(mha_path))
            frame = volume[frame_idx]
            img = Image.fromarray(frame).convert("L")
            img.save(output_path)
        except Exception as e:
            print(f"Error with {filename}, frame {frame_idx}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MHA scans based on CSV labels")
    parser.add_argument("--csv", type=str, required=True, help="Path to balanced CSV file")
    parser.add_argument("--mha_dir", type=str, required=True, help="Directory containing MHA scans")
    parser.add_argument("--output", type=str, required=True, help="Output base directory")
    args = parser.parse_args()

    extract_frames(args.csv, args.mha_dir, args.output)
