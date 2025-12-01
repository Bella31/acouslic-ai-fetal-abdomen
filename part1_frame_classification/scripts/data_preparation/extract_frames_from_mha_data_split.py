import glob
import os
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import gc, time
import argparse
import numpy as np


def save_split_info(output_base, train_lst, valid_lst, test_lst):
    """
    Save training, validation andtest data
    """
    split_path = os.path.join(output_base, 'data_split')
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    np.savetxt(os.path.join(split_path, 'training_ids.txt'), train_lst, fmt='%s')
    np.savetxt(os.path.join(split_path, 'valid_ids.txt'), valid_lst, fmt='%s')
    np.savetxt(os.path.join(split_path, 'test_ids.txt'), test_lst, fmt='%s')


def save_data_split(train_scans, val_scans, test_scans, df, mha_dir, output_base):
    """
    Exract images and save the data split to output_base folder
    """
    #save scans_info
    save_split_info(output_base, train_scans, val_scans, test_scans)
    split_map = {fn: 'train' for fn in train_scans}
    split_map.update({fn: 'val' for fn in val_scans})
    split_map.update({fn: 'test' for fn in test_scans})
    label_map = {0: "irrelevant", 1: "optimal", 2: "suboptimal"}

    EXT = ".jpg"
    saved_count = 0

    for fname, g in df.groupby("Filename", sort=False):
        t0 = time.time()

        split = split_map.get(fname)
        if not split:
            continue

        mha_path = os.path.join(mha_dir, fname)
        if not os.path.exists(mha_path):
            print(f"Missing scan: {mha_path}")
            continue

        # collect frames that still need saving
        needed = []
        for _, row in g.iterrows():
            label = row["Label"]
            frame_idx = int(row["Frame"])
            save_dir = os.path.join(output_base, split, label_map[label])  # unchanged
            os.makedirs(save_dir, exist_ok=True)
            base = f"{fname.replace('.mha', '')}_{frame_idx}"
            out_path = os.path.join(save_dir, base + EXT)
            if not os.path.exists(out_path):
                needed.append((frame_idx, out_path))

        if not needed:
            continue

        img = vol = im = None
        try:
            img = sitk.ReadImage(mha_path)  # load once
            vol = sitk.GetArrayViewFromImage(img)  # (z,y,x) zero-copy view

            for frame_idx, out_path in needed:
                frame = vol[frame_idx]  # 2D slice
                # # ---- Min–max normalization to 8-bit (as previously suggested) ----
                # f = frame.astype(np.float32, copy=False)
                # fmin, fmax = float(f.min()), float(f.max())
                # if fmax > fmin:
                #     frame8 = np.round((f - fmin) / (fmax - fmin) * 255).astype(np.uint8)
                # else:
                #     frame8 = np.zeros_like(frame, dtype=np.uint8)

                im = Image.fromarray(frame).convert("L")
                im.save(out_path)  # PNG = lossless
                im.close()
                saved_count += 1

        except Exception as e:
            print(f"Error with {fname}: {e}")
        finally:
            del vol, img, im
            gc.collect()

        print(f"{fname}: {time.time() - t0:.2f}s")

    print(f"Done. Saved {saved_count} frames.")


def extract_frames(csv_path, mha_dir, output_base, train_ratio=0.7, val_ratio=0.15):
    df = pd.read_csv(csv_path)

    # Validate CSV
    required_cols = {"Filename", "Frame", "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Create train/val/test split
    unique_scans = df["Filename"].unique()
    train_end = int(train_ratio * len(unique_scans))
    val_end = train_end + int(val_ratio * len(unique_scans))
    train_scans, val_scans, test_scans = np.split(unique_scans, [train_end, val_end])

    print("reading volumes..")
    save_data_split(train_scans, val_scans, test_scans, df, mha_dir, output_base)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MHA scans based on CSV labels")
    parser.add_argument("--csv", type=str, required=True, help="Path to balanced CSV file")
    parser.add_argument("--mha_dir", type=str, required=True, help="Directory containing MHA scans")
    parser.add_argument("--output", type=str, required=True, help="Output base directory")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of scans for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of scans for validation")
    args = parser.parse_args()

    extract_frames(args.csv, args.mha_dir, args.output, args.train_ratio, args.val_ratio)
