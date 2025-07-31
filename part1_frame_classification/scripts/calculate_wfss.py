import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import argparse

def calculate_wfss(predictions_csv, mask_dir):
    pred_df = pd.read_csv(predictions_csv)
    scores = []

    def get_frame_labels_from_mask(mask_path):
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        labels = []
        for frame in mask_array:
            unique = np.unique(frame)
            if 1 in unique:
                labels.append("optimal")
            elif 2 in unique:
                labels.append("suboptimal")
            else:
                labels.append("irrelevant")
        return labels

    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        scan_id = row["scan"]
        selected = int(row["best_frame"])
        mask_path = os.path.join(mask_dir, scan_id)
        if not os.path.exists(mask_path):
            continue
        frame_labels = get_frame_labels_from_mask(mask_path)
        if selected == -1 or selected >= len(frame_labels):
            scores.append(0.0)
            continue
        label = frame_labels[selected]
        has_optimal = "optimal" in frame_labels
        if label == "optimal":
            scores.append(1.0)
        elif label == "suboptimal" and has_optimal:
            scores.append(0.6)
        else:
            scores.append(0.0)

    wfss_score = np.mean(scores)
    print(f"WFSS: {wfss_score:.4f}")
    return wfss_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Weighted Frame Selection Score (WFSS)")
    parser.add_argument("--predictions", type=str, required=True, help="Path to CSV file with predictions")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to directory with masks")
    args = parser.parse_args()

    calculate_wfss(args.predictions, args.mask_dir)
