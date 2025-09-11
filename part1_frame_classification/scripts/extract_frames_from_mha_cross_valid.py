import os
import random
import pandas as pd
import argparse
import numpy as np
from part1_frame_classification.scripts.extract_frames_from_mha_data_split import save_data_split


def extract_frames_cross_valid(csv_path, mha_dir, output_dir, num_folds = 5, valid_fraction=0.25):
    """
    Create cross validation splits according to the number of folds
    Save split frames in image format
    """
    df = pd.read_csv(csv_path)

    # Validate CSV
    required_cols = {"Filename", "Frame", "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    unique_scans = df["Filename"].unique()
    random.shuffle(unique_scans)
    num_valid_test_scans = int(len(unique_scans)/num_folds)
    train_num = int(len(unique_scans) - num_valid_test_scans)
    valid_num = int(num_valid_test_scans*0.25)
    test_num = int(num_valid_test_scans-valid_num)
    print('number of training  scans is: ', train_num)
    print('number of validation  scans is: ', valid_num)
    print('number of test scans is: ', test_num)

    # Create train/val/test splits for each one of the folds
    ind = 0
    for fold in range(num_folds):
        first_split = ind #first split - up until that index it is training data
        second_split = first_split + valid_num #split for validation data
        third_split = second_split + test_num #split for test data
        train_scans1, val_scans, test_scans, train_scans2 = np.split(unique_scans, [first_split, second_split, third_split])
        train_scans = np.concatenate((train_scans1, train_scans2), axis=0)
        output_base = os.path.join(output_dir, str(fold))
        if os.path.exists(output_base) is False:
            os.mkdir(output_base)

        save_data_split(train_scans, val_scans, test_scans, df, mha_dir, output_base)
        ind = num_valid_test_scans #update index for next fold



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MHA scans based on CSV labels")
    parser.add_argument("--csv", type=str, required=True, help="Path to balanced CSV file")
    parser.add_argument("--mha_dir", type=str, required=True, help="Directory containing MHA scans")
    parser.add_argument("--output", type=str, required=True, help="Output base directory")
    parser.add_argument("--num_folds", type=float, default=5, help="number of cross validation folds")
    parser.add_argument("--valid_fraction", type=float, default=0.25, help="fraction of validation cases out of validation and test")
    args = parser.parse_args()

    extract_frames_cross_valid(args.csv, args.mha_dir, args.output, args.num_folds, args.valid_fraction)