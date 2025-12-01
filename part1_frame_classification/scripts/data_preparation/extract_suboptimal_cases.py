import os
import pandas as pd

from part1_frame_classification.src.utils import ParamsReadWrite


def prepare_balanced_dataset(input_csv, output_lst):
    # Load labeled frame CSV
    df = pd.read_csv(input_csv)

    # Validate required columns
    required_cols = {"Filename", "Frame", "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    subopt_cases = []

    # Group by scan
    for scan_name, group in df.groupby("Filename"):
        optimal = group[group["Label"] == 1]
        if len(optimal) == 0:
            print('adding case ' + scan_name)
            subopt_cases.append(scan_name)

    ParamsReadWrite.list_dump(subopt_cases, output_lst)

if __name__ == "__main__":
    input_csv = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'
    output_subopt_lst = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    prepare_balanced_dataset(input_csv, output_subopt_lst)