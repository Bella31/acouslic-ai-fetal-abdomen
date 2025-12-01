import pandas as pd

if __name__ == "__main__":
    input_csv = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'
    save_path = '/home/bella/Academy/results/data_stats.csv'

    df = pd.read_csv(input_csv)

    # Validate required columns
    required_cols = {"Filename", "Frame", "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    scan_stats = {}
    # Group by scan
    for scan_name, group in df.groupby("Filename"):
        optimal = group[group["Label"] == 1]
        suboptimal = group[group["Label"] == 2]
        irrelevant = group[group["Label"] == 0]
        scan_stats[scan_name] = {}
        scan_stats[scan_name]['optimal'] = len(optimal)
        scan_stats[scan_name]['suboptimal'] = len(suboptimal)
        scan_stats[scan_name]['irrelevant'] = len(irrelevant)

    stats_df = pd.DataFrame(scan_stats).T
    stats_df.to_csv(save_path)

