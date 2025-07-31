import pandas as pd
import numpy as np
import os

# Output file path
output_path = "/content/drive/MyDrive/per_scan_downsampled_3class_labels.csv"

# Check if the file already exists and has content
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    if not existing_df.empty:
        print(f"File already exists with {len(existing_df)} rows. Skipping generation.")
    else:
        generate = True
else:
    generate = True

# Only run generation if file doesn't exist or is empty
if 'generate' in locals() and generate:
    # Load 3-class labeled frame CSV
    df = pd.read_csv("/content/drive/MyDrive/per_scan_binary_labels.csv")

    # Store sampled rows here
    balanced_rows = []

    # Group by scan
    for scan_name, group in df.groupby("Filename"):
        optimal = group[group["Label"] == 1]
        suboptimal = group[group["Label"] == 2]
        irrelevant = group[group["Label"] == 0]

        if len(optimal) == 0 and len(suboptimal) == 0:
            continue  # skip scans with no good frames

        # Sample suboptimal (optional limit)
        suboptimal_sample = suboptimal.sample(n=min(len(suboptimal), 10), random_state=42)

        # Combine good frames
        good_frames = pd.concat([optimal, suboptimal_sample])

        # Sample same number of irrelevant frames
        irrelevant_sample = irrelevant.sample(n=min(len(irrelevant), len(good_frames)), random_state=42)

        # Combine and store
        balanced_rows.append(pd.concat([good_frames, irrelevant_sample]))

    # Combine everything and shuffle
    balanced_df = pd.concat(balanced_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to new CSV
    balanced_df.to_csv(output_path, index=False)
    print("Saved reduced 3-class dataset with", len(balanced_df), "frames.")
