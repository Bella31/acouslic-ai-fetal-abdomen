import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from part1_frame_classification.src.utils import load_mha_frames
from part1_frame_classification.scripts.train_classification import score_frames
from torchvision import transforms
from part1_frame_classification.src.model_resnet import get_finetune_resnet_model
from part1_frame_classification.scripts.calculate_wfss import calc_scan_wfss

def get_optimal_suboptimal(scan_id, labels_df):
    """
    Extract optimal and subopitmal frmas of a scan and output lists
    """
    scan_df = labels_df[labels_df['Filename']==scan_id + ".mha"]
    optimal_slices = scan_df.loc[labels_df["Label"] == 1, "Frame"].to_list()
    suboptimal_slices = scan_df.loc[labels_df["Label"] == 2, "Frame"].to_list()

    return optimal_slices, suboptimal_slices

def read_test_scans(val_scans_path):
    frames = os.listdir(os.path.join(val_scans_path, 'irrelevant'))
    frames.extend(os.listdir(os.path.join(val_scans_path, 'optimal')))
    frames.extend(os.listdir(os.path.join(val_scans_path, 'suboptimal')))
    scans_set = set()
    for frame_name in frames:
        name = frame_name.split('_')[0]
        scans_set.add(name)
    return scans_set

# def get_top_slices(probs, 5):
#     """
#     Calculate top 5 slices and their predicted quality
#     """



def process_scans(scan_dir, model, transform, val_scan_names, labels_df, threshold=0.0, device=None):
    results = []
    all_scan_paths = sorted(glob.glob(os.path.join(scan_dir, "*.mha")))
    for path in tqdm(all_scan_paths, desc="Processing validation scans"):
        scan_id = os.path.basename(path).replace('.mha', '')
        if scan_id not in val_scan_names:
            continue
        try:
            print('processing scan ' + scan_id)
            frames_tensor = load_mha_frames(path, transform)
            probs = score_frames(model, frames_tensor, device)
       #     top_5_slices, top_5_quality = get_top_slices(probs, 5)
            scores = probs[:, 1]  # class 1 = optimal
            max_score = np.max(scores)
            best_idx = int(np.argmax(scores)) if max_score >= threshold else -1
            optimal_frames, suboptimal_frames = get_optimal_suboptimal(scan_id, labels_df)
            wfss_score = calc_scan_wfss(scan_id, best_idx, labels_df)
            print("wfss_score is: " + str(wfss_score))
            results.append({"scan": scan_id + ".mha", "best_frame": best_idx, "score": max_score,
                            "optimal_frames":optimal_frames , "suboptimal_frames": suboptimal_frames,
                            "wfss": wfss_score})

        except Exception as e:
            print(f"Error processing {scan_id}: {e}")

    return pd.DataFrame(results)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frame selection")
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to .mha scans")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to path to labels.csv file")
    parser.add_argument("--output", type=str, required=True, help="Path to save CSV results")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
   # parser.add_argument("--val_scans", type=str, required=True, help="CSV with validation scan names")
    parser.add_argument("--test_scans_path", type=str, required=True, help="path to test scans")
    parser.add_argument("--use_gpu", type=str2bool, default='true', help="use GPU if there is anought GPU memory")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_path)
    # Load model
    device = torch.device("cpu")
    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_finetune_resnet_model(device=device)
    if device == "cpu":
        model = model.float()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load val scans
 #   val_df = pd.read_csv(args.val_scans)
#    val_scan_names = set([scan.replace('.mha', '') for scan in val_df['Filename'].unique()])
    val_scan_names = read_test_scans(args.test_scans_path)

    # Define transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Process
    out_dir = os.path.dirname(args.output)
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)
    df_results = process_scans(args.scan_dir, model, transform, val_scan_names, labels_df, device=device)
    df_results.to_csv(args.output, index=False)
    print(f"✅ Results saved to {args.output}")
