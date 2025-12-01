import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from part1_frame_classification.src.utils import load_mha_frames, ParamsReadWrite
from part1_frame_classification.scripts.train_classification import score_frames
from torchvision import transforms
from part1_frame_classification.src.model_resnet import get_finetune_resnet_model
from part1_frame_classification.scripts.calculate_wfss import  \
    calc_scan_wfss_accuracy, calc_top_k_scan_wfss_accuracy

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


def process_scans(scan_dir, model, model2, transform, val_scan_names, labels_df, threshold=0.0, device=None):
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
            idx3 = np.argpartition(scores, -3)[-3:]
            idx3 = idx3[np.argsort(scores[idx3])[::-1]]
            idx5 = np.argpartition(scores, -5)[-5:]
            idx5 = idx5[np.argsort(scores[idx5])[::-1]]  # sort by value desc
            #if there is a second network, use top 5
            if model2 is not None:
                probs2 = score_frames(model2, frames_tensor[idx5], device)
                scores = probs2[:, 1] #class 1- optimal
                max_score = np.max(scores)
                best_idx2 = int(np.argmax(scores))
                best_idx =   idx5[best_idx2]
                idx3 = idx5[np.argsort(scores)[::-1]][:3]
            else:
                max_score = np.max(scores)
                best_idx = int(np.argmax(scores)) if max_score >= threshold else -1
            optimal_frames, suboptimal_frames = get_optimal_suboptimal(scan_id, labels_df)
            wfss_score, accuracy = calc_scan_wfss_accuracy(scan_id, best_idx, labels_df)
            wfss_score_top_3, accuracy_top3 = calc_top_k_scan_wfss_accuracy(scan_id, idx3, labels_df)
            wfss_score_top_5, accuracy_top5 = calc_top_k_scan_wfss_accuracy(scan_id, idx5, labels_df)
            print("wfss_score is: " + str(wfss_score))
            print("Top 5 wfss_score is: " + str(wfss_score_top_5))
            results.append({"scan": scan_id + ".mha", "best_frame": best_idx, "top_5": idx5,"score": max_score,
                            "optimal_frames":optimal_frames , "suboptimal_frames": suboptimal_frames,
                            "wfss": wfss_score, "wfss_top3": wfss_score_top_3,"wfss_top5": wfss_score_top_5,
                            "accuracy": accuracy, "accuracy_top3": accuracy_top3,  "accuracy_top5": accuracy_top5})

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
    parser.add_argument("--model2", type=str, default=None, help="Path to the second trained model if exists")
   # parser.add_argument("--val_scans", type=str, required=True, help="CSV with validation scan names")
    parser.add_argument("--test_scans_path", type=str, default=None, help="path to test scans folder")
    parser.add_argument("--test_scans_lst_path", type=str, default=None, help="path to test scans list")
    parser.add_argument("--use_gpu", type=str2bool, default='true', help="use GPU if there is anought GPU memory")
    parser.add_argument("--num_classes", type=int, default=3, help="patience")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_path)
    # Load model
    device = torch.device("cpu")
    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_finetune_resnet_model(device=device, num_classes=args.num_classes)
    if device == "cpu":
        model = model.float()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if args.model2 is not None:
        model2 = get_finetune_resnet_model(device=device, num_classes=args.num_classes)
        if device == "cpu":
            model2 = model2.float()
        model2.load_state_dict(torch.load(args.model2, map_location=device))
        model2.eval()
    else:
        model2 = None
    #either provide test folder or test cases list
    if args.test_scans_path is not None:
        val_scan_names = read_test_scans(args.test_scans_path)
    else:
        val_scan_names = ParamsReadWrite.list_load(args.test_scans_lst_path)

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
    df_results = process_scans(args.scan_dir, model, model2, transform, val_scan_names, labels_df, device=device)
    df_results.to_csv(args.output, index=False)
    print(f"✅ Results saved to {args.output}")
