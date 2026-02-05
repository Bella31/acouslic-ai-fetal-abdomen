import json
import os
import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from part1_frame_classification.src.utils import load_mha_frames, ParamsReadWrite
from part1_frame_classification.scripts.train_classification import score_frames
from torchvision import transforms
from part1_frame_classification.src.models import get_finetune_resnet_model, get_finetune_efficientnetv2_s_gray, \
    get_finetune_convnext_small, get_finetune_densenet121_gray
from part1_frame_classification.scripts.calculate_wfss import  \
    calc_scan_wfss_accuracy, calc_top_k_scan_wfss_accuracy
from sklearn.metrics import confusion_matrix
from openpyxl import Workbook


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

def calc_per_slice_precision_recall(probs, optimal_frames, suboptimal_frames, probs2 = None, idx_rel = None):
    """
    Calculate precision, recall and F1 metrics from all scan slices
    Also produces confusion matrix for the scan
    """
    predictions = np.argmax(probs, axis=1)
    gt = np.zeros_like(predictions)
    gt[optimal_frames] = 1
    gt[suboptimal_frames] = 2
    if probs2 is not None:
        predictions2 = np.argmax(probs2, axis=1)
        predictions2[predictions2==0] = 2 #the 0 here corresponds to suboptimal slice with value of 2
        predictions[idx_rel] = predictions2
    #calculate precision and recall when only optimal examples are used
    mapping = {0: 0, 1: 1, 2: 0}
    y_true_merged = np.vectorize(mapping.get)(gt)
    y_pred_merged = np.vectorize(mapping.get)(predictions)
    conf_matrix_opt_only = confusion_matrix(y_true_merged, y_pred_merged)
    TP = conf_matrix_opt_only[1, 1]
    FN = conf_matrix_opt_only[1, 0]
    FP = conf_matrix_opt_only[0, 1]
    precision_opt_only = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_opt_only = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_opt_only = 2 * (precision_opt_only * recall_opt_only) / (precision_opt_only + recall_opt_only)

    #calculate precision and recall when both optimal and suboptimal are used
    mapping = {0: 0, 1: 1, 2: 1}
    y_true_merged = np.vectorize(mapping.get)(gt)
    y_pred_merged = np.vectorize(mapping.get)(predictions)
    conf_matrix_opt_only = confusion_matrix(y_true_merged, y_pred_merged)
    TP = conf_matrix_opt_only[1, 1]
    FN = conf_matrix_opt_only[1, 0]
    FP = conf_matrix_opt_only[0, 1]
    precision_opt_subopt = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_opt_subopt = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_opt_subopt = 2 * (precision_opt_subopt * recall_opt_subopt) / (precision_opt_subopt + recall_opt_subopt)

    return gt, predictions, precision_opt_only, recall_opt_only, f1_opt_only, precision_opt_subopt, recall_opt_subopt, f1_opt_subopt


def process_scans(scan_dir, model, model2, transform, val_scan_names, labels_df, threshold=0.0, device=None,
                  top_5 = True):
    """
    Apply inference and perform all evaluations
    Output 3 dataframes: one for top slices selection, ont with precision and recall and a total confusion matrix
    """
    results = []
    precision_recall_res = []
    cm_total = np.zeros((3, 3), dtype=np.int64)
    all_scan_paths = sorted(glob.glob(os.path.join(scan_dir, "*.mha")))
    for path in tqdm(all_scan_paths, desc="Processing validation scans"):
        scan_id = os.path.basename(path).replace('.mha', '')
        if scan_id not in val_scan_names:
            continue
        try:
            print('processing scan ' + scan_id)
            frames_tensor = load_mha_frames(path, transform)
            start = time.perf_counter()
            probs = score_frames(model, frames_tensor, device)
       #     top_5_slices, top_5_quality = get_top_slices(probs, 5)
            scores = probs[:, 1]  # class 1 = optimal
            idx5 = np.argpartition(scores, -5)[-5:]
            idx5 = idx5[np.argsort(scores[idx5])[::-1]]  # sort by value desc
            probs2 = None
            idx_all = None
            if model2 is not None:
                if top_5:
                    idx_all = idx5
                else: #apply inference of the second network on all relevant frames
                    predictions = np.argmax(probs, axis=1)
                    idx_all = np.where(predictions == 1)[0]
                probs2 = score_frames(model2, frames_tensor[idx_all], device)
                best_idx, idx3, idx5, max_score = calc_top_1_5(idx_all, probs2, scores, threshold)
            else:
                idx3 = np.argpartition(scores, -3)[-3:]
                idx3 = idx3[np.argsort(scores[idx3])[::-1]]
                max_score = np.max(scores)
                best_idx = int(np.argmax(scores))

            end = time.perf_counter()
            running_time = end-start
            print(f"running time is: {running_time}")
            optimal_frames, suboptimal_frames = get_optimal_suboptimal(scan_id, labels_df)
            (gt, predictions, precision_opt_only, recall_opt_only, f1_opt_only, precision_opt_subopt, recall_opt_subopt,
             f1_opt_subopt) = calc_per_slice_precision_recall(probs, optimal_frames, suboptimal_frames, probs2, idx_all)
            for t, p in zip(gt, predictions):
                cm_total[t, p] += 1
            precision_recall_res.append({"scan": scan_id + ".mha", "precision_opt_only": precision_opt_only,
                                         "recall_opt_only": recall_opt_only, "F1_opt_only": f1_opt_only,
                                         "precision_opt_subopt": precision_opt_subopt, "recall_opt_subopt": recall_opt_subopt,
                                         "F1_opt_subopt": f1_opt_subopt,"num_total": len(probs[:,0]),
                                         "num_opt": len(optimal_frames), "num_subopt": len(suboptimal_frames)})
            wfss_score, accuracy = calc_scan_wfss_accuracy(scan_id, best_idx, labels_df)
            wfss_score_top_3, accuracy_top3 = calc_top_k_scan_wfss_accuracy(scan_id, idx3, labels_df)
            wfss_score_top_5, accuracy_top5 = calc_top_k_scan_wfss_accuracy(scan_id, idx5, labels_df)
            print("wfss_score is: " + str(wfss_score))
            print("Top 5 wfss_score is: " + str(wfss_score_top_5))
            results.append({"scan": scan_id + ".mha", "best_frame": best_idx, "top_5": idx5,"score": max_score,
                            "optimal_frames":optimal_frames , "suboptimal_frames": suboptimal_frames,
                            "wfss": wfss_score, "wfss_top3": wfss_score_top_3,"wfss_top5": wfss_score_top_5,
                            "accuracy": accuracy, "accuracy_top3": accuracy_top3,  "accuracy_top5": accuracy_top5,
                            "inference_time": running_time})

        except Exception as e:
            print(f"Error processing {scan_id}: {e}")
    #create a dataframe from the total confusion matrix
    class_names = ["irrelevant", "optimal", "suboptimal"]
    cm_df = pd.DataFrame(cm_total, index=class_names, columns=class_names)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"
    return pd.DataFrame(results), pd.DataFrame(precision_recall_res), cm_df


def calc_top_1_5(idx_relevant, probs2, scores_base_network, threshold):
    """"
    Calculate top 1-5 in case of two networks
    """
    num_rel = len(probs2)
    scores = probs2[:, 1]  # class 1- optimal
    max_score = np.max(scores)
    best_idx2 = int(np.argmax(scores))
    if max_score < threshold:
        print(f"maximum score is {max_score}, which is smaller than threshold {threshold}")
        max_sub_scores = probs2[:, 0]
        best_idx2 = int(np.argmax(max_sub_scores))
    best_idx = idx_relevant[best_idx2]
    if num_rel>=3:
        idx3 = idx_relevant[np.argsort(scores)[::-1]][:3]
    else:
        idx3 = idx_relevant[np.argsort(scores_base_network)[::-1]][:3]
    if num_rel>=5:
        idx5 = idx_relevant[np.argsort(scores)[::-1]][:5]
    else:
        idx3 = idx_relevant[np.argsort(scores_base_network)[::-1]][:5]

    return best_idx, idx3, idx5, max_score


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_model(model_name, args):
    """
    Get relevant model
    """
    if model_name == 'resnet' or model_name is None:
        model = get_finetune_resnet_model(device=device, num_classes=args.num_classes)
    elif model_name == 'efficientnet':
        model = get_finetune_efficientnetv2_s_gray(
            num_classes=args.num_classes, pretrained=True, device=device)
    elif model_name == 'convnext':
        model = get_finetune_convnext_small(
            num_classes=args.num_classes, pretrained=True, device=device)
    elif model_name == "densenet":
        model = get_finetune_densenet121_gray(
            num_classes=args.num_classes, pretrained=True, device=device)
    else:
        print(f'model name {model_name} is not defined!')
        model = None
    print(f'model name is: {model_name}')
    return model

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
    parser.add_argument("--num_classes", type=int, default=3, help="number of classes")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold for the second network")
    parser.add_argument("--top5", type=str2bool, default='true', help="should use top 5 for the second"
                                                                       " network if applicable")


    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_path)
    # Load model
    device = torch.device("cpu")
    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = None
    #extract model name
    with open(os.path.join(os.path.dirname(args.model), 'config.json'), 'r') as f:
        config = json.load(f)
        if 'model_name' in config:
            model_name = config['model_name']
        else: model_name = None
    model = get_model(model_name, args)

    if device == "cpu":
        model = model.float()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if args.model2 is not None:
        with open(os.path.join(os.path.dirname(args.model2), 'config.json'), 'r') as f:
            config = json.load(f)
            if 'model_name' in config:
                model2_name = config['model_name']
            else:
                model2_name = 'Resnet'
        model2 = get_model(model2_name, args)
        if device == "cpu":
            model2 = model2.float()
        model2.load_state_dict(torch.load(args.model2, map_location=device))
        model2.eval()
    else:
        model2 = None
    #either provide test folder or test cases list
    if args.test_scans_path is not None and args.test_scans_path!='None':
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
    df_results, df_precision_recall_res, df_cm = process_scans(args.scan_dir, model, model2, transform, val_scan_names, labels_df, device=device,
                               threshold = args.threshold, top_5=args.top5)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="top1-5", index=False)
        df_precision_recall_res.to_excel(writer, sheet_name="precision_recall", index=False)
        df_cm.to_excel(writer, sheet_name="conf_matrix", index=True, index_label="True")

    print(f"✅ Results saved to {args.output}")
