import json
import os
import glob
import shutil
import sys
import time
import numpy as np
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
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


DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_TOPK_LIST = (3, 5)


def prepare_labels(labels_df):
    """
    Sort the labels by scan and frame number and check they can be indexed by frame.
    get_frame_labels takes the labels in row order and the frame index is used to look into that list, so
    rows that are out of order or duplicated silently shift every label of the scan.
    """
    required = {'Filename', 'Frame', 'Label'}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels file is missing the columns {sorted(missing)}. The frame level labels file "
                         f"with all frames is needed here, not a balanced or summarized one.")
    duplicated = labels_df.duplicated(subset=['Filename', 'Frame'])
    if duplicated.any():
        examples = labels_df.loc[duplicated, ['Filename', 'Frame']].head(5).values.tolist()
        raise ValueError(f"labels file holds {int(duplicated.sum())} duplicated (Filename, Frame) rows, "
                         f"for example {examples}")
    return labels_df.sort_values(['Filename', 'Frame']).reset_index(drop=True)


def validate_scan_labels(scan_id, labels_df, num_frames):
    """
    Check that the scan has one label per frame, numbered 0 to num_frames-1
    """
    frames = np.asarray(labels_df.loc[labels_df['Filename'] == scan_id + ".mha", 'Frame'].values)
    if len(frames) == 0:
        raise ValueError(f"scan {scan_id} has no rows in the labels file")
    if len(frames) != num_frames:
        raise ValueError(f"scan {scan_id} has {num_frames} frames but {len(frames)} label rows. The frame "
                         f"level labels file with all frames is needed here.")
    expected = np.arange(num_frames)
    if not np.array_equal(frames, expected):
        raise ValueError(f"scan {scan_id} labels do not cover frames 0..{num_frames - 1} exactly, "
                         f"they run from {frames.min()} to {frames.max()}")


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

def calc_precision_recall_from_conf_df(cm):
    """
    Calculate precision, recall and F1 confusion matrix with two classes
    """
    TP = cm.iloc[1, 1]
    FN = cm.iloc[1, 0]
    FP = cm.iloc[0, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return  precision, recall, f1

def calc_precision_recall_from_conf(cm):
    """
    Calculate precision, recall and F1 confusion matrix with two classes
    """
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return  precision, recall, f1


def calc_per_slice_precision_recall(probs, optimal_frames, suboptimal_frames, probs2 = None, idx_rel = None, rel_only=False):
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
    if rel_only is False:
        mapping = {0: 0, 1: 1, 2: 0}
        y_true_merged = np.vectorize(mapping.get)(gt)
        y_pred_merged = np.vectorize(mapping.get)(predictions)
        conf_matrix_opt_only = confusion_matrix(y_true_merged, y_pred_merged)
        precision_opt_only, recall_opt_only, f1_opt_only = calc_precision_recall_from_conf(conf_matrix_opt_only)
    else:
        precision_opt_only = -1
        recall_opt_only = -1
        f1_opt_only = -1
    #calculate precision and recall when both optimal and suboptimal are used
    mapping = {0: 0, 1: 1, 2: 1}
    y_true_merged = np.vectorize(mapping.get)(gt)
    y_pred_merged = np.vectorize(mapping.get)(predictions)
    conf_matrix_opt_subopt = confusion_matrix(y_true_merged, y_pred_merged)
    precision_opt_subopt, recall_opt_subopt, f1_opt_subopt = calc_precision_recall_from_conf(conf_matrix_opt_subopt)

    return gt, predictions, precision_opt_only, recall_opt_only, f1_opt_only, precision_opt_subopt, recall_opt_subopt, f1_opt_subopt


def get_top_k_indices(scores, k):
    """
    Indices of the k highest scores, sorted from the best score down.
    k is clipped to the number of available frames.
    """
    k = int(min(k, len(scores)))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def process_scans(scan_dir, model, model2, transform, val_scan_names, labels_df, threshold=0.0, device=None,
                  top_5 = True, rel_only = False, shortlist_size = DEFAULT_SHORTLIST_SIZE,
                  topk_list = DEFAULT_TOPK_LIST):
    """
    Apply inference and perform all evaluations
    Output 3 dataframes: one for top slices selection, ont with precision and recall and a total confusion matrix

    shortlist_size is the number of candidate frames the first network hands to the second one when top_5 is
    True. topk_list are the additional top-k frame selections to score besides the single best frame, which is
    always reported in the "wfss" and "accuracy" columns. A k larger than the shortlist cannot be selected, so
    its columns are left empty.
    """
    results = []
    precision_recall_res = []
    cm_total = np.zeros((3, 3), dtype=np.int64)
    shortlist_size = int(shortlist_size)
    topk_list = [int(k) for k in topk_list]
    #a shortlist of n frames cannot produce a top-k selection for k above n
    if top_5 and model2 is not None:
        skipped_ks = [k for k in topk_list if k > shortlist_size]
        if skipped_ks:
            print(f"shortlist size is {shortlist_size}, top {skipped_ks} will be left empty")
    all_scan_paths = sorted(glob.glob(os.path.join(scan_dir, "*.mha")))
    #a test scan with no .mha in scan_dir would otherwise be dropped without a word, leaving the fold
    #evaluated on fewer scans than it should be
    val_scan_names = set(str(name).replace('.mha', '') for name in val_scan_names)
    found_scan_names = set(os.path.basename(p).replace('.mha', '') for p in all_scan_paths)
    absent = sorted(val_scan_names - found_scan_names)
    if absent:
        raise FileNotFoundError(f"{len(absent)} of the {len(val_scan_names)} test scans are not in "
                                f"{scan_dir}, for example {absent[:5]}")
    processed_scan_names = set()
    for path in tqdm(all_scan_paths, desc="Processing validation scans"):
        scan_id = os.path.basename(path).replace('.mha', '')
        if scan_id not in val_scan_names:
            continue

        print('processing scan ' + scan_id)
        frames_tensor = load_mha_frames(path, transform)
        validate_scan_labels(scan_id, labels_df, len(frames_tensor))
        processed_scan_names.add(scan_id)
        start = time.perf_counter()
        probs = score_frames(model, frames_tensor, device)
        scores = probs[:, 1]  # class 1 = optimal
        shortlist = get_top_k_indices(scores, shortlist_size)
        probs2 = None
        idx_all = None
        if model2 is not None:
            if top_5:
                idx_all = shortlist
            else: #apply inference of the second network on all relevant frames
                predictions = np.argmax(probs, axis=1)
                idx_all = np.where(predictions == 1)[0]
            probs2 = score_frames(model2, frames_tensor[idx_all], device)
            best_idx, topk_indices, max_score = calc_topk_indices(idx_all, probs2, scores, threshold,
                                                                  [k for k in topk_list if k <= len(idx_all)])
        else:
            topk_indices = {k: get_top_k_indices(scores, k) for k in topk_list}
            max_score = np.max(scores)
            best_idx = int(np.argmax(scores))

        end = time.perf_counter()
        running_time = end-start
        print(f"running time is: {running_time}")
        optimal_frames, suboptimal_frames = get_optimal_suboptimal(scan_id, labels_df)
        (gt, predictions, precision_opt_only, recall_opt_only, f1_opt_only, precision_opt_subopt, recall_opt_subopt,
         f1_opt_subopt) = calc_per_slice_precision_recall(probs, optimal_frames, suboptimal_frames, probs2, idx_all, rel_only=rel_only)
        for t, p in zip(gt, predictions):
            cm_total[t, p] += 1
        precision_recall_res.append({"scan": scan_id + ".mha", "precision_opt_only": precision_opt_only,
                                     "recall_opt_only": recall_opt_only, "F1_opt_only": f1_opt_only,
                                     "precision_opt_subopt": precision_opt_subopt, "recall_opt_subopt": recall_opt_subopt,
                                     "F1_opt_subopt": f1_opt_subopt,"num_total": len(probs[:,0]),
                                     "num_opt": len(optimal_frames), "num_subopt": len(suboptimal_frames)})
        wfss_score, accuracy = calc_scan_wfss_accuracy(scan_id, best_idx, labels_df)
        print("wfss_score is: " + str(wfss_score))
        #the largest selection that the shortlist could actually serve, so this is filled for every
        #shortlist size rather than only for the largest requested k
        available_ks = sorted(topk_indices)
        scan_result = {"scan": scan_id + ".mha", "best_frame": best_idx,
                       "shortlist_size": shortlist_size if (top_5 and model2 is not None) else len(probs[:, 0]),
                       "top_frames": list(topk_indices[available_ks[-1]]) if available_ks else None,
                       "score": max_score,
                       "optimal_frames": optimal_frames, "suboptimal_frames": suboptimal_frames,
                       "wfss": wfss_score, "accuracy": accuracy}
        #score every requested top-k selection, leaving a k that the shortlist cannot serve empty,
        #and keep the ranked frames of each k so the metrics can be checked afterwards
        for k in topk_list:
            if k in topk_indices:
                wfss_score_top_k, accuracy_top_k = calc_top_k_scan_wfss_accuracy(scan_id, topk_indices[k], labels_df)
                print(f"Top {k} wfss_score is: " + str(wfss_score_top_k))
                scan_result[f"top{k}_frames"] = list(topk_indices[k])
            else:
                wfss_score_top_k, accuracy_top_k = np.nan, np.nan
                scan_result[f"top{k}_frames"] = None
            scan_result[f"wfss_top{k}"] = wfss_score_top_k
            scan_result[f"accuracy_top{k}"] = accuracy_top_k
        scan_result["inference_time"] = running_time
        results.append(scan_result)



    #every test scan must have gone through, otherwise the fold is evaluated on the wrong set
    not_processed = sorted(val_scan_names - processed_scan_names)
    if not_processed:
        raise RuntimeError(f"{len(not_processed)} test scans were not processed, "
                           f"for example {not_processed[:5]}")
    print(f"processed {len(processed_scan_names)} of {len(val_scan_names)} test scans")

    #create a dataframe from the total confusion matrix
    class_names = ["irrelevant", "optimal", "suboptimal"]
    cm_df = pd.DataFrame(cm_total, index=class_names, columns=class_names)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"
    return pd.DataFrame(results), pd.DataFrame(precision_recall_res), cm_df


def calc_topk_indices(idx_relevant, probs2, scores_base_network, threshold, topk_list):
    """"
    Re-rank the frames the first network selected using the second network and return the best frame
    together with the top k frames for every requested k.
    When fewer frames reached the second network than a requested k, the remaining places are filled with
    the best remaining frames of the first network.
    """
    idx_relevant = np.asarray(idx_relevant).ravel().astype(int)
    num_rel = len(probs2)
    scores = probs2[:, 1]  # class 1- optimal
    max_score = np.max(scores)
    best_idx2 = int(np.argmax(scores))
    if max_score < threshold:
        print(f"maximum score is {max_score}, which is smaller than threshold {threshold}")
        max_sub_scores = probs2[:, 0]
        best_idx2 = int(np.argmax(max_sub_scores))
    best_idx = int(idx_relevant[best_idx2])
    ranked = idx_relevant[np.argsort(scores)[::-1]]
    #frames of the first network, best first, used only to fill up a shortlist shorter than k
    padding = np.argsort(scores_base_network)[::-1]
    topk_indices = {}
    for k in topk_list:
        selected = ranked[:k]
        if num_rel < k:
            already_selected = set(selected.tolist())
            extra = [int(i) for i in padding if int(i) not in already_selected][:k - len(selected)]
            selected = np.concatenate([selected, np.asarray(extra, dtype=int)])
        topk_indices[k] = selected

    return best_idx, topk_indices, max_score


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2intlist(v):
    """
    Parse a comma separated list of k values, for example "3,5,10"
    """
    if isinstance(v, (list, tuple)):
        return [int(k) for k in v]
    values = [int(part) for part in str(v).replace(' ', '').split(',') if part != '']
    if not values:
        raise argparse.ArgumentTypeError('At least one k value is expected.')
    if min(values) < 1:
        raise argparse.ArgumentTypeError('k values must be positive.')
    return sorted(set(values))

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
    parser.add_argument("--top5", type=str2bool, default='true', help="should use a candidate shortlist for the"
                                                                       " second network if applicable")
    parser.add_argument("--shortlist_size", type=int, default=DEFAULT_SHORTLIST_SIZE,
                        help="number of candidate frames passed from the first to the second network when"
                             " --top5 is true")
    parser.add_argument("--topk_list", type=str2intlist, default=','.join(str(k) for k in DEFAULT_TOPK_LIST),
                        help="comma separated top-k selections to score besides the single best frame,"
                             " for example 3,5,10")
    parser.add_argument("--rel_only", type=str2bool, default='false', help="should only relevant cases be calculated")
    parser.add_argument("--overwrite", type=str2bool, default='false',
                        help="overwrite an existing output file, moving the old one to previous_versions."
                             " When false an existing output file is kept and the evaluation is skipped")



    args = parser.parse_args()

    if args.shortlist_size < 1:
        parser.error("--shortlist_size must be at least 1")

    #decided before inference runs, so a skip costs nothing
    if os.path.exists(args.output):
        if not args.overwrite:
            print(f"⏭  {args.output} already exists, skipping. Pass --overwrite true to replace it.")
            sys.exit(0)
        archive_dir = os.path.join(os.path.dirname(args.output), 'previous_versions')
        os.makedirs(archive_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base, ext = os.path.splitext(os.path.basename(args.output))
        archived = os.path.join(archive_dir, f'{base}_{stamp}{ext}')
        shutil.move(args.output, archived)
        print(f"moved the previous results to {archived}")

    labels_df = prepare_labels(pd.read_csv(args.labels_path))
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
        os.makedirs(out_dir, exist_ok=True)
    df_results, df_precision_recall_res, df_cm = process_scans(args.scan_dir, model, model2, transform, val_scan_names, labels_df, device=device,
                               threshold = args.threshold, top_5=args.top5, rel_only = args.rel_only,
                               shortlist_size = args.shortlist_size, topk_list = args.topk_list)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="top1-5", index=False)
        df_precision_recall_res.to_excel(writer, sheet_name="precision_recall", index=False)
        df_cm.to_excel(writer, sheet_name="conf_matrix", index=True, index_label="True")

    print(f"✅ Results saved to {args.output}")
