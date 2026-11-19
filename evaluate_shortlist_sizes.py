"""
Evaluate the two-step frame selection with several candidate shortlist sizes.

For every shortlist size the first network scores all frames of a scan, its top K frames are handed to the
second network, and the second network re-ranks them. K is the quantity under study here, so a separate
results file is written per K and per fold.

The first network is the one-step network and the second network is the second stage of the two-step
approach, matched by fold. Results are written into the results folder of the second network.

Usage:
    python evaluate_shortlist_sizes.py --scan_dir <stacked_fetal_ultrasound> --labels_path <labels.csv> \
        --test_scans_lst_path <subopt_cases.txt>

    # see the commands without running them
    python evaluate_shortlist_sizes.py ... --dry_run true

    # a different method pair or a different set of sizes
    python evaluate_shortlist_sizes.py ... --method "two-step Convnext" --shortlist_sizes 3,5,10,20
"""

import argparse
import json
import os
import shlex
import subprocess
import sys


DEFAULT_MODEL_DIR = ('/Users/bella/Academy/Research/ultrasound inexperienced operator/'
                     'Ultrasound slice selection/Networks/experiments_networks')
DEFAULT_SCAN_DIR = ('/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/'
                    'stacked_fetal_ultrasound')
DEFAULT_LABELS_PATH = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'
DEFAULT_SUBOPT_CASES = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'

# method name -> (first network per fold, second network per fold).
# The first network is the one-step network, the second one is the second stage of the two-step approach.
METHODS = {
    'two-step (ours)':        ([196, 197, 198, 199, 200], [201, 202, 203, 204, 205]),
    'two-step Convnext':      ([131, 132, 133, 134, 135], [160, 161, 162, 163, 164]),
    'two-step Efficientnet':  ([236, 237, 238, 239, 240], [241, 242, 243, 244, 245]),
    'two-step Resnet':        ([226, 227, 228, 229, 230], [231, 232, 233, 234, 235]),
}


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
    if isinstance(v, (list, tuple)):
        return [int(k) for k in v]
    values = [int(part) for part in str(v).replace(' ', '').split(',') if part != '']
    if not values:
        raise argparse.ArgumentTypeError('At least one value is expected.')
    if min(values) < 1:
        raise argparse.ArgumentTypeError('Shortlist sizes must be positive.')
    return sorted(set(values))


def get_test_scans_path(model2_path):
    """
    Test folder of the fold, read from the config of the second network
    """
    config_path = os.path.join(os.path.dirname(model2_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return os.path.join(config['data_dir'], 'test')


def build_command(model_path, model2_path, scan_dir, labels_path, results_path, num_classes, threshold,
                  shortlist_size, topk_list, test_scans_path=None, test_scans_lst_path=None, rel_only=False,
                  use_gpu=True, overwrite=False):
    #paths are quoted because they may contain spaces
    q = shlex.quote
    args = (
            '--model ' + q(model_path) + ' --model2 ' + q(model2_path) + ' --scan_dir ' + q(scan_dir) +
            ' --output ' + q(results_path) + ' --labels_path ' + q(labels_path) +
            ' --test_scans_lst_path ' + q(str(test_scans_lst_path)) +
            ' --test_scans_path ' + q(str(test_scans_path)) +
            ' --num_classes ' + str(num_classes) + ' --threshold ' + str(threshold) +
            ' --top5 true' + ' --shortlist_size ' + str(shortlist_size) +
            ' --topk_list ' + ','.join(str(k) for k in topk_list) +
            ' --rel_only ' + str(rel_only).lower() + ' --use_gpu ' + str(use_gpu).lower() +
            ' --overwrite ' + str(overwrite).lower())
    return "python -m part1_frame_classification.scripts.evaluate_frame_selection " + args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate several candidate shortlist sizes")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="folder holding the networks")
    parser.add_argument("--scan_dir", type=str, default=DEFAULT_SCAN_DIR, help="Path to .mha scans")
    parser.add_argument("--labels_path", type=str, default=DEFAULT_LABELS_PATH,
                        help="frame level labels.csv holding all frames, with Filename, Frame and Label columns")
    parser.add_argument("--test_scans_lst_path", type=str, default=DEFAULT_SUBOPT_CASES,
                        help="list of the suboptimal-only test cases")
    parser.add_argument("--method", type=str, default='two-step (ours)', choices=sorted(METHODS.keys()),
                        help="which network pair to evaluate")
    parser.add_argument("--shortlist_sizes", type=str2intlist, default='3,5,10',
                        help="comma separated candidate shortlist sizes to evaluate")
    parser.add_argument("--topk_list", type=str2intlist, default='3,5,10',
                        help="comma separated top-k selections to score besides the single best frame")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold for the second network")
    parser.add_argument("--use_gpu", type=str2bool, default='true', help="use GPU if there is enough GPU memory")
    parser.add_argument("--run_opt", type=str2bool, default='true', help="run the optimal-included evaluation")
    parser.add_argument("--run_subopt", type=str2bool, default='true', help="run the suboptimal-only evaluation")
    parser.add_argument("--overwrite", type=str2bool, default='false',
                        help="replace result files that already exist, keeping the old ones in"
                             " previous_versions. When false an existing file is kept and that evaluation"
                             " is skipped, so an interrupted sweep can simply be restarted")
    parser.add_argument("--dry_run", type=str2bool, default='false', help="print the commands without running them")

    args = parser.parse_args()

    model_dirnames, model2_dirnames = METHODS[args.method]
    print(f"method: {args.method}")
    print(f"first networks : {model_dirnames}")
    print(f"second networks: {model2_dirnames}")
    print(f"shortlist sizes: {args.shortlist_sizes}")

    commands = []
    for shortlist_size in args.shortlist_sizes:
        for fold in range(len(model2_dirnames)):
            model_path = os.path.join(args.model_dir, str(model_dirnames[fold]), 'model.pt')
            model2_path = os.path.join(args.model_dir, str(model2_dirnames[fold]), 'model.pt')
            results_dir = os.path.join(args.model_dir, str(model2_dirnames[fold]), 'results')
            #optimal-included evaluation, the test folder of the fold holds the scans
            if args.run_opt:
                results_path = os.path.join(
                    results_dir, f'predictions_{model2_dirnames[fold]}_k{shortlist_size}.xlsx')
                commands.append(build_command(
                    model_path, model2_path, args.scan_dir, args.labels_path, results_path, args.num_classes,
                    args.threshold, shortlist_size, args.topk_list,
                    test_scans_path=get_test_scans_path(model2_path), test_scans_lst_path=None,
                    rel_only=False, use_gpu=args.use_gpu, overwrite=args.overwrite))
            #suboptimal-only evaluation, over the cases that hold no optimal frame
            if args.run_subopt:
                results_path = os.path.join(
                    results_dir, f'predictions_{model2_dirnames[fold]}_subopt_k{shortlist_size}.xlsx')
                commands.append(build_command(
                    model_path, model2_path, args.scan_dir, args.labels_path, results_path, args.num_classes,
                    args.threshold, shortlist_size, args.topk_list,
                    test_scans_path=None, test_scans_lst_path=args.test_scans_lst_path,
                    rel_only=True, use_gpu=args.use_gpu, overwrite=args.overwrite))

    print(f"\n{len(commands)} evaluations to run\n")
    failures = []
    for i, command in enumerate(commands, start=1):
        print(f"--- [{i}/{len(commands)}] ---")
        print(command)
        if args.dry_run:
            continue
        return_code = subprocess.call(command, shell=True)
        if return_code != 0:
            print(f"evaluation failed with return code {return_code}")
            failures.append((i, command, return_code))

    if args.dry_run:
        print("\ndry run, nothing was executed")
    elif failures:
        print(f"\n{len(failures)} of {len(commands)} evaluations failed:")
        for i, command, return_code in failures:
            print(f"  [{i}] return code {return_code}: {command}")
        sys.exit(1)
    else:
        print(f"\nall {len(commands)} evaluations finished")
