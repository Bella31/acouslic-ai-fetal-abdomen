"""
Summarize the shortlist size sweep produced by evaluate_shortlist_sizes.py.

Reads predictions_{id}_k{K}.xlsx and predictions_{id}_subopt_k{K}.xlsx from the results folder of every
second network and writes one workbook comparing the shortlist sizes.

Each metric is first averaged over the scans of a fold and then over the five folds, the same aggregation
as overall_summary.xlsx written by summarize_results.py, so the numbers are directly comparable. The
standard deviation is taken over the five fold means.

The comparison between shortlist sizes is only meaningful when every size was evaluated on exactly the same
folds and the same scans, so that is checked before anything is averaged. A size that is missing a fold, or
that covers a different set of scans, stops the summary rather than producing a difference that comes from
the missing data. Use --allow_incomplete true to summarize anyway, for a look at a sweep still running.

Sheets:
    summary_opt      mean and SD over folds, optimal-included test set
    summary_subopt   mean and SD over folds, suboptimal-only test set
    folds_opt        the per fold means behind summary_opt
    folds_subopt     the per fold means behind summary_subopt

Usage:
    python summarize_shortlist_sizes.py --output <shortlist_size_comparison.xlsx>
"""

import argparse
import os
import sys

import pandas as pd

from evaluate_shortlist_sizes import METHODS, str2bool, str2intlist


DEFAULT_MODEL_DIR = ('/Users/bella/Academy/Research/ultrasound inexperienced operator/'
                     'Ultrasound slice selection/Networks/experiments_networks')


def collect_fold_means(model_dir, model2_dirnames, shortlist_sizes, metrics, subopt):
    """
    Average every metric over the scans of a fold, for every shortlist size
    Returns the per fold means, the files that were missing, and the set of scans behind each fold
    """
    rows = []
    missing = []
    scan_sets = {}
    for shortlist_size in shortlist_sizes:
        for fold, network_id in enumerate(model2_dirnames):
            postfix = '_subopt' if subopt else ''
            excel_path = os.path.join(model_dir, str(network_id), 'results',
                                      f'predictions_{network_id}{postfix}_k{shortlist_size}.xlsx')
            if not os.path.exists(excel_path):
                missing.append(excel_path)
                continue
            df = pd.read_excel(excel_path, sheet_name='top1-5')
            scan_sets[(shortlist_size, fold)] = frozenset(df['scan'].astype(str))
            row = {'shortlist_size': shortlist_size, 'fold': fold, 'network': network_id,
                   'num_scans': len(df)}
            for metric in metrics:
                #a metric the shortlist could not serve was written empty, it stays empty here
                row[metric] = df[metric].mean() if metric in df.columns else float('nan')
            rows.append(row)
    return pd.DataFrame(rows), missing, scan_sets


def check_comparable(shortlist_sizes, num_folds, missing, scan_sets, label):
    """
    Collect the reasons the shortlist sizes cannot be compared as they stand
    """
    problems = []
    if missing:
        problems.append('%s: %d result file(s) missing:\n    %s'
                        % (label, len(missing), '\n    '.join(missing)))
    #every size needs every fold, otherwise sizes are averaged over different folds
    for shortlist_size in shortlist_sizes:
        folds_present = sorted(fold for (k, fold) in scan_sets if k == shortlist_size)
        if len(folds_present) != num_folds:
            problems.append('%s: shortlist size %d has %d of %d folds (%s)'
                            % (label, shortlist_size, len(folds_present), num_folds, folds_present))
    #within a fold every size must cover the same scans, otherwise the comparison is not paired
    for fold in range(num_folds):
        sets = {k: scan_sets[(k, fold)] for k in shortlist_sizes if (k, fold) in scan_sets}
        if len(sets) < 2:
            continue
        reference_k = sorted(sets)[0]
        reference = sets[reference_k]
        for k in sorted(sets):
            if sets[k] != reference:
                only_here = sorted(sets[k] - reference)[:3]
                only_there = sorted(reference - sets[k])[:3]
                problems.append('%s: fold %d, shortlist size %d covers different scans than size %d '
                                '(%d vs %d scans; only in %d: %s; only in %d: %s)'
                                % (label, fold, k, reference_k, len(sets[k]), len(reference),
                                   k, only_here, reference_k, only_there))
    return problems


def summarize_over_folds(fold_means, metrics):
    """
    Mean and SD over the fold means, one row per shortlist size
    """
    if fold_means.empty:
        return pd.DataFrame()
    rows = []
    for shortlist_size, group in fold_means.groupby('shortlist_size'):
        row = {'shortlist_size': shortlist_size, 'folds': len(group),
               'scans_per_fold': group['num_scans'].min() if group['num_scans'].nunique() == 1
               else '%d-%d' % (group['num_scans'].min(), group['num_scans'].max())}
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_sd'] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('shortlist_size').reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the shortlist size sweep")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="folder holding the networks")
    parser.add_argument("--method", type=str, default='two-step (ours)', choices=sorted(METHODS.keys()),
                        help="which network pair was evaluated")
    parser.add_argument("--shortlist_sizes", type=str2intlist, default='3,5,10',
                        help="comma separated candidate shortlist sizes that were evaluated")
    parser.add_argument("--topk_list", type=str2intlist, default='3,5,10',
                        help="comma separated top-k selections that were scored")
    parser.add_argument("--allow_incomplete", type=str2bool, default='false',
                        help="summarize even when folds are missing or the shortlist sizes cover different"
                             " scans. The result is not a fair comparison between sizes")
    parser.add_argument("--output", type=str, required=True, help="path of the comparison workbook to write")

    args = parser.parse_args()

    _, model2_dirnames = METHODS[args.method]
    num_folds = len(model2_dirnames)
    #top-1 lives in the wfss and accuracy columns, the rest carry their k in the name
    wfss_metrics = ['wfss'] + [f'wfss_top{k}' for k in args.topk_list]
    accuracy_metrics = ['accuracy'] + [f'accuracy_top{k}' for k in args.topk_list]
    metrics_opt = wfss_metrics + accuracy_metrics + ['inference_time']
    #on the suboptimal-only cases every selected frame is suboptimal at best, so wfss carries no information
    metrics_subopt = accuracy_metrics + ['inference_time']

    folds_opt, missing_opt, scans_opt = collect_fold_means(
        args.model_dir, model2_dirnames, args.shortlist_sizes, metrics_opt, subopt=False)
    folds_subopt, missing_subopt, scans_subopt = collect_fold_means(
        args.model_dir, model2_dirnames, args.shortlist_sizes, metrics_subopt, subopt=True)

    problems = (check_comparable(args.shortlist_sizes, num_folds, missing_opt, scans_opt, 'optimal-included')
                + check_comparable(args.shortlist_sizes, num_folds, missing_subopt, scans_subopt,
                                   'suboptimal-only'))
    if problems:
        print('the shortlist sizes are not comparable as they stand:\n')
        for problem in problems:
            print('  - %s' % problem)
        if not args.allow_incomplete:
            print('\nnothing was written. Finish the sweep, or pass --allow_incomplete true to summarize'
                  ' what is there, remembering that differences between sizes may come from the missing data.')
            sys.exit(1)
        print('\n--allow_incomplete was given, summarizing anyway. The comparison between sizes is not fair.\n')

    summary_opt = summarize_over_folds(folds_opt, metrics_opt)
    summary_subopt = summarize_over_folds(folds_subopt, metrics_subopt)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary_opt.to_excel(writer, sheet_name="summary_opt", index=False)
        summary_subopt.to_excel(writer, sheet_name="summary_subopt", index=False)
        folds_opt.to_excel(writer, sheet_name="folds_opt", index=False)
        folds_subopt.to_excel(writer, sheet_name="folds_subopt", index=False)

    print("\noptimal-included:")
    print(summary_opt.to_string(index=False))
    print("\nsuboptimal-only:")
    print(summary_subopt.to_string(index=False))
    print(f"\nSaved to {args.output}")
