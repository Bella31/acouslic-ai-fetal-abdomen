"""
Section: frame selection results - the main comparison of supervision strategies.

Reproduces the paper's main results table and Fig. 2:
top-1/3/5 accuracy on the Optimal-included and Suboptimal-only datasets, and WFSS@1/3/5 on the
Optimal-included dataset, for the five compared methods on the DenseNet backbone:

    One-step              RelNet alone (relevant vs irrelevant, as in the leading challenge methods)
    Opt-only              optimal vs everything else
    Three-class           irrelevant / optimal / suboptimal
    Two-step              RelNet top-5 shortlist re-ranked by OptNet (the proposed method)
    Two-step (relevant)   every frame RelNet calls relevant re-ranked by OptNet

Numbers are means over the five fold models (each Optimal-included scan is tested once, by the
model of its fold; every Suboptimal-only scan is tested by all five).  The workbook also holds the
SDs over folds, the per-fold means, and a per-scan sheet for the statistical tests.

    python -m part1_frame_classification.scripts.paper_evaluation.evaluate_main_comparison \
        --networks <subject_disjoint_manifest.csv>

Writes main_comparison.xlsx in --out_dir.  See README.md in this folder for the full workflow.
"""

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     os.pardir, os.pardir, os.pardir)))
from part1_frame_classification.scripts.paper_evaluation import common as C

SECTION = 'main_comparison'
METHOD_NAMES = ['One-step', 'Opt-only', 'Three-class', 'Two-step', 'Two-step (relevant)']


def tables(summary, archs):
    acc_rows, wfss_rows = [], []
    for arch in archs:
        for name in METHOD_NAMES:
            if summary.empty or not ((summary['method'] == name) & (summary['arch'] == arch)).any():
                continue
            acc = {'Method': name, 'Backbone': C.ARCH_LABEL[arch]}
            for cohort, label in (('opt', 'Optimal'), ('subopt', 'Suboptimal')):
                for k, col in (('1', 'accuracy'), ('3', 'accuracy_top3'), ('5', 'accuracy_top5')):
                    acc['%s Top-%s' % (label, k)] = C.lookup(summary, name, arch, cohort, col)
            acc_rows.append(acc)
            w = {'Method': name, 'Backbone': C.ARCH_LABEL[arch]}
            for k, col in (('1', 'wfss'), ('3', 'wfss_top3'), ('5', 'wfss_top5')):
                w['Optimal WFSS@%s' % k] = C.lookup(summary, name, arch, 'opt', col)
            wfss_rows.append(w)
    return pd.DataFrame(acc_rows), pd.DataFrame(wfss_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    C.add_common_args(ap, default_archs=('densenet',))
    ap.add_argument('--methods', nargs='+', default=METHOD_NAMES, choices=METHOD_NAMES)
    args = ap.parse_args()

    methods = [C.METHODS[n] for n in args.methods]
    jobs, expected, checks = C.prepare(args, SECTION, methods)
    if args.dry_run:
        return
    fold_df, scan_df, runs, more = C.collect(jobs, expected, args.subject_csv)
    summary = C.summarize_folds(fold_df, C.OPT_METRICS)
    acc, wfss = tables(summary, args.archs)
    print('\nAccuracy (mean over folds)')
    print(acc.round(3).to_string(index=False))
    print('\nWFSS, Optimal-included')
    print(wfss.round(3).to_string(index=False))
    C.write_workbook(args, SECTION, [('table_accuracy', acc), ('table_wfss', wfss), ('summary', summary),
                                     ('folds', fold_df), ('per_scan', scan_df), ('runs', runs)],
                     checks + more)


if __name__ == '__main__':
    main()
