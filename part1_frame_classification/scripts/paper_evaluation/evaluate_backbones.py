"""
Section: backbone ablation.

Reproduces the Supplement's ablation tables (tab:ablation_opt_supp and tab:ablation_subopt_supp)
and the backbone paragraph of the Results: One-step, Opt-only and Two-step on four backbones -
DenseNet, EfficientNet, ResNet and ConvNeXt.

    Optimal-included   top-1/3/5 accuracy, WFSS@1/3/5, inference time (s per scan)
    Suboptimal-only    top-1/3/5 accuracy, inference time

Means over the five fold models; SDs and per-fold values are in the workbook.  The DenseNet rows
reuse the main comparison's predictions files.

    python -m part1_frame_classification.scripts.paper_evaluation.evaluate_backbones \
        --networks <subject_disjoint_manifest.csv>

    # while the ConvNeXt networks are still training
    ... --archs densenet efficientnet resnet

Writes backbones.xlsx in --out_dir.
"""

import argparse
import os
import sys
from collections import OrderedDict

import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     os.pardir, os.pardir, os.pardir)))
from part1_frame_classification.scripts.paper_evaluation import common as C

SECTION = 'backbones'
METHOD_NAMES = ['One-step', 'Opt-only', 'Two-step']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    C.add_common_args(ap, default_archs=C.ARCHS)
    args = ap.parse_args()

    methods = [C.METHODS[n] for n in METHOD_NAMES]
    jobs, expected, checks = C.prepare(args, SECTION, methods)
    if args.dry_run:
        return
    fold_df, scan_df, runs, more = C.collect(jobs, expected, args.subject_csv)
    summary = C.summarize_folds(fold_df, C.OPT_METRICS)

    tables = {}
    for cohort in C.COHORTS:
        cols = [('Top-1', 'accuracy'), ('Top-3', 'accuracy_top3'), ('Top-5', 'accuracy_top5')]
        if cohort == 'opt':
            cols += [('WFSS@1', 'wfss'), ('WFSS@3', 'wfss_top3'), ('WFSS@5', 'wfss_top5')]
        cols += [('Time (s)', 'inference_time')]
        rows = []
        for name in METHOD_NAMES:
            for arch in args.archs:
                row = OrderedDict([('Method', name), ('Backbone', C.ARCH_LABEL[arch])])
                for label, col in cols:
                    row[label] = C.lookup(summary, name, arch, cohort, col)
                rows.append(row)
        tables[cohort] = pd.DataFrame(rows)
        print('\n%s (mean over folds)' % C.COHORT_LABEL[cohort])
        print(tables[cohort].round(3).to_string(index=False))
    C.write_workbook(args, SECTION, [('table_optimal_included', tables['opt']),
                                     ('table_suboptimal_only', tables['subopt']), ('summary', summary),
                                     ('folds', fold_df), ('per_scan', scan_df), ('runs', runs)],
                     checks + more)


if __name__ == '__main__':
    main()
