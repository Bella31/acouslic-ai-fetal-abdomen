"""
Section: sensitivity to the candidate shortlist size K.

Reproduces the paper's Table 2: the two-step method with RelNet shortlists of K = 3, 5 and 10
frames re-ranked by OptNet, on the DenseNet backbone.

    Coverage   share of scans whose shortlist holds at least one target frame (optimal for
               Optimal-included scans, suboptimal for Suboptimal-only scans).  Re-ranking does
               not change which frames are in the shortlist, so this is the top-K accuracy of the
               K run, and equals RelNet's own top-K.
    Top-1/3    final selection accuracy after OptNet re-ranks the shortlist
    WFSS@1/3   Optimal-included only (it equals accuracy on Suboptimal-only scans)

Values are mean +- SD across the five fold models, the SD taken over the five fold means.

The K files are separate from the main two-step predictions (suffix _k3/_k5/_k10, scored at
top-3/5/10), so K=5 is evaluated again here; it gives the same selections as the main two-step.

    python -m part1_frame_classification.scripts.paper_evaluation.evaluate_shortlist_size \
        --networks <subject_disjoint_manifest.csv>

Writes shortlist_size.xlsx in --out_dir.
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

SECTION = 'shortlist_size'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    C.add_common_args(ap, default_archs=('densenet',))
    ap.add_argument('--shortlist_sizes', nargs='+', type=int, default=[3, 5, 10])
    args = ap.parse_args()

    methods = [C.shortlist_method(k) for k in args.shortlist_sizes]
    jobs, expected, checks = C.prepare(args, SECTION, methods)
    if args.dry_run:
        return
    fold_df, scan_df, runs, more = C.collect(jobs, expected, args.subject_csv)
    summary = C.summarize_folds(fold_df, C.OPT_METRICS)

    numeric, display = [], []
    for arch in args.archs:
        for cohort in C.COHORTS:
            for k in args.shortlist_sizes:
                name = 'Two-step K=%d' % k
                cols = [('Coverage', 'accuracy_top%d' % k), ('Top-1', 'accuracy'), ('Top-3', 'accuracy_top3')]
                if cohort == 'opt':
                    cols += [('WFSS@1', 'wfss'), ('WFSS@3', 'wfss_top3')]
                n = OrderedDict([('Dataset', C.COHORT_LABEL[cohort]), ('Backbone', C.ARCH_LABEL[arch]), ('K', k)])
                d = OrderedDict(n)
                for label, col in cols:
                    m = C.lookup(summary, name, arch, cohort, col)
                    s = C.lookup(summary, name, arch, cohort, col + '_sd')
                    n[label] = m
                    n[label + ' SD'] = s
                    d[label] = C.fmt_mean_sd(m, s)
                if cohort == 'subopt':
                    d['WFSS@1'] = d['WFSS@3'] = '---'
                numeric.append(n)
                display.append(d)
    numeric, display = pd.DataFrame(numeric), pd.DataFrame(display)
    print('\nTable 2 (mean +- SD over the five fold models)')
    print(display.to_string(index=False))
    C.write_workbook(args, SECTION, [('table', display), ('table_numeric', numeric), ('summary', summary),
                                     ('folds', fold_df), ('per_scan', scan_df), ('runs', runs)],
                     checks + more)


if __name__ == '__main__':
    main()
