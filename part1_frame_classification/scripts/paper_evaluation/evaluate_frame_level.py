"""
Section: frame-level classification - precision, recall and F1.

Reproduces the paper's F1 table, the precision-recall scatter (Fig. 3) and the Supplement's
three-class confusion matrices, for the DenseNet backbone:

    optimal vs non-optimal       Optimal-included dataset
    relevant vs irrelevant       both datasets (optimal and suboptimal frames grouped as relevant)

The proposed method enters as "Two-step (relevant)", which forwards every frame RelNet calls
relevant to OptNet: the standard two-step method only re-labels the five shortlisted frames, so it
has no meaningful scan-wide recall (it is still listed, for completeness).

Aggregation:
  - Optimal-included: the per-scan 3x3 confusion matrices are summed over all scans of all folds,
    merged to the 2-class task, and precision/recall/F1 computed once from the pooled counts
  - Suboptimal-only: per-scan precision/recall/F1 are averaged within each fold model and then
    over the five models

F1 of a scan with target frames but no true positive is 0, not undefined.  A predictions file may
leave F1 empty there, and a mean that skips those scans is too high, so the per-scan F1 is
recomputed here from the per-scan precision and recall; the checks sheet says how many scans this
affected.

These come from the same predictions files as the main comparison, so after
evaluate_main_comparison.py has run this needs no inference at all.

    python -m part1_frame_classification.scripts.paper_evaluation.evaluate_frame_level \
        --networks <subject_disjoint_manifest.csv>

Writes frame_level.xlsx in --out_dir.
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     os.pardir, os.pardir, os.pardir)))
from part1_frame_classification.scripts.paper_evaluation import common as C

SECTION = 'frame_level'
METHOD_NAMES = ['Opt-only', 'Three-class', 'One-step', 'Two-step (relevant)', 'Two-step']
TABLE_METHODS = ['Opt-only', 'Three-class', 'One-step', 'Two-step (relevant)']
CLASSES = ['irrelevant', 'optimal', 'suboptimal']


def read_conf(path):
    cm = pd.read_excel(path, sheet_name='conf_matrix')
    if 'True' in cm.columns:
        cm = cm.set_index('True')
    cm.index = cm.index.astype(str)
    return cm.reindex(index=CLASSES, columns=CLASSES).fillna(0).astype(float)


def prf(tp, fp, fn):
    """As precision_recall_f1 in evaluate_frame_selection.py: F1 = 0 with no true positive."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    else:
        f = 0.0 if (tp + fp + fn) > 0 else np.nan
    return p, r, f


def per_scan_f1(pr):
    """Relevant-vs-irrelevant F1 per scan from its precision and recall; 0 when both are 0 and the
    scan has relevant frames.  Returns (f1, number of scans whose file F1 was empty but is 0)."""
    p = pd.to_numeric(pr['precision_opt_subopt'], errors='coerce')
    r = pd.to_numeric(pr['recall_opt_subopt'], errors='coerce')
    has_target = (pd.to_numeric(pr['num_opt'], errors='coerce') +
                  pd.to_numeric(pr['num_subopt'], errors='coerce')) > 0
    denom = (p + r).where((p + r) > 0)
    f1 = (2 * p * r / denom).where(denom.notna(), np.where(has_target, 0.0, np.nan))
    was_empty = pd.to_numeric(pr['F1_opt_subopt'], errors='coerce').isna()
    return f1, int((was_empty & f1.notna()).sum())


def tasks_from_conf(cm):
    """Rows are the true class, columns the predicted one (as the evaluator writes them)."""
    o, i, s = 'optimal', 'irrelevant', 'suboptimal'
    opt = prf(tp=cm.loc[o, o], fp=cm.loc[i, o] + cm.loc[s, o], fn=cm.loc[o, i] + cm.loc[o, s])
    rel_tp = cm.loc[o, o] + cm.loc[o, s] + cm.loc[s, o] + cm.loc[s, s]
    rel = prf(tp=rel_tp, fp=cm.loc[i, o] + cm.loc[i, s], fn=cm.loc[o, i] + cm.loc[s, i])
    return opt, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    C.add_common_args(ap, default_archs=('densenet',))
    args = ap.parse_args()

    methods = [C.METHODS[n] for n in METHOD_NAMES]
    jobs, expected, checks = C.prepare(args, SECTION, methods)
    if args.dry_run:
        return
    _, _, runs, more = C.collect(jobs, expected, None)

    pooled = OrderedDict()          # (method, arch, cohort) -> summed confusion matrix
    subopt_rows = []
    zero_f1 = OrderedDict()         # (method, arch) -> scans whose empty F1 is counted as 0
    for j in jobs:
        if not (j.output and os.path.exists(j.output)):
            continue
        key = (j.method.name, j.arch, j.cohort)
        cm = read_conf(j.output)
        pooled[key] = cm if key not in pooled else pooled[key] + cm
        if j.cohort == 'subopt':
            pr = pd.read_excel(j.output, sheet_name='precision_recall')
            f1, n_zero = per_scan_f1(pr)
            zero_f1[(j.method.name, j.arch)] = zero_f1.get((j.method.name, j.arch), 0) + n_zero
            subopt_rows.append(OrderedDict([('method', j.method.name), ('arch', j.arch), ('fold', j.fold)] +
                                           [(c, pd.to_numeric(pr[c], errors='coerce').mean())
                                            for c in ('precision_opt_subopt', 'recall_opt_subopt')] +
                                           [('F1_opt_subopt', f1.mean()), ('scans_F1_counted_as_0', n_zero)]))

    opt_rows, conf_rows = [], []
    for (name, arch, cohort), cm in pooled.items():
        for t in CLASSES:
            conf_rows.append(OrderedDict([('method', name), ('arch', arch), ('cohort', cohort), ('true', t)] +
                                         [('pred_' + c, int(cm.loc[t, c])) for c in CLASSES]))
        if cohort != 'opt':
            continue
        (po, ro, fo), (pr_, rr, fr) = tasks_from_conf(cm)
        opt_rows.append(OrderedDict([('method', name), ('arch', arch),
                                     ('precision_optimal', po), ('recall_optimal', ro), ('F1_optimal', fo),
                                     ('precision_relevant', pr_), ('recall_relevant', rr), ('F1_relevant', fr)]))
    pr_opt = pd.DataFrame(opt_rows)
    sub_folds = pd.DataFrame(subopt_rows)
    pr_subopt = (sub_folds.groupby(['method', 'arch'], sort=False)
                 [['precision_opt_subopt', 'recall_opt_subopt', 'F1_opt_subopt']].mean().reset_index()
                 if len(sub_folds) else pd.DataFrame())
    if len(pr_subopt):
        pr_subopt.columns = ['method', 'arch', 'precision_relevant', 'recall_relevant', 'F1_relevant']

    table = []
    for arch in args.archs:
        for name in TABLE_METHODS:
            row = OrderedDict([('Method', name), ('Backbone', C.ARCH_LABEL[arch])])
            o = pr_opt[(pr_opt['method'] == name) & (pr_opt['arch'] == arch)] if len(pr_opt) else pr_opt
            s = (pr_subopt[(pr_subopt['method'] == name) & (pr_subopt['arch'] == arch)]
                 if len(pr_subopt) else pr_subopt)
            row['Optimal vs suboptimal, Optimal-included F1'] = o['F1_optimal'].iloc[0] if len(o) else np.nan
            row['Relevant vs irrelevant, Optimal-included F1'] = o['F1_relevant'].iloc[0] if len(o) else np.nan
            row['Relevant vs irrelevant, Suboptimal-only F1'] = s['F1_relevant'].iloc[0] if len(s) else np.nan
            table.append(row)
    table = pd.DataFrame(table)
    print('\nF1 (table)')
    print(table.round(3).to_string(index=False))
    print('\nprecision / recall, Optimal-included (pooled over all scans)')
    print(pr_opt.round(3).to_string(index=False))
    more = more + [('note', '%s %s: %d Suboptimal-only scan-fold values had no true positive and an empty F1 '
                             'in the predictions file; counted as F1 = 0' % (name, C.ARCH_LABEL[arch], n))
                   for (name, arch), n in zero_f1.items() if n]
    C.write_workbook(args, SECTION, [('table_f1', table), ('pr_opt_pooled', pr_opt),
                                     ('pr_subopt_mean', pr_subopt), ('pr_subopt_folds', sub_folds),
                                     ('confusion_pooled', pd.DataFrame(conf_rows)), ('runs', runs)],
                     checks + more)


if __name__ == '__main__':
    main()
