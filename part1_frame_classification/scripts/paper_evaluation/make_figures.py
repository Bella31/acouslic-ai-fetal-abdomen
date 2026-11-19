"""
Figures of the paper, drawn from the evaluation workbooks with the plotting functions in
part1_frame_classification/scripts/evaluation_plots.

  Fig2.pdf              top-frame accuracy at Top-1/3/5, mean and SD over the five fold models, both
                        datasets                                  main_comparison.xlsx, sheet folds
  Fig3.pdf              precision-recall scatter: optimal classification (Optimal-included, pooled
                        over all scans) and relevant classification (both datasets)   frame_level.xlsx
  FigureS1.pdf / .png   confusion matrices of Three-class and Two-step (relevant), Optimal-included,
                        pooled over all scans                                         frame_level.xlsx
  FigureS2.pdf          WFSS score distributions at Top-1/3/5, Optimal-included
                                                                  main_comparison.xlsx, sheet per_scan

DenseNet only. Method names are mapped to figure labels by LABEL (Two-step is "two-step (ours)").
A method missing from a workbook is left out of the figure, with a message; a figure that cannot
be drawn without it (the confusion matrices need Three-class) is skipped.

Usage, from the repository root:
  python -m part1_frame_classification.scripts.paper_evaluation.make_figures --networks <networks table>
  python -m part1_frame_classification.scripts.paper_evaluation.make_figures --out_dir <workbook folder>
Writes to <workbook folder>/figures/; figures already there are moved to figures/previous_versions/.
run_paper_evaluation.py runs it after the frame-level section.
"""

import argparse
import glob
import os
import shutil
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')   # write files, never open windows
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     os.pardir, os.pardir, os.pardir)))
from part1_frame_classification.scripts.evaluation_plots import confusion_matrices as conf_plot
from part1_frame_classification.scripts.evaluation_plots import line_plot_2_panel as bar_plot
from part1_frame_classification.scripts.evaluation_plots import precision_recall_scatter as pr_plot
from part1_frame_classification.scripts.evaluation_plots import wfss_freq as wfss_plot

ARCH = 'densenet'
CLASSES = ['irrelevant', 'optimal', 'suboptimal']
# evaluation method name -> figure label
LABEL = OrderedDict([('One-step', 'One-step'), ('Opt-only', 'Opt-only'), ('Three-class', 'Three-class'),
                     ('Two-step', 'two-step (ours)'), ('Two-step (relevant)', 'two-step (relevant)')])


def say(text=''):
    print(text, flush=True)


def note_missing(figure, wanted, present):
    missing = [m for m in wanted if m not in set(present)]
    if missing:
        say('  %s: %s not in the workbook, left out' % (figure, ', '.join(missing)))
    return missing


def fig2(main_wb, out):
    folds = pd.read_excel(main_wb, sheet_name='folds')
    folds = folds[(folds['arch'] == ARCH) & folds['method'].isin(LABEL)]
    note_missing('Fig2', LABEL, folds['method'])
    parts = []
    for cohort, dataset in (('opt', 'optimal'), ('subopt', 'suboptimal')):
        d = folds[folds['cohort'] == cohort].copy()
        d['name'] = d['method'].map(LABEL)
        long = bar_plot.wide_to_long_topk(d, method_col='name', fold_col='fold')
        long['dataset'] = dataset
        parts.append(long)
    bar_plot.plot_topk_grouped_bars_mean_sd(pd.concat(parts, ignore_index=True), results_folder=out,
                                            out_name='Fig2.pdf')
    plt.close('all')
    return ['Fig2.pdf']


def fig3(frame_wb, out):
    sheets = pd.read_excel(frame_wb, sheet_name=None)
    pooled = sheets['pr_opt_pooled']
    pooled = pooled[pooled['arch'] == ARCH].set_index('method')
    subopt = sheets['pr_subopt_mean']
    subopt = subopt[subopt['arch'] == ARCH].set_index('method')

    # left panel: optimal-frame classification, Optimal-included dataset
    left = ['Opt-only', 'Three-class', 'One-step', 'Two-step (relevant)']
    note_missing('Fig3 (optimal)', left, pooled.index)
    df_opt = pd.DataFrame([OrderedDict([('method', LABEL[m]),
                                        ('precision_opt_only', pooled.loc[m, 'precision_optimal']),
                                        ('recall_opt_only', pooled.loc[m, 'recall_optimal'])])
                           for m in left if m in pooled.index])
    # right panel: relevant-frame classification, both datasets; One-step and Two-step share RelNet,
    # hence one point
    right = [('Opt-only', 'Opt-only'), ('Three-class', 'Three-class'), ('One-step', 'One- and two-step')]
    note_missing('Fig3 (relevant)', [m for m, _ in right], [m for m in pooled.index if m in subopt.index])
    df_rel = pd.DataFrame([OrderedDict([('method', label),
                                        ('precision_rel_Opt_data', pooled.loc[m, 'precision_relevant']),
                                        ('recall_rel_opt_data', pooled.loc[m, 'recall_relevant']),
                                        ('precision_rel_subopt_data', subopt.loc[m, 'precision_relevant']),
                                        ('recall_rel_subopt_data', subopt.loc[m, 'recall_relevant'])])
                           for m, label in right if m in pooled.index and m in subopt.index])
    fig = pr_plot.plot_pr(df_rel, df_opt)
    fig.savefig(os.path.join(out, 'Fig3.pdf'), bbox_inches='tight')
    plt.close('all')
    return ['Fig3.pdf']


def fig_s1(frame_wb, out):
    conf = pd.read_excel(frame_wb, sheet_name='confusion_pooled')
    conf = conf[(conf['arch'] == ARCH) & (conf['cohort'] == 'opt')]
    wanted = ['Three-class', 'Two-step (relevant)']
    if note_missing('FigureS1', wanted, conf['method']):
        say('  FigureS1: skipped (it needs both methods)')
        return []
    methods, matrices = [], []
    for m in wanted:
        d = conf[conf['method'] == m].set_index('true').reindex(CLASSES)
        methods.append(LABEL[m])
        matrices.append(d[['pred_' + c for c in CLASSES]].values.astype(float))
    conf_plot.plot_selected_confusions(methods, matrices, CLASSES, methods,
                                       out_basename=os.path.join(out, 'FigureS1'))
    plt.close('all')
    return ['FigureS1.pdf', 'FigureS1.png']


def fig_s2(main_wb, out):
    cols = OrderedDict([('Top-1', 'wfss'), ('Top-3', 'wfss_top3'), ('Top-5', 'wfss_top5')])
    scans = pd.read_excel(main_wb, sheet_name='per_scan')
    scans = scans[(scans['arch'] == ARCH) & (scans['cohort'] == 'opt') & scans['method'].isin(LABEL)]
    note_missing('FigureS2', LABEL, scans['method'])
    data = pd.DataFrame({'Algorithm': scans['method'].map(LABEL).values})
    for c in cols.values():
        data[c] = pd.to_numeric(scans[c], errors='coerce').round(3).values
    cfg = wfss_plot.WFSSPlotConfig(excel_path=main_wb, wfss_cols=cols, round_decimals=3,
                                   colormap='tab10', figsize=(7.6, 8.8))
    wfss_plot.plot_wfss_three_panels(cfg, savepath=os.path.join(out, 'FigureS2.pdf'), data=data)
    plt.close('all')
    return ['FigureS2.pdf']


def main():
    ap = argparse.ArgumentParser(description='Draw the paper figures from the evaluation workbooks.')
    ap.add_argument('--networks', help='networks table, to find the workbook folder as the other sections do')
    ap.add_argument('--out_dir', help='folder with main_comparison.xlsx and frame_level.xlsx')
    ap.add_argument('--main_comparison', help='default: <out_dir>/main_comparison.xlsx')
    ap.add_argument('--frame_level', help='default: <out_dir>/frame_level.xlsx')
    args = ap.parse_args()

    folder = args.out_dir
    if not folder and args.networks:
        stem = os.path.splitext(os.path.basename(args.networks))[0]
        folder = os.path.join(os.path.dirname(os.path.abspath(args.networks)), 'paper_evaluation_' + stem)
    if not folder and not (args.main_comparison and args.frame_level):
        ap.error('give --out_dir or --networks (or both workbook paths)')
    main_wb = args.main_comparison or os.path.join(folder, 'main_comparison.xlsx')
    frame_wb = args.frame_level or os.path.join(folder, 'frame_level.xlsx')
    out = os.path.join(os.path.dirname(os.path.abspath(main_wb)), 'figures')

    if not os.path.isdir(out):
        os.makedirs(out)
    old = [f for f in glob.glob(os.path.join(out, '*')) if os.path.isfile(f)]
    if old:
        prev = os.path.join(out, 'previous_versions', time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(prev)
        for f in old:
            shutil.move(f, prev)

    written, failed = [], []
    for name, make, workbook in (('Fig2', fig2, main_wb), ('Fig3', fig3, frame_wb),
                                 ('FigureS1', fig_s1, frame_wb), ('FigureS2', fig_s2, main_wb)):
        say('%s  from %s' % (name, os.path.basename(workbook)))
        if not os.path.exists(workbook):
            say('  skipped: %s not found' % workbook)
            failed.append(name)
            continue
        try:
            written += make(workbook, out)
        except Exception as e:                      # one broken figure should not stop the others
            say('  FAILED: %s: %s' % (type(e).__name__, e))
            failed.append(name)
    say('\nwrote %s to %s' % (', '.join(written) or 'nothing', out))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
