"""
Run the evaluation sections of the paper, in order, for one set of trained networks.

  main_comparison   main results table and WFSS (DenseNet, five methods)
  statistics        Two-step vs Opt-only tests (subject-clustered bootstrap, Holm), from the main comparison
  frame_level       F1 table, precision-recall, confusion matrices (reuses the predictions above)
  figures           Fig2, Fig3, FigureS1, FigureS2 from the two workbooks above (make_figures.py)
  shortlist_size    Table 2, shortlist sizes K = 3, 5, 10
  backbones         Supplement ablation tables, four backbones

Each section runs only the evaluations whose prediction files are missing, then writes its workbook
(by default to paper_evaluation_<networks table name>/ next to the networks table). A section that fails
does not stop the next one, except that statistics and figures are skipped when a section whose
workbook they read failed in the same run. The script exits with status 1 at the end if any section
failed. Running it again continues where it stopped.

A subject shared between a network's training/validation scans and the scans it is evaluated on stops
the run, so the subject table is required.

Usage, from the repository root (default paths are set at the top of this script; see --help):

  # DenseNet sections for the subject-disjoint networks (inference, tables, statistics, figures):
  # first see what would run and how long
  python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py \
      --sections main_comparison statistics frame_level figures shortlist_size --dry_run
  nohup python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py \
      --sections main_comparison statistics frame_level figures shortlist_size \
      > <output folder>/eval_densenet.log 2>&1 &

  # backbone tables, once the other backbones are trained (--backbones selects a subset)
  nohup python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py \
      --sections backbones > <output folder>/eval_backbones.log 2>&1 &

  # the reference networks in published_networks.csv (trained on per-scan splits that share
  # subjects, so the overlap is listed in the checks sheet instead of stopping the run)
  python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py \
      --networks part1_frame_classification/scripts/paper_evaluation/published_networks.csv \
      --allow_subject_overlap

  # tables only, from prediction files that already exist (no GPU, no scans needed)
  python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --summarize_only
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
GPU_OUT = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output'
GPU_SCANS = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'

SECTIONS = ['main_comparison', 'statistics', 'frame_level', 'figures', 'shortlist_size', 'backbones']   # run order
# sections that only read workbooks written by earlier sections: module name and the sections they read
READERS = {'statistics': ('evaluate_statistics', ['main_comparison']),
           'figures': ('make_figures', ['main_comparison', 'frame_level'])}
ARCHS = ['densenet', 'efficientnet', 'resnet', 'convnext']


def say(text=''):
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run the evaluation sections of the paper, in order.')
    parser.add_argument('--sections', nargs='+', default=SECTIONS, choices=SECTIONS,
                        help='default: all of them')
    parser.add_argument('--backbones', nargs='+', default=ARCHS, choices=ARCHS,
                        help='backbones in the backbones section; default: all four')
    parser.add_argument('--dry_run', action='store_true',
                        help='run the checks and print what would be evaluated, with a time estimate')
    parser.add_argument('--summarize_only', action='store_true',
                        help='no inference: write the tables from the prediction files that exist')
    parser.add_argument('--allow_subject_overlap', action='store_true',
                        help='list subject overlap in the checks sheet instead of stopping (networks '
                             'trained on per-scan splits, e.g. published_networks.csv)')
    parser.add_argument('--out', default=GPU_OUT,
                        help='folder with the network folder, labels, Suboptimal-only list and subject table; '
                             'default: %(default)s')
    parser.add_argument('--networks', help='networks table; default: <log_dir>/subject_disjoint_manifest.csv')
    parser.add_argument('--log_dir', help='folder with the run folders; default: <out>/network')
    parser.add_argument('--out_dir', help='where the workbooks go; default: next to the networks table')
    parser.add_argument('--scan_dir', default=GPU_SCANS, help='the .mha scans; default: %(default)s')
    parser.add_argument('--labels_path', help='frame labels; default: <out>/labels.csv')
    parser.add_argument('--subopt_list', help='the 113 Suboptimal-only scans; default: <out>/subopt_cases.txt')
    parser.add_argument('--subject_csv', help='default: <out>/fetal_abdominal_circumferences_per_sweep.csv')
    parser.add_argument('--splits_root',
                        help='only when the split lists are not at the data_dir recorded in each run\'s config.json')
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(args.out, 'network')
    networks = args.networks or os.path.join(log_dir, 'subject_disjoint_manifest.csv')
    labels = args.labels_path or os.path.join(args.out, 'labels.csv')
    subopt_list = args.subopt_list or os.path.join(args.out, 'subopt_cases.txt')
    subject_csv = args.subject_csv or os.path.join(args.out, 'fetal_abdominal_circumferences_per_sweep.csv')
    for path, what in ((networks, 'networks table'), (subject_csv, 'subject table')):
        if not os.path.exists(path):
            say('ERROR: %s not found: %s' % (what, path))
            sys.exit(1)

    common = ['--networks', networks, '--log_dir', log_dir, '--scan_dir', args.scan_dir,
              '--labels_path', labels, '--subopt_list', subopt_list, '--subject_csv', subject_csv,
              '--python', sys.executable]
    if args.out_dir:
        common += ['--out_dir', args.out_dir]
    if args.splits_root:
        common += ['--splits_root', args.splits_root]
    for flag in ('dry_run', 'summarize_only', 'allow_subject_overlap'):
        if getattr(args, flag):
            common.append('--' + flag)

    failed = []
    for section in [s for s in SECTIONS if s in args.sections]:
        say('\n################ %s  (%s)' % (section, time.strftime('%Y-%m-%d %H:%M:%S')))
        if section in READERS:
            module, sources = READERS[section]
            command = [sys.executable, '-m', 'part1_frame_classification.scripts.paper_evaluation.' + module]
            # these sections read workbooks only, so they need only the workbook folder
            if args.dry_run:
                say('runs after %s, from the workbook(s); takes under a minute' % ' and '.join(sources))
                continue
            # a workbook its source section failed to write in this run would be missing or out of date
            failed_sources = [s for s in sources if s in failed]
            if failed_sources:
                say('!! %s skipped: %s did not finish cleanly in this run' % (section, ', '.join(failed_sources)))
                failed.append(section)
                continue
            arguments = ['--networks', networks] + (['--out_dir', args.out_dir] if args.out_dir else [])
        else:
            command = [sys.executable, '-m', 'part1_frame_classification.scripts.paper_evaluation.evaluate_' + section]
            arguments = common + (['--archs'] + args.backbones if section == 'backbones' else [])
        if subprocess.call(command + arguments, cwd=REPO_ROOT) != 0:
            say('!! %s did not finish cleanly' % section)
            failed.append(section)

    say('\ndone (%s)%s' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                          ', sections that failed: ' + ', '.join(failed) if failed else ', all sections finished'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
