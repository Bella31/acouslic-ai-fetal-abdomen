"""
Shared machinery for the paper's evaluation scripts.

Each evaluate_<section>.py script in this folder reproduces one part of the paper.  They all work
the same way:

    1. read a NETWORKS TABLE - which trained run folder is which network, for each backbone and fold
    2. check every run's config.json really is that network, and that the test cohorts are what the
       paper says (paired across methods, disjoint across folds, Suboptimal-only never trained on)
    3. run evaluate_frame_selection.py for every (method, backbone, fold, cohort) whose predictions
       file does not exist yet
    4. read the predictions files back and write that section's tables to one workbook

The networks table is a CSV with at least the columns  run_dir, arch, network, fold :

    run_dir   run folder name inside --log_dir (as created by train_classification.py)
    arch      densenet | resnet | efficientnet | convnext
    network   relnet | optnet | optonly | threeclass
    fold      0..4

train_subject_disjoint.py writes exactly this file as it trains (subject_disjoint_manifest.csv).
published_networks.csv in this folder lists reference networks trained on per-scan splits; their
splits share subjects, so they are evaluated with --allow_subject_overlap.

Predictions files go inside the run folder of the network that owns the result:

    <log_dir>/<run>/results/predictions_<run>[_subopt][<suffix>].xlsx

and they are shared between sections: the DenseNet two-step predictions serve the main comparison,
the frame-level table and the backbone table, and are computed once.  An existing file is never
recomputed unless --overwrite is given, so an interrupted run can simply be restarted.

Nothing here imports torch: the inference happens in evaluate_frame_selection.py, run as a
subprocess, so the summaries can also be produced on a machine without a GPU.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
EVALUATOR_MODULE = 'part1_frame_classification.scripts.evaluate_frame_selection'
EVALUATOR_FILE = os.path.join(REPO_ROOT, 'part1_frame_classification', 'scripts', 'evaluate_frame_selection.py')

# default paths; override them on the command line
GPU_OUT = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output'
DEFAULT_LOG_DIR = GPU_OUT + '/network'
DEFAULT_SCAN_DIR = ('/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/'
                    'stacked_fetal_ultrasound')
DEFAULT_LABELS = GPU_OUT + '/labels.csv'
DEFAULT_SUBOPT_LIST = GPU_OUT + '/subopt_cases.txt'

ARCHS = ('densenet', 'efficientnet', 'resnet', 'convnext')        # the Supplement's order
ARCH_LABEL = {'densenet': 'DenseNet', 'efficientnet': 'EfficientNet', 'resnet': 'ResNet',
              'convnext': 'ConvNeXt'}
FOLDS = (0, 1, 2, 3, 4)
COHORTS = ('opt', 'subopt')
COHORT_LABEL = {'opt': 'Optimal-included', 'subopt': 'Suboptimal-only'}

# what each trained network is, as recorded in its config.json
NETWORK_TYPES = {
    'relnet':     dict(num_classes=2, opt_only=False, pos_only=False, family='opt_using_subopt'),
    'optnet':     dict(num_classes=2, opt_only=False, pos_only=True,  family='opt_using_subopt'),
    'optonly':    dict(num_classes=2, opt_only=True,  pos_only=False, family='opt_no_subopt'),
    'threeclass': dict(num_classes=3, opt_only=False, pos_only=False, family='opt_using_subopt'),
}

# How each compared method is evaluated.
#   first / second  network types; second=None is a single-network method
#   shortlist       None for a single network; K = the first network's top-K go to the second
#                   network (the paper's two-step, K=5); 0 = every frame the first network calls
#                   relevant goes to the second network ("Two-step (relevant)")
#   topk            top-k selections scored besides the single best frame
#   suffix          predictions file suffix
Method = namedtuple('Method', 'name first second num_classes shortlist topk suffix')
METHODS = OrderedDict((m.name, m) for m in [
    Method('One-step',            'relnet',     None,     2, None, (3, 5), ''),
    Method('Opt-only',            'optonly',    None,     2, None, (3, 5), ''),
    Method('Three-class',         'threeclass', None,     3, None, (3, 5), ''),
    Method('Two-step',            'relnet',     'optnet', 2, 5,    (3, 5), ''),
    Method('Two-step (relevant)', 'relnet',     'optnet', 2, 0,    (3, 5), '_all_rel'),
])


def shortlist_method(k):
    """The two-step method with candidate shortlist size k (Table 2), scored at top-3/5/10."""
    return Method('Two-step K=%d' % k, 'relnet', 'optnet', 2, int(k), (3, 5, 10), '_k%d' % int(k))


# rough per-scan inference seconds, plus reading the .mha; used only for the time estimate
# printed before running
SECONDS_PER_SCAN = {'densenet': 3.0, 'resnet': 2.4, 'efficientnet': 2.6, 'convnext': 8.7}
LOAD_SECONDS = 1.0

Job = namedtuple('Job', 'method arch fold cohort first_run second_run owner_run output')


# ------------------------------------------------------------------------------ arguments ----

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_common_args(ap, default_archs):
    ap.add_argument('--networks', required=True,
                    help='networks table CSV (run_dir, arch, network, fold): the manifest written by '
                         'train_subject_disjoint.py, or published_networks.csv')
    ap.add_argument('--log_dir', default=DEFAULT_LOG_DIR, help='folder holding the run folders')
    ap.add_argument('--out_dir', default=None,
                    help='where the summary workbooks go (default: <folder of --networks>/'
                         'paper_evaluation_<name of --networks>)')
    ap.add_argument('--archs', nargs='+', default=list(default_archs), choices=ARCHS,
                    help='backbones to include (default: %s)' % ' '.join(default_archs))
    ap.add_argument('--scan_dir', default=DEFAULT_SCAN_DIR, help='folder of the full .mha scans')
    ap.add_argument('--labels_path', default=DEFAULT_LABELS,
                    help='frame-level labels CSV with EVERY frame (Filename, Frame, Label)')
    ap.add_argument('--subopt_list', default=DEFAULT_SUBOPT_LIST,
                    help='the 113 Suboptimal-only scan ids, one per line')
    ap.add_argument('--splits_root', default=None,
                    help='read the fold split lists from <splits_root>/<family>/<fold>/data_split '
                         'instead of <data_dir>/data_split of each run (for running elsewhere)')
    ap.add_argument('--subject_csv', default=None,
                    help='per-sweep circumference CSV with uuid and subject_id.  Required: nothing runs if '
                         'a subject is shared between a network\'s training/validation scans and the scans '
                         'it is evaluated on, or between its training and validation scans, or if a scan '
                         'has no subject_id.  Also adds subject_id to the per-scan sheet')
    ap.add_argument('--allow_subject_overlap', action='store_true',
                    help='report subject overlap instead of stopping (for networks trained on per-scan '
                         'splits, e.g. published_networks.csv); the overlap is listed in the checks sheet')
    ap.add_argument('--python', default=sys.executable, help='interpreter that runs the evaluator')
    ap.add_argument('--use_gpu', type=str2bool, default=True)
    ap.add_argument('--summarize_only', action='store_true',
                    help='do not run any inference; summarize the predictions files that exist')
    ap.add_argument('--overwrite', action='store_true',
                    help='recompute predictions that already exist (the evaluator moves the existing '
                         'file to previous_versions/)')
    ap.add_argument('--dry_run', action='store_true',
                    help='run the checks and print what would be evaluated, with a time estimate')
    ap.add_argument('--allow_incomplete', action='store_true',
                    help='write the tables even if files are missing or cohorts do not match; the '
                         'workbook is then named *_INCOMPLETE.xlsx')
    return ap


def resolve_out_dir(args):
    if args.out_dir:
        return args.out_dir
    stem = os.path.splitext(os.path.basename(args.networks))[0]
    return os.path.join(os.path.dirname(os.path.abspath(args.networks)), 'paper_evaluation_' + stem)


# --------------------------------------------------------------------------- networks table ---

def load_networks(path):
    """{(arch, network, fold): run_dir}, refusing duplicates and unknown values."""
    if not os.path.exists(path):
        raise SystemExit('networks table not found: %s' % path)
    table = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        need = {'run_dir', 'arch', 'network', 'fold'}
        if not need.issubset(reader.fieldnames or []):
            raise SystemExit('%s needs the columns %s (found %s)'
                             % (path, ', '.join(sorted(need)), ', '.join(reader.fieldnames or [])))
        for i, row in enumerate(reader, start=2):
            arch, net = row['arch'].strip().lower(), row['network'].strip().lower()
            run, fold = row['run_dir'].strip(), row['fold'].strip()
            if arch not in ARCHS or net not in NETWORK_TYPES or not fold.isdigit() or not run:
                raise SystemExit('%s line %d: not a valid row: %s' % (path, i, dict(row)))
            key = (arch, net, int(fold))
            if key in table and table[key] != run:
                raise SystemExit('%s: %s %s fold %s is listed twice (runs %s and %s)'
                                 % (path, arch, net, fold, table[key], run))
            table[key] = run
    return table


def read_config(log_dir, run):
    p = os.path.join(log_dir, str(run), 'config.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def check_networks(table, needed, log_dir, require_model):
    """Every needed (arch, network, fold) is in the table and its config.json agrees.

    require_model is the set of keys whose model.pt must exist (the ones with evaluations still
    to run).  Returns (configs {key: config}, problems [str])."""
    configs, problems = {}, []
    for key in sorted(needed):
        arch, net, fold = key
        if key not in table:
            problems.append('no %s %s network for fold %d in the networks table' % (arch, net, fold))
            continue
        run = table[key]
        cfg = read_config(log_dir, run)
        if cfg is None:
            problems.append('run %s (%s %s fold %d): no config.json in %s' % (run, arch, net, fold, log_dir))
            continue
        spec = NETWORK_TYPES[net]
        got = OrderedDict([
            ('model_name', str(cfg.get('model_name') or 'resnet').lower()),   # missing = resnet
            ('num_classes', int(cfg.get('num_classes', -1))),
            ('opt_only', str2bool(cfg.get('opt_only', False))),
            ('pos_only', str2bool(cfg.get('pos_only', False))),
        ])
        want = OrderedDict([('model_name', arch), ('num_classes', spec['num_classes']),
                            ('opt_only', spec['opt_only']), ('pos_only', spec['pos_only'])])
        bad = ['%s=%s (expected %s)' % (k, got[k], want[k]) for k in want if got[k] != want[k]]
        data_dir = str(cfg.get('data_dir', '')).rstrip('/')
        family = os.path.basename(os.path.dirname(data_dir))
        if spec['family'] not in family:
            bad.append('trained on family %r (expected one containing %r)' % (family, spec['family']))
        if os.path.basename(data_dir) != str(fold):
            bad.append('trained on fold %r (expected %d)' % (os.path.basename(data_dir), fold))
        if bad:
            problems.append('run %s is not %s %s fold %d: %s' % (run, arch, net, fold, '; '.join(bad)))
        if key in require_model and not os.path.exists(os.path.join(log_dir, str(run), 'model.pt')):
            problems.append('run %s (%s %s fold %d): model.pt missing' % (run, arch, net, fold))
        configs[key] = cfg
    return configs, problems


# ---------------------------------------------------------------------------------- jobs -----

def predictions_path(log_dir, run, cohort, suffix):
    name = 'predictions_%s%s%s.xlsx' % (run, '_subopt' if cohort == 'subopt' else '', suffix)
    return os.path.join(log_dir, str(run), 'results', name)


def plan_jobs(table, methods, archs, log_dir, cohorts=COHORTS, folds=FOLDS):
    """All (method, arch, fold, cohort) evaluations, plus the set of networks they need."""
    jobs, needed = [], set()
    for method in methods:
        for arch in archs:
            for fold in folds:
                for net in (method.first, method.second):
                    if net:
                        needed.add((arch, net, fold))
                first = table.get((arch, method.first, fold))
                second = table.get((arch, method.second, fold)) if method.second else None
                owner = second if method.second else first
                for cohort in cohorts:
                    out = predictions_path(log_dir, owner, cohort, method.suffix) if owner else None
                    jobs.append(Job(method, arch, fold, cohort, first, second, owner, out))
    return jobs, needed


# ------------------------------------------------------------------------------- cohorts -----

def read_id_list(path):
    with open(path) as fh:
        out = []
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(ln[:-4] if ln.endswith('.mha') else ln)
    return out


def split_list_path(cfg, which, splits_root):
    data_dir = str(cfg.get('data_dir', '')).rstrip('/')
    if splits_root:
        base = os.path.join(splits_root, os.path.basename(os.path.dirname(data_dir)),
                            os.path.basename(data_dir), 'data_split')
    else:
        base = os.path.join(data_dir, 'data_split')
    return os.path.join(base, '%s_ids.txt' % which)


def subject_map(path):
    df = pd.read_csv(path, dtype={'uuid': str, 'subject_id': str})
    for col in ('uuid', 'subject_id'):
        if col not in df.columns:
            raise SystemExit('%s has no %r column' % (path, col))
    return dict(zip(df['uuid'].str.strip(), df['subject_id'].str.strip()))


def check_cohorts(configs, archs, subopt_list, splits_root, subject_csv, required, allow_overlap=False):
    """Check the evaluation cohorts behind the paired design.

    Subject overlap (and scans without a subject_id) are problems, which stop the run, unless
    allow_overlap is set; then they are listed as notes.

    Returns (expected, problems, notes), where expected = {'opt': {(arch, fold): set},
    'subopt': set} or None when the split lists are not readable and not required."""
    problems, notes = [], []
    lists = {}
    missing = []
    for key, cfg in configs.items():
        for which in ('training', 'valid', 'test'):
            p = split_list_path(cfg, which, splits_root)
            if os.path.exists(p):
                lists[(key, which)] = set(read_id_list(p))
            else:
                missing.append(p)
    if missing:
        msg = '%d split lists not found, e.g. %s' % (len(missing), missing[0])
        if required:
            problems.append(msg + ' (use --splits_root if they live elsewhere)')
        else:
            notes.append(msg + ': cohort checks skipped')
        return None, problems, notes

    opt = {}
    for arch in archs:
        for fold in FOLDS:
            tests = OrderedDict()
            for (key, which), ids in lists.items():
                if which == 'test' and key[0] == arch and key[2] == fold:
                    tests[key] = ids
            if not tests:
                continue
            ref_key, ref = next(iter(tests.items()))
            for key, ids in tests.items():
                if ids != ref:
                    problems.append('%s fold %d: %s and %s have different test sets - the methods would '
                                    'not be compared on the same scans' % (arch, fold, ref_key[1], key[1]))
            opt[(arch, fold)] = ref
        folds_here = [f for f in FOLDS if (arch, f) in opt]
        seen = set()
        for f in folds_here:
            dup = seen & opt[(arch, f)]
            if dup:
                problems.append('%s: fold %d test scans also tested in another fold: %s'
                                % (arch, f, ', '.join(sorted(dup)[:3])))
            seen |= opt[(arch, f)]
        if folds_here:
            notes.append('%s: %d Optimal-included test scans over %d folds'
                         % (arch, len(seen), len(folds_here)))

    subopt = None
    if subopt_list:
        if not os.path.exists(subopt_list):
            (problems if required else notes).append('Suboptimal-only list not found: %s' % subopt_list)
        else:
            subopt = set(read_id_list(subopt_list))
            notes.append('Suboptimal-only list: %d scans' % len(subopt))
            for (key, which), ids in lists.items():
                both = subopt & ids
                if both:
                    problems.append('run of %s %s fold %d has %d Suboptimal-only scans in its %s list, '
                                    'e.g. %s' % (key[0], key[1], key[2], len(both), which, sorted(both)[0]))

    # subject overlap stops the run unless it is explicitly allowed (networks trained on per-scan splits)
    flag = notes if allow_overlap else problems
    hint = '' if allow_overlap else ' (--allow_subject_overlap only for networks trained on per-scan splits)'
    if not subject_csv:
        flag.append('subject overlap NOT checked: no --subject_csv' + hint)
    elif not os.path.exists(subject_csv):
        flag.append('subject overlap NOT checked: --subject_csv not found: %s' % subject_csv)
    else:
        subj = subject_map(subject_csv)
        unknown = set()
        overlap_rows = tv_rows = 0
        for key in configs:
            arch, net, fold = key
            train, valid = lists[(key, 'training')], lists[(key, 'valid')]
            evaluated = set(opt.get((arch, fold), set())) | (subopt or set())
            for u in train | valid | evaluated:
                if u not in subj:
                    unknown.add(u)
            shared = {subj.get(u) for u in train | valid} & {subj.get(u) for u in evaluated} - {None}
            if shared:
                overlap_rows += 1
                flag.append('SUBJECT OVERLAP: %s %s fold %d was trained/validated on %d subjects it is also '
                            'evaluated on (%s)%s' % (arch, net, fold, len(shared), ', '.join(sorted(shared)[:5]), hint))
            tv = {subj.get(u) for u in train} & {subj.get(u) for u in valid} - {None}
            if tv:
                tv_rows += 1
                flag.append('SUBJECT OVERLAP: %s %s fold %d has %d subjects in both training and validation '
                            '(%s)%s' % (arch, net, fold, len(tv), ', '.join(sorted(tv)[:5]), hint))
        if unknown:
            flag.append('%d scans have no subject_id in %s, e.g. %s - their subjects cannot be checked'
                        % (len(unknown), subject_csv, ', '.join(sorted(unknown)[:3])))
        if not overlap_rows and not tv_rows and not unknown:
            notes.append('subject check passed: training, validation and evaluated scans are subject-disjoint '
                         'for every network')

    return {'opt': opt, 'subopt': subopt}, problems, notes


# ------------------------------------------------------------------------------- running -----

def evaluator_command(args, job, cohort_list):
    m = job.method
    cmd = [args.python, '-m', EVALUATOR_MODULE,
           '--model', os.path.join(args.log_dir, str(job.first_run), 'model.pt'),
           '--scan_dir', args.scan_dir,
           '--labels_path', args.labels_path,
           '--output', job.output,
           '--test_scans_lst_path', cohort_list,
           '--num_classes', str(m.num_classes),
           '--threshold', '0',
           '--top5', 'false' if m.shortlist == 0 else 'true',
           '--shortlist_size', str(m.shortlist or 5),
           '--topk_list', ','.join(str(k) for k in m.topk),
           '--rel_only', 'true' if job.cohort == 'subopt' else 'false',
           '--use_gpu', 'true' if args.use_gpu else 'false',
           '--overwrite', 'true' if args.overwrite else 'false']
    if m.second:
        cmd[5:5] = ['--model2', os.path.join(args.log_dir, str(job.second_run), 'model.pt')]
    return cmd


def run_jobs(args, section, jobs, configs, expected):
    """Run every job whose predictions file is missing (or all, with --overwrite)."""
    with open(EVALUATOR_FILE) as fh:
        if '--shortlist_size' not in fh.read():
            raise SystemExit('%s has no --shortlist_size option: update it to the current repository '
                             'version first.' % EVALUATOR_FILE)
    for p, what in ((args.scan_dir, '--scan_dir'), (args.labels_path, '--labels_path'),
                    (args.subopt_list, '--subopt_list')):
        if not os.path.exists(p):
            raise SystemExit('%s not found: %s' % (what, p))

    todo = [j for j in jobs if j.output and (args.overwrite or not os.path.exists(j.output))]
    est = 0.0
    for j in todo:
        if j.cohort == 'subopt':
            n = len(expected['subopt']) if expected and expected['subopt'] else 113
        else:
            n = len(expected['opt'].get((j.arch, j.fold), ())) if expected else 28
        est += n * (SECONDS_PER_SCAN.get(j.arch, 3.0) + LOAD_SECONDS)
    print('%s: %d evaluations, %d already done, %d to run, roughly %.1f h'
          % (section, len(jobs), len(jobs) - len(todo), len(todo), est / 3600.0))
    if args.dry_run or not todo:
        for j in todo:
            print('  would run: %-22s %-12s fold %d %-6s -> %s'
                  % (j.method.name, j.arch, j.fold, j.cohort, os.path.relpath(j.output, args.log_dir)))
        return []

    log_dir = os.path.join(resolve_out_dir(args), 'logs', section)
    os.makedirs(log_dir, exist_ok=True)
    tmp_lists = os.path.join(resolve_out_dir(args), 'logs', 'cohort_lists')
    os.makedirs(tmp_lists, exist_ok=True)
    failures = []
    for i, j in enumerate(todo, start=1):
        if j.cohort == 'subopt':
            cohort_list = args.subopt_list
        else:
            # the test list of the fold, as recorded with the network, written out without
            # extensions so it does not depend on how the evaluator treats them
            cohort_list = os.path.join(tmp_lists, '%s_fold%d_test.txt' % (j.arch, j.fold))
            with open(cohort_list, 'w') as fh:
                for u in sorted(expected['opt'][(j.arch, j.fold)]):
                    fh.write(u + '\n')
        cmd = evaluator_command(args, j, cohort_list)
        tag = '%s_%s_fold%d_%s' % (j.method.name.replace(' ', '_').replace('(', '').replace(')', '')
                                   .replace('=', ''), j.arch, j.fold, j.cohort)
        log_path = os.path.join(log_dir, tag + '.log')
        t0 = time.time()
        print('[%d/%d] %-22s %-12s fold %d %-6s ...' % (i, len(todo), j.method.name, j.arch, j.fold, j.cohort),
              end='', flush=True)
        with open(log_path, 'w') as log:
            log.write(' '.join(cmd) + '\n\n')
            log.flush()
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        secs = time.time() - t0
        if rc != 0 or not os.path.exists(j.output):
            print(' FAILED (%ds), log: %s' % (secs, log_path))
            with open(log_path) as fh:
                tail = fh.readlines()[2:][-12:]          # skip the command line written first
            tail = tail or ['(no output)\n']
            print('    ' + '    '.join(tail))
            failures.append((j, log_path))
        else:
            print(' done in %dm %02ds' % (secs // 60, secs % 60))
    if failures:
        print('\n%d evaluations failed; their logs are in %s' % (len(failures), log_dir))
    return failures


def prepare(args, section, methods, default_cohorts=COHORTS):
    """Load and check everything, then run what is missing.  Returns (jobs, expected, checks)."""
    table = load_networks(args.networks)
    jobs, needed = plan_jobs(table, methods, args.archs, args.log_dir, default_cohorts)
    running = not args.summarize_only
    # a model is only needed for an evaluation that still has to run
    to_run = set()
    if running and not args.dry_run:
        for j in jobs:
            if j.output and (args.overwrite or not os.path.exists(j.output)):
                to_run.add((j.arch, j.method.first, j.fold))
                if j.method.second:
                    to_run.add((j.arch, j.method.second, j.fold))
    configs, problems = check_networks(table, needed, args.log_dir, require_model=to_run)
    expected, cproblems, notes = check_cohorts(configs, args.archs, args.subopt_list, args.splits_root,
                                               args.subject_csv, required=running,
                                               allow_overlap=args.allow_subject_overlap)
    problems += cproblems
    # which OptNet definition the table holds: trained with irrelevant frames as negatives (the default;
    # also when the config has no exclude_irrelevant) or on relevant frames only.  Never mixed in one table.
    optnet_modes = sorted({str2bool(cfg.get('exclude_irrelevant', False))
                           for (arch, net, fold), cfg in configs.items() if net == 'optnet'})
    if len(optnet_modes) > 1:
        problems.append('the OptNet runs in %s mix two training definitions (exclude_irrelevant true and '
                        'false); evaluate them separately' % args.networks)
    elif optnet_modes:
        notes.append('OptNet trained on %s' % ('relevant frames only (optimal vs suboptimal)' if optnet_modes[0]
                                               else 'all frames, irrelevant counted with suboptimal '
                                                    '(default definition)'))
    print('== %s ==' % section)
    for n in notes:
        print('  ' + n)
    if problems:
        print('  PROBLEMS:')
        for p in problems:
            print('    ' + p)
        if running or not args.allow_incomplete:
            raise SystemExit('fix the problems above first%s'
                             % ('' if running else ', or pass --allow_incomplete to summarize anyway'))
    checks = [('note', n) for n in notes] + [('problem', p) for p in problems]
    if running:
        failures = run_jobs(args, section, jobs, configs, expected)
        checks += [('failed evaluation', '%s %s fold %d %s, log %s'
                    % (j.method.name, j.arch, j.fold, j.cohort, log)) for j, log in failures]
    return jobs, expected, checks


# ------------------------------------------------------------------------------ summaries ----

OPT_METRICS = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'accuracy_top10',
               'wfss', 'wfss_top3', 'wfss_top5', 'wfss_top10', 'inference_time']


def strip_mha(name):
    """Scan identifier without the .mha extension, as in the split lists."""
    name = str(name)
    return name[:-4] if name.endswith('.mha') else name


def read_top_sheet(path):
    df = pd.read_excel(path, sheet_name='top1-5')
    df['scan'] = df['scan'].map(strip_mha)
    return df


def collect(jobs, expected, subject_csv=None):
    """Per-fold means and per-scan rows for every job, with coverage checks.

    Every metric is averaged over the scans of a fold first and then over the folds; the SD is
    over the fold means."""
    subj = subject_map(subject_csv) if subject_csv and os.path.exists(subject_csv) else None
    folds, scans, runs, checks = [], [], [], []
    for j in jobs:
        exists = bool(j.output) and os.path.exists(j.output)
        runs.append(OrderedDict([('method', j.method.name), ('arch', j.arch), ('fold', j.fold),
                                 ('cohort', j.cohort), ('first_network_run', j.first_run),
                                 ('second_network_run', j.second_run or ''),
                                 ('predictions_file', j.output or ''), ('exists', exists)]))
        if not exists:
            checks.append(('missing predictions', '%s %s fold %d %s: %s'
                           % (j.method.name, j.arch, j.fold, j.cohort, j.output)))
            continue
        df = read_top_sheet(j.output)
        got = set(df['scan'])
        if expected:
            want = expected['subopt'] if j.cohort == 'subopt' else expected['opt'].get((j.arch, j.fold))
            if want is not None and got != want:
                checks.append(('wrong scans', '%s %s fold %d %s: %d scans, %d missing, %d unexpected'
                               % (j.method.name, j.arch, j.fold, j.cohort, len(got), len(want - got),
                                  len(got - want))))
        row = OrderedDict([('method', j.method.name), ('arch', j.arch), ('cohort', j.cohort),
                           ('fold', j.fold), ('n_scans', len(df))])
        for m in OPT_METRICS:
            if m in df.columns:
                v = pd.to_numeric(df[m], errors='coerce')
                # a metric missing for SOME scans would be averaged over fewer scans than the other
                # methods; missing for all (e.g. top-10 of a K=5 shortlist) just means not applicable
                if v.isna().any() and v.notna().any():
                    checks.append(('missing values', '%s %s fold %d %s: %s is missing for %d of %d scans, so '
                                   'its mean would use fewer scans than the other methods'
                                   % (j.method.name, j.arch, j.fold, j.cohort, m, int(v.isna().sum()), len(v))))
                row[m] = v.mean()
        folds.append(row)
        keep = ['scan'] + [m for m in OPT_METRICS if m in df.columns]
        part = df[keep].copy()
        part.insert(0, 'fold', j.fold)
        part.insert(0, 'cohort', j.cohort)
        part.insert(0, 'arch', j.arch)
        part.insert(0, 'method', j.method.name)
        if subj is not None:
            part.insert(5, 'subject_id', part['scan'].map(subj))
        scans.append(part)
    fold_df = pd.DataFrame(folds)
    scan_df = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame()
    return fold_df, scan_df, pd.DataFrame(runs), checks


def summarize_folds(fold_df, metrics):
    """mean and SD over the fold means, per (method, arch, cohort), in the order first seen."""
    rows = []
    if fold_df.empty:
        return pd.DataFrame(rows)
    seen = OrderedDict()
    for _, r in fold_df.iterrows():
        seen.setdefault((r['method'], r['arch'], r['cohort']), None)
    for key in seen:
        sub = fold_df[(fold_df['method'] == key[0]) & (fold_df['arch'] == key[1]) & (fold_df['cohort'] == key[2])]
        row = OrderedDict([('method', key[0]), ('arch', key[1]), ('cohort', key[2]),
                           ('n_folds', len(sub)), ('n_scans', int(sub['n_scans'].sum()))])
        for m in metrics:
            if m in sub.columns and sub[m].notna().any():
                row[m] = sub[m].mean()
                row[m + '_sd'] = sub[m].std(ddof=1) if sub[m].notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def lookup(summary, method, arch, cohort, metric):
    if summary.empty or metric not in summary.columns:
        return np.nan
    hit = summary[(summary['method'] == method) & (summary['arch'] == arch) & (summary['cohort'] == cohort)]
    return hit[metric].iloc[0] if len(hit) else np.nan


def write_workbook(args, section, sheets, checks):
    """Write the section workbook; an existing one is moved to previous_versions/ first."""
    out_dir = resolve_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    bad = [c for c in checks if c[0] != 'note']
    if bad and not args.allow_incomplete:
        print('\nNOT writing %s tables:' % section)
        for kind, what in bad:
            print('  %s: %s' % (kind, what))
        print('Run the missing evaluations, or pass --allow_incomplete to write them anyway.')
        sys.exit(1)
    name = section + ('_INCOMPLETE' if bad else '') + '.xlsx'
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        prev = os.path.join(out_dir, 'previous_versions')
        os.makedirs(prev, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(os.path.getmtime(path)))
        shutil.move(path, os.path.join(prev, '%s_%s.xlsx' % (os.path.splitext(name)[0], stamp)))
    info = pd.DataFrame([('networks table', os.path.abspath(args.networks)),
                         ('log_dir', args.log_dir), ('backbones', ' '.join(args.archs)),
                         ('written', time.strftime('%Y-%m-%d %H:%M'))] + list(checks),
                        columns=['item', 'value'])
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for sheet, df in sheets:
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
        info.to_excel(xw, sheet_name='checks', index=False)
    print('\nwrote %s' % path)
    if bad:
        print('INCOMPLETE - see the checks sheet:')
        for kind, what in bad:
            print('  %s: %s' % (kind, what))
    return path


def fmt_mean_sd(m, s, digits=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return ''
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '%.*f' % (digits, m)
    return '%.*f ± %.*f' % (digits, m, digits, s)
