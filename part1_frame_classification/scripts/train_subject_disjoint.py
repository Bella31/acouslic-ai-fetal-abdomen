"""
Retrain the frame-selection networks on subject-disjoint cross-validation splits.

Why: the 300 ACOUSLIC-AI development scans come from 281 subjects; 19 subjects were scanned twice.
The published folds were split by scan, so a subject could have one scan in training and the other
in validation or test. The corrected folds keep every test set exactly as published and withhold the
7-10 scans per fold that share a subject with an evaluated scan or with the fold's validation set
(data_preparation/make_subject_disjoint_splits.py).

Steps, in order:
  1. split lists  --make_splits writes the corrected split lists. Existing lists are always checked
                  against what the split script produces now; lists from an older version stop the run.
  2. frames       --extract_frames extracts the frame images for those lists, each family from its own
                  balanced CSV.
  3. frame check  the extracted images must match the original extraction file for file (catches
                  leftover images of withheld scans and a wrong balanced CSV).
  4. settings     every training setting is compared with the config.json of the published runs of
                  the same backbone and network. Any difference stops the run.
  5. training     every requested network that is not yet in the manifest is trained, and each finished
                  run is added to the manifest (the evaluation scripts read it). Finished runs are
                  skipped, so an interrupted run can simply be started again.

Networks:
  relnet      relevant vs irrelevant (One-step, and stage 1 of Two-step)
  optnet      optimal vs not optimal: suboptimal and irrelevant frames together form the negative
              class, as in the published runs (stage 2 of Two-step)
  optonly     optimal vs everything else, on the opt_no_subopt family (Opt-only)
  threeclass  irrelevant / optimal / suboptimal (Three-class; DenseNet only in the paper)
Two-step (relevant) and the shortlist sizes K reuse relnet + optnet, so they need no training.

Usage, from the repository root (the default paths are the GPU machine's; see --help):

  # first time: split lists, frames, checks, then the DenseNet networks (20 networks, about 8.5 h)
  nohup python part1_frame_classification/scripts/train_subject_disjoint.py --make_splits \
      --extract_frames --archs densenet --nets relnet optnet optonly threeclass > train_sd_1.log 2>&1 &

  # then the other backbones (45 networks, about 62 h; ConvNeXt is about 43 h of it, so it goes last)
  nohup python part1_frame_classification/scripts/train_subject_disjoint.py \
      --archs resnet efficientnet convnext --nets relnet optnet optonly > train_sd_2.log 2>&1 &

  # see what would run and how long it would take (checks run; nothing is extracted or trained)
  python part1_frame_classification/scripts/train_subject_disjoint.py --archs densenet --dry_run

To train a network again, delete its line from the manifest.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
GPU_OUT = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output'
GPU_SCANS = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
PUBLISHED = os.path.join(REPO_ROOT, 'part1_frame_classification', 'scripts', 'paper_evaluation',
                         'published_networks.csv')

# training settings, copied from the config.json of the published runs (checked in step 4)
SETTINGS = {'epochs': 50, 'batch_size': 32, 'lr': 3e-4, 'weight_decay': 1e-5,
            'patience': 20, 'min_epoch': 10, 'loss': 'CE'}

NETWORKS = {
    'relnet':     {'family': 'opt_using_subopt', 'num_classes': 2, 'opt_only': False, 'pos_only': False},
    'optnet':     {'family': 'opt_using_subopt', 'num_classes': 2, 'opt_only': False, 'pos_only': True},
    'optonly':    {'family': 'opt_no_subopt',    'num_classes': 2, 'opt_only': True,  'pos_only': False},
    'threeclass': {'family': 'opt_using_subopt', 'num_classes': 3, 'opt_only': False, 'pos_only': False},
}

# minutes per network on the GPU machine, for the opt_using_subopt and opt_no_subopt families
MINUTES = {'densenet': (28, 18), 'resnet': (27, 20), 'efficientnet': (58, 38), 'convnext': (195, 125)}

MANIFEST_COLUMNS = ['run_dir', 'arch', 'network', 'fold', 'data_dir', 'started', 'finished', 'seconds']


def say(text=''):
    print(text, flush=True)


def stop(message):
    say('\nERROR: ' + message)
    sys.exit(1)


def run_module(module, *args):
    """Run python -m part1_frame_classification.scripts.<module> from the repository root."""
    command = [sys.executable, '-m', 'part1_frame_classification.scripts.' + module] + [str(a) for a in args]
    return subprocess.call(command, cwd=REPO_ROOT)


def require(path, what):
    if not os.path.exists(path):
        stop('%s not found: %s' % (what, path))


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------------------------------

def step1_split_lists(args, families, original, corrected):
    say('=== 1. subject-disjoint split lists ===')
    require(args.subject_csv, 'subject table (--subject_csv)')
    for family in families:
        require(original[family], 'original split folder')
        has_lists = os.path.exists(os.path.join(corrected[family], '0', 'data_split', 'training_ids.txt'))
        if not has_lists:
            if not args.make_splits:
                stop('no split lists in %s. Run with --make_splits.' % corrected[family])
            say('--- writing ' + corrected[family])
            if run_module('data_preparation.make_subject_disjoint_splits', '--subject_csv', args.subject_csv,
                          '--existing_split_path', original[family], '--output', corrected[family]) != 0:
                stop('writing the split lists failed for ' + corrected[family])
        say('--- checking ' + corrected[family])
        if run_module('data_preparation.make_subject_disjoint_splits', '--subject_csv', args.subject_csv,
                      '--existing_split_path', original[family], '--output', corrected[family],
                      '--check_only') != 0:
            stop('the split lists in %s are not the current subject-disjoint ones (see above)'
                 % corrected[family])


def step2_extract_frames(args, families, corrected):
    say('\n=== 2. frame extraction ===')
    if not args.extract_frames:
        say('--- skipped (no --extract_frames)')
        return
    require(args.scan_dir, 'scan folder (--scan_dir)')
    for family in families:
        folder = corrected[family]
        balanced_csv = os.path.join(args.out, 'labels_balanced_%s.csv' % family)
        require(balanced_csv, 'balanced CSV for ' + family)
        # the extraction script reads every entry of the folder as a fold
        if os.path.exists(os.path.join(folder, '.DS_Store')):
            os.remove(os.path.join(folder, '.DS_Store'))
        stray = [e for e in os.listdir(folder) if not os.path.isdir(os.path.join(folder, e))]
        if stray:
            stop('files in %s would be read as folds: %s' % (folder, ', '.join(stray)))
        say('--- %s with %s' % (folder, os.path.basename(balanced_csv)))
        if args.dry_run:
            say('    [dry run] not extracting')
            continue
        if run_module('data_preparation.extract_frames_from_mha_cross_valid', '--csv', balanced_csv,
                      '--mha_dir', args.scan_dir, '--output', folder, '--existing_split_path', folder) != 0:
            stop('frame extraction failed for ' + folder)


def step3_check_frames(args, families, original, corrected):
    say('\n=== 3. extracted frames vs the original extraction ===')
    for family in families:
        if not os.path.isdir(os.path.join(corrected[family], '0', 'train')):
            if args.dry_run:
                say('--- %s: no frames yet, check skipped (dry run)' % corrected[family])
                continue
            stop('no frames in %s. Run with --extract_frames.' % corrected[family])
        say('--- ' + corrected[family])
        if run_module('data_preparation.verify_frames_match_splits', '--frames_dir', corrected[family],
                      '--reference_frames_dir', original[family]) != 0:
            stop('the frames in %s do not match the original extraction (see above); not training on them'
                 % corrected[family])


def read_threeclass_loss(args):
    """The loss of the published three-class runs, unless --threeclass_loss is given."""
    if args.threeclass_loss:
        return args.threeclass_loss
    result = subprocess.run(
        [sys.executable, '-m', 'part1_frame_classification.scripts.check_against_published_configs',
         '--published', PUBLISHED, '--log_dir', args.log_dir, '--print_loss', 'threeclass'],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, universal_newlines=True)
    loss = result.stdout.strip()
    if result.returncode != 0 or not loss:
        stop('could not read the loss of the published three-class runs in %s. Give --threeclass_loss.'
             % args.log_dir)
    return loss


def step4_compare_settings(args, threeclass_loss):
    say('\n=== 4. training settings vs the published runs ===')
    command = ['--published', PUBLISHED, '--log_dir', args.log_dir,
               '--archs'] + args.archs + ['--nets'] + args.nets + [
               '--epochs', SETTINGS['epochs'], '--batch_size', SETTINGS['batch_size'], '--lr', SETTINGS['lr'],
               '--weight_decay', SETTINGS['weight_decay'], '--patience', SETTINGS['patience'],
               '--min_epoch', SETTINGS['min_epoch'], '--apply_mixup', 'False', '--loss', SETTINGS['loss']]
    if threeclass_loss:
        command += ['--threeclass_loss', threeclass_loss]
    if run_module('check_against_published_configs', *command) != 0:
        stop('the training settings differ from the published runs (see above)')
    if 'optnet' in args.nets:
        say('OptNet: optimal vs not optimal (suboptimal + irrelevant), as in the published runs')


def latest_run_folder(log_dir):
    numbers = [int(d) for d in os.listdir(log_dir) if d.isdigit() and os.path.isdir(os.path.join(log_dir, d))]
    return max(numbers) if numbers else None


def step5_train(args, corrected, threeclass_loss):
    os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.manifest):
        with open(args.manifest, 'w', newline='') as fh:
            csv.writer(fh).writerow(MANIFEST_COLUMNS)
    with open(args.manifest, newline='') as fh:
        done = {(r['arch'], r['network'], r['fold']) for r in csv.DictReader(fh)}

    requested = [(a, n, f) for a in args.archs for n in args.nets for f in args.folds]
    todo = [(a, n, f) for a, n, f in requested if (a, n, str(f)) not in done]
    minutes = sum(MINUTES[a][0 if NETWORKS[n]['family'] == 'opt_using_subopt' else 1] for a, n, f in todo)
    say('\n=== 5. training: %d networks to train, %d already in %s ==='
        % (len(todo), len(requested) - len(todo), args.manifest))
    say('estimated GPU time: about %dh %02dm' % (minutes // 60, minutes % 60))

    failed = []
    for arch, net, fold in todo:
        spec = NETWORKS[net]
        data_dir = os.path.join(corrected[spec['family']], str(fold))
        loss = threeclass_loss if net == 'threeclass' else SETTINGS['loss']
        say('\n=== %s / %s / fold %d  (%s) ===' % (arch, net, fold, now()))
        say('    data_dir=%s  classes=%d  opt_only=%s  pos_only=%s  loss=%s'
            % (data_dir, spec['num_classes'], spec['opt_only'], spec['pos_only'], loss))
        if args.dry_run:
            say('    [dry run] would train')
            continue
        require(os.path.join(data_dir, 'train'), 'extracted frames')

        before = latest_run_folder(args.log_dir)   # the new run folder is the highest number + 1
        started, t0 = now(), time.time()
        code = run_module('train_classification', '--data_dir', data_dir, '--model_name', arch,
                          '--num_classes', spec['num_classes'], '--opt_only', spec['opt_only'],
                          '--pos_only', spec['pos_only'], '--loss', loss,
                          '--epochs', SETTINGS['epochs'], '--batch_size', SETTINGS['batch_size'],
                          '--lr', SETTINGS['lr'], '--weight_decay', SETTINGS['weight_decay'],
                          '--patience', SETTINGS['patience'], '--min_epoch', SETTINGS['min_epoch'],
                          '--apply_mixup', 'False', '--log_dir', args.log_dir)
        seconds = int(time.time() - t0)
        run = latest_run_folder(args.log_dir)

        if code != 0:
            say('    FAILED (exit %d) after %d s; going on with the rest' % (code, seconds))
            failed.append('%s/%s/fold %d' % (arch, net, fold))
            continue
        if run is None or run == before or not os.path.exists(os.path.join(args.log_dir, str(run), 'model.pt')):
            say('    WARNING: no new run folder with a model.pt in %s; not added to the manifest' % args.log_dir)
            failed.append('%s/%s/fold %d' % (arch, net, fold))
            continue
        with open(args.manifest, 'a', newline='') as fh:
            csv.writer(fh).writerow([run, arch, net, fold, data_dir, started, now(), seconds])
        say('    done in %dm %02ds -> run folder %d' % (seconds // 60, seconds % 60, run))
    return failed


# ---------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Retrain the networks on subject-disjoint splits.')
    parser.add_argument('--archs', nargs='+', default=['densenet'], choices=sorted(MINUTES),
                        help='default: densenet')
    parser.add_argument('--nets', nargs='+', default=['relnet', 'optnet', 'optonly'], choices=sorted(NETWORKS),
                        help='default: relnet optnet optonly')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='default: 0 1 2 3 4')
    parser.add_argument('--make_splits', action='store_true', help='step 1: write missing split lists')
    parser.add_argument('--extract_frames', action='store_true', help='step 2: extract the frame images')
    parser.add_argument('--dry_run', action='store_true',
                        help='run the checks and print the plan; extract and train nothing')
    parser.add_argument('--out', default=GPU_OUT,
                        help='folder with the split families, the subject table and the balanced CSVs; '
                             'default: the GPU machine output folder')
    parser.add_argument('--scan_dir', default=GPU_SCANS, help='the .mha scans (step 2); default: the GPU machine')
    parser.add_argument('--subject_csv', help='default: <out>/fetal_abdominal_circumferences_per_sweep.csv')
    parser.add_argument('--log_dir', help='network run folders; default: <out>/network')
    parser.add_argument('--manifest', help='default: <log_dir>/subject_disjoint_manifest.csv')
    parser.add_argument('--threeclass_loss', help='default: read from the published three-class runs')
    args = parser.parse_args()
    args.subject_csv = args.subject_csv or os.path.join(args.out, 'fetal_abdominal_circumferences_per_sweep.csv')
    args.log_dir = args.log_dir or os.path.join(args.out, 'network')
    args.manifest = args.manifest or os.path.join(args.log_dir, 'subject_disjoint_manifest.csv')

    families = sorted({NETWORKS[n]['family'] for n in args.nets})
    original = {f: os.path.join(args.out, 'cross_valid_folds_' + f) for f in families}
    corrected = {f: os.path.join(args.out, 'cross_valid_folds_%s_subjdisjoint' % f) for f in families}
    threeclass_loss = read_threeclass_loss(args) if 'threeclass' in args.nets else None

    say('%s  %s' % (now(), 'DRY RUN' if args.dry_run else ''))
    say('networks: %s x %s x folds %s' % (' '.join(args.archs), ' '.join(args.nets),
                                         ' '.join(str(f) for f in args.folds)))
    step1_split_lists(args, families, original, corrected)
    step2_extract_frames(args, families, corrected)
    step3_check_frames(args, families, original, corrected)
    step4_compare_settings(args, threeclass_loss)
    failed = step5_train(args, corrected, threeclass_loss)

    say('\n' + '=' * 70)
    if failed:
        say('%d networks failed: %s' % (len(failed), ', '.join(failed)))
    say('run folders are listed in %s (the evaluation scripts read this file)' % args.manifest)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
