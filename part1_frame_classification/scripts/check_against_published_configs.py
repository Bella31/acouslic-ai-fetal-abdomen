"""
Compare the training settings of a retraining run with the saved config.json of the published runs.

train_subject_disjoint.py calls this before training, so the retrained networks differ from the
published ones only where intended (the split lists, and OptNet's frames if chosen), not in a
hyper-parameter copied wrongly.  The published run folders are found through
paper_evaluation/published_networks.csv in --log_dir.

    # print the loss the published networks of one type were trained with (all must agree)
    python -m part1_frame_classification.scripts.check_against_published_configs \
        --log_dir <network folder> --print_loss threeclass

    # compare every setting for the networks about to be trained
    python -m part1_frame_classification.scripts.check_against_published_configs \
        --log_dir <network folder> --archs densenet --nets relnet optnet optonly threeclass \
        --epochs 50 --batch_size 32 --lr 3e-4 --weight_decay 1e-5 --patience 20 --min_epoch 10 \
        --loss CE --threeclass_loss CE

Exit 1 on a mismatch, or when --print_loss finds no published config or disagreeing ones.
Published configs that are not in --log_dir are reported and skipped (exit 0), unless --strict.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUBLISHED = os.path.join(HERE, 'paper_evaluation', 'published_networks.csv')

# what each network type is (the same definitions as paper_evaluation/common.py)
NETWORK_TYPES = {
    'relnet':     dict(num_classes=2, opt_only=False, pos_only=False, family='opt_using_subopt'),
    'optnet':     dict(num_classes=2, opt_only=False, pos_only=True,  family='opt_using_subopt'),
    'optonly':    dict(num_classes=2, opt_only=True,  pos_only=False, family='opt_no_subopt'),
    'threeclass': dict(num_classes=3, opt_only=False, pos_only=False, family='opt_using_subopt'),
}


def as_bool(v):
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def published_runs(path):
    """{(arch, network): [run_dir, ...]} from the published networks table."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.setdefault((row['arch'].strip(), row['network'].strip()), []).append(row['run_dir'].strip())
    return out


def read_config(log_dir, run):
    p = os.path.join(log_dir, run, 'config.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def same(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--published', default=DEFAULT_PUBLISHED, help='published networks table')
    ap.add_argument('--log_dir', required=True, help='folder holding the published run folders')
    ap.add_argument('--print_loss', choices=sorted(NETWORK_TYPES),
                    help='print the loss of the published networks of this type and exit')
    ap.add_argument('--archs', nargs='+', default=[])
    ap.add_argument('--nets', nargs='+', default=[], choices=sorted(NETWORK_TYPES))
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--weight_decay', type=float)
    ap.add_argument('--patience', type=int)
    ap.add_argument('--min_epoch', type=int)
    ap.add_argument('--apply_mixup', default='False')
    ap.add_argument('--loss', help='loss of the binary networks')
    ap.add_argument('--threeclass_loss', help='loss of the three-class network')
    ap.add_argument('--strict', action='store_true', help='fail when a published config is not found')
    args = ap.parse_args()

    runs = published_runs(args.published)

    if args.print_loss:
        losses, found = set(), []
        for (arch, net), rr in sorted(runs.items()):
            if net != args.print_loss:
                continue
            for r in rr:
                cfg = read_config(args.log_dir, r)
                if cfg is not None:
                    losses.add(str(cfg.get('loss')))
                    found.append(r)
        if not found:
            sys.stderr.write('no config.json of a published %s run in %s\n' % (args.print_loss, args.log_dir))
            sys.exit(1)
        if len(losses) != 1:
            sys.stderr.write('published %s runs disagree on the loss: %s (runs %s)\n'
                             % (args.print_loss, ', '.join(sorted(losses)), ', '.join(found)))
            sys.exit(1)
        sys.stderr.write('loss of the published %s runs (%s): %s\n'
                         % (args.print_loss, ', '.join(found), next(iter(losses))))
        print(next(iter(losses)))
        return

    mismatches, missing, checked = [], [], 0
    for arch in args.archs:
        for net in args.nets:
            spec = NETWORK_TYPES[net]
            want = {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
                    'weight_decay': args.weight_decay, 'patience': args.patience,
                    'min_epoch': args.min_epoch, 'apply_mixup': args.apply_mixup,
                    'loss': args.threeclass_loss if net == 'threeclass' else args.loss,
                    'num_classes': spec['num_classes'], 'opt_only': spec['opt_only'],
                    'pos_only': spec['pos_only'], 'model_name': arch}
            rr = runs.get((arch, net), [])
            if not rr:
                missing.append('%s %s: not in %s' % (arch, net, args.published))
                continue
            for r in rr:
                cfg = read_config(args.log_dir, r)
                if cfg is None:
                    missing.append('%s %s run %s: no config.json in %s' % (arch, net, r, args.log_dir))
                    continue
                checked += 1
                for k, v in want.items():
                    if v is None:
                        continue
                    got = cfg.get(k, 'resnet' if k == 'model_name' else None)
                    if k in ('opt_only', 'pos_only'):
                        ok = as_bool(got) == v
                    else:
                        ok = same(got, v)
                    if not ok:
                        mismatches.append('%s %s run %s: %s = %s in the published config, %s here'
                                          % (arch, net, r, k, got, v))
                family = os.path.basename(os.path.dirname(str(cfg.get('data_dir', '')).rstrip('/')))
                if spec['family'] not in family:
                    mismatches.append('%s %s run %s: published run used family %r, expected %r'
                                      % (arch, net, r, family, spec['family']))

    print('compared with %d published configs' % checked)
    for m in missing:
        print('  not compared: ' + m)
    if mismatches:
        print('SETTINGS DIFFER FROM THE PUBLISHED RUNS:')
        for m in mismatches:
            print('  ' + m)
        sys.exit(1)
    if missing and args.strict:
        sys.exit(1)
    print('settings match the published runs')


if __name__ == '__main__':
    main()
