"""
Check extracted frame folders against their split lists and against the original frame folders.

Run this after extract_frames_from_mha_cross_valid.py --existing_split_path and before training.
It is the last chance to catch two mistakes that would otherwise only show up as quietly wrong
results:

  1. LEFTOVER IMAGES.  Frame extraction only ever adds images and never deletes them.  Extracting
     corrected splits into a folder that already held the original frames leaves the withheld
     scans' images in train/ - and the network trains on them, so the leakage the correction was
     meant to remove is still there.

  2. WRONG BALANCED CSV.  The two split families were built from different balanced CSVs
     (labels_balanced_opt_using_subopt.csv and labels_balanced_opt_no_subopt.csv).  Extracting a
     family with the other one silently changes which frames, and how many suboptimal frames, the
     network sees.

Checks, per fold and per split (train / val / test):

  a. every image belongs to a scan listed for that split in <fold>/data_split/*_ids.txt
  b. every listed scan has at least one image
  c. with --reference_frames_dir (the ORIGINAL family): for every scan, the image files are exactly
     the ones the original extraction produced for that scan, class folder by class folder.  The
     corrected splits keep each scan in its original role, so they must match file for file.

Exit status 0 when everything matches, 1 otherwise.

    python -m part1_frame_classification.scripts.data_preparation.verify_frames_match_splits \
        --frames_dir <.../cross_valid_folds_opt_using_subopt_subjdisjoint> \
        --reference_frames_dir <.../cross_valid_folds_opt_using_subopt>
"""

import argparse
import os
import sys
from collections import defaultdict

SPLITS = (('train', 'training_ids.txt'), ('val', 'valid_ids.txt'), ('test', 'test_ids.txt'))
CLASSES = ('irrelevant', 'optimal', 'suboptimal')
IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def read_ids(path):
    if not os.path.exists(path):
        raise SystemExit('missing split list: %s' % path)
    with open(path) as fh:
        return {ln.strip()[:-4] if ln.strip().endswith('.mha') else ln.strip() for ln in fh if ln.strip()}


def images_by_scan(split_dir):
    """{scan: {(class, filename), ...}} for one train/val/test folder.  Files are <scan>_<frame>.jpg"""
    out = defaultdict(set)
    if not os.path.isdir(split_dir):
        return out
    for cls in CLASSES:
        d = os.path.join(split_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith('.') or not f.lower().endswith(IMAGE_EXT):
                continue
            out[f.rsplit('_', 1)[0]].add((cls, f))
    return out


def fold_names(root):
    names = [n for n in os.listdir(root) if not n.startswith('.') and os.path.isdir(os.path.join(root, n))]
    try:
        return sorted(names, key=int)
    except ValueError:
        return sorted(names)


def main():
    ap = argparse.ArgumentParser(description='Verify extracted frames against split lists and the original frames.')
    ap.add_argument('--frames_dir', required=True, help='extracted family to check, <fold>/{data_split,train,val,test}')
    ap.add_argument('--reference_frames_dir', default=None,
                    help='the ORIGINAL extracted family, for the file-by-file comparison (recommended)')
    ap.add_argument('--max_report', type=int, default=10, help='problems listed per kind')
    args = ap.parse_args()

    if not os.path.isdir(args.frames_dir):
        raise SystemExit('not a directory: %s' % args.frames_dir)
    stray = [n for n in os.listdir(args.frames_dir)
             if not n.startswith('.') and not os.path.isdir(os.path.join(args.frames_dir, n))]
    problems = defaultdict(list)
    if stray:
        problems['non-fold entries in frames_dir (frame extraction treats them as folds)'].extend(stray)

    totals = defaultdict(int)
    for fold in fold_names(args.frames_dir):
        base = os.path.join(args.frames_dir, fold)
        for split, list_name in SPLITS:
            listed = read_ids(os.path.join(base, 'data_split', list_name))
            have = images_by_scan(os.path.join(base, split))
            totals[split] += sum(len(v) for v in have.values())

            for scan in sorted(set(have) - listed):
                problems['images of a scan NOT listed for %s (leftover / leakage)' % split].append(
                    'fold %s: %s (%d images)' % (fold, scan, len(have[scan])))
            for scan in sorted(listed - set(have)):
                problems['listed %s scan with no images' % split].append('fold %s: %s' % (fold, scan))

            if args.reference_frames_dir:
                ref = images_by_scan(os.path.join(args.reference_frames_dir, fold, split))
                for scan in sorted(listed & set(have)):
                    if scan not in ref:
                        problems['scan not in this split of the reference'].append(
                            'fold %s %s: %s' % (fold, split, scan))
                        continue
                    if have[scan] != ref[scan]:
                        extra, missing = have[scan] - ref[scan], ref[scan] - have[scan]
                        problems['images differ from the reference (wrong balanced CSV?)'].append(
                            'fold %s %s: %s  +%d / -%d files, e.g. %s'
                            % (fold, split, scan, len(extra), len(missing),
                               sorted(extra or missing)[0]))

    print('images checked: %s' % ', '.join('%s %d' % (s, totals[s]) for s, _ in SPLITS))
    print('compared file by file with: %s' % (args.reference_frames_dir or '(no reference given)'))
    if problems:
        print('\nVERIFY FAILED')
        for kind, items in problems.items():
            print('  %s: %d' % (kind, len(items)))
            for it in items[:args.max_report]:
                print('      %s' % it)
            if len(items) > args.max_report:
                print('      ... %d more' % (len(items) - args.max_report))
        sys.exit(1)
    print('VERIFY OK: every image belongs to a scan listed for its split, every listed scan has images'
          + (', and every scan matches the original extraction file for file.' if args.reference_frames_dir
             else '.'))


if __name__ == '__main__':
    main()
