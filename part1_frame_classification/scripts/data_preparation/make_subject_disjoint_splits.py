"""
Make the saved cross-validation splits subject-disjoint, without redesigning them.

WHY
    The ACOUSLIC-AI public development set is 300 cases from 281 subjects: 19 subjects were
    examined twice.  The split code keys on the scan filename ("Filename" in the balanced CSV),
    so it cannot know that, and in the saved folds 11 of the 140 evaluated Optimal-included scans
    have the same subject's other scan in their own fold's training or validation set.  Six of
    the 113 Suboptimal-only scans are in the same position.

    The subject identifier is NOT in the imaging data.  It is the `subject_id` column of the
    per-sweep circumference file released with the dataset
    (Data/fetal_abdominal_circumferences_per_sweep.csv).

WHAT THIS DOES
    For each fold it removes from TRAINING and VALIDATION every scan that shares a subject with
    a scan the fold evaluates, where "evaluates" means that fold's own test set plus the
    Suboptimal-only cohort (which every fold model is evaluated on).

    Test sets are never touched.  The Suboptimal-only cohort is never touched.  So the corrected
    folds evaluate exactly the same 140 + 113 scans as the published ones, and the only change is
    that a handful of training scans are withheld - typically 6 to 9 of 159 per fold.

    That is the point of doing it this way.  Excluding contaminated scans from the EVALUATION
    instead would shrink the test cohort to a subset selected on "subject was examined once",
    whose composition may differ from the whole; withholding training scans leaves the evaluation
    cohort intact and removes the overlap at its source.

    The original fold assignment is otherwise preserved exactly, so this is a controlled
    correction of the existing design, not a new random split.  That matters because the shuffle
    in extract_frames_from_mha_cross_valid.py is unseeded and the published folds survive only as
    the saved *_ids.txt files.

TRAIN/VALIDATION OVERLAP
    A subject with one scan in training and the other in validation contaminates model selection
    (early stopping), not the test estimate.  It is reported but left alone by default, because
    validation holds only 9 scans and dropping from it costs more than the overlap does.  Pass
    --separate_train_valid to remove it as well.

AFTER RUNNING THIS
    The output has the same layout the rest of the pipeline expects,
    <output>/<fold>/data_split/{training,valid,test}_ids.txt, so the frame folders are rebuilt with

        python -m part1_frame_classification.scripts.data_preparation.extract_frames_from_mha_cross_valid \
            --csv <balanced.csv> --mha_dir <mha> --output <frames_out> --existing_split_path <output>

    Each family has its OWN balanced CSV: labels_balanced_opt_using_subopt.csv for
    cross_valid_folds_opt_using_subopt and labels_balanced_opt_no_subopt.csv for
    cross_valid_folds_opt_no_subopt.  Using the wrong one silently changes the training images.

    That step lists <output> with os.listdir and treats every entry as a fold, so nothing but fold
    directories may sit in it - a stray file such as .DS_Store makes it crash.  This script writes
    only fold directories there; the withheld-scans report goes NEXT TO <output>, as
    <output>_withheld_scans.csv.  --verify re-reads what was written.

    Afterwards, verify_frames_match_splits.py checks the extracted images against the original
    frame folders, which catches a wrong balanced CSV and leftover images from withheld scans.

USAGE
    Run once per split family (they are separate folder trees):

        python make_subject_disjoint_splits.py \
            --subject_csv  .../Data/fetal_abdominal_circumferences_per_sweep.csv \
            --existing_split_path .../Data/cross_valid_folds_opt_using_subopt \
            --output .../Data/cross_valid_folds_opt_using_subopt_subjdisjoint

    Deterministic and idempotent: the same inputs always give the same output, and running it on
    an already-corrected tree changes nothing.
"""

import argparse
import csv
import os
import shutil
import sys
from collections import OrderedDict, defaultdict

ROLE_FILES = OrderedDict([('training', 'train'), ('valid', 'valid'), ('test', 'test')])


def load_subject_map(csv_path, uuid_col='uuid', subject_col='subject_id'):
    """case uuid (no extension) -> subject id, from the released circumference file."""
    if not os.path.exists(csv_path):
        raise SystemExit('subject mapping not found: %s' % csv_path)
    mapping = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for col in (uuid_col, subject_col):
            if col not in reader.fieldnames:
                raise SystemExit('column %r not in %s (found: %s)'
                                 % (col, csv_path, ', '.join(reader.fieldnames or [])))
        for row in reader:
            uid = (row[uuid_col] or '').strip()
            sid = (row[subject_col] or '').strip()
            if not uid:
                continue
            if not sid:
                raise SystemExit('row for %s has an empty %s; refusing to guess' % (uid, subject_col))
            if uid in mapping and mapping[uid] != sid:
                raise SystemExit('uuid %s maps to two subjects (%s, %s)' % (uid, mapping[uid], sid))
            mapping[uid] = sid
    if not mapping:
        raise SystemExit('no rows read from %s' % csv_path)
    return mapping


def stem(name):
    """'<uuid>.mha' -> '<uuid>'. The id files carry the extension; the CSV does not."""
    return name[:-4] if name.endswith('.mha') else name


def read_ids(path):
    if not os.path.exists(path):
        raise SystemExit('missing split file: %s' % path)
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def write_ids(path, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for i in ids:
            fh.write('%s\n' % i)


def fold_names(split_path):
    """Fold directories, in numeric order when they are numeric. Ignores files and dotfiles."""
    if not os.path.isdir(split_path):
        raise SystemExit('not a directory: %s' % split_path)
    names = [n for n in os.listdir(split_path)
             if not n.startswith('.') and os.path.isdir(os.path.join(split_path, n))]
    if not names:
        raise SystemExit('no fold directories under %s' % split_path)
    try:
        return sorted(names, key=int)
    except ValueError:
        return sorted(names)


def main():
    ap = argparse.ArgumentParser(
        description='Withhold training/validation scans that share a subject with an evaluated scan.')
    ap.add_argument('--subject_csv', required=True,
                    help='released per-sweep circumference CSV, carrying uuid and subject_id')
    ap.add_argument('--existing_split_path', required=True,
                    help='split family to correct, holding <fold>/data_split/*_ids.txt')
    ap.add_argument('--output', required=True, help='where to write the corrected family')
    ap.add_argument('--uuid_col', default='uuid')
    ap.add_argument('--subject_col', default='subject_id')
    ap.add_argument('--separate_train_valid', action='store_true',
                    help='also remove subject overlap between training and validation '
                         '(drops from validation; off by default, see the module docstring)')
    ap.add_argument('--report', default=None,
                    help='CSV listing every withheld scan (default: <output>_withheld_scans.csv, next to '
                         'the output folder; it may not go inside it, see the module docstring)')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace the output directory if it already exists')
    ap.add_argument('--verify', action='store_true', default=True,
                    help='re-read the written files and confirm subject disjointness (default on)')
    args = ap.parse_args()

    subject_of = load_subject_map(args.subject_csv, args.uuid_col, args.subject_col)
    scans_of = defaultdict(list)
    for uid, sid in subject_of.items():
        scans_of[sid].append(uid)

    folds = fold_names(args.existing_split_path)

    # ---- read the family -------------------------------------------------------------------
    original = OrderedDict()
    everything_in_splits = set()
    for fold in folds:
        base = os.path.join(args.existing_split_path, fold, 'data_split')
        roles = OrderedDict()
        for file_stem in ROLE_FILES:
            ids = read_ids(os.path.join(base, '%s_ids.txt' % file_stem))
            roles[file_stem] = ids
            everything_in_splits.update(stem(i) for i in ids)
        original[fold] = roles

    unknown = sorted(u for u in everything_in_splits if u not in subject_of)
    if unknown:
        raise SystemExit('%d scans in the splits have no row in %s, e.g. %s\n'
                         'Refusing to proceed: a missing subject id cannot be treated as unique.'
                         % (len(unknown), args.subject_csv, ', '.join(unknown[:3])))

    # scans in the dataset but in no split file of this family: the Suboptimal-only cohort,
    # which every fold model is evaluated on, so no fold may train on their subjects
    subopt_only = sorted(set(subject_of) - everything_in_splits)
    subopt_subjects = {subject_of[u] for u in subopt_only}

    # ---- correct each fold -----------------------------------------------------------------
    src_abs = os.path.abspath(args.existing_split_path).rstrip(os.sep)
    dst_abs = os.path.abspath(args.output).rstrip(os.sep)
    if dst_abs == src_abs or dst_abs.startswith(src_abs + os.sep) or src_abs.startswith(dst_abs + os.sep):
        raise SystemExit('--output must be a separate folder from --existing_split_path')
    if os.path.exists(args.output):
        if not args.overwrite:
            raise SystemExit('output already exists: %s\nPass --overwrite to replace it.' % args.output)
        # only ever delete a tree this script could have written: fold folders holding data_split.
        # Once frames have been extracted into it, it also holds train/val/test images, and those
        # are not this script's to delete.
        foreign = []
        for entry in sorted(os.listdir(args.output)):
            p = os.path.join(args.output, entry)
            if entry == '.DS_Store':
                continue
            if not os.path.isdir(p):
                foreign.append(entry)
                continue
            extra = [e for e in os.listdir(p) if e not in ('data_split', '.DS_Store')]
            if extra:
                foreign.append('%s/{%s}' % (entry, ','.join(sorted(extra))))
        if foreign:
            raise SystemExit('refusing to delete %s: it holds more than split lists (%s).\n'
                             'Frames were probably extracted into it. Write the splits to a new folder.'
                             % (args.output, '; '.join(foreign[:5])))
        shutil.rmtree(args.output)

    withheld_rows = []
    tv_overlap_total = 0
    print('fold   train+valid   ->   kept    withheld     (test kept intact)')
    for fold in folds:
        roles = original[fold]
        test_ids = roles['test']
        protected = {subject_of[stem(i)] for i in test_ids} | subopt_subjects

        kept = OrderedDict()
        kept['test'] = list(test_ids)        # never touched

        dropped_here = []
        for file_stem in ('training', 'valid'):
            keep_list = []
            for i in roles[file_stem]:
                sid = subject_of[stem(i)]
                if sid in protected:
                    dropped_here.append((file_stem, i, sid))
                else:
                    keep_list.append(i)
            kept[file_stem] = keep_list

        # optional: remove training/validation subject overlap by dropping from validation
        tv_overlap = sorted({subject_of[stem(i)] for i in kept['valid']} &
                            {subject_of[stem(i)] for i in kept['training']})
        tv_overlap_total += len(tv_overlap)
        if args.separate_train_valid and tv_overlap:
            overlap = set(tv_overlap)
            remaining = []
            for i in kept['valid']:
                if subject_of[stem(i)] in overlap:
                    dropped_here.append(('valid', i, subject_of[stem(i)]))
                else:
                    remaining.append(i)
            kept['valid'] = remaining

        before = len(roles['training']) + len(roles['valid'])
        after = len(kept['training']) + len(kept['valid'])
        print('  %-4s %6d        ->  %5d   %5d          %d -> %d'
              % (fold, before, after, before - after, len(test_ids), len(kept['test'])))

        for file_stem, scan, sid in dropped_here:
            withheld_rows.append(OrderedDict([
                ('fold', fold), ('removed_from', file_stem), ('scan', scan), ('subject_id', sid),
                ('reason', 'subject evaluated by this fold')]))

        base_out = os.path.join(args.output, fold, 'data_split')
        for file_stem in ROLE_FILES:
            write_ids(os.path.join(base_out, '%s_ids.txt' % file_stem), kept[file_stem])

    # ---- report ----------------------------------------------------------------------------
    out_abs = os.path.abspath(args.output).rstrip(os.sep)
    report_path = os.path.abspath(args.report or out_abs + '_withheld_scans.csv')
    if report_path.startswith(out_abs + os.sep):
        raise SystemExit('--report may not be inside the output folder: frame extraction treats every '
                         'entry there as a fold')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=['fold', 'removed_from', 'scan', 'subject_id', 'reason'])
        w.writeheader()
        for row in withheld_rows:
            w.writerow(row)

    print('')
    print('dataset: %d scans, %d subjects, %d subjects with more than one scan'
          % (len(subject_of), len(scans_of), sum(1 for s in scans_of.values() if len(s) > 1)))
    print('Suboptimal-only cohort in this family: %d scans (untouched)' % len(subopt_only))
    print('withheld in total: %d scan-fold entries, %d distinct scans'
          % (len(withheld_rows), len({r['scan'] for r in withheld_rows})))
    print('training/validation subject overlap: %d subject-fold entries%s'
          % (tv_overlap_total, ' (removed)' if args.separate_train_valid else ' (left in place)'))
    print('report: %s' % report_path)

    # ---- verify ----------------------------------------------------------------------------
    if args.verify:
        problems = []
        for fold in fold_names(args.output):
            base = os.path.join(args.output, fold, 'data_split')
            got = {s: read_ids(os.path.join(base, '%s_ids.txt' % s)) for s in ROLE_FILES}

            if got['test'] != original[fold]['test']:
                problems.append('fold %s: test set changed' % fold)

            seen = {subject_of[stem(i)] for i in got['training']} | \
                   {subject_of[stem(i)] for i in got['valid']}
            bad = sorted(seen & {subject_of[stem(i)] for i in got['test']})
            if bad:
                problems.append('fold %s: %d subjects in both test and train/valid: %s'
                                % (fold, len(bad), ', '.join(bad)))
            bad_sub = sorted(seen & subopt_subjects)
            if bad_sub:
                problems.append('fold %s: %d Suboptimal-only subjects still trained on: %s'
                                % (fold, len(bad_sub), ', '.join(bad_sub)))
            overlap = {i for i in got['training']} & {i for i in got['valid']}
            if overlap:
                problems.append('fold %s: %d scans in both training and validation' % (fold, len(overlap)))

        print('')
        if problems:
            print('VERIFY FAILED')
            for p in problems:
                print('  ' + p)
            sys.exit(1)
        print('VERIFY OK: every fold is subject-disjoint between train/valid and '
              'test + Suboptimal-only, and all test sets are byte-identical to the originals.')


if __name__ == '__main__':
    main()
