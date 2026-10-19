"""
Collect every Excel file that summarize_results.py reads.

Two read sites, 60 runs each (120 files total):
    line 250: {networks_path}/{id}/results/predictions_{id}{postfix}.xlsx
    line 258: {networks_path}/{id}/results/predictions_{id}_subopt{postfix}.xlsx

Both are read with sheet_name=["top1-5", "precision_recall", "conf_matrix"].
(overall_summary.xlsx at line 153 is written by the script, not read, so it is
not collected here.)

Usage:
    # just list them (and report which are missing)
    python collect_prediction_excels.py

    # copy them into a folder, renamed by run description + fold
    python collect_prediction_excels.py --dest ~/excels_for_export --flat-names

    # line 250 only
    python collect_prediction_excels.py --opt-only
"""

import argparse
import os
import shutil
import sys

# --- copied verbatim from summarize_results.py -------------------------------
NETWORKS_PATH = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'

# folder id -> [fold number, experiment description]
RESULTS_INFO = {
    101: [0, 'Opt-only Resnet'],
    102: [1, 'Opt-only Resnet'],
    103: [2, 'Opt-only Resnet'],
    104: [3, 'Opt-only Resnet'],
    105: [4, 'Opt-only Resnet'],
    165: [0, 'Opt-only Efficientnet'],
    166: [1, 'Opt-only Efficientnet'],
    167: [2, 'Opt-only Efficientnet'],
    168: [3, 'Opt-only Efficientnet'],
    169: [4, 'Opt-only Efficientnet'],
    155: [0, 'Opt-only Convnext'],
    156: [1, 'Opt-only Convnext'],
    157: [2, 'Opt-only Convnext'],
    158: [3, 'Opt-only Convnext'],
    159: [4, 'Opt-only Convnext'],
    170: [0, 'Opt-only Densenet'],
    171: [1, 'Opt-only Densenet'],
    172: [2, 'Opt-only Densenet'],
    173: [3, 'Opt-only Densenet'],
    174: [4, 'Opt-only Densenet'],
    226: [0, 'One-step Resnet'],
    227: [1, 'One-step Resnet'],
    228: [2, 'One-step Resnet'],
    229: [3, 'One-step Resnet'],
    230: [4, 'One-step Resnet'],
    236: [0, 'One-step Efficientnet'],
    237: [1, 'One-step Efficientnet'],
    238: [2, 'One-step Efficientnet'],
    239: [3, 'One-step Efficientnet'],
    240: [4, 'One-step Efficientnet'],
    131: [0, 'One-step Convnext'],
    132: [1, 'One-step Convnext'],
    133: [2, 'One-step Convnext'],
    134: [3, 'One-step Convnext'],
    135: [4, 'One-step Convnext'],
    196: [0, 'One-step Densenet'],
    197: [1, 'One-step Densenet'],
    198: [2, 'One-step Densenet'],
    199: [3, 'One-step Densenet'],
    200: [4, 'One-step Densenet'],
    231: [0, 'two-step Resnet'],
    232: [1, 'two-step Resnet'],
    233: [2, 'two-step Resnet'],
    234: [3, 'two-step Resnet'],
    235: [4, 'two-step Resnet'],
    241: [0, 'two-step Efficientnet'],
    242: [1, 'two-step Efficientnet'],
    243: [2, 'two-step Efficientnet'],
    244: [3, 'two-step Efficientnet'],
    245: [4, 'two-step Efficientnet'],
    160: [0, 'two-step Convnext'],
    161: [1, 'two-step Convnext'],
    162: [2, 'two-step Convnext'],
    163: [3, 'two-step Convnext'],
    164: [4, 'two-step Convnext'],
    201: [0, 'two-step (ours)'],
    202: [1, 'two-step (ours)'],
    203: [2, 'two-step (ours)'],
    204: [3, 'two-step (ours)'],
    205: [4, 't wo-step (ours)'],   # typo preserved from the original
}
# -----------------------------------------------------------------------------


def split_run_id(run_id):
    """Same network_id / postfix logic as summarize_results.py lines 239-245."""
    if isinstance(run_id, str) and '_' in run_id:
        parts = run_id.split('_')
        return parts[0], "_" + "_".join(parts[1:])
    return run_id, ""


def build_paths(networks_path, include_subopt=False):
    """Return a list of (run_id, fold, description, kind, path)."""
    rows = []
    for run_id, (fold, description) in RESULTS_INFO.items():
        network_id, postfix = split_run_id(run_id)
        rows.append((
            run_id, fold, description, 'opt',
            os.path.join(networks_path, str(network_id), 'results',
                         f"predictions_{network_id}{postfix}.xlsx"),
        ))
        if include_subopt:
            rows.append((
                run_id, fold, description, 'subopt',
                os.path.join(networks_path, str(network_id), 'results',
                             f"predictions_{network_id}_subopt{postfix}.xlsx"),
            ))
    return rows


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--networks-path', default=NETWORKS_PATH,
                    help='root network output folder (default: path from summarize_results.py)')
    ap.add_argument('--dest', default=None,
                    help='if given, copy the files into this folder')
    ap.add_argument('--opt-only', action='store_true',
                    help='collect only the line-250 files, skipping the *_subopt files from line 258')
    ap.add_argument('--manifest', default=None,
                    help='optional path for a CSV manifest of run_id, fold, description, kind, source, exists')
    ap.add_argument('--flat-names', action='store_true',
                    help='when copying, rename to <description>_fold<N>_<kind>.xlsx instead of keeping '
                         'the original predictions_<id>.xlsx name')
    args = ap.parse_args()

    rows = build_paths(args.networks_path, include_subopt=not args.opt_only)
    missing = [r for r in rows if not os.path.isfile(r[4])]

    if args.dest:
        os.makedirs(args.dest, exist_ok=True)

    for run_id, fold, description, kind, src in rows:
        exists = os.path.isfile(src)
        print(f"{'OK ' if exists else 'MISS'}  {src}")
        if args.dest and exists:
            if args.flat_names:
                out_name = f"{safe_name(description)}_fold{fold}_{kind}.xlsx"
            else:
                out_name = os.path.basename(src)
            shutil.copy2(src, os.path.join(args.dest, out_name))

    if args.manifest:
        import csv
        with open(args.manifest, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['run_id', 'fold', 'description', 'kind', 'source_path', 'exists'])
            for run_id, fold, description, kind, src in rows:
                w.writerow([run_id, fold, description, kind, src, os.path.isfile(src)])
        print(f"\nmanifest written to {args.manifest}")

    print(f"\n{len(rows)} expected file(s), {len(rows) - len(missing)} found, {len(missing)} missing.")
    if args.dest:
        print(f"copied into {args.dest}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
