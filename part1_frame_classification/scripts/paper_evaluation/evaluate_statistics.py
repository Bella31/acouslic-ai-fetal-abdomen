"""
Statistical comparison of Two-step and Opt-only (DenseNet, shortlist K = 5).

Reads the per_scan sheet of main_comparison.xlsx (written by evaluate_main_comparison.py) and writes
statistics.xlsx to the same folder.

Four pre-specified comparisons, each as the difference Two-step minus Opt-only: Optimal-included
Top-1 and WFSS@1, and Suboptimal-only Top-1 and Top-3. The differences are scan-level means, as in
the results tables. Every Suboptimal-only scan is evaluated by all five fold models, and its
difference is first averaged over them.

Main analysis (sheet subject_level): subject-clustered bootstrap. Some subjects were scanned twice
and their two scans are not independent, so subjects are resampled, each with all of its scans, and
the difference is recomputed over the scans of the drawn subjects. The estimate therefore stays
scan-weighted. All four comparisons, Top-1 included, take their p-values from this bootstrap.

Sensitivity analysis (sheet scan_level): each scan treated as independent.
  Optimal-included Top-1   exact McNemar test; 95% CI from a paired case bootstrap
  Optimal-included WFSS@1  paired case bootstrap
  Suboptimal-only Top-1    scan-clustered bootstrap: scans resampled, each averaged over the folds
  Suboptimal-only Top-3    scan-clustered bootstrap

Both analyses: 100,000 resamples, seed 20260908, percentile 95% CI, two-sided p-value from the
null-centred bootstrap distribution, Holm adjustment across the four comparisons, significance at
adjusted p < 0.01. The checks sheet states whether the two analyses reach the same significance
decisions.

The resampling is deterministic: a fixed seed, units in a fixed order (scans by fold, then scan;
subjects sorted), and the comparisons drawn in the order above. Bootstrap p-values are Monte Carlo
estimates with a standard error of sqrt(p (1 - p) / 100,000).

--check_published compares the scan_level sheet with the reference results of the networks listed
in published_networks.csv (see check_published).

Usage, from the repository root:
  python -m part1_frame_classification.scripts.paper_evaluation.evaluate_statistics \
      --networks <networks table> [--out_dir <folder with main_comparison.xlsx>]
  python -m part1_frame_classification.scripts.paper_evaluation.evaluate_statistics \
      --workbook <path>/main_comparison.xlsx
run_paper_evaluation.py runs it right after the main comparison.
"""

import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

B = 100000
SEED = 20260908
ALPHA = 0.01
ARCH = 'densenet'
TWO, BASE = 'Two-step', 'Opt-only'
COMPARISONS = ['Optimal-included Top-1', 'Optimal-included WFSS@1', 'Suboptimal-only Top-1', 'Suboptimal-only Top-3']

# reference results of the networks in published_networks.csv, used by --check_published:
# raw p-values, and (difference, 95% CI low, 95% CI high, Holm-adjusted p) to 3 decimals
PUBLISHED_P_RAW = [0.391528, 0.15812, 0.00122, 0.00176]
PUBLISHED_TABLE = [(0.043, -0.036, 0.121, 0.392), (0.043, -0.014, 0.103, 0.316),
                   (0.080, 0.034, 0.127, 0.0049), (0.065, 0.027, 0.106, 0.0053)]


def say(text=''):
    print(text, flush=True)


# ----------------------------------------------------------------------------------- tests ---

def quantile_type7(values, probability):
    """Quantile with linear interpolation between order statistics (Hyndman-Fan type 7)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(differences, rng, replicates):
    """Mean difference resampled over units; returns CI low, CI high, null-centred two-sided p."""
    observed = sum(differences) / len(differences)
    n = len(differences)
    bootstrapped = []
    for _ in range(replicates):
        bootstrapped.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    null_centered = [value - observed for value in bootstrapped]
    p_value = sum(abs(value) >= abs(observed) for value in null_centered) / replicates
    return quantile_type7(bootstrapped, 0.025), quantile_type7(bootstrapped, 0.975), p_value


def comb(n, k):
    """Binomial coefficient (math.comb needs Python 3.8)."""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def exact_mcnemar(two_step, baseline):
    """Exact (binomial) McNemar test on paired 0/1 outcomes; returns b, c and the two-sided p-value.
    b: Two-step correct and Opt-only wrong; c: the reverse."""
    b = sum(a == 1.0 and z == 0.0 for a, z in zip(two_step, baseline))
    c = sum(a == 0.0 and z == 1.0 for a, z in zip(two_step, baseline))
    if b + c == 0:
        return b, c, 1.0
    lower_tail = sum(comb(b + c, k) for k in range(min(b, c) + 1)) / (2 ** (b + c))
    return b, c, min(1.0, 2.0 * lower_tail)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running, m = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


# ------------------------------------------------------------------------------------- data ---

def load_pairs(workbook):
    """Pair Two-step and Opt-only per scan.
    Optimal-included: one record per scan (each scan is tested in one fold), ordered by fold, then scan.
    Suboptimal-only: one record per scan with the values of all fold models, ordered by scan.
    Returns the two lists and the problems found."""
    d = pd.read_excel(workbook, sheet_name='per_scan', dtype={'subject_id': str, 'scan': str})
    d = d[d['arch'] == ARCH]
    problems = []
    for method in (TWO, BASE):
        if not (d['method'] == method).any():
            problems.append('no %s rows for %s in %s' % (method, ARCH, workbook))
    if problems:
        return None, None, problems
    has_subjects = 'subject_id' in d.columns and d['subject_id'].notna().all()
    if not has_subjects:
        problems.append('subject_id missing for some scans (run the evaluation with --subject_csv)')

    def table(method, cohort):
        t = d[(d['method'] == method) & (d['cohort'] == cohort)]
        return {(int(r['fold']), r['scan']): r for _, r in t.iterrows()}

    # Optimal-included: each scan once, in its own fold; folds in order, scans sorted within a fold
    two, base = table(TWO, 'opt'), table(BASE, 'opt')
    if set(two) != set(base):
        problems.append('Optimal-included: Two-step and Opt-only were evaluated on different scans')
    optimal, seen = [], set()
    for fold in sorted({f for f, _ in two}):
        scans = sorted(s for f, s in two if f == fold)
        if seen.intersection(scans):
            problems.append('Optimal-included: fold %d shares test scans with an earlier fold' % fold)
        seen.update(scans)
        for s in scans:
            if (fold, s) in base:
                a, z = two[(fold, s)], base[(fold, s)]
                optimal.append({'fold': fold, 'scan': s, 'subject': a.get('subject_id'),
                                'two_top1': float(a['accuracy']), 'base_top1': float(z['accuracy']),
                                'two_wfss': float(a['wfss']), 'base_wfss': float(z['wfss'])})

    # Suboptimal-only: every scan evaluated by every fold model
    two, base = table(TWO, 'subopt'), table(BASE, 'subopt')
    if set(two) != set(base):
        problems.append('Suboptimal-only: Two-step and Opt-only were evaluated on different scans')
    folds = sorted({f for f, _ in two})
    suboptimal = OrderedDict()
    for s in sorted({s for _, s in two}):
        rows = [(two[(f, s)], base[(f, s)]) for f in folds if (f, s) in two and (f, s) in base]
        if len(rows) != len(folds):
            problems.append('Suboptimal-only: scan %s is not paired in all %d folds' % (s, len(folds)))
        suboptimal[s] = {'subject': rows[0][0].get('subject_id') if rows else None,
                         'two_top1': [float(a['accuracy']) for a, _ in rows],
                         'base_top1': [float(z['accuracy']) for _, z in rows],
                         'two_top3': [float(a['accuracy_top3']) for a, _ in rows],
                         'base_top3': [float(z['accuracy_top3']) for _, z in rows]}
    return optimal, suboptimal, problems


# ------------------------------------------------------------------------------- analyses ---

def scan_level(optimal, suboptimal):
    """Sensitivity analysis, scans treated as independent (sheet scan_level); one generator, drawn in
    comparison order."""
    rng = random.Random(SEED)
    d_top1 = [r['two_top1'] - r['base_top1'] for r in optimal]
    d_wfss = [r['two_wfss'] - r['base_wfss'] for r in optimal]
    top1_low, top1_high, _ = paired_bootstrap(d_top1, rng, B)
    b, c, p_top1 = exact_mcnemar([r['two_top1'] for r in optimal], [r['base_top1'] for r in optimal])
    wfss_low, wfss_high, p_wfss = paired_bootstrap(d_wfss, rng, B)
    sub = {}
    for m in ('top1', 'top3'):
        diffs = [sum(a - z for a, z in zip(v['two_' + m], v['base_' + m])) / len(v['two_' + m])
                 for v in suboptimal.values()]
        sub[m] = paired_bootstrap(diffs, rng, B)

    def mean_opt(key):
        return sum(r[key] for r in optimal) / len(optimal)

    def mean_sub(key):
        values = [x for v in suboptimal.values() for x in v[key]]
        return sum(values) / len(values)

    n_pred = sum(len(v['two_top1']) for v in suboptimal.values())
    rows = [
        (COMPARISONS[0], 'Exact McNemar / paired bootstrap CI', len(optimal), mean_opt('two_top1'),
         mean_opt('base_top1'), top1_low, top1_high, p_top1, 'b=%d, c=%d' % (b, c)),
        (COMPARISONS[1], 'Paired case bootstrap', len(optimal), mean_opt('two_wfss'), mean_opt('base_wfss'),
         wfss_low, wfss_high, p_wfss, ''),
        (COMPARISONS[2], 'Scan-clustered bootstrap', len(suboptimal), mean_sub('two_top1'), mean_sub('base_top1'),
         sub['top1'][0], sub['top1'][1], sub['top1'][2], '%d fold predictions' % n_pred),
        (COMPARISONS[3], 'Scan-clustered bootstrap', len(suboptimal), mean_sub('two_top3'), mean_sub('base_top3'),
         sub['top3'][0], sub['top3'][1], sub['top3'][2], '%d fold predictions' % n_pred),
    ]
    return finish(rows, 'scans')


def subject_bootstrap(per_unit, rng):
    """per_unit: [(subject, difference)]; resample subjects; mean over the units of the drawn subjects."""
    subjects = sorted({s for s, _ in per_unit})
    pos = {s: i for i, s in enumerate(subjects)}
    total = np.zeros(len(subjects))
    count = np.zeros(len(subjects))
    for s, diff in per_unit:
        total[pos[s]] += diff
        count[pos[s]] += 1
    observed = total.sum() / count.sum()
    boot = np.empty(B)
    for start in range(0, B, 10000):
        idx = rng.randint(0, len(subjects), size=(min(10000, B - start), len(subjects)))
        boot[start:start + len(idx)] = total[idx].sum(1) / count[idx].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p = float(np.mean(np.abs(boot - observed) >= abs(observed) - 1e-12))
    return low, high, p, len(subjects), int((count > 1).sum())


def subject_level(optimal, suboptimal):
    """Main analysis, subject-clustered bootstrap (sheet subject_level)."""
    rng = np.random.RandomState(SEED)
    rows = []
    for name, key in ((COMPARISONS[0], 'top1'), (COMPARISONS[1], 'wfss')):
        units = [(r['subject'], r['two_' + key] - r['base_' + key]) for r in optimal]
        low, high, p, n_subj, n_multi = subject_bootstrap(units, rng)
        two = sum(r['two_' + key] for r in optimal) / len(optimal)
        base = sum(r['base_' + key] for r in optimal) / len(optimal)
        rows.append((name, 'Subject-clustered bootstrap', len(optimal), two, base, low, high, p,
                     '%d subjects, %d with two scans' % (n_subj, n_multi)))
    for name, key in ((COMPARISONS[2], 'top1'), (COMPARISONS[3], 'top3')):
        units = [(v['subject'], float(np.mean(np.array(v['two_' + key]) - np.array(v['base_' + key]))))
                 for v in suboptimal.values()]
        low, high, p, n_subj, n_multi = subject_bootstrap(units, rng)
        two = float(np.mean([x for v in suboptimal.values() for x in v['two_' + key]]))
        base = float(np.mean([x for v in suboptimal.values() for x in v['base_' + key]]))
        rows.append((name, 'Subject-clustered bootstrap', len(suboptimal), two, base, low, high, p,
                     '%d subjects, %d with two scans' % (n_subj, n_multi)))
    return finish(rows, 'subjects')


def finish(rows, unit):
    """Result table: adds the difference, the Holm-adjusted p-values and the significance decision."""
    table = pd.DataFrame(rows, columns=['Comparison', 'Test', 'n scans', 'Two-step', 'Opt-only', 'CI low',
                                        'CI high', 'p', 'Note'])
    table.insert(5, 'Difference', table['Two-step'] - table['Opt-only'])
    table['p (Holm)'] = holm_adjust(list(table['p']))
    table['Significant (p<0.01)'] = np.where(table['p (Holm)'] < ALPHA, 'yes', 'no')
    table['Resampled unit'] = unit
    cols = ['Comparison', 'Test', 'n scans', 'Two-step', 'Opt-only', 'Difference', 'CI low', 'CI high',
            'p', 'p (Holm)', 'Significant (p<0.01)', 'Note', 'Resampled unit']
    return table[cols]


def same_decisions(main, sensitivity):
    """Whether the two analyses reach the same significance decision for every comparison."""
    differ = [c for c, a, b in zip(main['Comparison'], main['Significant (p<0.01)'],
                                   sensitivity['Significant (p<0.01)']) if a != b]
    if not differ:
        return 'yes: same significance decisions for all four comparisons'
    return 'NO: the significance decision differs for ' + ', '.join(differ)


def check_published(by_scan):
    """Compare the scan-level table with the reference results. Differences and 95% CIs must agree to
    3 decimals, the McNemar p-value exactly, the bootstrap p-values within 4 Monte Carlo standard
    errors, and the significance decisions must be the same. Returns the disagreements."""
    problems = []
    for i, r in by_scan.iterrows():
        diff, low, high, p_holm = PUBLISHED_TABLE[i]
        got = (round(r['Difference'], 3), round(r['CI low'], 3), round(r['CI high'], 3))
        if got != (diff, low, high):
            problems.append('%s: difference and CI %s, reference %s' % (r['Comparison'], got, (diff, low, high)))
        published_p = PUBLISHED_P_RAW[i]
        if r['Test'].startswith('Exact McNemar'):
            ok = abs(r['p'] - published_p) < 5e-7                       # exact test: must be identical
        else:
            ok = abs(r['p'] - published_p) <= 4 * math.sqrt(published_p * (1 - published_p) / B)
        if not ok:
            problems.append('%s: raw p %.5f, reference %.5f' % (r['Comparison'], r['p'], published_p))
        if (r['p (Holm)'] < ALPHA) != (p_holm < ALPHA):
            problems.append('%s: significance differs (Holm p %.4f, reference %g)' % (r['Comparison'], r['p (Holm)'], p_holm))
    return problems


# ---------------------------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description='Statistical comparisons of Two-step vs Opt-only (DenseNet).')
    ap.add_argument('--workbook', help='main_comparison.xlsx; default: <out_dir>/main_comparison.xlsx')
    ap.add_argument('--networks', help='networks table, to find the default out_dir as the other sections do')
    ap.add_argument('--out_dir', help='folder with main_comparison.xlsx; statistics.xlsx is written there')
    ap.add_argument('--check_published', action='store_true',
                    help='exit with an error unless the result agrees with the reference results of the '
                         'networks in published_networks.csv')
    args = ap.parse_args()

    workbook = args.workbook
    if not workbook:
        if args.out_dir:
            folder = args.out_dir
        elif args.networks:
            stem = os.path.splitext(os.path.basename(args.networks))[0]
            folder = os.path.join(os.path.dirname(os.path.abspath(args.networks)), 'paper_evaluation_' + stem)
        else:
            ap.error('give --workbook, --out_dir or --networks')
        workbook = os.path.join(folder, 'main_comparison.xlsx')
    if not os.path.exists(workbook):
        say('ERROR: %s not found (run the main comparison first)' % workbook)
        sys.exit(1)

    say('reading %s' % workbook)
    optimal, suboptimal, problems = load_pairs(workbook)
    if problems or optimal is None:
        say('ERROR: the two methods cannot be paired by scan and subject:\n  ' + '\n  '.join(problems))
        sys.exit(1)
    say('%d Optimal-included scans, %d Suboptimal-only scans x %d fold models; %d bootstrap resamples'
        % (len(optimal), len(suboptimal), len(next(iter(suboptimal.values()))['two_top1']), B))

    t0 = time.time()
    by_subject = subject_level(optimal, suboptimal)
    by_scan = scan_level(optimal, suboptimal)
    say('computed in %.0f s' % (time.time() - t0))
    agreement = same_decisions(by_subject, by_scan)

    pd.set_option('display.width', 250)
    show = ['Comparison', 'Two-step', 'Opt-only', 'Difference', 'CI low', 'CI high', 'p (Holm)', 'Significant (p<0.01)']
    say('\nMain analysis, subject-clustered bootstrap (sheet subject_level):')
    say(by_subject[show].round(4).to_string(index=False))
    say('\nSensitivity analysis, scans treated as independent (sheet scan_level):')
    say(by_scan[show].round(4).to_string(index=False))
    say('\nSame conclusions in both analyses: ' + agreement)

    checks = [('workbook', workbook), ('arch', ARCH), ('methods', '%s vs %s' % (TWO, BASE)),
              ('Optimal-included scans', len(optimal)), ('Suboptimal-only scans', len(suboptimal)),
              ('bootstrap resamples', B), ('seed', SEED), ('significance', 'Holm-adjusted p < %g' % ALPHA),
              ('same conclusions in both analyses', agreement)]
    if args.check_published:
        mismatch = check_published(by_scan)
        verdict = ('reproduced: differences, CIs and McNemar exactly, bootstrap p-values within resampling '
                   'noise, same significance decisions')
        checks.append(('reference results', verdict if not mismatch else '; '.join(mismatch)))
        say('\nreference results: ' + (verdict if not mismatch else 'NOT reproduced:\n  ' + '\n  '.join(mismatch)))

    out = os.path.join(os.path.dirname(os.path.abspath(workbook)), 'statistics.xlsx')
    if os.path.exists(out):
        prev = os.path.join(os.path.dirname(out), 'previous_versions')
        if not os.path.isdir(prev):
            os.makedirs(prev)
        shutil.move(out, os.path.join(prev, 'statistics_%s.xlsx' % time.strftime('%Y%m%d_%H%M%S')))
    with pd.ExcelWriter(out) as xw:
        by_subject.to_excel(xw, sheet_name='subject_level', index=False)
        by_scan.to_excel(xw, sheet_name='scan_level', index=False)
        pd.DataFrame(checks, columns=['item', 'value']).to_excel(xw, sheet_name='checks', index=False)
    say('\nwrote %s' % out)
    if args.check_published and check_published(by_scan):
        sys.exit(1)


if __name__ == '__main__':
    main()
