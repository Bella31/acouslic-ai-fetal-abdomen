"""
Check the frame ranking logic of evaluate_frame_selection.py.

The functions under test are pulled out of the source file and run on their own, so the test needs only
numpy and pandas and does not load torch, the networks or the scans. Run it before spending GPU time on
the sweep.

Usage:
    python test_shortlist_logic.py
"""

import os
import sys

import numpy as np


SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'part1_frame_classification', 'scripts', 'evaluate_frame_selection.py')
FUNCTIONS = ['rank_frames', 'get_top_k_indices', 'calc_topk_indices', 'str2intlist', 'prepare_labels',
             'validate_scan_labels']


def load_functions(source_path, names):
    """
    Exec the named top level functions of a source file without importing the module
    """
    with open(source_path, 'r') as f:
        lines = f.readlines()
    #a top level def runs until the next line that starts in column zero and is not blank or a comment
    namespace = {'np': np, 'pd': __import__('pandas'), 'argparse': __import__('argparse')}
    for name in names:
        start = None
        for i, line in enumerate(lines):
            if line.startswith('def ' + name + '('):
                start = i
                break
        if start is None:
            raise AssertionError('function %s was not found in %s' % (name, source_path))
        end = len(lines)
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip() == '' or line[0] in ' \t#':
                continue
            end = i
            break
        exec(''.join(lines[start:end]), namespace)
    return namespace


def main():
    ns = load_functions(SOURCE, FUNCTIONS)
    get_top_k_indices = ns['get_top_k_indices']
    calc_topk_indices = ns['calc_topk_indices']
    str2intlist = ns['str2intlist']

    failures = []

    def check(name, condition, detail=''):
        if condition:
            print('  ok    %s' % name)
        else:
            print('  FAIL  %s %s' % (name, detail))
            failures.append(name)

    #twelve frames, frame 7 is the strongest for the first network
    scores = np.array([0.10, 0.20, 0.30, 0.05, 0.60, 0.15, 0.25, 0.95, 0.40, 0.50, 0.01, 0.35])
    expected_order = list(np.argsort(scores)[::-1])

    print('get_top_k_indices')
    for k in (1, 3, 5, 10):
        got = list(get_top_k_indices(scores, k))
        check('k=%d returns the k best frames, best first' % k, got == expected_order[:k],
              'got %s expected %s' % (got, expected_order[:k]))
    check('k larger than the scan is clipped to the scan',
          list(get_top_k_indices(scores, 50)) == expected_order,
          'got %s' % list(get_top_k_indices(scores, 50)))
    check('k=1 agrees with argmax', int(get_top_k_indices(scores, 1)[0]) == int(np.argmax(scores)))

    print('\ncalc_topk_indices, shortlist of 10 re-ranked by the second network')
    shortlist = get_top_k_indices(scores, 10)
    #the second network reverses the order of the shortlist
    probs2 = np.zeros((len(shortlist), 2))
    probs2[:, 1] = np.linspace(0.1, 0.9, len(shortlist))
    probs2[:, 0] = 1 - probs2[:, 1]
    best_idx, topk, max_score = calc_topk_indices(shortlist, probs2, scores, 0.0, [3, 5, 10])
    check('best frame is the second network argmax', best_idx == int(shortlist[int(np.argmax(probs2[:, 1]))]),
          'got %s' % best_idx)
    check('top-3 holds 3 frames', len(topk[3]) == 3, 'got %s' % list(topk[3]))
    check('top-5 holds 5 frames', len(topk[5]) == 5, 'got %s' % list(topk[5]))
    check('top-10 holds 10 frames', len(topk[10]) == 10, 'got %s' % list(topk[10]))
    check('top-3 is ranked by the second network, not the first',
          list(topk[3]) == list(shortlist[np.argsort(probs2[:, 1])[::-1]][:3]), 'got %s' % list(topk[3]))
    check('top-3 is a prefix of top-5', list(topk[3]) == list(topk[5])[:3])
    check('every selection is made of real frame indices',
          all(0 <= int(i) < len(scores) for k in topk for i in topk[k]))
    check('selections hold no duplicates', all(len(set(topk[k].tolist())) == len(topk[k]) for k in topk))

    print('\ncalc_topk_indices, shortlist of 3, must return without raising')
    shortlist3 = get_top_k_indices(scores, 3)
    probs2_3 = np.zeros((3, 2))
    probs2_3[:, 1] = np.array([0.3, 0.9, 0.5])
    probs2_3[:, 0] = 1 - probs2_3[:, 1]
    best_idx3, topk3, _ = calc_topk_indices(shortlist3, probs2_3, scores, 0.0, [3])
    check('a shortlist of 3 returns without raising', set(topk3.keys()) == {3})
    check('top-3 holds the 3 shortlisted frames', sorted(topk3[3].tolist()) == sorted(shortlist3.tolist()),
          'got %s' % list(topk3[3]))
    check('best frame is the second network argmax', best_idx3 == int(shortlist3[1]), 'got %s' % best_idx3)

    print('\ncalc_topk_indices, fewer relevant frames than k, the all-relevant mode')
    few = np.array([4, 7])
    probs2_few = np.array([[0.4, 0.6], [0.2, 0.8]])
    _, topk_few, _ = calc_topk_indices(few, probs2_few, scores, 0.0, [3, 5])
    check('top-3 is padded up to 3 frames', len(topk_few[3]) == 3, 'got %s' % list(topk_few[3]))
    check('top-5 is padded up to 5 frames', len(topk_few[5]) == 5, 'got %s' % list(topk_few[5]))
    check('the two re-ranked frames come first', list(topk_few[3])[:2] == [7, 4], 'got %s' % list(topk_few[3]))
    check('padding holds no duplicates', len(set(topk_few[5].tolist())) == 5, 'got %s' % list(topk_few[5]))
    check('padding uses the best remaining first network frames',
          int(topk_few[3][2]) == int([i for i in np.argsort(scores)[::-1] if i not in (4, 7)][0]),
          'got %s' % list(topk_few[3]))

    print('\ncalc_topk_indices, threshold fallback')
    probs2_low = np.zeros((5, 2))
    probs2_low[:, 1] = np.array([0.1, 0.2, 0.15, 0.05, 0.12])
    probs2_low[:, 0] = 1 - probs2_low[:, 1]
    shortlist5 = get_top_k_indices(scores, 5)
    best_low, _, max_low = calc_topk_indices(shortlist5, probs2_low, scores, 0.5, [3, 5])
    check('below the threshold the best suboptimal frame is taken',
          best_low == int(shortlist5[int(np.argmax(probs2_low[:, 0]))]), 'got %s' % best_low)
    check('max_score still reports the optimal probability', abs(max_low - 0.2) < 1e-9, 'got %s' % max_low)

    print('\nregression against a fixed top-5 reference implementation at shortlist size 5')
    #reference implementation of a fixed top-5 shortlist with top-3 and top-5 re-ranking, restricted to a
    #shortlist of exactly 5, where no fallback for shorter shortlists is needed
    def original_shortlist(scores_in):
        idx5 = np.argpartition(scores_in, -5)[-5:]
        return idx5[np.argsort(scores_in[idx5])[::-1]]

    def original_calc_top_1_5(idx_relevant, probs2_in, scores_base_network, threshold_in):
        num_rel = len(probs2_in)
        scores_in = probs2_in[:, 1]
        max_score_in = np.max(scores_in)
        best_idx2_in = int(np.argmax(scores_in))
        if max_score_in < threshold_in:
            max_sub_scores = probs2_in[:, 0]
            best_idx2_in = int(np.argmax(max_sub_scores))
        best_idx_in = idx_relevant[best_idx2_in]
        idx3_in = idx_relevant[np.argsort(scores_in)[::-1]][:3] if num_rel >= 3 else None
        idx5_in = idx_relevant[np.argsort(scores_in)[::-1]][:5] if num_rel >= 5 else None
        return best_idx_in, idx3_in, idx5_in, max_score_in

    rng = np.random.RandomState(20261028)
    mismatches = 0
    for trial in range(2000):
        n_frames = int(rng.randint(20, 900))
        trial_scores = rng.rand(n_frames)
        old_shortlist = original_shortlist(trial_scores)
        new_shortlist = get_top_k_indices(trial_scores, 5)
        trial_probs2 = np.zeros((5, 2))
        trial_probs2[:, 1] = rng.rand(5)
        trial_probs2[:, 0] = 1 - trial_probs2[:, 1]
        old_best, old_idx3, old_idx5, old_max = original_calc_top_1_5(old_shortlist, trial_probs2,
                                                                      trial_scores, 0.0)
        new_best, new_topk, new_max = calc_topk_indices(new_shortlist, trial_probs2, trial_scores, 0.0, [3, 5])
        if (list(old_shortlist) != list(new_shortlist) or int(old_best) != int(new_best)
                or list(old_idx3) != list(new_topk[3]) or list(old_idx5) != list(new_topk[5])
                or abs(old_max - new_max) > 1e-12):
            mismatches += 1
    check('2000 random scans give byte-identical selections at shortlist size 5', mismatches == 0,
          'mismatches: %d' % mismatches)

    print('\nlabel handling, labels follow the frame order whatever the csv row order')
    import pandas as pd
    prepare_labels = ns['prepare_labels']
    validate_scan_labels = ns['validate_scan_labels']

    #a scan of 8 frames whose frame 3 is the optimal one
    n = 8
    rows = [{'Filename': 'scan_a.mha', 'Frame': i, 'Label': (1 if i == 3 else 0)} for i in range(n)]
    ordered = pd.DataFrame(rows)
    reversed_rows = pd.DataFrame(list(reversed(rows)))

    def frame_labels_in_row_order(df, scan_id='scan_a'):
        """a lookup that reads the labels in csv row order, without sorting by frame"""
        scan_df = df[df['Filename'] == scan_id + '.mha']
        mapping = {0: 'irrelevant', 1: 'optimal', 2: 'suboptimal'}
        return [mapping.get(v, v) for v in list(scan_df['Label'])]

    check('reversing the csv rows does change the row order lookup',
          frame_labels_in_row_order(ordered)[3] == 'optimal'
          and frame_labels_in_row_order(reversed_rows)[3] != 'optimal')
    check('prepare_labels restores the frame order',
          frame_labels_in_row_order(prepare_labels(reversed_rows))[3] == 'optimal')
    check('prepare_labels leaves already ordered labels alone',
          list(prepare_labels(ordered)['Frame']) == list(range(n)))

    for bad_name, bad_df, pattern in (
            ('missing columns', pd.DataFrame({'Row Labels': ['scan_a.mha']}), 'columns'),
            ('duplicated frames', pd.DataFrame(rows + [rows[3]]), 'duplicated')):
        try:
            prepare_labels(bad_df)
            check('prepare_labels rejects %s' % bad_name, False)
        except ValueError as e:
            check('prepare_labels rejects %s' % bad_name, pattern in str(e), 'message: %s' % e)

    good = prepare_labels(ordered)
    try:
        validate_scan_labels('scan_a', good, n)
        check('validate_scan_labels accepts complete labels', True)
    except Exception as e:
        check('validate_scan_labels accepts complete labels', False, str(e))
    for bad_name, df_in, n_frames in (
            ('a frame count mismatch', good, n + 100),
            ('a scan with no rows', good.assign(Filename='other.mha'), n),
            ('gaps in the frame numbers', prepare_labels(pd.DataFrame(
                [r for r in rows if r['Frame'] != 5])), n - 1)):
        try:
            validate_scan_labels('scan_a', df_in, n_frames)
            check('validate_scan_labels rejects %s' % bad_name, False)
        except ValueError:
            check('validate_scan_labels rejects %s' % bad_name, True)

    print('\ntop_frames is filled for every shortlist size')
    for K in (3, 5, 10):
        requested = [3, 5, 10]
        shortlist_k = get_top_k_indices(scores, K)
        probs2_k = np.zeros((K, 2))
        probs2_k[:, 1] = np.linspace(0.2, 0.8, K)
        probs2_k[:, 0] = 1 - probs2_k[:, 1]
        _, topk_k, _ = calc_topk_indices(shortlist_k, probs2_k, scores, 0.0,
                                         [k for k in requested if k <= K])
        available = sorted(topk_k)
        top_frames = list(topk_k[available[-1]]) if available else None
        check('K=%d fills top_frames with %d frames' % (K, K),
              top_frames is not None and len(top_frames) == K, 'got %s' % top_frames)
        check('K=%d keeps a per-k column for every served k' % K,
              all(len(topk_k[k]) == k for k in available))

    print('\nsummarizer refuses an incomparable sweep')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from summarize_shortlist_sizes import check_comparable
    sizes, folds = [3, 5, 10], 5
    complete = {(k, f): frozenset({'a.mha', 'b.mha'}) for k in sizes for f in range(folds)}
    check('a complete matching sweep passes',
          check_comparable(sizes, folds, [], complete, 'test') == [])
    short = dict(complete)
    del short[(10, 4)]
    check('a missing fold is caught', len(check_comparable(sizes, folds, [], short, 'test')) > 0)
    check('a missing file is caught',
          len(check_comparable(sizes, folds, ['/some/file.xlsx'], complete, 'test')) > 0)
    mismatched = dict(complete)
    mismatched[(10, 2)] = frozenset({'a.mha'})
    problems = check_comparable(sizes, folds, [], mismatched, 'test')
    check('different scans across sizes are caught', len(problems) > 0)
    check('the mismatch message names the fold and sizes',
          any('fold 2' in p and 'shortlist size 10' in p for p in problems),
          'got %s' % problems)

    print('\nstr2intlist')
    check('parses a comma separated list', str2intlist('3,5,10') == [3, 5, 10])
    check('sorts and de-duplicates', str2intlist('10,3,5,3') == [3, 5, 10])
    check('tolerates spaces', str2intlist(' 3 , 5 ') == [3, 5])
    for bad in ('', '0,3', '-1'):
        try:
            str2intlist(bad)
            check('rejects %r' % bad, False)
        except Exception:
            check('rejects %r' % bad, True)

    print('')
    if failures:
        print('%d check(s) FAILED: %s' % (len(failures), ', '.join(failures)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
