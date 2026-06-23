"""Pilot: Signal A (a_gt, logged-future conflict) convergent with s_inter, the
component it measures. Stratified ~240-scene sample across the s_inter range.
Two-part stats per F4_VALIDATION_PROTOCOL: group contrast (s_inter=0 vs >0) +
Spearman within s_inter>0. Reports coverage too.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/pilot_signal_a.py
"""
from __future__ import annotations
import random
import numpy as np

from scene_loader import build_scenarios, index_by_token, load_f4_scores
from signal_a_gt_conflict import a_gt


def stratified_tokens(f4, per_bin=60, seed=0):
    rng = random.Random(seed)
    bins = {'zero': [], 'low': [], 'med': [], 'high': []}
    for t, v in f4.items():
        si = v.get('s_inter')
        if si is None:
            continue
        if si < 1e-6:
            bins['zero'].append(t)
        elif si <= 0.33:
            bins['low'].append(t)
        elif si <= 0.66:
            bins['med'].append(t)
        else:
            bins['high'].append(t)
    sel = []
    for b, toks in bins.items():
        rng.shuffle(toks)
        sel += toks[:per_bin]
        print(f'  bin {b}: {len(toks)} avail, took {min(per_bin, len(toks))}')
    return sel


def main():
    f4 = load_f4_scores()
    toks = stratified_tokens(f4)
    print(f'pilot tokens: {len(toks)}; building scenarios...')
    by_tok = index_by_token(build_scenarios(tokens=toks))

    rows = []
    n_nan = 0
    for t in toks:
        if t not in by_tok:
            continue
        r = a_gt(by_tok[t])
        ag = r['a_gt']
        si = f4[t]['s_inter']
        if ag != ag:  # nan
            n_nan += 1
            continue
        rows.append((si, ag, r.get('reason'), r.get('n_conflict', 0)))

    si_arr = np.array([r[0] for r in rows])
    ag_arr = np.array([r[1] for r in rows])
    print(f'\nscored {len(rows)} scenes, {n_nan} nan (no route)')

    # group contrast: s_inter==0 vs s_inter>0
    g0 = ag_arr[si_arr < 1e-6]
    g1 = ag_arr[si_arr >= 1e-6]
    print(f'  a_gt | s_inter=0 (n={len(g0)}): mean={g0.mean():.3f} median={np.median(g0):.3f} '
          f'frac>0={np.mean(g0>1e-6):.2f}')
    print(f'  a_gt | s_inter>0 (n={len(g1)}): mean={g1.mean():.3f} median={np.median(g1):.3f} '
          f'frac>0={np.mean(g1>1e-6):.2f}')

    try:
        from scipy.stats import spearmanr, mannwhitneyu
        U, p = mannwhitneyu(g1, g0, alternative='greater')
        # rank-biserial effect size
        rbc = 2 * U / (len(g1) * len(g0)) - 1
        print(f'  group contrast Mann-Whitney (s_inter>0 > s_inter=0): p={p:.2e} rank-biserial={rbc:.3f}')
        m = si_arr >= 1e-6
        rho, prho = spearmanr(si_arr[m], ag_arr[m])
        print(f'  within s_inter>0 Spearman(s_inter, a_gt): rho={rho:.3f} p={prho:.2e} (n={m.sum()})')
        rho_all, p_all = spearmanr(si_arr, ag_arr)
        print(f'  all-scenes Spearman: rho={rho_all:.3f} p={p_all:.2e}')
    except ImportError:
        print('  [scipy missing — raw separation only]')


if __name__ == '__main__':
    main()
