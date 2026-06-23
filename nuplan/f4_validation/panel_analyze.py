"""Signal B pilot analysis: 4-persona AI rater panel vs F4 and its components.
Ratings were produced blind (anonymized scene_NN.png) by 4 persona subagents.
Reports inter-rater reliability (Cronbach alpha + mean pairwise corr) and the
panel score's convergence with F4 / S_inter / S_branch.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/panel_analyze.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
MAN = REPO / 'data/scene_renders/panel_pilot/manifest_PRIVATE.json'
RATE_DIR = REPO / 'data/panel_pilot_ratings'

RATINGS = {
 'cautious':  {"scene_00":0.8,"scene_01":0.4,"scene_02":0.1,"scene_03":0.3,"scene_04":0.5,"scene_05":0.3,"scene_06":0.65,"scene_07":0.5,"scene_08":0.45,"scene_09":0.3,"scene_10":0.95,"scene_11":0.7,"scene_12":0.55,"scene_13":0.7,"scene_14":0.65,"scene_15":0.9,"scene_16":0.8,"scene_17":0.3,"scene_18":0.8,"scene_19":0.35},
 'assertive': {"scene_00":0.8,"scene_01":0.2,"scene_02":0.1,"scene_03":0.2,"scene_04":0.25,"scene_05":0.35,"scene_06":0.6,"scene_07":0.4,"scene_08":0.35,"scene_09":0.25,"scene_10":0.85,"scene_11":0.65,"scene_12":0.6,"scene_13":0.6,"scene_14":0.25,"scene_15":0.85,"scene_16":0.6,"scene_17":0.15,"scene_18":0.65,"scene_19":0.25},
 'safety':    {"scene_00":0.8,"scene_01":0.3,"scene_02":0.1,"scene_03":0.2,"scene_04":0.5,"scene_05":0.3,"scene_06":0.6,"scene_07":0.55,"scene_08":0.45,"scene_09":0.3,"scene_10":0.85,"scene_11":0.7,"scene_12":0.35,"scene_13":0.7,"scene_14":0.3,"scene_15":0.9,"scene_16":0.7,"scene_17":0.15,"scene_18":0.7,"scene_19":0.3},
 'av_eng':    {"scene_00":0.7,"scene_01":0.25,"scene_02":0.1,"scene_03":0.3,"scene_04":0.65,"scene_05":0.4,"scene_06":0.6,"scene_07":0.35,"scene_08":0.35,"scene_09":0.15,"scene_10":0.9,"scene_11":0.65,"scene_12":0.5,"scene_13":0.75,"scene_14":0.55,"scene_15":0.8,"scene_16":0.6,"scene_17":0.25,"scene_18":0.7,"scene_19":0.3},
}


def cronbach_alpha(M):  # M: (n_items, k_raters)
    k = M.shape[1]
    item_var = M.var(axis=1, ddof=1)  # not used; alpha uses across-item var per rater
    var_raters = M.var(axis=0, ddof=1).sum()
    total = M.sum(axis=1)
    var_total = total.var(ddof=1)
    return (k / (k - 1)) * (1 - var_raters / var_total)


def main():
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    for name, r in RATINGS.items():
        (RATE_DIR / f'{name}.json').write_text(json.dumps(r))
    man = json.loads(MAN.read_text())
    scenes = sorted(man.keys())
    stem = {s: s.replace('.png', '') for s in scenes}
    raters = list(RATINGS.keys())
    M = np.array([[RATINGS[r][stem[s]] for r in raters] for s in scenes])  # (20,4)

    print(f'panel pilot: {len(scenes)} scenes x {len(raters)} personas')
    # reliability
    alpha = cronbach_alpha(M)
    pair = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            pair.append(np.corrcoef(M[:, i], M[:, j])[0, 1])
    print(f'  Cronbach alpha = {alpha:.3f}   mean pairwise Pearson = {np.mean(pair):.3f} '
          f'(range {min(pair):.2f}-{max(pair):.2f})')

    panel = M.mean(axis=1)
    f4 = np.array([man[s]['f4'] for s in scenes])
    si = np.array([man[s]['s_inter'] for s in scenes])
    sb = np.array([man[s].get('s_branch') or 0.0 for s in scenes])

    print('\n  convergence (Spearman, n=20 PILOT):')
    for label, x in [('F4 combined', f4), ('S_inter', si), ('S_branch', sb)]:
        rho, p = spearmanr(panel, x)
        print(f'    panel vs {label:12s}: rho={rho:.3f}  p={p:.2e}')

    # group contrast f4=0 vs f4>0
    g0 = panel[f4 < 1e-6]; g1 = panel[f4 >= 1e-6]
    U, p = mannwhitneyu(g1, g0, alternative='greater')
    rbc = 2 * U / (len(g1) * len(g0)) - 1
    print(f'\n  group contrast panel: f4=0 (n={len(g0)}) mean {g0.mean():.2f} vs '
          f'f4>0 (n={len(g1)}) mean {g1.mean():.2f}; MW p={p:.2e} rank-biserial={rbc:.3f}')

    # per-band means
    print('\n  panel mean by design band:')
    bands = {}
    for s in scenes:
        bands.setdefault(man[s]['band'], []).append(panel[scenes.index(s)])
    for b, vs in sorted(bands.items()):
        print(f'    {b:10s} n={len(vs)}  panel mean {np.mean(vs):.2f}')


if __name__ == '__main__':
    main()
