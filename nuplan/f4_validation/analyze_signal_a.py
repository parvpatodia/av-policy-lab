"""Full Signal A analysis: a_gt (logged-future conflict) convergent with S_inter,
the component it measures. Two-part stats per F4_VALIDATION_PROTOCOL on the real
(non-stratified) 5,604-scene distribution:
  1. group contrast s_inter=0 vs s_inter>0: Cliff's delta + bootstrap CI + perm p
  2. within s_inter>0: Spearman rho + bootstrap CI + permutation p
  3. partial Spearman(s_inter, a_gt | n_par, v0)  -- ambiguity beyond busyness
  4. Holm across the primary p-values
Also: the s_inter=0-but-a_gt>0 "constant-turn-rate blind spot" rate.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/analyze_signal_a.py
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
RNG = np.random.default_rng(0)
B = 2000


def cliffs_delta(x, y):
    # P(x>y) - P(x<y); x, y are 1D arrays. O(n log n) via ranks.
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    r = rankdata(allv)
    rx = r[:nx].sum()
    # Mann-Whitney U for x over y
    U = rx - nx * (nx + 1) / 2.0
    return 2.0 * U / (nx * ny) - 1.0


def partial_spearman(a, b, controls):
    """Spearman partial correlation of a,b controlling for columns in `controls`."""
    ra = rankdata(a); rb = rankdata(b)
    C = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in controls])
    # residualize
    beta_a, *_ = np.linalg.lstsq(C, ra, rcond=None)
    beta_b, *_ = np.linalg.lstsq(C, rb, rcond=None)
    res_a = ra - C @ beta_a
    res_b = rb - C @ beta_b
    if res_a.std() < 1e-12 or res_b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(res_a, res_b)[0, 1])


def boot_ci(stat_fn, n, b=B):
    vals = []
    for _ in range(b):
        idx = RNG.integers(0, n, n)
        vals.append(stat_fn(idx))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    f4 = json.loads((REPO / 'data/f4_scores_v11.json').read_text())
    sa = json.loads((REPO / 'data/signal_a_results.json').read_text())

    toks = [t for t in f4 if t in sa and sa[t].get('a_gt') is not None
            and sa[t]['a_gt'] == sa[t]['a_gt']]
    si = np.array([f4[t]['s_inter'] for t in toks])
    ag = np.array([sa[t]['a_gt'] for t in toks])
    npar = np.array([f4[t].get('n_par') or 0 for t in toks], dtype=float)
    v0 = np.array([f4[t].get('v0') or 0.0 for t in toks])
    print(f'N = {len(toks)} scenes (real distribution)')
    print(f'  s_inter: frac=0 {np.mean(si<1e-6):.3f}  frac>0 {np.mean(si>=1e-6):.3f}')

    m0 = si < 1e-6
    m1 = ~m0
    g0, g1 = ag[m0], ag[m1]
    print('\n[1] GROUP CONTRAST  a_gt: s_inter=0 vs s_inter>0')
    print(f'  s_inter=0 (n={m0.sum()}): mean {g0.mean():.3f}  median {np.median(g0):.3f}  frac>0 {np.mean(g0>1e-6):.3f}')
    print(f'  s_inter>0 (n={m1.sum()}): mean {g1.mean():.3f}  median {np.median(g1):.3f}  frac>0 {np.mean(g1>1e-6):.3f}')
    delta = cliffs_delta(g1, g0)

    def delta_boot(idx):
        s = si[idx]; a = ag[idx]
        a0 = a[s < 1e-6]; a1 = a[s >= 1e-6]
        if len(a0) < 2 or len(a1) < 2:
            return np.nan
        return cliffs_delta(a1, a0)
    dlo, dhi = boot_ci(delta_boot, len(toks))
    # permutation: shuffle group labels
    obs = g1.mean() - g0.mean()
    perm = []
    lab = m1.astype(int)
    for _ in range(B):
        p = RNG.permutation(lab)
        perm.append(ag[p == 1].mean() - ag[p == 0].mean())
    pval = (np.sum(np.array(perm) >= obs) + 1) / (B + 1)
    print(f"  Cliff's delta {delta:.3f}  95% CI [{dlo:.3f}, {dhi:.3f}]")
    print(f'  permutation p (mean diff, one-sided) = {pval:.2e}')

    print('\n[2] WITHIN s_inter>0  Spearman(s_inter, a_gt)')
    rho, _ = spearmanr(si[m1], ag[m1])
    idx1 = np.flatnonzero(m1)

    def rho_boot(idx):
        ii = idx1[RNG.integers(0, len(idx1), len(idx1))]
        return spearmanr(si[ii], ag[ii])[0]
    rlo, rhi = boot_ci(lambda _: rho_boot(_), len(idx1))
    # permutation null
    permr = [spearmanr(si[m1], RNG.permutation(ag[m1]))[0] for _ in range(B)]
    prho = (np.sum(np.array(permr) >= rho) + 1) / (B + 1)
    print(f'  rho {rho:.3f}  95% CI [{rlo:.3f}, {rhi:.3f}]  perm p {prho:.2e}  (n={m1.sum()})')

    print('\n[3] PARTIAL Spearman(s_inter, a_gt | n_par, v0)  -- beyond busyness')
    pr = partial_spearman(si, ag, [npar, v0])
    rho_all, _ = spearmanr(si, ag)
    print(f'  zero-order all-scenes rho {rho_all:.3f}   partial rho {pr:.3f}')

    print('\n[4] HOLM across primary p-values (group, within)')
    ps = sorted([('group_contrast', pval), ('within_conflict', prho)], key=lambda x: x[1])
    m = len(ps)
    for i, (name, p) in enumerate(ps):
        adj = min(p * (m - i), 1.0)
        print(f'  {name}: p={p:.2e}  holm-adj={adj:.2e}  {"PASS" if adj<0.05 else "n.s."}')

    print('\n[blind spot] s_inter=0 but a_gt>0 (constant-turn-rate missed a real crossing)')
    print(f'  rate among s_inter=0: {np.mean(g0>1e-6):.3f}  (n={int(np.sum(g0>1e-6))}/{m0.sum()})')
    print(f'  rate among s_inter=0, a_gt>0.5 (strong miss): {np.mean(g0>0.5):.3f}')


if __name__ == '__main__':
    main()
