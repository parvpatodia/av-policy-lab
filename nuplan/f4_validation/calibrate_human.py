"""Calibrate the AI panel against Parv's human ratings, and compute the gold
human-vs-F4 convergent validity. Handles the hidden test-retest repeats.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/calibrate_human.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
VAL = REPO / 'data/scene_renders/validation_set'
tasklist = json.loads((VAL / 'human_tasklist.json').read_text())
manifest = json.loads((VAL / 'manifest_PRIVATE.json').read_text())
panel = {nm: json.loads((REPO / f'data/panel_ratings_n80/{nm}.json').read_text())
         for nm in ['cautious', 'assertive', 'safety', 'av_eng']}

# human csv
hr = {}
with open(REPO / 'human_ratings.csv') as f:
    for row in csv.DictReader(f):
        if row['rating'] != '':
            hr[row['display_id']] = float(row['rating'])

# group human ratings by val file (repeats -> 2 ratings)
by_file = {}
for t in tasklist:
    did = t['display_id']
    if did in hr:
        by_file.setdefault(t['file'], []).append(hr[did])

# test-retest: files rated twice
retest = [(v[0], v[1]) for v in by_file.values() if len(v) >= 2]
print(f'human rated {len(by_file)} unique scenes; {len(retest)} test-retest pairs')
if retest:
    a = np.array([x[0] for x in retest]); b = np.array([x[1] for x in retest])
    exact = np.mean(a == b)
    mad = np.mean(np.abs(a - b))
    print(f'  test-retest: exact-agree {exact:.2f}, mean abs diff {mad:.3f}, '
          f'Pearson {np.corrcoef(a,b)[0,1]:.3f}' if len(set(a))>1 else
          f'  test-retest: exact-agree {exact:.2f}, mean abs diff {mad:.3f}')

# one human score per val (mean of its ratings)
files = sorted(by_file, key=lambda s: int(s.split('_')[1].split('.')[0]))
human = np.array([np.mean(by_file[f]) for f in files])
stem = [f.replace('.png', '') for f in files]
pmean = np.array([np.mean([panel[nm][s] for nm in panel]) for s in stem])
f4 = np.array([manifest[f]['f4'] for f in files])
si = np.array([manifest[f]['s_inter'] for f in files])
sb = np.array([manifest[f].get('s_branch') or 0.0 for f in files])
bands = [manifest[f]['band'] for f in files]
print(f'\nhuman rating distribution: mean {human.mean():.2f}, '
      f'max {human.max():.2f}, frac=0 {np.mean(human<1e-6):.2f}')

def rep(label, x, y):
    rho, p = spearmanr(x, y)
    print(f'  {label:26s}: Spearman rho={rho:.3f}  p={p:.2e}')

print('\n[GOLD] human vs F4 and components (n={}):'.format(len(files)))
rep('human vs F4 combined', human, f4)
rep('human vs S_inter', human, si)
rep('human vs S_branch', human, sb)

print('\n[CALIBRATION] panel vs human (is the AI panel a faithful proxy?):')
rho, p = spearmanr(pmean, human)
print(f'  panel_mean vs human: Spearman rho={rho:.3f} p={p:.2e}  '
      f'Pearson {np.corrcoef(pmean,human)[0,1]:.3f}  mean|panel-human| {np.mean(np.abs(pmean-human)):.3f}')
print('  per-persona vs human:')
for nm in panel:
    pv = np.array([panel[nm][s] for s in stem])
    print(f'    {nm:10s}: Spearman {spearmanr(pv,human)[0]:.3f}  bias(mean {np.mean(pv):.2f} vs human {human.mean():.2f})')

print('\n[group contrast] human f4=0 vs f4>0:')
g0 = human[f4 < 1e-6]; g1 = human[f4 >= 1e-6]
if len(g0) and len(g1):
    U, p = mannwhitneyu(g1, g0, alternative='greater'); rbc = 2*U/(len(g1)*len(g0))-1
    print(f'  f4=0 (n={len(g0)}) mean {g0.mean():.2f} vs f4>0 (n={len(g1)}) mean {g1.mean():.2f}; '
          f'MW p={p:.2e} rank-biserial={rbc:.3f}')

print('\nhuman mean by band:')
for b in ['zero', 'mid', 'hi_branch', 'hi_inter']:
    vs = [human[i] for i in range(len(files)) if bands[i] == b]
    if vs:
        print(f'  {b:10s} n={len(vs)} human mean {np.mean(vs):.2f}  (panel {np.mean([pmean[i] for i in range(len(files)) if bands[i]==b]):.2f})')


if __name__ == '__main__':
    pass
