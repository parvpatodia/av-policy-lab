"""Decisive test: human ratings on TEMPORAL renders (futures shown) vs F4.
If still null -> the over-fire conclusion is locked (not a render artifact).
Also: did the temporal render change Parv's ratings vs the static ones (same scenes)?

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/decisive_human_temporal.py
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
VAL = REPO / 'data/scene_renders/validation_set'
TVAL = REPO / 'data/scene_renders/validation_set_temporal'
man = json.loads((VAL / 'manifest_PRIVATE.json').read_text())

# ht_NN -> val file: parse the ITEMS array embedded in the temporal tool (exact mapping)
html = (TVAL / 'rate_temporal.html').read_text()
items = json.loads(re.search(r'const ITEMS=(\[.*?\]);', html).group(1))
ht2file = {it['id']: it['file'] for it in items}

# human temporal ratings
hT = {r['display_id']: float(r['rating']) for r in csv.DictReader(open(REPO / 'human_ratings_temporal.csv')) if r['rating'] != ''}
# human STATIC ratings (via the static tasklist) -> per val file
stask = json.loads((VAL / 'human_tasklist.json').read_text())
hS_raw = {r['display_id']: float(r['rating']) for r in csv.DictReader(open(REPO / 'human_ratings.csv')) if r['rating'] != ''}
hS_file = {}
for t in stask:
    if t['display_id'] in hS_raw:
        hS_file.setdefault(t['file'], []).append(hS_raw[t['display_id']])

# align on val files present in the temporal rating set
files = [ht2file[k] for k in sorted(hT)]
ht = np.array([hT[k] for k in sorted(hT)])
f4 = np.array([man[f]['f4'] for f in files])
si = np.array([man[f]['s_inter'] for f in files])
sb = np.array([man[f].get('s_branch') or 0.0 for f in files])
hs = np.array([np.mean(hS_file[f]) if f in hS_file else np.nan for f in files])

def rr(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    rho, p = spearmanr(a, b); return f'rho={rho:+.3f} (p={p:.2f})'

print(f'n={len(files)} temporal-rated scenes')
print(f'human-temporal distribution: mean {ht.mean():.2f}, max {ht.max():.2f}, frac=0 {np.mean(ht<1e-6):.2f}\n')
print('[DECISIVE] human-on-TEMPORAL vs F4 / components:')
print(f'  vs F4 combined : {rr(ht, f4)}')
print(f'  vs S_inter     : {rr(ht, si)}')
print(f'  vs S_branch    : {rr(ht, sb)}')
print(f'\n  (recall human-on-STATIC vs F4 was rho=+0.02; vs s_inter -0.05)')

g0, g1 = ht[f4 < 1e-6], ht[f4 >= 1e-6]
if len(g0) and len(g1):
    U, p = mannwhitneyu(g1, g0, alternative='greater'); rbc = 2*U/(len(g1)*len(g0))-1
    print(f'\n  group contrast f4=0 (n={len(g0)}) mean {g0.mean():.2f} vs f4>0 (n={len(g1)}) mean {g1.mean():.2f}; MW p={p:.2f} rbc={rbc:.3f}')

m = ~np.isnan(hs)
print(f'\n[render effect] human static vs temporal on the same {m.sum()} scenes:')
print(f'  Spearman(static, temporal) = {rr(ht, hs, m)}   means: static {np.nanmean(hs):.2f} -> temporal {ht.mean():.2f}')
print(f'  mean |temporal - static| = {np.mean(np.abs(ht[m]-hs[m])):.3f}')

# did high-s_inter scenes move up with futures shown?
hi = si > 0.6
if hi.sum():
    print(f'\n  on s_inter>0.6 (n={hi.sum()}): human static {np.nanmean(hs[hi]):.2f} -> temporal {ht[hi].mean():.2f}')
