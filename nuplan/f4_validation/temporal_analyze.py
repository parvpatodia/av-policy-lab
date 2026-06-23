"""Does the temporal overlay (logged futures) fix the panel/human<->F4 null?
Compares static-panel, temporal-panel, and human against F4/s_inter on the same 40.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/temporal_analyze.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
VAL = REPO / 'data/scene_renders/validation_set'
man = json.loads((VAL / 'manifest_PRIVATE.json').read_text())
tasklist = json.loads((VAL / 'human_tasklist.json').read_text())
static = {nm: json.loads((REPO / f'data/panel_ratings_n80/{nm}.json').read_text())
          for nm in ['cautious', 'assertive', 'safety', 'av_eng']}

T = {
 'cautious': {"val_00":0.2,"val_07":0.1,"val_09":0.3,"val_10":0.2,"val_11":0.2,"val_12":0.4,"val_13":0.3,"val_14":0.2,"val_17":0.2,"val_18":0.2,"val_19":0.2,"val_21":0.4,"val_22":0.3,"val_24":0.2,"val_26":0.2,"val_32":0.4,"val_35":0.6,"val_36":0.5,"val_41":0.5,"val_42":0.2,"val_46":0.2,"val_47":0.3,"val_49":0.4,"val_50":0.4,"val_52":0.5,"val_54":0.2,"val_55":0.3,"val_56":0.4,"val_57":0.6,"val_59":0.2,"val_60":0.2,"val_63":0.2,"val_65":0.5,"val_66":0.3,"val_67":0.4,"val_69":0.4,"val_71":0.2,"val_73":0.2,"val_78":0.2,"val_79":0.1},
 'assertive': {"val_00":0.2,"val_07":0.1,"val_09":0.4,"val_10":0.2,"val_11":0.2,"val_12":0.5,"val_13":0.2,"val_14":0.3,"val_17":0.3,"val_18":0.2,"val_19":0.1,"val_21":0.3,"val_22":0.2,"val_24":0.2,"val_26":0.2,"val_32":0.4,"val_35":0.5,"val_36":0.6,"val_41":0.5,"val_42":0.1,"val_46":0.3,"val_47":0.5,"val_49":0.3,"val_50":0.2,"val_52":0.4,"val_54":0.2,"val_55":0.3,"val_56":0.3,"val_57":0.5,"val_59":0.1,"val_60":0.2,"val_63":0.2,"val_65":0.6,"val_66":0.2,"val_67":0.4,"val_69":0.5,"val_71":0.2,"val_73":0.3,"val_78":0.2,"val_79":0.0},
 'safety': {"val_00":0.3,"val_07":0.15,"val_09":0.25,"val_10":0.2,"val_11":0.2,"val_12":0.4,"val_13":0.55,"val_14":0.25,"val_17":0.2,"val_18":0.25,"val_19":0.2,"val_21":0.45,"val_22":0.25,"val_24":0.2,"val_26":0.35,"val_32":0.45,"val_35":0.55,"val_36":0.45,"val_41":0.45,"val_42":0.2,"val_46":0.15,"val_47":0.3,"val_49":0.5,"val_50":0.55,"val_52":0.5,"val_54":0.3,"val_55":0.25,"val_56":0.5,"val_57":0.7,"val_59":0.25,"val_60":0.3,"val_63":0.2,"val_65":0.5,"val_66":0.3,"val_67":0.5,"val_69":0.45,"val_71":0.25,"val_73":0.3,"val_78":0.3,"val_79":0.1},
 'av_eng': {"val_00":0.2,"val_07":0.1,"val_09":0.5,"val_10":0.2,"val_11":0.2,"val_12":0.4,"val_13":0.3,"val_14":0.2,"val_17":0.3,"val_18":0.2,"val_19":0.1,"val_21":0.3,"val_22":0.2,"val_24":0.2,"val_26":0.3,"val_32":0.5,"val_35":0.4,"val_36":0.6,"val_41":0.3,"val_42":0.6,"val_46":0.3,"val_47":0.3,"val_49":0.4,"val_50":0.3,"val_52":0.3,"val_54":0.2,"val_55":0.4,"val_56":0.3,"val_57":0.5,"val_59":0.2,"val_60":0.2,"val_63":0.2,"val_65":0.6,"val_66":0.4,"val_67":0.4,"val_69":0.5,"val_71":0.2,"val_73":0.5,"val_78":0.4,"val_79":0.3},
}

# human per val
hr = {r['display_id']: float(r['rating']) for r in csv.DictReader(open(REPO / 'human_ratings.csv')) if r['rating'] != ''}
hf = {}
for t in tasklist:
    if t['display_id'] in hr:
        hf.setdefault(t['file'], []).append(hr[t['display_id']])

files = sorted(T['cautious'].keys(), key=lambda s: int(s.split('_')[1]))
pngs = [f + '.png' for f in files]
f4 = np.array([man[p]['f4'] for p in pngs])
si = np.array([man[p]['s_inter'] for p in pngs])
human = np.array([np.mean(hf[p]) for p in pngs])
sp = np.array([np.mean([static[nm][f] for nm in static]) for f in files])
tp = np.array([np.mean([T[nm][f] for nm in T]) for f in files])

def r(a, b):
    rho, p = spearmanr(a, b); return f'{rho:+.3f} (p={p:.2f})'

print(f'n={len(files)} (the human-rated subset)\n')
print('                       vs F4        vs s_inter    vs human')
print(f'static panel    : {r(sp,f4):>14} {r(sp,si):>14} {r(sp,human):>14}')
print(f'temporal panel  : {r(tp,f4):>14} {r(tp,si):>14} {r(tp,human):>14}')
print(f'human           : {r(human,f4):>14} {r(human,si):>14}        --')
print(f'\nmeans: static panel {sp.mean():.2f}, temporal panel {tp.mean():.2f}, human {human.mean():.2f}')
print(f'temporal panel vs static panel agreement: {r(tp,sp)}')
# did high-s_inter scenes drop with temporal info? (over-fire check)
hi = si > 0.6
print(f'\non s_inter>0.6 scenes (n={hi.sum()}): static panel {sp[hi].mean():.2f} -> temporal {tp[hi].mean():.2f}, human {human[hi].mean():.2f}')
print(f'on s_inter=0 scenes  (n={int((si<1e-6).sum())}): static {sp[si<1e-6].mean():.2f} -> temporal {tp[si<1e-6].mean():.2f}, human {human[si<1e-6].mean():.2f}')
