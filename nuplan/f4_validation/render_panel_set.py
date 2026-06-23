"""Render a stratified, ANONYMIZED scene set for the Signal B rater panel.
Filenames carry no F4 info (scene_NN.png); a private manifest maps them back to
tokens + F4 components for scoring after the panel rates them blind.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/render_panel_set.py [N_per_band]
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from scene_loader import build_scenarios, index_by_token, load_f4_scores
from render_scene import render

OUT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders/panel_pilot')
MANIFEST = OUT / 'manifest_PRIVATE.json'


def pick(f4, per_band=5, seed=7):
    rng = random.Random(seed)
    bands = {'zero': [], 'hi_inter': [], 'hi_branch': [], 'mid': []}
    for t, v in f4.items():
        si = v.get('s_inter') or 0.0
        sb = v.get('s_branch') or 0.0
        f = v.get('f4') or 0.0
        if f < 1e-6:
            bands['zero'].append(t)
        elif si > 0.6 and sb < 0.34:
            bands['hi_inter'].append(t)
        elif sb > 0.6 and si < 0.2:
            bands['hi_branch'].append(t)
        elif 0.3 <= f <= 0.7:
            bands['mid'].append(t)
    sel = []
    for b, toks in bands.items():
        rng.shuffle(toks)
        for t in toks[:per_band]:
            sel.append((b, t))
    rng.shuffle(sel)  # mix bands so the order leaks nothing
    return sel


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    f4 = load_f4_scores()
    sel = pick(f4, per_band=per)
    print(f'rendering {len(sel)} scenes...')
    by_tok = index_by_token(build_scenarios(tokens=[t for _, t in sel]))
    manifest = {}
    for i, (band, t) in enumerate(sel):
        if t not in by_tok:
            continue
        name = f'scene_{i:02d}.png'
        render(by_tok[t], OUT / name)
        v = f4[t]
        manifest[name] = {'token': t, 'band': band, 'f4': v['f4'],
                          's_inter': v['s_inter'], 's_branch': v.get('s_branch'),
                          'scenario_type': v.get('scenario_type')}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f'  rendered {len(manifest)} -> {OUT}\n  manifest -> {MANIFEST}')
    print('  files:', sorted(p.name for p in OUT.glob('scene_*.png')))


if __name__ == '__main__':
    main()
