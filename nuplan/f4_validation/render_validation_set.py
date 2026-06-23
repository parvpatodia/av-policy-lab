"""Render the shared Signal B validation set (N=80, stratified 20/band), anonymized.
Emits:
  - manifest_PRIVATE.json : val_NN.png -> token + F4 components (for scoring; NOT shown to raters)
  - human_tasklist.json   : 40-scene human subset (10/band) + 10 test-retest repeats,
                            shuffled, display ids h_NN -> val file (repeats hidden)
The AI panel rates all 80 val_*.png; the human rates the 50-item tasklist. Overlap
gives direct human<->panel calibration; repeats give intra-rater reliability.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/render_validation_set.py
"""
from __future__ import annotations
import json, random
from pathlib import Path
from scene_loader import build_scenarios, index_by_token, load_f4_scores
from render_scene import render

OUT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders/validation_set')
PER_BAND = 20
HUMAN_PER_BAND = 10
N_REPEAT = 10
SEED = 13


def bands_of(f4):
    b = {'zero': [], 'hi_inter': [], 'hi_branch': [], 'mid': []}
    for t, v in f4.items():
        si = v.get('s_inter') or 0.0; sb = v.get('s_branch') or 0.0; f = v.get('f4') or 0.0
        if f < 1e-6: b['zero'].append(t)
        elif si > 0.6 and sb < 0.34: b['hi_inter'].append(t)
        elif sb > 0.6 and si < 0.2: b['hi_branch'].append(t)
        elif 0.3 <= f <= 0.7: b['mid'].append(t)
    return b


def main():
    rng = random.Random(SEED)
    f4 = load_f4_scores()
    b = bands_of(f4)
    for k, v in b.items():
        rng.shuffle(v)
        print(f'  band {k}: {len(v)} avail')
    sel = []  # (band, token)
    for k in ['zero', 'hi_inter', 'hi_branch', 'mid']:
        for t in b[k][:PER_BAND]:
            sel.append((k, t))
    rng.shuffle(sel)

    by_tok = index_by_token(build_scenarios(tokens=[t for _, t in sel]))
    manifest = {}
    val_by_band = {}
    for i, (band, t) in enumerate(sel):
        if t not in by_tok:
            continue
        name = f'val_{i:02d}.png'
        render(by_tok[t], OUT / name)
        manifest[name] = {'token': t, 'band': band, 'f4': f4[t]['f4'],
                          's_inter': f4[t]['s_inter'], 's_branch': f4[t].get('s_branch'),
                          'scenario_type': f4[t].get('scenario_type')}
        val_by_band.setdefault(band, []).append(name)
    (OUT / 'manifest_PRIVATE.json').write_text(json.dumps(manifest, indent=2))

    # human subset: 10/band + repeats
    human_files = []
    for band, files in val_by_band.items():
        rng.shuffle(files)
        human_files += files[:HUMAN_PER_BAND]
    repeats = rng.sample(human_files, min(N_REPEAT, len(human_files)))
    items = [{'file': f, 'is_repeat': False} for f in human_files] + \
            [{'file': f, 'is_repeat': True} for f in repeats]
    rng.shuffle(items)
    tasklist = [{'display_id': f'h_{i:02d}', 'file': it['file'], 'is_repeat': it['is_repeat']}
                for i, it in enumerate(items)]
    (OUT / 'human_tasklist.json').write_text(json.dumps(tasklist, indent=2))

    print(f'rendered {len(manifest)} val scenes -> {OUT}')
    print(f'human tasklist: {len(tasklist)} items ({len(human_files)} unique + {len(repeats)} repeats)')


if __name__ == '__main__':
    main()
