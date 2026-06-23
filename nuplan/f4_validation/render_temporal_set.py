"""Re-render the 40 human-rated scenes WITH the temporal overlay (logged agent
futures + ego route), same val_NN names, into validation_set_temporal/.
Tests whether showing the space-time conflict fixes the human/panel<->F4 null.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/render_temporal_set.py
"""
from __future__ import annotations
import json
from pathlib import Path
from scene_loader import build_scenarios, index_by_token
from render_scene import render

VAL = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders/validation_set')
OUT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders/validation_set_temporal')


def main():
    man = json.loads((VAL / 'manifest_PRIVATE.json').read_text())
    tasklist = json.loads((VAL / 'human_tasklist.json').read_text())
    human_files = sorted({t['file'] for t in tasklist})  # 40 unique
    tok_of = {f: man[f]['token'] for f in human_files}
    by_tok = index_by_token(build_scenarios(tokens=list(tok_of.values())))
    n = 0
    for f, tok in tok_of.items():
        if tok in by_tok:
            render(by_tok[tok], OUT / f, show_futures=True)
            n += 1
    print(f'rendered {n} temporal scenes -> {OUT}')
    print('files:', ' '.join(sorted(f.replace('.png', '') for f in human_files)))


if __name__ == '__main__':
    main()
