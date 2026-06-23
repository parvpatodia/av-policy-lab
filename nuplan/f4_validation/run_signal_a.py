"""Full Signal A run: compute a_gt (logged-future conflict) for all 5,604 F4 scenes.
Saves incrementally to data/signal_a_results.json so partial progress survives.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/run_signal_a.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from scene_loader import build_scenarios, index_by_token, load_f4_scores
from signal_a_gt_conflict import a_gt

OUT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/signal_a_results.json')


def main():
    f4 = load_f4_scores()
    toks = list(f4.keys())
    t0 = time.time()
    print(f'building {len(toks)} scenarios...', flush=True)
    by_tok = index_by_token(build_scenarios(tokens=toks))
    print(f'  built {len(by_tok)} in {time.time()-t0:.0f}s', flush=True)

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text())  # resume
        print(f'  resuming: {len(results)} already done', flush=True)

    done = 0
    for i, t in enumerate(toks):
        if t in results:
            continue
        if t not in by_tok:
            results[t] = {'a_gt': None, 'reason': 'not_built'}
            continue
        try:
            r = a_gt(by_tok[t])
        except Exception as exc:
            r = {'a_gt': None, 'reason': f'err:{type(exc).__name__}'}
        # store compactly
        results[t] = {'a_gt': r.get('a_gt'), 'reason': r.get('reason'),
                      'n_conflict': r.get('n_conflict', 0)}
        done += 1
        if done % 500 == 0:
            OUT.write_text(json.dumps(results))
            el = time.time() - t0
            print(f'  {len(results)}/{len(toks)} done ({el:.0f}s, {el/max(done,1):.2f}s/new)', flush=True)

    OUT.write_text(json.dumps(results))
    scored = sum(1 for v in results.values() if v.get('a_gt') is not None
                 and v['a_gt'] == v['a_gt'])
    print(f'DONE: {len(results)} total, {scored} scored, {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
