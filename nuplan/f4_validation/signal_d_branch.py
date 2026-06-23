"""Signal D — independent BEHAVIORAL check for S_branch (route-choice ambiguity).
F4's S_branch counts route-corridor lane branches from the map graph. Independent
question: at the ego's location, do OTHER agents actually fan out across different
turn directions (left/straight/right)? A junction that genuinely admits multiple
routes shows agents taking different ones. d_branch = turn-direction entropy of
nearby agents' LOGGED futures (no ego-future, no f4_map_branch reuse).

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/signal_d_branch.py
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from scene_loader import F4_ITERATION, build_scenarios, index_by_token, load_f4_scores
from signal_a_gt_conflict import agent_future_paths

OUT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/signal_d_results.json')
NEAR_M = 40.0
TURN_DEG = 20.0   # |net turn| above this counts as a left/right; else straight


def net_turn_deg(path: np.ndarray) -> float | None:
    """Net heading change over a path via its two halves. None if too short/slow."""
    if len(path) < 3:
        return None
    mid = len(path) // 2
    v1 = path[mid] - path[0]
    v2 = path[-1] - path[mid]
    if np.linalg.norm(v1) < 2.0 or np.linalg.norm(v2) < 2.0:  # essentially stationary
        return None
    a = math.atan2(v1[1], v1[0]); b = math.atan2(v2[1], v2[0])
    return math.degrees(math.atan2(math.sin(b - a), math.cos(b - a)))


def d_branch(scenario, iteration: int = F4_ITERATION) -> dict:
    ego = scenario.get_ego_state_at_iteration(iteration)
    e0 = np.array([ego.rear_axle.x, ego.rear_axle.y])
    bins = {'L': 0, 'S': 0, 'R': 0}
    n_used = 0
    for path, t in agent_future_paths(scenario, iteration):
        if np.linalg.norm(path[0] - e0) > NEAR_M:
            continue
        turn = net_turn_deg(path)
        if turn is None:
            continue
        n_used += 1
        if turn > TURN_DEG:
            bins['L'] += 1
        elif turn < -TURN_DEG:
            bins['R'] += 1
        else:
            bins['S'] += 1
    counts = np.array(list(bins.values()), dtype=float)
    if n_used < 2 or counts.sum() == 0:
        return {'d_branch': float('nan'), 'n_used': n_used}
    p = counts / counts.sum()
    p = p[p > 0]
    ent = -(p * np.log(p)).sum() / math.log(3)  # normalized entropy over 3 bins
    return {'d_branch': float(ent), 'n_used': n_used, 'bins': bins}


def main():
    f4 = load_f4_scores()
    toks = list(f4.keys())
    t0 = time.time()
    by_tok = index_by_token(build_scenarios(tokens=toks))
    print(f'built {len(by_tok)} in {time.time()-t0:.0f}s')
    res = {}
    for i, tk in enumerate(toks):
        if tk in by_tok:
            try:
                res[tk] = d_branch(by_tok[tk])
            except Exception as e:
                res[tk] = {'d_branch': float('nan'), 'err': type(e).__name__}
        if (i + 1) % 1500 == 0:
            print(f'  {i+1}/{len(toks)} ({time.time()-t0:.0f}s)')
    OUT.write_text(json.dumps(res))

    # correlate with s_branch / b_r
    from scipy.stats import spearmanr, mannwhitneyu
    toks_ok = [t for t in toks if t in res and res[t]['d_branch'] == res[t]['d_branch']]
    db = np.array([res[t]['d_branch'] for t in toks_ok])
    sb = np.array([f4[t].get('s_branch') or 0.0 for t in toks_ok])
    br = np.array([f4[t].get('b_r') or 0 for t in toks_ok], dtype=float)
    print(f'\nscored {len(toks_ok)}/{len(toks)} (rest: <2 turning agents nearby)')
    print(f'  Spearman(d_branch, s_branch) = {spearmanr(db, sb)[0]:.3f} (p={spearmanr(db,sb)[1]:.2e})')
    print(f'  Spearman(d_branch, b_r)      = {spearmanr(db, br)[0]:.3f}')
    g0 = db[sb < 1e-6]; g1 = db[sb >= 1e-6]
    if len(g0) and len(g1):
        U, p = mannwhitneyu(g1, g0, alternative='greater'); rbc = 2*U/(len(g1)*len(g0))-1
        print(f'  group: d_branch s_branch=0 (n={len(g0)}) mean {g0.mean():.3f} vs '
              f's_branch>0 (n={len(g1)}) mean {g1.mean():.3f}; MW p={p:.2e} rbc={rbc:.3f}')


if __name__ == '__main__':
    main()
