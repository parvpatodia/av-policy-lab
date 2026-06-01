"""
Phase 3c''''' — DualHorizonRouteMapBC: train + the experiment's rigor gate.

WHY this experiment (the chain that led here):
  3c''  SpeedAdaptiveRouteMapBC: tied with IDM; 4 intersection failures dominate.
  3c''' RoadblockRouteMapBC: gave the CORRECT turn direction, still tied (15 fixed/13 broke).
        Finding: a correct goal is necessary but not sufficient.
  Look-ahead analysis (notes/phase3_roadmap.md): a single goal at the near horizon
        (speed*0.8s ~= 3.5m) shows the turn in only ~6% of turning windows. The turn
        only enters a single goal point at 16-24m. => The turn was NEVER IN THE INPUT.
        A multi-modal head (Diffusion) alone cannot fix a missing-information problem.

THE FIX (this script): condition on TWO goals, like a real reference-path controller:
  near = speed*0.8s arc-length  (precise local tracking — what already works on straights)
  far  = fixed 20m arc-length   (turn anticipation — the information that was missing)
Input becomes 10-dim. Architecture, optimizer, targets, protocol otherwise identical to
GoalBC — a clean ablation that ADDS the far-preview goal (near goal held fixed).

EXPECTED OUTCOMES (both decisive):
  - Turn execution improves (4 tail failures drop) -> bottleneck was INPUT INFORMATION;
    deployable fix; deterministic MLP is sufficient when told about the turn.
  - No improvement -> the MLP cannot represent junction bimodality even when informed
    -> Phase 3d (Diffusion Policy) is justified with hard evidence.

Run:
    conda activate nuplan
    python nuplan/train_dual_horizon.py --sanity   # fast rigor gate (no training)
    python nuplan/train_dual_horizon.py            # ~20 min on M-series, writes checkpoint

Then eval (unchanged planner, new weights):
    python nuplan/eval_production.py --n_scenarios 30 --planners idm,speedadaptive,dualhorizon
    python nuplan/statistical_analysis.py --a DualHorizonRouteMapBCPlanner --b SpeedAdaptiveRouteMapBCPlanner
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from planners import GoalBCPolicy, _GOAL_LOOKAHEAD_S, _FAR_LOOKAHEAD_M

DB_DIR       = Path('/Users/parvpatodia/nuplan-devkit/data/cache/mini')
CKPT_DIR     = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/checkpoints')
CKPT_OUT     = CKPT_DIR / 'trained_dual_horizon.pt'

FUTURE_STEPS  = 16
STRIDE        = 10
BATCH_SIZE    = 512
LR            = 1e-3
EPOCHS        = 50
MIN_LOOKAHEAD = 0.05
NEAR_S        = _GOAL_LOOKAHEAD_S    # 0.8 s (speed-adaptive near goal)
FAR_M         = _FAR_LOOKAHEAD_M     # 20 m (fixed far preview goal)

DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')


def quat_to_yaw(qw, qx, qy, qz):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))


def _load_ego(db_path):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        'SELECT x, y, qw, qx, qy, qz, vx, vy, acceleration_x, acceleration_y '
        'FROM ego_pose ORDER BY timestamp'
    ).fetchall()
    conn.close()
    if len(rows) < FUTURE_STEPS + 1:
        return None
    a = np.array(rows, dtype=np.float64)
    return dict(x=a[:, 0], y=a[:, 1], yaw=quat_to_yaw(a[:, 2], a[:, 3], a[:, 4], a[:, 5]),
                vx=a[:, 6], vy=a[:, 7], ax=a[:, 8], ay=a[:, 9])


def _goal_at_arclength(x, y, yaw, cum_arc, i, N, look_ahead_m):
    """Ego-frame goal at `look_ahead_m` arc-length ahead along the expert path.
    Identical rule to RouteMapBCPlanner._get_route_goal — train/inference matched."""
    gi = min(int(np.searchsorted(cum_arc, cum_arc[i] + look_ahead_m)), N - 1)
    cyaw = yaw[i]; cos_h, sin_h = np.cos(-cyaw), np.sin(-cyaw)
    dxw, dyw = x[gi] - x[i], y[gi] - y[i]
    return cos_h * dxw - sin_h * dyw, sin_h * dxw + cos_h * dyw


def extract_from_db(db_path):
    """X (n,10): state(6) + near_goal(2) + far_goal(2);  Y (n,48): 16-step ego traj."""
    ego = _load_ego(db_path)
    if ego is None:
        return None, None
    x, y, yaw, vx, vy, ax, ay = (ego['x'], ego['y'], ego['yaw'],
                                 ego['vx'], ego['vy'], ego['ax'], ego['ay'])
    N = len(x)
    cum = np.zeros(N); cum[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    X, Y = [], []
    for i in range(0, N - FUTURE_STEPS, STRIDE):
        near_la = max(MIN_LOOKAHEAD, float(np.hypot(vx[i], vy[i])) * NEAR_S)
        dxn, dyn = _goal_at_arclength(x, y, yaw, cum, i, N, near_la)
        dxf, dyf = _goal_at_arclength(x, y, yaw, cum, i, N, FAR_M)
        X.append(np.array([np.sin(yaw[i]), np.cos(yaw[i]), vx[i], vy[i], ax[i], ay[i],
                           dxn, dyn, dxf, dyf], dtype=np.float32))
        cyaw = yaw[i]; cos_h, sin_h = np.cos(-cyaw), np.sin(-cyaw)
        tgt = np.zeros(FUTURE_STEPS * 3, dtype=np.float32)
        for j in range(FUTURE_STEPS):
            fi = i + j + 1
            dxw, dyw = x[fi] - x[i], y[fi] - y[i]
            tgt[j*3]   = cos_h * dxw - sin_h * dyw
            tgt[j*3+1] = sin_h * dxw + cos_h * dyw
            tgt[j*3+2] = (yaw[fi] - cyaw + np.pi) % (2 * np.pi) - np.pi
        Y.append(tgt)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def run_sanity_check(db_files):
    """Rigor gate: prove the FAR goal encodes turns the NEAR goal misses.
    If it does not, the dual-horizon input adds no turn information and we must not train."""
    near_ang, far_ang = [], []
    near_turn, far_turn = [], []   # restricted to turning windows
    for db in db_files[:20]:
        ego = _load_ego(db)
        if ego is None:
            continue
        x, y, yaw, vx, vy = ego['x'], ego['y'], ego['yaw'], ego['vx'], ego['vy']
        N = len(x); cum = np.zeros(N); cum[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
        for i in range(0, N - FUTURE_STEPS, STRIDE):
            near_la = max(MIN_LOOKAHEAD, float(np.hypot(vx[i], vy[i])) * NEAR_S)
            dxn, dyn = _goal_at_arclength(x, y, yaw, cum, i, N, near_la)
            dxf, dyf = _goal_at_arclength(x, y, yaw, cum, i, N, FAR_M)
            an = abs(np.degrees(np.arctan2(dyn, dxn)))
            af = abs(np.degrees(np.arctan2(dyf, dxf)))
            near_ang.append(an); far_ang.append(af)
            # turning window: heading changes >20deg over next 20m of arc
            j20 = min(int(np.searchsorted(cum, cum[i] + 20)), N - 1)
            if abs((yaw[j20] - yaw[i] + np.pi) % (2*np.pi) - np.pi) > np.radians(20):
                near_turn.append(an); far_turn.append(af)
    near_ang, far_ang = np.array(near_ang), np.array(far_ang)
    near_turn, far_turn = np.array(near_turn), np.array(far_turn)
    pct = lambda a, t=15: 100 * (a > t).mean()

    print('=' * 72)
    print(f'SANITY GATE — does the FAR goal ({FAR_M:.0f}m) encode turns the NEAR goal misses?')
    print('=' * 72)
    print(f'{"goal":<22}{"mean|ang|":>11}{"%>15deg(all)":>14}{"%>15deg(turning)":>18}')
    print('-' * 72)
    print(f'{"near (speed*0.8s)":<22}{near_ang.mean():>10.1f}°{pct(near_ang):>13.1f}%{pct(near_turn):>17.1f}%')
    print(f'{"far  (%.0fm fixed)" % FAR_M:<22}{far_ang.mean():>10.1f}°{pct(far_ang):>13.1f}%{pct(far_turn):>17.1f}%')
    print('-' * 72)
    if pct(far_turn) > 3 * max(pct(near_turn), 1e-6):
        print(f'VERDICT: PASS — on turning windows the far goal exposes the turn '
              f'{pct(far_turn)/max(pct(near_turn),1e-6):.0f}x more often\n'
              f'         ({pct(far_turn):.0f}% vs {pct(near_turn):.0f}%). The dual-horizon input '
              f'adds real turn information. Train.')
    else:
        print('VERDICT: FAIL — far goal does not add turn information. Do not train; rethink.')
    print('=' * 72)


class _DS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float(); self.Y = torch.from_numpy(Y).float()
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    args = ap.parse_args()
    db_files = sorted(DB_DIR.glob('*.db'))
    print(f'DB files: {len(db_files)}   device: {DEVICE}   near=speed*{NEAR_S}s  far={FAR_M}m')

    run_sanity_check(db_files)        # rigor gate always runs first
    if args.sanity:
        return

    print('\nExtracting dual-horizon training set ...')
    Xs, Ys = [], []
    for db in db_files:
        X, Y = extract_from_db(db)
        if X is not None:
            Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    print(f'Dataset: {X.shape[0]:,} windows | X{X.shape[1]} Y{Y.shape[1]}')

    Xm, Xsd = X.mean(0).astype(np.float32), (X.std(0) + 1e-6).astype(np.float32)
    Ym, Ysd = Y.mean(0).astype(np.float32), (Y.std(0) + 1e-6).astype(np.float32)
    Xn, Yn = (X - Xm) / Xsd, (Y - Ym) / Ysd
    np.random.seed(42)
    idx = np.random.permutation(len(Xn)); ntr = int(0.9 * len(Xn))
    tr, va = idx[:ntr], idx[ntr:]
    tdl = DataLoader(_DS(Xn[tr], Yn[tr]), batch_size=BATCH_SIZE, shuffle=True)
    vdl = DataLoader(_DS(Xn[va], Yn[va]), batch_size=BATCH_SIZE, shuffle=False)
    print(f'Train {len(tr):,} | Val {len(va):,}')

    model = GoalBCPolicy(in_dim=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss(); best = float('inf')
    for ep in range(EPOCHS):
        model.train(); tl = 0.0
        for xb, yb in tdl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()
        tl /= len(tdl)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in vdl:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        vl /= len(vdl); sched.step(vl)
        if vl < best:
            best = vl
            torch.save({'model': model.state_dict(),
                        'X_mean': Xm, 'X_std': Xsd, 'Y_mean': Ym, 'Y_std': Ysd}, CKPT_OUT)
        if (ep + 1) % 5 == 0:
            print(f'epoch {ep+1:3d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}  best={best:.4f}')
    print(f'\nBest val {best:.4f}  ->  {CKPT_OUT}')
    print('Eval: python nuplan/eval_production.py --n_scenarios 30 --planners idm,speedadaptive,dualhorizon')


if __name__ == '__main__':
    main()
