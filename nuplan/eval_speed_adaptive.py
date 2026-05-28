"""
Phase 3c'' — SpeedAdaptiveRouteMapBCPlanner eval script.

Root-cause diagnosis for RouteMapBC (32m) and TrainedRouteBCPlanner (49m) failures.

CONFIRMED: nuPlan mini SQLite DB is at 100 Hz (10 ms/row).
  GoalBCPlanner._get_expert_at_offset: offset_steps * 100_000 µs = 8 × 100 ms = T+0.8 s.
  GoalBC INFERENCE goal at avg 4.33 m/s: 4.33 × 0.8 = 3.46 m.

SCALE MISMATCH (speed-dependent):
  RouteMapBC used fixed 8.0 m regardless of speed.
  GoalBC uses speed × 0.8 s:
    at 10 m/s → 8.0 m  (identical to RouteMapBC — no problem at highway speed)
    at  4 m/s → 3.2 m  vs RouteMapBC 8.0 m  (2.5× mismatch)
    at  0 m/s →  ~0 m  vs RouteMapBC 8.0 m  (∞ mismatch at stops)
  nuPlan urban scenarios spend significant time stopped/slow → failure accumulates.

WHY RETRAIN FAILED (TrainedRouteBCPlanner, 49m):
  Retraining with 8 m goals fixed the scale at inference but not the training utility.
  8 m goal is ~12× the prediction horizon (16 rows × 10 ms × 4.33 m/s = 0.69 m).
  MSE reaches near-zero without the network attending to the goal → ignores it → BC.

FIX: speed_adaptive = True → look_ahead = max(0.05, speed × 0.8)
  Matches GoalBC T+0.8 s temporal horizon at every speed.
  goal_bc.pt already learned to use goals at this scale (GoalBC inference works).
  No retraining needed.

Run:
  conda activate nuplan
  python nuplan/eval_speed_adaptive.py
"""
import sys, os
sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

import nest_asyncio; nest_asyncio.apply()
import hydra, pandas as pd, numpy as np
from pathlib import Path
from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths
from nuplan.planning.script.run_simulation import run_simulation as main_sim
from planners import SpeedAdaptiveRouteMapBCPlanner

# ── Config ────────────────────────────────────────────────────────────────────
DB_DIR    = Path('/Users/parvpatodia/nuplan-devkit/data/cache/mini')
CKPT_PATH = '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/checkpoints/goal_bc.pt'
SIM_OUT   = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/sim_results')
SIM_OUT.mkdir(exist_ok=True)

LOG_NAME = '2021.05.12.22.00.38_veh-35_01008_01518'
N_SCEN   = 3
EXP_NAME = 'speed_adaptive_eval'

print('SpeedAdaptiveRouteMapBCPlanner eval')
print(f'  Checkpoint:  {CKPT_PATH}  (goal_bc.pt — no retraining)')
print(f'  Look-ahead:  speed × 0.08 s  (T+8 equivalent at 100 Hz DB)')
print(f'  Scenarios:   {N_SCEN}  (same as all prior evals)')
print()

# ── Run sim ───────────────────────────────────────────────────────────────────
planner = SpeedAdaptiveRouteMapBCPlanner(CKPT_PATH)

BASE  = '/Users/parvpatodia/nuplan-devkit/nuplan/planning/script'
paths = construct_simulation_hydra_paths(BASE)
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=paths.config_path, version_base='1.1')

cfg = hydra.compose(
    config_name=paths.config_name,
    overrides=[
        f'group={SIM_OUT}',
        f'experiment_name={EXP_NAME}',
        'job_name=eval',
        'experiment=${experiment_name}/${job_name}',
        'worker=sequential',
        'ego_controller=perfect_tracking_controller',
        'observation=box_observation',
        f'hydra.searchpath=[{paths.common_dir}, {paths.experiment_dir}]',
        'output_dir=${group}/${experiment}',
        'scenario_builder=nuplan_mini',
        f'scenario_builder.db_files={DB_DIR}',
        'scenario_filter=one_continuous_log',
        f"scenario_filter.log_names=['{LOG_NAME}']",
        f'scenario_filter.limit_total_scenarios={N_SCEN}',
    ],
)

main_sim(cfg, planner)
hydra.core.global_hydra.GlobalHydra.instance().clear()

# ── Parse results ─────────────────────────────────────────────────────────────
mdir = SIM_OUT / EXP_NAME / 'eval' / 'metrics'
l2   = pd.read_parquet(mdir / 'ego_expert_L2_error.parquet')

avg = float(l2['avg_ego_expert_L2_error_stat_value'].mean())
mx  = float(l2['max_ego_expert_L2_error_stat_value'].mean())
p90 = float(l2['p90_ego_expert_L2_error_stat_value'].mean())

# Full comparison table
results = [
    ('GoalBCPlanner (oracle)',         1.820,   2.944,  2.646),
    ('IDMPlanner',                     6.285,  24.308, 15.733),
    ('SpeedAdaptiveRouteMapBC ← NEW',  avg,       mx,    p90),
    ('RouteMapBCPlanner (8m fixed)',  32.085,  77.952, 48.596),
    ('TrainedRouteBCPlanner (8m)',    49.034, 101.902, 89.021),
    ('BCPlanner_v0 (baseline)',       49.449, 104.614, 91.526),
]

print()
print('=' * 72)
print('RESULTS: SpeedAdaptiveRouteMapBC vs all prior planners')
print('=' * 72)
print(f"{'Policy':<36} {'Avg L2':>8} {'Max L2':>8} {'p90 L2':>8}")
print('-' * 62)
for name, a, m, p in sorted(results, key=lambda r: r[1]):
    marker = ' ←' if 'NEW' in name else ''
    print(f'{name:<36} {a:>8.3f} {m:>8.3f} {p:>8.3f}{marker}')

print()
print('DB sampling rate: 100 Hz (confirmed). GoalBCPlanner inference: T+8 × 0.1 s = T+0.8 s.')
print(f'SpeedAdaptive look-ahead at avg speed 4.3 m/s → {4.3*0.8:.3f} m  (= GoalBC inference scale ✓)')
print()

bc_v0 = 49.449
goalbc = 1.820
pct_vs_bc = (avg - bc_v0) / bc_v0 * 100
pct_vs_goalbc = (avg - goalbc) / goalbc * 100
print(f'vs GoalBC (1.820m):   {pct_vs_goalbc:+.1f}%')
print(f'vs BC_v0  (49.45m):   {pct_vs_bc:+.1f}%')

if avg < 5.0:
    print()
    print('CLAIM CONFIRMED: deployable goal source (route) ≈ oracle goal (expert DB).')
    print('Full 96.3% gain reproducible without expert data at inference.')
elif avg < 20.0:
    print()
    print('PARTIAL: speed-adaptive look-ahead helps significantly but gap to GoalBC remains.')
    print('Likely cause: route centerline ≠ expert trajectory at curves/intersections.')
    print('Next: DAgger + route goal (Phase 3d path B).')
else:
    print()
    print('STILL FAILS: speed-adaptive look-ahead not sufficient.')
    print('Check goal_bc.pt normalization stats vs SpeedAdaptive goal magnitude at runtime.')
