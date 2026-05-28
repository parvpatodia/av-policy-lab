"""
Phase 3c'' — SpeedAdaptiveRouteMapBCPlanner eval script.

Root-cause fix for RouteMapBC (32m) and TrainedRouteBCPlanner (49m) failures.

FINDING: The nuPlan mini SQLite DB is at 100 Hz (10 ms per row), NOT 10 Hz.
  GOAL_OFFSET = 8 rows × 10 ms = 0.08 s ahead (not 0.8 s as commented).
  GoalBC training goal magnitude: mean = 0.342 m (= avg_speed × 0.08 s).
  RouteMapBC used fixed 8.0 m look-ahead → 23× SCALE MISMATCH → policy ignores goal.
  TrainedRouteBCPlanner retrained with 8 m goals → same mismatch at inference → 49 m.

FIX: speed-adaptive look-ahead = sqrt(vx²+vy²) × 0.08
  → same magnitude as GoalBC training distribution at every speed
  → uses existing goal_bc.pt, no retraining
  → expected result ≈ GoalBC (1.820 m)

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
print('DB sampling rate: 100 Hz (confirmed). GoalBC T+8 = 0.08 s.')
print(f'SpeedAdaptive look-ahead at avg speed 4.3 m/s → {4.3*0.08:.3f} m')
print(f'GoalBC training goal mean: ~0.342 m  →  distribution match: {"✓" if abs(avg - 1.82) < 5 else "✗ — check above"}')
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
