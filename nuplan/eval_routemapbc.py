"""
Phase 3c evaluation: RouteMapBCPlanner only.
Run once — results written to nuplan/sim_results/closed_loop_RouteMapBCPlanner/
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

import hydra
import nest_asyncio
nest_asyncio.apply()

from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

CKPT_ROUTEMAPBC = '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/checkpoints/goal_bc.pt'
DB_DIR   = '/Users/parvpatodia/nuplan-devkit/data/cache/mini'
SAVE_DIR = '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/sim_results'
LOG_NAME = '2021.05.12.22.00.38_veh-35_01008_01518'


def build_cfg(planner_name, save_dir, n_scenarios=3):
    BASE  = '/Users/parvpatodia/nuplan-devkit/nuplan/planning/script'
    paths = construct_simulation_hydra_paths(BASE)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=paths.config_path, version_base='1.1')
    return hydra.compose(
        config_name=paths.config_name,
        overrides=[
            f'group={save_dir}',
            f'experiment_name=closed_loop_{planner_name}',
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
            f'scenario_filter.limit_total_scenarios={n_scenarios}',
        ],
    )


def run_and_parse(planner, save_dir, n_scenarios=3):
    from nuplan.planning.script.run_simulation import run_simulation as main_sim
    cfg = build_cfg(planner.name(), save_dir, n_scenarios)
    print(f'\n>>> {planner.name()} — {n_scenarios} scenarios ... ', flush=True)
    main_sim(cfg, planner)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Parse L2 from parquet
    res_dir = Path(save_dir) / f'closed_loop_{planner.name()}' / 'eval' / 'metrics'
    parq = list(res_dir.glob('ego_expert_L2_error*.parquet'))
    if not parq:
        print('  [WARN] L2 parquet not found — check sim_results/')
        return None
    df = pd.read_parquet(parq[0])
    vals = df['metric_score'].dropna().values if 'metric_score' in df.columns else df.iloc[:, -1].dropna().values
    avg_l2  = float(np.mean(vals))
    max_l2  = float(np.max(vals))
    p90_l2  = float(np.percentile(vals, 90))
    print(f'  avg L2 = {avg_l2:.3f} m  |  max = {max_l2:.3f} m  |  p90 = {p90_l2:.3f} m')
    return dict(avg=avg_l2, max=max_l2, p90=p90_l2)


if __name__ == '__main__':
    from planners import RouteMapBCPlanner
    Path(SAVE_DIR).mkdir(exist_ok=True)
    planner = RouteMapBCPlanner(CKPT_ROUTEMAPBC)
    results = run_and_parse(planner, SAVE_DIR, n_scenarios=3)
    if results:
        print(f'\n=== Phase 3c RouteMapBC result ===')
        print(f'  avg L2 : {results["avg"]:.3f} m')
        print(f'  max L2 : {results["max"]:.3f} m')
        print(f'  p90 L2 : {results["p90"]:.3f} m')
        print(f'\nInterpretation:')
        if results["avg"] < 5.0:
            print('  🟢 EXCELLENT — approaches GoalBC (1.82m). Global route recovers almost all gain.')
        elif results["avg"] < 15.0:
            print('  🟡 PARTIAL WIN — route helps but some scenarios still drift. Debug _build_route().')
        elif results["avg"] < 40.0:
            print('  🟠 WEAK — route constructed but goal-following weak. Check ego-frame transform.')
        else:
            print('  🔴 FAIL — similar to MapBC. Route construction likely failing. Check successor chaining.')
