"""
Closed-loop simulation runner for all planners:
  BCPlanner (v0, v1 DAgger iter1, v2 DAgger iter2), IDMPlanner, BEVPlanner, MILEPlanner.

Planners whose checkpoint does not exist are skipped with a warning — run the
corresponding notebook first (dagger.ipynb, bev_cnn.ipynb, mile_policy.ipynb).

Usage (from nuplan-devkit root):
    python /Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/closed_loop_eval.py

Results written to nuplan/sim_results/.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

import hydra
from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

_CKPT_ROOT  = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/checkpoints')
CKPT_BC_V0  = str(_CKPT_ROOT / 'bc_best.pt')          # BC pure imitation
CKPT_BC_V1  = str(_CKPT_ROOT / 'bc_dagger_v1.pt')     # DAgger iter 1 (745 samples)
CKPT_BC_V2  = str(_CKPT_ROOT / 'bc_dagger_v2.pt')     # DAgger iter 2 (~15K samples)
CKPT_BEV    = str(_CKPT_ROOT / 'bev_cnn.pt')          # BEV CNN
CKPT_MILE   = str(_CKPT_ROOT / 'mile_policy.pt')      # MILE world model
CKPT_GOALBC = str(_CKPT_ROOT / 'goal_bc.pt')          # Phase 3a: Goal-conditioned BC
CKPT_MAPBC  = str(_CKPT_ROOT / 'goal_bc.pt')          # Phase 3b: MapBC reuses GoalBC weights
DB_DIR      = '/Users/parvpatodia/nuplan-devkit/data/cache/mini'
# WHY: GoalBCPlanner needs a specific DB file to build expert T+8 lookup.
# We use the first sorted DB file — same log as the simulation scenarios.
_DB_FILES   = sorted(Path(DB_DIR).glob('*.db'))
GOALBC_DB   = str(_DB_FILES[0]) if _DB_FILES else ''
SAVE_DIR    = '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/sim_results'
LOG_NAME    = '2021.05.12.22.00.38_veh-35_01008_01518'


def build_cfg(planner_name: str, save_dir: str, n_scenarios: int = 3):
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
            # WHY: db_files overrides data_root; our mini data is at data/cache/mini
            f'scenario_builder.db_files={DB_DIR}',
            'scenario_filter=one_continuous_log',
            f"scenario_filter.log_names=['{LOG_NAME}']",
            f'scenario_filter.limit_total_scenarios={n_scenarios}',
        ],
    )


def run_simulation(planner, save_dir: str, n_scenarios: int = 3):
    from nuplan.planning.script.run_simulation import run_simulation as main_sim
    cfg = build_cfg(planner.name(), save_dir, n_scenarios)
    print(f'\n>>> {planner.name()} — {n_scenarios} scenarios')
    main_sim(cfg, planner)
    print(f'>>> Done.')
    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == '__main__':
    from planners import BCPlanner, IDMPlanner, BEVPlanner, MILEPlanner, GoalBCPlanner, MapBCPlanner

    Path(SAVE_DIR).mkdir(exist_ok=True)

    # ── Always available ───────────────────────────────────────────────────
    run_simulation(BCPlanner(CKPT_BC_V0), SAVE_DIR, n_scenarios=3)  # baseline
    run_simulation(IDMPlanner(),          SAVE_DIR, n_scenarios=3)  # reactive

    # ── DAgger checkpoints (skip if not yet collected) ─────────────────────
    for label, ckpt in [('v1', CKPT_BC_V1), ('v2', CKPT_BC_V2)]:
        if Path(ckpt).exists():
            run_simulation(BCPlanner(ckpt), SAVE_DIR, n_scenarios=3)
        else:
            print(f'[SKIP] BCPlanner {label} — checkpoint not found: {ckpt}')
            print(f'       Run dagger.ipynb Cell 4 (iter1) or Cell 4+5 (iter2) first.')

    # ── BEV CNN (skip if not yet trained) ──────────────────────────────────
    if Path(CKPT_BEV).exists():
        run_simulation(BEVPlanner(CKPT_BEV), SAVE_DIR, n_scenarios=3)
    else:
        print(f'[SKIP] BEVPlanner — checkpoint not found: {CKPT_BEV}')
        print(f'       Run bev_cnn.ipynb Cells 3-7 first.')

    # ── MILE world model (skip if not yet trained) ─────────────────────────
    if Path(CKPT_MILE).exists():
        run_simulation(MILEPlanner(CKPT_MILE), SAVE_DIR, n_scenarios=3)
    else:
        print(f'[SKIP] MILEPlanner — checkpoint not found: {CKPT_MILE}')
        print(f'       Run mile_policy.ipynb Cells 2-7 first.')

    # ── Goal-conditioned BC (Phase 3a) — skip if not yet trained ──────────
    # WHY: GoalBCPlanner needs both the checkpoint and a DB path for the expert
    #      T+8 goal lookup. GOALBC_DB uses the first sorted mini DB file.
    if Path(CKPT_GOALBC).exists() and GOALBC_DB:
        run_simulation(GoalBCPlanner(CKPT_GOALBC, GOALBC_DB), SAVE_DIR, n_scenarios=3)
    else:
        print(f'[SKIP] GoalBCPlanner — checkpoint not found: {CKPT_GOALBC}')
        print(f'       Run goal_bc.ipynb Cells 1-5 first.')

    # ── MapBC (Phase 3b) — centerline goal, no expert at inference ────────
    # WHY reuse goal_bc.pt: MapBC and GoalBC have identical training (same data,
    # same architecture, same T+8 expert goals). Only inference differs.
    # MapBCPlanner uses nuPlan map_api (injected via initialize()) for goal.
    if Path(CKPT_MAPBC).exists():
        run_simulation(MapBCPlanner(CKPT_MAPBC), SAVE_DIR, n_scenarios=3)
    else:
        print(f'[SKIP] MapBCPlanner — checkpoint not found: {CKPT_MAPBC}')
        print(f'       Run goal_bc.ipynb Cells 1-5 first (shares goal_bc.pt).')
