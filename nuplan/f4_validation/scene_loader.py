"""Load nuPlan scenarios by token at the F4 scoring iteration (19) and expose
everything the three validation signals need: ego state, tracked objects, logged
agent futures, map api, route, traffic lights, and a minimal PlannerInput.

WHY this module: F4 scored 5,604 mini scenarios at iteration 19. The validation
signals must be computed on the SAME scenes at the SAME frame. This builds nuPlan
scenario handles once (lazy heavy data) and pulls per-scene fields on demand.

Env/path setup mirrors check_roadblock_availability.py (nuPlan reads env at import).
Run the self-test:
    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/scene_loader.py
(from /Users/parvpatodia/Desktop/diffusion-policy-zoo)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

import nest_asyncio
nest_asyncio.apply()

import json
from pathlib import Path

import hydra

F4_ITERATION = 19  # F4 scored every scene at this fixed iteration

# F4 scores live on HPC; a local copy is synced to data/ for the validation run.
_F4_LOCAL = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/f4_scores_v11.json')


def build_scenarios(tokens=None, limit: int = 100000):
    """Build mini scenario handles (lazy data). Returns list of AbstractScenario.

    WHY token filtering: the all_scenarios filter samples a different scenario set
    than the one F4 was scored on (only ~1.6k/5.6k overlap). Setting scenario_tokens
    to the exact F4 tokens loads precisely the scored scenes. scenario_types and the
    per-type/total limits are nulled so the token list is not intersected away.
    """
    from omegaconf import open_dict
    from eval_production import build_cfg
    from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
    from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
    from nuplan.planning.script.builders.worker_pool_builder import build_worker

    cfg = build_cfg(experiment_name='f4_validation', n_scenarios=limit)
    with open_dict(cfg.scenario_filter):
        cfg.scenario_filter.shuffle = False
        if tokens is not None:
            cfg.scenario_filter.scenario_tokens = list(tokens)
            cfg.scenario_filter.scenario_types = None
            cfg.scenario_filter.num_scenarios_per_type = None
            cfg.scenario_filter.limit_total_scenarios = None
            cfg.scenario_filter.remove_invalid_goals = False
            cfg.scenario_filter.expand_scenarios = False

    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass
    return scenarios


def index_by_token(scenarios) -> dict:
    out = {}
    for s in scenarios:
        try:
            out[s.token] = s
        except Exception:
            pass
    return out


def load_f4_scores() -> dict:
    return json.loads(_F4_LOCAL.read_text())


def _self_test():
    f4 = load_f4_scores()
    f4_tokens = list(f4.keys())
    print(f'building {len(f4_tokens)} F4-token scenarios via scenario_tokens filter...')
    scenarios = build_scenarios(tokens=f4_tokens)
    print(f'  built {len(scenarios)} scenario handles')
    by_tok = index_by_token(scenarios)
    built_tokens = set(by_tok.keys())
    covered = set(f4_tokens) & built_tokens
    print(f'  COVERAGE: {len(covered)}/{len(f4_tokens)} F4 scenes are loadable')
    missing = set(f4_tokens) - built_tokens
    if missing:
        print(f'  missing example tokens: {list(missing)[:5]}')

    # eyeball one high-F4 scene: confirm iteration 19 exists and basic getters work
    hi = max(f4_tokens, key=lambda t: f4[t].get('f4') or 0.0)
    if hi in by_tok:
        s = by_tok[hi]
        n_iter = s.get_number_of_iterations()
        ego = s.get_ego_state_at_iteration(F4_ITERATION)
        tracks = s.get_tracked_objects_at_iteration(F4_ITERATION)
        n_obj = len(tracks.tracked_objects.tracked_objects)
        print(f'  sample hi-F4 token {hi}: f4={f4[hi]["f4"]:.3f} type={s.scenario_type} '
              f'n_iter={n_iter} n_obj@19={n_obj} ego_v={ego.dynamic_car_state.speed:.2f}')


if __name__ == '__main__':
    _self_test()
