"""
Fast diagnostic: does nuPlan mini expose non-empty route_roadblock_ids?

WHY this script exists:
    RoadblockRouteMapBCPlanner uses initialization.route_roadblock_ids to pick the
    correct lane at intersections. Those ids come from the scenario via
    AbstractScenario.get_route_roadblock_ids(). If the mini DB scenarios return
    EMPTY lists, the planner is byte-for-byte identical to its parent
    (SpeedAdaptiveRouteMapBCPlanner) and a 30-scenario closed-loop eval is wasted.

    This script builds the SAME 30 scenarios the production eval uses, then ONLY
    reads get_route_roadblock_ids() off each scenario. No simulation, no metric
    engine, no planner inference. It should finish in well under a minute.

Run:
    /opt/anaconda3/envs/nuplan/bin/python3.9 nuplan/check_roadblock_availability.py
    (from /Users/parvpatodia/Desktop/diffusion-policy-zoo)
"""

import os
import sys

# WHY set env vars before any nuPlan import — nuPlan reads these at module load time.
os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

# WHY nest_asyncio.apply() before nuplan imports: nuPlan's builders/worker use
# asyncio internally; without this, running inside an already-running event loop
# raises RuntimeError. Matches eval_production.py.
import nest_asyncio
nest_asyncio.apply()

import traceback

import numpy as np

# WHY reuse build_cfg from eval_production: it already encodes the EXACT scenario
# selection the production eval uses (scenario_builder=nuplan_mini, all_scenarios,
# shuffle=true, limit_total_scenarios=N). Reusing it guarantees we check the same
# scenarios that a real eval would run — no drift between this check and the eval.
from eval_production import build_cfg, DB_DIR

import hydra
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.worker_pool_builder import build_worker


N_SCENARIOS = 30


def get_route_ids(scenario):
    """
    Read route_roadblock_ids off a scenario, robust to API shape.

    Returns (ids_list, access_method_str). ids_list is [] on any failure.

    WHY try multiple access shapes: the devkit abstract API declares
    get_route_roadblock_ids() as a method, but some scenario subclasses or
    versions may expose it as a property or under a slightly different name.
    We try the documented method first, then fall back, and report which worked.
    """
    # 1) Documented method call.
    try:
        attr = getattr(scenario, 'get_route_roadblock_ids')
        if callable(attr):
            ids = attr()
            return list(ids) if ids is not None else [], 'get_route_roadblock_ids()'
        # Not callable -> it's actually a property/attribute holding the value.
        return list(attr) if attr is not None else [], 'get_route_roadblock_ids (property)'
    except AttributeError:
        pass
    except Exception as exc:
        # Method exists but raised — try the property fallback before giving up.
        method_err = f'get_route_roadblock_ids() raised: {type(exc).__name__}: {exc}'
    else:
        method_err = None

    # 2) Bare attribute fallback.
    try:
        attr = getattr(scenario, 'route_roadblock_ids')
        return list(attr) if attr is not None else [], 'route_roadblock_ids (attr)'
    except Exception:
        pass

    raise RuntimeError(
        method_err or 'no get_route_roadblock_ids() method or route_roadblock_ids attr found'
    )


def build_scenarios():
    """
    Build the same N scenarios the production eval uses — WITHOUT simulation.

    WHY this path (builders, not run_simulation): build_cfg() produces the full
    hydra config, but we only invoke the three builders needed to materialize the
    scenario list (filter + builder + worker). We never touch run_simulation,
    metric engine, or planner. This is the fast part of the pipeline.
    """
    # WHY pass any experiment_name: build_cfg needs one for output_dir interpolation,
    # but we never write output. The scenario selection overrides are what matter.
    cfg = build_cfg(experiment_name='roadblock_check', n_scenarios=N_SCENARIOS)

    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg)
    # WHY cfg.scenario_filter (not cfg): build_scenario_filter reads scenario_tokens
    # off the node it is given and hydra-instantiates it directly. The token field
    # lives under the scenario_filter subconfig, so we must pass that subnode.
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    # WHY get_scenarios over build_scenarios(): the lower-level scenario_builder
    # API takes (filter, worker) and returns the scenario objects directly,
    # avoiding the model/feature-precompute path in script.builders.build_scenarios.
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    # Clear hydra so a subsequent run in the same process starts clean (matches
    # eval_production convention).
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass

    return scenarios


def main():
    print()
    print('=' * 70)
    print('ROADBLOCK AVAILABILITY CHECK (no simulation)')
    print('=' * 70)
    print(f'  DB dir            : {DB_DIR}')
    print(f'  Scenarios target  : {N_SCENARIOS} (all_scenarios, shuffle=true)')
    print(f'  Checking          : scenario.get_route_roadblock_ids()')
    print()

    try:
        scenarios = build_scenarios()
    except Exception:
        print('[ERROR] Failed to build scenarios. No verdict possible.')
        traceback.print_exc()
        print()
        print('  Check:')
        print(f'    - DB files present in {DB_DIR}')
        print('    - NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT env vars resolve to real dirs')
        print('    - nuplan-devkit importable on sys.path')
        return

    n = len(scenarios)
    if n == 0:
        print('[ERROR] Scenario builder returned 0 scenarios. Cannot form a verdict.')
        print('  Check scenario_filter=all_scenarios and DB files exist.')
        return

    counts = []
    methods_used = set()
    errored = 0
    samples = []  # (token, first5_ids) for the first 3 scenarios

    for i, scenario in enumerate(scenarios):
        try:
            ids, method = get_route_ids(scenario)
            methods_used.add(method)
        except Exception as exc:
            errored += 1
            ids = []
            if i < 3:
                samples.append((_safe_token(scenario), f'<ERROR: {exc}>'))
            continue

        counts.append(len(ids))

        if i < 3:
            samples.append((_safe_token(scenario), ids[:5]))

    if not counts:
        print(f'[ERROR] Could not read route_roadblock_ids on ANY of {n} scenarios '
              f'({errored} raised). Cannot form a verdict.')
        return

    counts_arr = np.array(counts)
    n_checked = len(counts)
    n_nonempty = int(np.sum(counts_arr > 0))
    cmin = int(counts_arr.min())
    cmax = int(counts_arr.max())
    cmean = float(counts_arr.mean())
    # Mean count over the scenarios that actually have a route (the interesting number).
    nonempty_mean = float(counts_arr[counts_arr > 0].mean()) if n_nonempty > 0 else 0.0

    # ── Sample format eyeball (first 3 scenarios) ──────────────────────────────
    print('  Sample (first 3 scenarios) — token + first 5 roadblock ids:')
    for token, sample_ids in samples:
        if isinstance(sample_ids, list) and sample_ids:
            elem_type = type(sample_ids[0]).__name__
            print(f'    {token}  | n_shown={len(sample_ids)} type={elem_type} | {sample_ids}')
        else:
            print(f'    {token}  | {sample_ids if sample_ids else "[] (empty)"}')
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print('  Summary:')
    print(f'    scenarios checked        : {n_checked} (of {n} built; {errored} errored)')
    print(f'    access method            : {sorted(methods_used)}')
    print(f'    non-empty route ids      : {n_nonempty}/{n_checked}')
    print(f'    ids per scenario (min/mean/max): {cmin} / {cmean:.1f} / {cmax}')
    if n_nonempty > 0:
        print(f'    mean ids (non-empty only): {nonempty_mean:.1f}')
    print()

    # ── Verdict ──────────────────────────────────────────────────────────────────
    # WHY threshold "majority populated": the planner only differs from its parent
    # on scenarios that actually carry route_roadblock_ids. If most scenarios have
    # them, the full eval can show a difference; if all/most are empty, it cannot.
    populated_majority = n_nonempty >= max(1, n_checked // 2)

    print('-' * 70)
    if n_nonempty > 0 and populated_majority:
        print(f'VERDICT: route_roadblock_ids POPULATED on {n_nonempty}/{n_checked} scenarios '
              f'(mean {nonempty_mean:.1f} ids) — RoadblockRouteMapBC CAN differ from parent. '
              f'Full eval worthwhile.')
    elif n_nonempty > 0:
        print(f'VERDICT: route_roadblock_ids POPULATED on only {n_nonempty}/{n_checked} scenarios '
              f'(mean {nonempty_mean:.1f} ids) — RoadblockRouteMapBC can differ on a MINORITY of '
              f'scenarios only. Full eval may show a small/no aggregate difference; consider a '
              f'map-geometry turn-detection fix for the empty-route scenarios.')
    else:
        print(f'VERDICT: route_roadblock_ids EMPTY on all/most scenarios '
              f'(0/{n_checked} populated) — RoadblockRouteMapBC will be identical to '
              f'SpeedAdaptive. Need a different intersection fix (map-geometry turn detection).')
    print('-' * 70)
    print()


def _safe_token(scenario):
    """Best-effort scenario identifier for eyeballing. Never raises."""
    for attr in ('token', 'scenario_name'):
        try:
            val = getattr(scenario, attr)
            if val:
                return str(val)
        except Exception:
            pass
    return '<unknown-token>'


if __name__ == '__main__':
    main()
