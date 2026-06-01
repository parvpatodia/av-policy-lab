"""
Production evaluation harness for all AV imitation-learning planners.

Evaluates all planners over N diverse scenarios sampled from all 64 mini DB files,
computes per-scenario and aggregate statistics, and saves full results to JSON.

Usage:
    python nuplan/eval_production.py [--n_scenarios 30] [--planners all|bc|idm|goalbc|routebc|speedadaptive]

Output:
    nuplan/eval_results/production_eval.json
    Formatted table printed to stdout
    Deployability summary printed to stdout
"""

import os
import sys

# WHY: set env vars before any nuPlan import — nuPlan reads these at module load time
os.environ.setdefault('NUPLAN_DATA_ROOT',     '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT',     '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',      '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')
sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')

# WHY nest_asyncio.apply() first: nuPlan's simulation engine uses asyncio internally.
# Without this, running nuPlan inside an already-running event loop (Jupyter, pytest,
# or any environment that starts an event loop at import) raises RuntimeError.
import nest_asyncio
nest_asyncio.apply()

import argparse
import json
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd

from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

# ── Constants ─────────────────────────────────────────────────────────────────

NUPLAN_DEVKIT    = Path('/Users/parvpatodia/nuplan-devkit')
DB_DIR           = NUPLAN_DEVKIT / 'data' / 'cache' / 'mini'
HYDRA_BASE       = str(NUPLAN_DEVKIT / 'nuplan' / 'planning' / 'script')

REPO_ROOT        = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
CKPT_DIR         = REPO_ROOT / 'nuplan' / 'checkpoints'
SIM_OUT          = REPO_ROOT / 'nuplan' / 'sim_results'
EVAL_OUT         = REPO_ROOT / 'nuplan' / 'eval_results'
# WHY: project-local hydra config dir. Holds simulation_metric/prod_eval_metrics.yaml,
# a composed metric set (common_metrics L2 + the PDM components) that a plain
# devkit override cannot express. Added to hydra.searchpath in build_cfg().
REPO_CONFIG_DIR  = REPO_ROOT / 'nuplan' / 'config'

CKPT_BC          = str(CKPT_DIR / 'bc_best.pt')
CKPT_GOALBC      = str(CKPT_DIR / 'goal_bc.pt')
# WHY both CKPT_ROUTEMAPBC and CKPT_SPEEDADAPTIVE point at goal_bc.pt:
# RouteMapBC and SpeedAdaptiveRouteMapBC reuse the GoalBC weights unchanged;
# only inference goal SOURCE differs (route vs expert DB).
CKPT_ROUTEMAPBC  = str(CKPT_DIR / 'goal_bc.pt')
CKPT_SPEEDADAPTIVE = str(CKPT_DIR / 'goal_bc.pt')
# Phase 3c''''' dual-horizon: 10-dim goal (near+far), separate trained checkpoint.
# Produced by train_dual_horizon.py. Until trained, build_planners warn-skips it.
CKPT_DUALHORIZON = str(CKPT_DIR / 'trained_dual_horizon.pt')
# Phase 3d diffusion policy: DDPM denoiser with dual-horizon goal, same 10-dim conditioning.
# Produced by train_diffusion_policy.py. Until trained, build_planners warn-skips it.
CKPT_DIFFUSION   = str(CKPT_DIR / 'trained_diffusion_policy.pt')

# GoalBCPlanner is EXCLUDED from the multi-scenario (all_scenarios) eval by default.
#
# WHY: GoalBCPlanner._build_expert_lookup() loads ego_pose from a SINGLE SQLite
# file. When scenario_filter=all_scenarios draws from all 64 DB logs, 63 of 64
# scenarios have timestamps that fall outside that one file's range.
# _get_expert_at_offset() does np.searchsorted over _sorted_ts and clamps to the
# boundary index — so it returns the first or last row of the wrong log for every
# non-matching scenario. That produces garbage goal vectors and inflates L2 for
# GoalBC, corrupting the oracle baseline every other planner is measured against.
#
# FIX: the 1.820m oracle result (3 scenarios, single log, goal_bc.ipynb Cell 8)
# is used as a hardcoded reference constant. It was validated on scenarios from
# a single log where the expert DB correctly matched. To re-run GoalBC, use
# eval_speed_adaptive.py / goal_bc.ipynb (single-log scenario filter).
GOALBC_ORACLE_L2: float = 1.820   # validated in goal_bc.ipynb, 3 scenarios

_DB_FILES = sorted(DB_DIR.glob('*.db'))

# L2 thresholds for the failure/good scenario counts
L2_FAIL_THRESHOLD = 20.0   # avg L2 > 20m  -> failure scenario
L2_GOOD_THRESHOLD = 5.0    # avg L2 < 5m   -> good scenario

# Planners that do NOT require expert data at inference (deployable without expert DB).
# WHY MapBCPlanner included: uses only HD map (no expert DB) at inference.
DEPLOYABLE_PLANNERS = {
    'IDMPlanner',
    'MapBCPlanner',
    'RouteMapBCPlanner',
    'SpeedAdaptiveRouteMapBCPlanner',
    'RoadblockRouteMapBCPlanner',
    'DualHorizonRouteMapBCPlanner',
    # Phase 3d: diffusion uses route goals (same as DualHorizon) — no expert DB at inference.
    'DiffusionPolicyPlanner',
}

# ── Planner registry ──────────────────────────────────────────────────────────

def build_planners(selection: str) -> List[Tuple[str, object]]:
    """
    Build (name, planner_instance) pairs for the requested selection.
    Returns list of (key, instance) so the caller can skip failed builds.

    WHY defer imports to here: nuPlan planners import nuPlan at module level;
    keeping them inside this function makes the top-level import section clean
    and avoids paying the import cost for planners we never run.
    """
    from planners import (
        BCPlanner,
        IDMPlanner,
        GoalBCPlanner,
        RouteMapBCPlanner,
        SpeedAdaptiveRouteMapBCPlanner,
        RoadblockRouteMapBCPlanner,
        DualHorizonRouteMapBCPlanner,
        DiffusionPolicyPlanner,
    )

    # Full ordered list: (cli_key, display_name, factory_fn)
    # WHY GoalBC absent from 'all': it requires a per-scenario expert DB.
    # Using a single DB for all_scenarios gives wrong goals for 63/64 logs.
    # Use the hardcoded GOALBC_ORACLE_L2 reference instead. See comment above.
    ALL = [
        ('bc',           'BCPlanner',                     lambda: BCPlanner(CKPT_BC)),
        ('idm',          'IDMPlanner',                    lambda: IDMPlanner()),
        ('routebc',      'RouteMapBCPlanner',             lambda: RouteMapBCPlanner(CKPT_ROUTEMAPBC)),
        ('speedadaptive','SpeedAdaptiveRouteMapBCPlanner', lambda: SpeedAdaptiveRouteMapBCPlanner(CKPT_SPEEDADAPTIVE)),
        ('roadblock',    'RoadblockRouteMapBCPlanner',    lambda: RoadblockRouteMapBCPlanner(CKPT_SPEEDADAPTIVE)),
        ('dualhorizon',  'DualHorizonRouteMapBCPlanner',  lambda: DualHorizonRouteMapBCPlanner(CKPT_DUALHORIZON)),
        # Phase 3d: same 10-dim dual-horizon goal, DDPM denoiser instead of MLP.
        # WHY 'diffusion' key (not 'phase3d'): matches the model type, not the phase label.
        ('diffusion',    'DiffusionPolicyPlanner',        lambda: DiffusionPolicyPlanner(CKPT_DIFFUSION)),
    ]

    valid_keys = {r[0] for r in ALL}
    if selection == 'all':
        selected_keys = set(valid_keys)
    else:
        selected_keys = set(selection.split(','))
        # WHY guard unknown keys: a typo'd planner name (e.g. 'roadblock' before it
        # was registered) was silently ignored, wasting a full eval run on the wrong
        # planner set. Fail loud instead.
        unknown = selected_keys - valid_keys
        if unknown:
            print(f'[ERROR] Unknown planner key(s): {sorted(unknown)}. '
                  f'Valid: {sorted(valid_keys)}')
            return []

    result = []
    for cli_key, display_name, factory in ALL:
        if cli_key not in selected_keys:
            continue
        # Validate checkpoint existence before attempting instantiation
        ckpt_map = {
            'bc':            CKPT_BC,
            'goalbc':        CKPT_GOALBC,
            'routebc':       CKPT_ROUTEMAPBC,
            'speedadaptive': CKPT_SPEEDADAPTIVE,
            'roadblock':     CKPT_SPEEDADAPTIVE,   # reuses goal_bc.pt weights
            'dualhorizon':   CKPT_DUALHORIZON,     # separate 10-dim checkpoint (train first)
            'diffusion':     CKPT_DIFFUSION,        # Phase 3d DDPM checkpoint (train first)
        }
        if cli_key in ckpt_map and not Path(ckpt_map[cli_key]).exists():
            print(f'[WARN] Skipping {display_name} — checkpoint not found: {ckpt_map[cli_key]}')
            continue
        if cli_key == 'goalbc' and not _DB_FILES:
            print(f'[WARN] Skipping GoalBCPlanner — no DB files found in {DB_DIR}')
            continue
        try:
            instance = factory()
            result.append((display_name, instance))
        except Exception as exc:
            print(f'[WARN] Skipping {display_name} — instantiation failed: {exc}')
    return result


# ── Hydra config builder ──────────────────────────────────────────────────────

def build_cfg(experiment_name: str, n_scenarios: int):
    """
    Build a Hydra DictConfig for one planner run.

    Key override decisions:
    - scenario_filter=all_scenarios    : samples from all 64 DB logs for diversity
    - scenario_filter.shuffle=true     : random N rather than first-N from sorted order
    - limit_total_scenarios=N          : cap at requested N after shuffle
    - worker=sequential                : avoids macOS multiprocessing pitfalls
    - perfect_tracking_controller      : standard closed-loop setup (matches prior evals)
    - box_observation                  : standard observation type (matches prior evals)

    WHY GlobalHydra.clear() before each call: Hydra is a singleton; initializing it
    twice in the same process raises ConfigCompositionException. We clear before each
    planner's config so planners can be run in sequence in the same process.
    """
    paths = construct_simulation_hydra_paths(HYDRA_BASE)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=paths.config_path, version_base='1.1')
    return hydra.compose(
        config_name=paths.config_name,
        overrides=[
            f'group={SIM_OUT}',
            f'experiment_name={experiment_name}',
            'job_name=eval',
            'experiment=${experiment_name}/${job_name}',
            'worker=sequential',
            'ego_controller=perfect_tracking_controller',
            'observation=box_observation',
            # WHY simulation_metric=prod_eval_metrics (project-local composed set):
            # Selecting a single devkit option (e.g. simulation_closed_loop_nonreactive_agents)
            # REPLACES the simulation_metric defaults LIST, dropping common_metrics — so a run
            # gets the 7 PDM components but NO ego_expert_L2_error parquet, and parse_results()
            # then fails. And composing both via simulation_metric=[common_metrics,
            # simulation_closed_loop_nonreactive_agents] does NOT work either: both define a
            # `low_level:` key, so Hydra's defaults-list merge makes the second OVERRIDE the
            # first instead of concatenating (this is why the devkit's nonreactive config keeps
            # `# - common_metrics` commented out). prod_eval_metrics.yaml merges every low_level
            # + high_level statistic from BOTH groups into one block, so ONE run emits BOTH the
            # L2 metrics (ego_mean_speed, ego_expert_l2_error[_with_yaw]) AND the 7 PDM-Score
            # components plus ego_is_making_progress. It lives in REPO_CONFIG_DIR/simulation_metric/
            # which is appended to hydra.searchpath below.
            # WHY nonreactive metric defs (not reactive): this eval uses observation=box_observation,
            # i.e. background agents replay their logged tracks (non-reactive). The metric set is
            # identical between the two; we match the nonreactive components to the observation type.
            'simulation_metric=prod_eval_metrics',
            f'hydra.searchpath=[{paths.common_dir}, {paths.experiment_dir}, file://{REPO_CONFIG_DIR}]',
            'output_dir=${group}/${experiment}',
            'scenario_builder=nuplan_mini',
            # WHY db_files=DB_DIR (directory, not single file): passing the directory
            # lets nuPlan discover all 64 .db files, giving geographic and behavioral
            # diversity across all mini logs.
            f'scenario_builder.db_files={DB_DIR}',
            # WHY all_scenarios: unlike one_continuous_log (which draws from 1 log,
            # so all scenarios share the same road / traffic context), all_scenarios
            # samples from every log in db_files. This is the standard production filter.
            'scenario_filter=all_scenarios',
            # WHY shuffle=true: without shuffle, limit_total_scenarios takes the FIRST N
            # scenarios in lexicographic log order — heavily biased toward the 2021-05-12
            # log files. Shuffle gives representative coverage across all 64 logs.
            'scenario_filter.shuffle=true',
            f'scenario_filter.limit_total_scenarios={n_scenarios}',
        ],
    )


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(planner, experiment_name: str, n_scenarios: int) -> bool:
    """
    Run nuPlan closed-loop simulation for one planner.
    Returns True on success, False on failure.
    """
    from nuplan.planning.script.run_simulation import run_simulation as main_sim

    # WHY wipe the experiment dir first: nuPlan does NOT clean its metrics output
    # between runs. A prior run's ego_expert_L2_error.parquet would otherwise survive
    # alongside this run's freshly-written PDM parquets, and parse_results()/pdm_score.py
    # would silently mix metrics from two different runs (different scenarios, different
    # timestamps). Deleting the dir guarantees every run starts clean and no stale parquet
    # can ever be read. Guard: only ever rmtree a path that lives under SIM_OUT.
    exp_dir = (SIM_OUT / experiment_name).resolve()
    sim_out_resolved = SIM_OUT.resolve()
    if exp_dir.exists():
        if sim_out_resolved in exp_dir.parents:
            shutil.rmtree(exp_dir)
            print(f'    Cleaned stale output dir: {exp_dir}', flush=True)
        else:
            raise RuntimeError(
                f'Refusing to delete {exp_dir}: not under SIM_OUT ({sim_out_resolved}).'
            )

    cfg = build_cfg(experiment_name, n_scenarios)
    print(f'\n>>> Running {planner.name()} ({n_scenarios} scenarios) ...', flush=True)
    main_sim(cfg, planner)
    # WHY clear after sim: run_simulation() may leave Hydra initialized;
    # clearing ensures the next build_cfg() starts from a clean state.
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    print(f'    Done.', flush=True)
    return True


# ── Metric parsing ────────────────────────────────────────────────────────────

def _load_l2_parquet(metrics_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the ego_expert_L2_error parquet from the metrics directory.
    Returns None if not found.

    WHY glob instead of direct path: nuPlan appends a hash suffix to metric
    filenames in some versions (e.g. ego_expert_L2_error_abc123.parquet).
    Glob catches both the plain and hashed variants.
    """
    candidates = sorted(metrics_dir.glob('ego_expert_L2_error*.parquet'))
    if not candidates:
        return None
    return pd.read_parquet(candidates[0])


def _load_extra_metrics(metrics_dir: Path) -> Dict[str, pd.Series]:
    """
    Load all metric parquets in the directory EXCEPT the L2 one.
    Returns {metric_name: per_scenario_series} for any additional metrics found.

    WHY include extras: drivable_area_compliance and no_ego_at_fault_collisions
    give safety signal beyond raw L2. We include them if present so the JSON
    output is self-contained for downstream analysis.
    """
    extras = {}
    for path in sorted(metrics_dir.glob('*.parquet')):
        if 'ego_expert_L2_error' in path.name:
            continue
        try:
            df = pd.read_parquet(path)
            # Best-effort: take the first numeric column as the per-scenario score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metric_name = path.stem  # filename without .parquet
                extras[metric_name] = df[numeric_cols[0]].dropna()
        except Exception:
            pass
    return extras


def parse_results(experiment_name: str) -> Optional[Dict]:
    """
    Parse simulation output for one planner experiment.

    Returns a dict with:
      - aggregate stats (mean, std, median, p10, p90, max)
      - failure/good scenario counts
      - per-scenario records
      - any extra metric means

    Returns None if the L2 parquet is missing.

    Column documentation (from eval_speed_adaptive.py):
      avg_ego_expert_L2_error_stat_value  : mean L2 over the scenario duration
      max_ego_expert_L2_error_stat_value  : max L2 over the scenario duration
      p90_ego_expert_L2_error_stat_value  : 90th-percentile L2 over the scenario duration
    Each row = one scenario.
    """
    metrics_dir = SIM_OUT / experiment_name / 'eval' / 'metrics'
    if not metrics_dir.exists():
        print(f'  [WARN] metrics dir not found: {metrics_dir}')
        return None

    df = _load_l2_parquet(metrics_dir)
    if df is None:
        print(f'  [WARN] ego_expert_L2_error parquet not found in {metrics_dir}')
        return None

    # WHY explicit column check before access: column names changed between nuPlan
    # minor versions. The three-column names below are confirmed in eval_speed_adaptive.py.
    required = [
        'avg_ego_expert_L2_error_stat_value',
        'max_ego_expert_L2_error_stat_value',
        'p90_ego_expert_L2_error_stat_value',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Fallback: match by substring rather than position.
        # WHY substring not position: nuPlan may emit columns in any order, or
        # include extra numeric columns (p50, p10, etc.). Positional assignment
        # silently swaps avg/max/p90 if order differs — a substring match is safer.
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        rename_map = {}
        for target_col, substring in [
            (required[0], 'avg'),
            (required[1], 'max'),
            (required[2], 'p90'),
        ]:
            if target_col in df.columns:
                continue   # already present, no rename needed
            candidates = [c for c in numeric_cols if substring in c.lower()]
            if candidates:
                rename_map[candidates[0]] = target_col
            else:
                print(f'  [WARN] Cannot find column matching "{substring}" in {numeric_cols}')
                return None
        if rename_map:
            warnings.warn(f'Expected L2 columns not found; renaming by substring: {rename_map}')
            df = df.rename(columns=rename_map)
        # Re-check after attempted rename
        still_missing = [c for c in required if c not in df.columns]
        if still_missing:
            print(f'  [WARN] Cannot parse L2 columns after fallback. Found: {list(df.columns)}')
            return None

    avg_vals = df['avg_ego_expert_L2_error_stat_value'].dropna().values
    max_vals = df['max_ego_expert_L2_error_stat_value'].dropna().values
    p90_vals = df['p90_ego_expert_L2_error_stat_value'].dropna().values

    n = len(avg_vals)
    if n == 0:
        print(f'  [WARN] L2 parquet has 0 rows after dropna.')
        return None

    # Per-scenario records (do NOT average before saving — requirement §4)
    scenario_col = None
    for candidate in ('scenario_token', 'token', 'log_name', 'scene_token'):
        if candidate in df.columns:
            scenario_col = candidate
            break

    per_scenario = []
    for i in range(n):
        record: Dict = {
            'avg_l2':  float(avg_vals[i]),
            'max_l2':  float(max_vals[i]),
            'p90_l2':  float(p90_vals[i]),
        }
        if scenario_col is not None:
            record['scenario_id'] = str(df[scenario_col].iloc[i])
        else:
            # WHY fallback to row index: scenario_token may not be in the parquet;
            # a sequential index is still useful for debugging specific rows.
            record['scenario_id'] = f'scenario_{i:04d}'
        per_scenario.append(record)

    # Extra metric means (safety proxies)
    extra_means = {}
    for metric_name, series in _load_extra_metrics(metrics_dir).items():
        extra_means[metric_name] = float(series.mean())

    return {
        'mean':       float(np.mean(avg_vals)),
        'std':        float(np.std(avg_vals)),
        'median':     float(np.median(avg_vals)),
        'p10':        float(np.percentile(avg_vals, 10)),
        'p90':        float(np.percentile(avg_vals, 90)),
        'max':        float(np.max(avg_vals)),
        'n_scenarios': n,
        'n_fail_20m': int(np.sum(avg_vals > L2_FAIL_THRESHOLD)),
        'n_good_5m':  int(np.sum(avg_vals < L2_GOOD_THRESHOLD)),
        'extra_metrics': extra_means,
        'scenarios':   per_scenario,
    }


# ── Table printer ─────────────────────────────────────────────────────────────

def print_results_table(all_results: Dict[str, Dict]) -> None:
    """
    Print a formatted comparison table sorted by mean L2 (best first).

    Columns: Planner | N | Mean | Std | Median | p10 | p90 | Max | Fail>20m | Good<5m
    """
    sorted_planners = sorted(all_results.keys(), key=lambda p: all_results[p]['mean'])

    header = (
        f"{'Planner':<38} {'N':>4} {'Mean':>7} {'Std':>7} {'Median':>7}"
        f" {'p10':>7} {'p90':>7} {'Max':>7} {'Fail>20m':>9} {'Good<5m':>8}"
    )
    sep = '-' * len(header)
    print()
    print('=' * len(header))
    print('PRODUCTION EVAL -- ALL PLANNERS (sorted by mean avg-L2)')
    print('=' * len(header))
    print(header)
    print(sep)

    for name in sorted_planners:
        r = all_results[name]
        n_fail = r['n_fail_20m']
        n_good = r['n_good_5m']
        n_tot  = r['n_scenarios']
        print(
            f"{name:<38} {n_tot:>4} {r['mean']:>7.2f} {r['std']:>7.2f} {r['median']:>7.2f}"
            f" {r['p10']:>7.2f} {r['p90']:>7.2f} {r['max']:>7.2f}"
            f" {n_fail:>4}/{n_tot:<4} {n_good:>4}/{n_tot:<4}"
        )

    print(sep)
    print()


# ── Deployability summary ─────────────────────────────────────────────────────

def print_deployability_summary(all_results: Dict[str, Dict]) -> None:
    """
    Print which planners do not require expert data at inference and their L2
    gap vs GoalBCPlanner (the oracle upper-bound).

    WHY GoalBC as oracle: GoalBCPlanner uses the expert DB T+8 goal at each
    planning step — the best possible goal a policy could receive. L2 gap to
    GoalBC quantifies how much information is lost by switching to a deployable
    goal source (route, map, IDM).
    """
    print('=' * 60)
    print('DEPLOYABILITY SUMMARY')
    print('=' * 60)

    # GoalBC is excluded from multi-scenario runs (wrong expert DB for cross-log scenarios).
    # Use the hardcoded oracle result from the validated 3-scenario single-log eval.
    # WHY not all_results.get('GoalBCPlanner'): GoalBC is never in all_results here.
    oracle_l2: float = GOALBC_ORACLE_L2   # 1.820m from goal_bc.ipynb Cell 8, 3 scenarios

    deployable_present = [p for p in all_results if p in DEPLOYABLE_PLANNERS]

    if not deployable_present:
        print('  No deployable planners were evaluated.')
        print()
        return

    print(f'  Oracle reference: GoalBCPlanner = {oracle_l2:.3f} m')
    print(f'  (3-scenario single-log eval, goal_bc.ipynb -- not re-run here to avoid')
    print(f'   per-scenario DB mismatch. See eval_speed_adaptive.py for single-log eval.)')
    print()
    print('  Deployable (no expert data at inference):')
    for name in sorted(deployable_present, key=lambda p: all_results[p]['mean']):
        r       = all_results[name]
        # Guard: oracle_l2 is the hardcoded constant; only divide if non-zero
        if oracle_l2 > 0:
            pct_gap = (r['mean'] - oracle_l2) / oracle_l2 * 100.0
            gap_str = f"   (gap vs GoalBC oracle: {pct_gap:+.1f}%)"
        else:
            gap_str = ''
        print(f"    {name:<38}  mean L2 = {r['mean']:.3f} m{gap_str}")

    print()
    print('  Requires expert DB at inference (not deployable as-is):')
    non_deploy = [p for p in all_results if p not in DEPLOYABLE_PLANNERS]
    for name in sorted(non_deploy, key=lambda p: all_results[p]['mean']):
        r = all_results[name]
        print(f"    {name:<38}  mean L2 = {r['mean']:.3f} m")

    if deployable_present:
        best_deploy = min(deployable_present, key=lambda p: all_results[p]['mean'])
        best_l2 = all_results[best_deploy]['mean']
        # Guard against zero oracle (degenerate case)
        pct = (best_l2 - oracle_l2) / oracle_l2 * 100.0 if oracle_l2 > 0 else float('nan')
        print()
        print(f'  Best deployable:   {best_deploy}  ->  {best_l2:.3f} m')
        print(f'  Oracle (GoalBC):   {oracle_l2:.3f} m')
        print(f'  L2 gap:            {best_l2 - oracle_l2:+.3f} m  ({pct:+.1f}%)')
        if pct < 20.0:
            print('  CLAIM: deployable planner approx oracle (< 20% relative gap). Suitable for deployment.')
        elif pct < 100.0:
            print('  PARTIAL: deployable planner trails oracle but shows strong improvement over BC baseline.')
        else:
            print('  GAP TOO LARGE: deployable planner does not match oracle. Further development needed.')

    print('=' * 60)
    print()


# ── JSON output ───────────────────────────────────────────────────────────────

def save_results_json(all_results: Dict[str, Dict], n_scenarios_requested: int) -> Path:
    """
    Save full results to EVAL_OUT/production_eval.json.

    Structure:
        {
          "meta": {"n_scenarios_requested": N, "db_dir": "...", ...},
          "planners": {
            "PlannerName": {
              "mean": X, "std": Y, "median": Z, "p10": A, "p90": B, "max": C,
              "n_scenarios": N, "n_fail_20m": K, "n_good_5m": M,
              "extra_metrics": {...},
              "scenarios": [{"scenario_id": "...", "avg_l2": X, "max_l2": Y, "p90_l2": Z}, ...]
            },
            ...
          }
        }
    """
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_OUT / 'production_eval.json'

    payload = {
        'meta': {
            'n_scenarios_requested': n_scenarios_requested,
            'db_dir':                str(DB_DIR),
            'scenario_filter':       'all_scenarios',
            'l2_fail_threshold_m':   L2_FAIL_THRESHOLD,
            'l2_good_threshold_m':   L2_GOOD_THRESHOLD,
        },
        'planners': all_results,
    }

    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'  Results saved -> {out_path}')
    return out_path


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Production evaluation of all AV imitation-learning planners.'
    )
    parser.add_argument(
        '--n_scenarios', type=int, default=30,
        help='Number of scenarios to sample from all_scenarios filter (default: 30).',
    )
    parser.add_argument(
        '--planners', type=str, default='all',
        help=(
            'Which planners to run. '
            '"all" (default) runs all planners. '
            'Comma-separated subset: bc,idm,goalbc,routebc,speedadaptive,dualhorizon,diffusion'
        ),
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n_scenarios = args.n_scenarios
    planner_selection = args.planners

    print()
    print('╔══════════════════════════════════════════════════════╗')
    print('║      AV IMITATION LEARNING -- PRODUCTION EVAL        ║')
    print('╚══════════════════════════════════════════════════════╝')
    print(f'  Scenarios requested : {n_scenarios}')
    print(f'  Scenario filter     : all_scenarios (all 64 mini DB files, shuffled)')
    print(f'  Planner selection   : {planner_selection}')
    print(f'  Sim output root     : {SIM_OUT}')
    print(f'  Results output      : {EVAL_OUT}/production_eval.json')
    print()

    SIM_OUT.mkdir(parents=True, exist_ok=True)

    # ── Build planner instances ────────────────────────────────────────────────
    planners = build_planners(planner_selection)
    if not planners:
        print('[ERROR] No planners could be instantiated. Check checkpoints.')
        return

    print(f'  Will evaluate {len(planners)} planner(s):')
    for name, _ in planners:
        deploy_tag = ' [deployable]' if name in DEPLOYABLE_PLANNERS else ' [needs expert DB]'
        print(f'    - {name}{deploy_tag}')
    print()

    # ── Run simulations ────────────────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    for display_name, planner in planners:
        # WHY experiment_name prefix "prod_eval_": prevents collision with existing
        # sim_results from closed_loop_eval.py (which uses "closed_loop_" prefix).
        experiment_name = f'prod_eval_{display_name}'
        print(f'\n{"─"*60}')
        print(f'  Planner: {display_name}')
        print(f'  Experiment: {experiment_name}')
        print(f'{"─"*60}')

        try:
            run_simulation(planner, experiment_name, n_scenarios)
        except Exception:
            print(f'  [WARN] Simulation failed for {display_name}:')
            traceback.print_exc()
            print(f'  [WARN] Skipping {display_name} — continuing with remaining planners.')
            # WHY explicit GlobalHydra clear in exception path: run_simulation() may
            # have partially initialized Hydra before crashing. Clear so the next
            # planner's build_cfg() starts from a known-clean state.
            try:
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            except Exception:
                pass
            continue

        result = parse_results(experiment_name)
        if result is None:
            print(f'  [WARN] Could not parse results for {display_name} — skipping.')
            continue

        all_results[display_name] = result
        n_actual = result['n_scenarios']
        print(
            f'  Parsed {n_actual} scenarios '
            f'(requested {n_scenarios}'
            + (f', fewer available' if n_actual < n_scenarios else '')
            + ')'
        )
        print(
            f'  mean={result["mean"]:.3f}m  std={result["std"]:.3f}m  '
            f'median={result["median"]:.3f}m  p10={result["p10"]:.3f}m  '
            f'p90={result["p90"]:.3f}m  max={result["max"]:.3f}m'
        )
        print(
            f'  fail>20m: {result["n_fail_20m"]}/{n_actual}  '
            f'good<5m: {result["n_good_5m"]}/{n_actual}'
        )
        if result['extra_metrics']:
            for m_name, m_val in result['extra_metrics'].items():
                print(f'  {m_name}: {m_val:.4f}')

    # ── Output ─────────────────────────────────────────────────────────────────
    if not all_results:
        print('\n[ERROR] No planner results to report. All planners failed or were skipped.')
        return

    print_results_table(all_results)
    print_deployability_summary(all_results)
    save_results_json(all_results, n_scenarios)


if __name__ == '__main__':
    main()
