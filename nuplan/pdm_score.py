"""
PDM-Score component metrics parser & aggregator for the AV eval harness.

Parses nuPlan's official closed-loop driving-quality metrics (the PDM-Score
component metrics) from the per-experiment metric parquet files written by
eval_production.py, and aggregates them into the PDM-Score composite.

REF: Dauner et al. 2023, "Parting with Misconceptions about Learning-based
     Vehicle Motion Planning", arXiv:2306.07962 (PDM-Score / PDMS).

The composite formula and the exact weights are NOT guessed: they are taken
directly from nuPlan's own metric-aggregator config
    nuplan/planning/script/config/simulation/metric_aggregator/
        closed_loop_nonreactive_agents_weighted_average.yaml
which is the canonical implementation of the closed-loop scenario score used
in the nuPlan planning challenge (the PDM-Score Dauner et al. report on).

    PDM-Score(scenario) =
        ( PROD over m in MULTIPLICATIVE: score_m )          # hard penalties [0/0.5/1]
        x ( SUM over m in WEIGHTED: w_m * score_m ) / ( SUM over m in WEIGHTED: w_m )

    MULTIPLICATIVE (any 0 -> whole score 0):
        no_ego_at_fault_collisions, drivable_area_compliance,
        driving_direction_compliance, ego_is_making_progress
    WEIGHTED average:
        ego_progress_along_expert_route   w = 5.0
        time_to_collision_within_bound    w = 5.0
        speed_limit_compliance            w = 4.0
        ego_is_comfortable                w = 2.0

The final composite reported here is the MEAN of the per-scenario PDM-Score
across all scenarios in the experiment (matching nuPlan's per-scenario-then-mean
scoring), and we ALSO report each component's mean separately so the composite
choice is fully transparent.

Per-component value source
--------------------------
Each nuPlan metric parquet has a `metric_score` column: the canonical per-scenario
score in [0, 1] that nuPlan itself feeds into the aggregator. We read THAT column
(not the raw stat columns) so our composite matches nuPlan's definition exactly.
We fall back to the `avg_<name>_stat_value` column if `metric_score` is absent
(older devkit versions).

CLI
---
    python nuplan/pdm_score.py
        Scans nuplan/sim_results/<experiment>/eval/metrics/ for every experiment
        found, parses available PDM components, and prints the PDM table.

Degrades gracefully: if a component parquet is missing (e.g. the experiment was
run before the metric override was added to eval_production.py), it warns and
skips that component rather than crashing.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
SIM_OUT = REPO_ROOT / 'nuplan' / 'sim_results'

# PDM component metric NAMES. These are the `name:` fields in the nuPlan metric
# configs, which become the parquet filename stems (e.g. drivable_area_compliance
# -> drivable_area_compliance.parquet).
# WHY a glob per name (in _load_component): nuPlan capitalizes some filenames
# differently (the L2 file is ego_expert_L2_error.parquet) and some devkit
# versions append a hash suffix; a case-insensitive glob catches every variant.

# Hard multiplicative penalties — any 0 zeroes the whole scenario score.
# REF: closed_loop_nonreactive_agents_weighted_average.yaml -> multiple_metrics
MULTIPLICATIVE_METRICS: List[str] = [
    'no_ego_at_fault_collisions',
    'drivable_area_compliance',
    'driving_direction_compliance',
    'ego_is_making_progress',
]

# Weighted-average metrics and their weights.
# REF: closed_loop_nonreactive_agents_weighted_average.yaml -> metric_weights
WEIGHTED_METRICS: Dict[str, float] = {
    'ego_progress_along_expert_route': 5.0,
    'time_to_collision_within_bound': 5.0,
    'speed_limit_compliance': 4.0,
    'ego_is_comfortable': 2.0,
}

# All PDM components in display order.
ALL_PDM_COMPONENTS: List[str] = MULTIPLICATIVE_METRICS + list(WEIGHTED_METRICS.keys())

# Short labels for the printed table (full names are long).
COMPONENT_LABELS: Dict[str, str] = {
    'no_ego_at_fault_collisions': 'no_collision',
    'drivable_area_compliance': 'drivable',
    'driving_direction_compliance': 'direction',
    'ego_is_making_progress': 'makes_prog',
    'ego_progress_along_expert_route': 'route_prog',
    'time_to_collision_within_bound': 'ttc',
    'speed_limit_compliance': 'speed_lim',
    'ego_is_comfortable': 'comfort',
}


# ── Single-component loader ─────────────────────────────────────────────────────

def _load_component_series(metrics_dir: Path, metric_name: str) -> Optional[pd.Series]:
    """
    Load the per-scenario score series for one PDM component metric.

    Returns a pandas Series of per-scenario scores in [0, 1], or None if the
    parquet for this metric is not present (graceful degradation).

    WHY read `metric_score` first: that is the canonical [0,1] score nuPlan's own
    aggregator consumes. We only fall back to a stat_value column if it's missing.
    """
    # Case-insensitive match: build a lowercase lookup of all parquet files.
    target = metric_name.lower()
    match: Optional[Path] = None
    for path in sorted(metrics_dir.glob('*.parquet')):
        stem = path.stem.lower()
        # Exact stem, or stem with a trailing hash suffix (e.g. name_abc123).
        if stem == target or stem.startswith(target + '_'):
            match = path
            break
    if match is None:
        return None

    try:
        df = pd.read_parquet(match)
    except Exception as exc:
        print(f'  [WARN] failed to read {match.name}: {exc}')
        return None

    if 'metric_score' in df.columns:
        series = pd.to_numeric(df['metric_score'], errors='coerce').dropna()
        if len(series) > 0:
            return series
        # metric_score present but all-null -> fall through to stat_value fallback.

    # Fallback: avg_<name>_stat_value (older devkit, or score column unpopulated).
    # WHY substring 'avg' + name: stat columns are prefixed by the metric name and
    # the statistic, e.g. avg_speed_limit_compliance_stat_value.
    fallback_cols = [
        c for c in df.columns
        if c.endswith('_stat_value') and 'avg' in c.lower()
    ]
    if fallback_cols:
        series = pd.to_numeric(df[fallback_cols[0]], errors='coerce').dropna()
        if len(series) > 0:
            print(f'  [INFO] {metric_name}: using fallback column "{fallback_cols[0]}" '
                  f'(metric_score absent/empty)')
            return series

    print(f'  [WARN] {metric_name}: no usable score column in {match.name}')
    return None


# ── Per-experiment parser ────────────────────────────────────────────────────

def parse_pdm_metrics(experiment_name: str) -> Optional[Dict]:
    """
    Parse all available PDM component parquets for one experiment.

    Returns a dict:
        {
          'experiment': str,
          'n_scenarios': int,                 # max rows seen across components
          'components': {name: mean_score},   # mean per component (present only)
          'missing': [name, ...],             # components with no parquet
          'composite': float | None,          # mean per-scenario PDM-Score
          'composite_note': str,              # how composite was computed
        }
    Returns None if the metrics dir does not exist at all.
    """
    metrics_dir = SIM_OUT / experiment_name / 'eval' / 'metrics'
    if not metrics_dir.exists():
        print(f'  [WARN] metrics dir not found: {metrics_dir}')
        return None

    # Load every component's per-scenario series (None if missing).
    series_by_name: Dict[str, pd.Series] = {}
    missing: List[str] = []
    for name in ALL_PDM_COMPONENTS:
        s = _load_component_series(metrics_dir, name)
        if s is None:
            missing.append(name)
        else:
            series_by_name[name] = s

    component_means: Dict[str, float] = {
        name: float(s.mean()) for name, s in series_by_name.items()
    }

    n_scenarios = max((len(s) for s in series_by_name.values()), default=0)

    composite, note = _compute_composite(series_by_name, missing)

    return {
        'experiment': experiment_name,
        'n_scenarios': n_scenarios,
        'components': component_means,
        'missing': missing,
        'composite': composite,
        'composite_note': note,
    }


def _compute_composite(series_by_name: Dict[str, pd.Series],
                       missing: List[str]) -> (Optional[float], str):
    """
    Compute the mean per-scenario PDM-Score from the per-component series.

    REF: arXiv:2306.07962 + nuPlan closed_loop_nonreactive_agents_weighted_average.

    Per scenario:
        score = PROD(multiplicative_m) * weighted_avg(weighted_m)
    Then average over scenarios.

    WHY per-scenario then mean (not mean-then-combine): nuPlan computes the
    composite for each scenario independently and averages the scenario scores.
    Combining component means first would not equal the mean of the products.

    Graceful degradation: if ANY component is missing we cannot compute the true
    composite (a missing multiplicative penalty would silently default to 1.0,
    and a missing weighted term changes the denominator). We return None and a
    note explaining which components are absent, but the per-component breakdown
    is still reported by the caller.
    """
    if missing:
        return None, (
            f'composite not computed — {len(missing)} component(s) missing: '
            f'{", ".join(missing)}. Re-run eval_production.py with the '
            f'simulation_metric override to populate them.'
        )

    # Align all components by row index. nuPlan writes one row per scenario per
    # metric in the same scenario order, so positional alignment is valid.
    # WHY reset_index: the series come from different parquets; we align by
    # position 0..n-1, which is the shared scenario ordering.
    n = min(len(s) for s in series_by_name.values())
    if n == 0:
        return None, 'composite not computed — zero scenarios.'

    mult_arrays = [
        series_by_name[m].reset_index(drop=True).iloc[:n].to_numpy()
        for m in MULTIPLICATIVE_METRICS
    ]
    # Element-wise product across the multiplicative penalties.
    penalty = np.ones(n, dtype=float)
    for arr in mult_arrays:
        penalty = penalty * arr

    # Weighted average of the weighted metrics.
    total_weight = sum(WEIGHTED_METRICS.values())
    weighted_sum = np.zeros(n, dtype=float)
    for name, w in WEIGHTED_METRICS.items():
        arr = series_by_name[name].reset_index(drop=True).iloc[:n].to_numpy()
        weighted_sum = weighted_sum + w * arr
    weighted_avg = weighted_sum / total_weight

    per_scenario_score = penalty * weighted_avg
    composite = float(np.mean(per_scenario_score))
    return composite, f'mean per-scenario PDM-Score over {n} scenarios (all components present)'


# ── Table printer ─────────────────────────────────────────────────────────────

def print_pdm_table(all_results: Dict[str, Dict]) -> None:
    """
    Print the PDM component breakdown + composite per planner/experiment.

    One row per experiment. Columns: each PDM component (mean), then composite.
    Missing components print as '  -  '.
    """
    if not all_results:
        print('No experiments with parsable metrics found.')
        return

    # Header.
    comp_w = 11   # column width per component
    name_w = 30
    header_cells = [f"{'Experiment':<{name_w}}", f"{'N':>3}"]
    for name in ALL_PDM_COMPONENTS:
        header_cells.append(f"{COMPONENT_LABELS[name]:>{comp_w}}")
    header_cells.append(f"{'PDM-Score':>11}")
    header = ' '.join(header_cells)
    sep = '-' * len(header)

    print()
    print('=' * len(header))
    print('PDM-SCORE COMPONENT METRICS  (REF: Dauner et al. 2023, arXiv:2306.07962)')
    print('Composite = PROD(no_collision, drivable, direction, makes_prog)'
          ' x weighted_avg(route_prog:5, ttc:5, speed_lim:4, comfort:2)')
    print('=' * len(header))
    print(header)
    print(sep)

    # Sort by composite (best first); experiments with no composite go last.
    def sort_key(exp):
        c = all_results[exp]['composite']
        return (-c if c is not None else 1.0,)  # None -> sorts after all numbers

    for exp in sorted(all_results, key=sort_key):
        r = all_results[exp]
        cells = [f"{exp:<{name_w}}", f"{r['n_scenarios']:>3}"]
        for name in ALL_PDM_COMPONENTS:
            if name in r['components']:
                cells.append(f"{r['components'][name]:>{comp_w}.3f}")
            else:
                cells.append(f"{'-':>{comp_w}}")
        if r['composite'] is not None:
            cells.append(f"{r['composite']:>11.4f}")
        else:
            cells.append(f"{'n/a':>11}")
        print(' '.join(cells))

    print(sep)

    # Footnotes: any experiment missing components.
    any_missing = False
    for exp in all_results:
        miss = all_results[exp]['missing']
        if miss:
            if not any_missing:
                print()
                print('Missing components (composite = n/a until re-run):')
                any_missing = True
            print(f'  {exp}: {", ".join(miss)}')

    if not any_missing:
        print()
        print('All components present for all experiments.')
    else:
        print()
        print('NOTE: experiments above are missing PDM components because they were')
        print('      run before the simulation_metric override was added to')
        print('      eval_production.py. Re-run eval_production.py to populate them;')
        print('      pdm_score.py will then compute the full composite automatically.')
    print()


# ── Experiment discovery ────────────────────────────────────────────────────

def discover_experiments() -> List[str]:
    """
    Return all experiment names under sim_results/ that have an eval/metrics dir.

    WHY check for the metrics dir specifically: sim_results may contain partial
    runs (sim logs but no metrics); we only list experiments we can actually parse.
    """
    if not SIM_OUT.exists():
        return []
    experiments = []
    for child in sorted(SIM_OUT.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'eval' / 'metrics').exists():
            experiments.append(child.name)
    return experiments


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print('Scanning for PDM-Score component metrics ...')
    print(f'  sim_results root: {SIM_OUT}')

    experiments = discover_experiments()
    if not experiments:
        print(f'  No experiments with eval/metrics found under {SIM_OUT}.')
        return

    print(f'  Found {len(experiments)} experiment(s).')
    print()

    all_results: Dict[str, Dict] = {}
    for exp in experiments:
        print(f'Parsing {exp} ...')
        r = parse_pdm_metrics(exp)
        if r is not None:
            all_results[exp] = r

    print_pdm_table(all_results)


if __name__ == '__main__':
    main()
