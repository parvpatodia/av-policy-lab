"""
trajectory_viz.py — Trajectory visualization and analysis for AV imitation learning.

Reads nuPlan simulation results from sim_results/ parquet files and produces four plots:
  Plot 1 : Trajectory comparison grid  (per-scenario, per-planner side-by-side)
           NOTE: nuPlan closed-loop parquets do not log raw ego/expert (x,y) trajectories;
           they log per-timestep L2 error as a scalar time series. Plot 1 therefore shows
           L2-error time series per scenario/planner arranged in a grid, with the ego
           trajectory reconstructed as a 1-D "deviation" signal. When full nuPlan
           SimulationLog output is available (e.g. via NuBoardSimulationScenario), raw
           (x,y) can be loaded and the trajectory-drawing blocks below activated.
  Plot 2 : L2 error over time — mean ± std band across all scenarios per planner.
  Plot 3 : Failure distribution — initial speed vs avg L2 (one point per scenario×planner).
  Plot 4 : Goal vector visualization — placeholder (raw goal vectors not in parquet output).

Usage:
    python nuplan/trajectory_viz.py

Output (saved to nuplan/eval_results/):
    trajectory_comparison_scenario_{N}.png   — one grid per scenario
    l2_over_time.png
    failure_vs_speed.png
    goal_vectors_representative.png          — placeholder if goal data unavailable

Data sources:
    sim_results/<exp>/eval/metrics/ego_expert_L2_error.parquet
      Columns used:
        scenario_name                        : scenario token (string)
        scenario_type                        : scenario type label
        avg_ego_expert_L2_error_stat_value   : mean L2 over scenario (float, metres)
        max_ego_expert_L2_error_stat_value   : max L2 over scenario (float, metres)
        p90_ego_expert_L2_error_stat_value   : p90 L2 over scenario (float, metres)
        time_series_timestamps               : list of timestamps (microseconds)
        time_series_values                   : list of per-timestep L2 values (metres)

    sim_results/<exp>/eval/metrics/ego_mean_speed.parquet
      Columns used:
        scenario_name                        : scenario token (matches L2 parquet)
        ego_mean_speed_value_stat_value      : mean speed over scenario (m/s) — used as
                                               proxy for initial speed (exact initial speed
                                               not stored in parquet output)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # WHY: headless — no display needed; avoids NSException on macOS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SIM_ROOT  = Path(__file__).parent / 'sim_results'
EVAL_OUT  = Path(__file__).parent / 'eval_results'

# ── Color scheme (project-wide convention) ────────────────────────────────────
# BC=gray, IDM=purple, GoalBC=blue, RouteMapBC=orange, SpeedAdaptive=green,
# TrainedRouteBC=red, MapBC=brown, BEV=teal, MILE=magenta
PLANNER_COLOR: Dict[str, str] = {
    'BCPlanner':                         '#888888',   # gray
    'IDMPlanner':                        '#7B2D8B',   # purple
    'GoalBCPlanner':                     '#1F77B4',   # blue
    'MapBCPlanner':                      '#8B4513',   # brown
    'RouteMapBCPlanner':                 '#FF7F0E',   # orange
    'SpeedAdaptiveRouteMapBCPlanner':    '#2CA02C',   # green
    'TrainedRouteBCPlanner':             '#D62728',   # red
    'BEVPlanner':                        '#17BECF',   # teal
    'MILEPlanner':                       '#E377C2',   # magenta
    'DAggerPlanner':                     '#BCBD22',   # olive
}

# Canonical display order (best → worst roughly)
DISPLAY_ORDER = [
    'GoalBCPlanner',
    'IDMPlanner',
    'SpeedAdaptiveRouteMapBCPlanner',
    'RouteMapBCPlanner',
    'TrainedRouteBCPlanner',
    'MapBCPlanner',
    'BEVPlanner',
    'MILEPlanner',
    'BCPlanner',
    'DAggerPlanner',
]

# Experiments to include in primary analysis (canonical Phase 3 planners)
# Maps experiment_dir_name -> planner display name
PRIMARY_EXPERIMENTS: Dict[str, str] = {
    'closed_loop_BCPlanner':              'BCPlanner',
    'closed_loop_IDMPlanner':             'IDMPlanner',
    'closed_loop_GoalBCPlanner':          'GoalBCPlanner',
    'closed_loop_MapBCPlanner':           'MapBCPlanner',
    'closed_loop_RouteMapBCPlanner':      'RouteMapBCPlanner',
    'speed_adaptive_eval':                'SpeedAdaptiveRouteMapBCPlanner',
    'trained_route_bc_eval':              'TrainedRouteBCPlanner',
    'closed_loop_BEVPlanner':             'BEVPlanner',
    'closed_loop_MILEPlanner':            'MILEPlanner',
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_l2_parquet(metrics_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the ego_expert_L2_error parquet from a metrics directory.

    Returns None with a warning if the file is not found. Glob is used because
    nuPlan appends hash suffixes in some versions.
    """
    candidates = sorted(metrics_dir.glob('ego_expert_L2_error*.parquet'))
    if not candidates:
        warnings.warn(f'[trajectory_viz] L2 parquet not found in {metrics_dir}')
        return None
    return pd.read_parquet(candidates[0])


def _load_speed_parquet(metrics_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the ego_mean_speed parquet from a metrics directory.
    Returns None with a warning if not found.
    """
    candidates = sorted(metrics_dir.glob('ego_mean_speed*.parquet'))
    if not candidates:
        warnings.warn(f'[trajectory_viz] Speed parquet not found in {metrics_dir}')
        return None
    return pd.read_parquet(candidates[0])


def load_all_results(
    sim_root: Path = SIM_ROOT,
    experiments: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """
    Load L2 and speed data for all available experiments.

    Args:
        sim_root:    Root directory containing experiment subdirectories.
        experiments: Dict mapping experiment_dir_name -> planner_display_name.
                     If None, uses PRIMARY_EXPERIMENTS.

    Returns:
        Dict keyed by planner_display_name, each containing:
            'scenarios': List[Dict] with keys:
                scenario_name (str), scenario_type (str),
                avg_l2 (float), max_l2 (float), p90_l2 (float),
                mean_speed (float),   -- proxy for initial speed
                ts_times (np.ndarray of float, seconds since start),
                ts_l2    (np.ndarray of float, metres)
            'planner_name': str (from parquet)
    """
    if experiments is None:
        experiments = PRIMARY_EXPERIMENTS

    all_data: Dict[str, Dict] = {}

    for exp_dir_name, display_name in experiments.items():
        metrics_dir = sim_root / exp_dir_name / 'eval' / 'metrics'
        if not metrics_dir.exists():
            warnings.warn(f'[trajectory_viz] Skipping {exp_dir_name} — metrics dir not found')
            continue

        df_l2 = _load_l2_parquet(metrics_dir)
        if df_l2 is None:
            continue

        df_spd = _load_speed_parquet(metrics_dir)

        # Build per-scenario records
        scenarios = []
        for _, row in df_l2.iterrows():
            scen_name  = str(row.get('scenario_name', 'unknown'))
            scen_type  = str(row.get('scenario_type',  'unknown'))
            avg_l2     = float(row.get('avg_ego_expert_L2_error_stat_value', np.nan))
            max_l2     = float(row.get('max_ego_expert_L2_error_stat_value', np.nan))
            p90_l2     = float(row.get('p90_ego_expert_L2_error_stat_value', np.nan))

            # Time series: timestamps are in microseconds, values are L2 in metres
            raw_ts  = row.get('time_series_timestamps', [])
            raw_vs  = row.get('time_series_values',     [])
            ts_arr  = np.array(raw_ts, dtype=np.float64)
            l2_arr  = np.array(raw_vs, dtype=np.float64)

            if len(ts_arr) > 0:
                ts_sec = (ts_arr - ts_arr[0]) / 1e6   # WHY: convert µs to seconds
            else:
                ts_sec = np.array([])

            # Mean speed as proxy for initial speed
            # WHY: exact initial speed is not stored in parquet output.
            # ego_mean_speed_value_stat_value is the scenario mean (m/s), which is
            # correlated with but not identical to initial speed.
            mean_speed = np.nan
            if df_spd is not None:
                spd_row = df_spd[df_spd['scenario_name'] == scen_name]
                if len(spd_row) > 0:
                    mean_speed = float(spd_row['ego_mean_speed_value_stat_value'].iloc[0])

            scenarios.append({
                'scenario_name':  scen_name,
                'scenario_type':  scen_type,
                'avg_l2':         avg_l2,
                'max_l2':         max_l2,
                'p90_l2':         p90_l2,
                'mean_speed':     mean_speed,
                'ts_times':       ts_sec,
                'ts_l2':          l2_arr,
            })

        # WHY .get() not direct key access: 'planner_name' column is absent from
        # some nuPlan parquet versions. Direct access raises KeyError; fall back
        # to the display_name derived from the experiment directory.
        planner_col_val = (
            str(df_l2['planner_name'].iloc[0])
            if 'planner_name' in df_l2.columns and len(df_l2) > 0
            else display_name
        )
        all_data[display_name] = {
            'scenarios':    scenarios,
            'planner_name': planner_col_val,
        }
        print(f'  [load] {display_name}: {len(scenarios)} scenarios')

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Trajectory comparison grid
# ─────────────────────────────────────────────────────────────────────────────

def _get_ordered_planners(all_data: Dict[str, Dict]) -> List[str]:
    """Return planners in canonical display order, filtering to those present in data."""
    ordered = [p for p in DISPLAY_ORDER if p in all_data]
    # Append any planners not in DISPLAY_ORDER at the end
    for p in sorted(all_data.keys()):
        if p not in ordered:
            ordered.append(p)
    return ordered


def plot_trajectory_comparison_grid(
    all_data: Dict[str, Dict],
    out_dir: Path = EVAL_OUT,
) -> None:
    """
    Plot L2-error time series in a grid: one row per scenario, one column per planner.

    NOTE ON TRAJECTORY DATA:
        nuPlan's closed-loop simulation does not write raw ego/expert (x,y) trajectories
        to the metrics parquet output by default. The parquet contains:
          - Per-scenario aggregate stats (avg/max/p90 L2)
          - time_series_values: per-timestep L2 scalar (|ego_pos - expert_pos| in metres)
          - time_series_timestamps: corresponding timestamps in microseconds

        To get raw (x,y) trajectories you would need to:
          1. Open the .nuboard file with nuPlan's NuBoardSimulationScenario API
          2. Or instrument the planner to log ego state at each step during simulation
          3. Or access the simulation log via nuplan.planning.simulation.log.simulation_log

        Until that data is available, this function plots the L2 scalar over time,
        which clearly shows WHEN each planner drifts, how quickly errors compound,
        and how planners compare within each scenario.

    The color of each line encodes the per-timestep L2 value using a green→red colormap.
    The avg L2 for that scenario appears in the subplot title.

    Args:
        all_data:  Output of load_all_results().
        out_dir:   Directory to save PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    planners = _get_ordered_planners(all_data)
    if not planners:
        warnings.warn('[plot_trajectory_comparison_grid] No planner data to plot.')
        return

    # Collect all unique scenario names across all planners
    all_scenarios: Dict[str, str] = {}   # name -> type
    for pdata in all_data.values():
        for scen in pdata['scenarios']:
            all_scenarios[scen['scenario_name']] = scen['scenario_type']

    scenario_names = sorted(all_scenarios.keys())
    if not scenario_names:
        warnings.warn('[plot_trajectory_comparison_grid] No scenarios found.')
        return

    n_cols     = len(planners)
    cmap_l2    = cm.RdYlGn_r   # green=low L2, red=high L2
    # WHY: RdYlGn_r uses green for 0 (good) and red for high values (bad),
    # matching the intuition that red = error / failure.

    for scen_idx, scen_name in enumerate(scenario_names):
        scen_type = all_scenarios[scen_name]
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)
        fig.suptitle(
            f'Scenario {scen_idx}: {scen_name[:12]}... | type: {scen_type}',
            fontsize=12, fontweight='bold', y=1.02,
        )

        for col_idx, planner_name in enumerate(planners):
            ax = axes[0][col_idx]
            color = PLANNER_COLOR.get(planner_name, '#333333')

            if planner_name not in all_data:
                ax.set_visible(False)
                continue

            pdata    = all_data[planner_name]
            scen_rec = next(
                (s for s in pdata['scenarios'] if s['scenario_name'] == scen_name),
                None,
            )

            if scen_rec is None:
                # WHY NOT ax.set_visible(False) here: we've already written text to
                # the axis; hiding it makes the 'no data' annotation invisible. Keep
                # the axis visible so the user sees the gap and knows data is missing.
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.set_title(f'{planner_name}\nN/A', fontsize=8)
                continue

            ts    = scen_rec['ts_times']
            l2    = scen_rec['ts_l2']
            avg_l2 = scen_rec['avg_l2']

            if len(ts) == 0 or len(l2) == 0:
                ax.text(0.5, 0.5, 'empty time series',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
            else:
                # Color-code each segment by its L2 value
                # WHY segment coloring: a single-color line hides the dynamics.
                # Segment coloring shows WHERE in time the error first spikes.
                l2_max = max(float(np.nanmax(l2)), 1e-3)
                norm   = mcolors.Normalize(vmin=0, vmax=l2_max)

                for i in range(len(ts) - 1):
                    seg_l2  = (l2[i] + l2[i + 1]) / 2.0
                    seg_col = cmap_l2(norm(seg_l2))
                    ax.plot(ts[i:i+2], l2[i:i+2], color=seg_col, linewidth=1.5, solid_capstyle='round')

                # Mark start (circle) and end (triangle)
                ax.plot(ts[0],  l2[0],  'o', color=color, markersize=6, zorder=5,
                        label='start', markeredgecolor='white', markeredgewidth=0.5)
                ax.plot(ts[-1], l2[-1], '^', color=color, markersize=7, zorder=5,
                        label='end',   markeredgecolor='white', markeredgewidth=0.5)

                # Horizontal reference lines
                ax.axhline(avg_l2, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
                ax.fill_between(ts, 0, l2, alpha=0.08, color=color)

                # Colorbar for this axis
                sm = cm.ScalarMappable(cmap=cmap_l2, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
                cbar.set_label('L2 (m)', fontsize=7)
                cbar.ax.tick_params(labelsize=7)

            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('L2 error (m)', fontsize=8)
            ax.tick_params(labelsize=7)

            short_name = planner_name.replace('Planner', '').replace('RouteMap', 'RMap').replace('SpeedAdaptive', 'SpdAdp')
            ax.set_title(f'{short_name}\navg={avg_l2:.2f} m', fontsize=9, color=color, fontweight='bold')

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.tight_layout()
        out_path = out_dir / f'trajectory_comparison_scenario_{scen_idx}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: L2 error over time (mean ± std band)
# ─────────────────────────────────────────────────────────────────────────────

def plot_l2_over_time(
    all_data: Dict[str, Dict],
    out_dir: Path = EVAL_OUT,
) -> None:
    """
    Plot mean ± std L2 error over time across all scenarios, all planners overlaid.

    Each planner is represented by:
      - A solid line at the mean per-timestep L2 across scenarios
      - A shaded ±1 std band

    Because scenarios have slightly different lengths (149 vs 150 timesteps),
    we interpolate all time series onto a common time grid at 0.1 s resolution.

    The plot reveals:
      - How quickly L2 compounds for drifting planners (BC family: nearly linear growth)
      - Whether GoalBC maintains a near-zero floor throughout
      - Whether SpeedAdaptive closes the gap at later timesteps (drift correction)

    Args:
        all_data:  Output of load_all_results().
        out_dir:   Directory to save the PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    planners = _get_ordered_planners(all_data)

    max_t_global = 0.0

    plot_handles = []
    plot_labels  = []

    for planner_name in planners:
        pdata = all_data[planner_name]
        series_list = []
        max_t_planner = 0.0

        for scen in pdata['scenarios']:
            ts = scen['ts_times']
            l2 = scen['ts_l2']
            if len(ts) > 0:
                max_t_planner = max(max_t_planner, float(ts[-1]))
                series_list.append((ts, l2))

        if not series_list:
            continue

        max_t_global = max(max_t_global, max_t_planner)

        # Interpolate all time series onto a common 0.1 s grid
        t_grid = np.arange(0, max_t_planner + 0.05, 0.1)
        interp_matrix = []
        for ts, l2 in series_list:
            if len(ts) < 2:
                continue
            l2_interp = np.interp(t_grid, ts, l2, left=l2[0], right=l2[-1])
            interp_matrix.append(l2_interp)

        if not interp_matrix:
            continue

        mat  = np.array(interp_matrix)    # (n_scenarios, n_timesteps)
        mean = np.mean(mat, axis=0)
        std  = np.std(mat,  axis=0)

        color = PLANNER_COLOR.get(planner_name, '#333333')
        lw    = 2.5 if planner_name in ('GoalBCPlanner', 'SpeedAdaptiveRouteMapBCPlanner') else 1.8

        h, = ax.plot(t_grid, mean, color=color, linewidth=lw,
                     label=f'{planner_name} ({np.mean([s["avg_l2"] for s in pdata["scenarios"]]):.2f} m)')
        # WHY clip lower bound not upper: fill_between(lower, upper).
        # mean+std is always ≥ mean ≥ 0 so the upper bound never needs clipping.
        # mean-std CAN go negative for low-L2 planners (e.g. GoalBC at 1.82m with
        # small std). Clipping the lower bound at 0 matches ax.set_ylim(bottom=0).
        ax.fill_between(t_grid, np.maximum(mean - std, 0), mean + std,
                        color=color, alpha=0.12)

        plot_handles.append(h)
        plot_labels.append(planner_name)

    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('L2 error to expert (metres)', fontsize=13)
    ax.set_title('L2 Error Over Time: Mean ± Std Across All Scenarios', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)

    # Annotation: highlight the compounding-drift inflection point (roughly at t=3s for BC)
    ax.text(
        0.98, 0.97,
        'Shaded bands = ±1 std across scenarios\nLine label = planner avg L2 (m)',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
    )

    plt.tight_layout()
    out_path = out_dir / 'l2_over_time.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Failure distribution (speed vs avg L2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_vs_speed(
    all_data: Dict[str, Dict],
    out_dir: Path = EVAL_OUT,
) -> None:
    """
    Scatter plot: mean speed (proxy for initial speed) vs avg L2 per scenario×planner.

    One point per (scenario, planner) combination. Points are colored by planner.
    This visualizes whether failures correlate with driving speed, validating the
    SpeedAdaptiveRouteMapBC hypothesis: RouteMapBC (fixed 8m goal) should fail most
    at LOW speeds, where the fixed 8m goal is far outside the T+0.8 s horizon.

    WHY mean speed as x-axis:
        The exact initial speed is not stored in the parquet output. The mean speed
        over the scenario is a valid proxy because scenario mean speed strongly
        correlates with initial speed (nuPlan scenarios are short: ~15 s). A scenario
        labeled "stationary_in_traffic" will have mean_speed ≈ 0; "high_magnitude_speed"
        will have mean_speed >> 10 m/s.

    Args:
        all_data:  Output of load_all_results().
        out_dir:   Directory to save the PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    planners = _get_ordered_planners(all_data)

    any_plotted = False
    for planner_name in planners:
        pdata  = all_data[planner_name]
        speeds = []
        l2s    = []
        types  = []

        for scen in pdata['scenarios']:
            spd = scen['mean_speed']
            l2  = scen['avg_l2']
            if np.isnan(spd) or np.isnan(l2):
                continue
            speeds.append(spd)
            l2s.append(l2)
            types.append(scen['scenario_type'])

        if not speeds:
            continue

        color = PLANNER_COLOR.get(planner_name, '#333333')
        short  = planner_name.replace('Planner', '').replace('RouteMap', 'RMap').replace('SpeedAdaptive', 'SpdAdp')

        sc = ax.scatter(speeds, l2s, color=color, alpha=0.85, s=90,
                        label=short, zorder=5, edgecolors='white', linewidths=0.5)

        # Annotate each point with its scenario type (abbreviated)
        for spd, l2, stype in zip(speeds, l2s, types):
            abbrev = ''.join(w[0].upper() for w in stype.replace('_', ' ').split())[:4]
            ax.annotate(abbrev, (spd, l2), textcoords='offset points',
                        xytext=(3, 3), fontsize=6, color=color, alpha=0.8)

        any_plotted = True

    if not any_plotted:
        warnings.warn('[plot_failure_vs_speed] No data to plot.')
        plt.close(fig)
        return

    # Reference lines
    ax.axhline(20.0, color='red',    linestyle='--', linewidth=0.9, alpha=0.6, label='20 m failure threshold')
    ax.axhline(5.0,  color='green',  linestyle='--', linewidth=0.9, alpha=0.6, label='5 m good threshold')
    ax.axvline(4.33, color='gray',   linestyle=':',  linewidth=0.9, alpha=0.5, label='4.33 m/s avg scenario speed')

    ax.set_xlabel('Mean scenario speed (m/s) — proxy for initial speed', fontsize=13)
    ax.set_ylabel('Avg L2 error to expert (metres)', fontsize=13)
    ax.set_title(
        'Failure Distribution vs Speed\n'
        '(SpeedAdaptiveRouteMapBC hypothesis: RouteMapBC fails most at low speed)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)

    # Explanation text
    ax.text(
        0.02, 0.97,
        'Point labels: NHV=near_high_speed_vehicle,\nHMS=high_magnitude_speed, SIT=stationary_in_traffic',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
    )

    plt.tight_layout()
    out_path = out_dir / 'failure_vs_speed.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Goal vector visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_goal_vectors(
    all_data: Dict[str, Dict],
    out_dir: Path = EVAL_OUT,
) -> None:
    """
    Goal vector visualization for a representative scenario.

    CURRENT STATUS: Goal vectors (route centerline, expert T+8 positions) are not
    logged in the nuPlan metrics parquet output. They would need to be captured
    by instrumenting the planner's compute_planner_trajectory() method to write
    per-step goal positions to disk, then loading that log here.

    WHAT THIS FUNCTION WOULD DO (when goal data is available):
        1. Load per-timestep ego position (x,y) and goal position (x,y) for each planner
        2. Find the scenario with highest avg L2 spread across planners (most informative)
        3. Plot route centerline as a light gray background curve
        4. Every 10th timestep: draw an arrow from ego pos in the direction of goal offset
        5. Color arrows by planner (GoalBC=blue, RouteMapBC=orange, SpeedAdaptive=green)
        6. Show how goal directions diverge between planners as speed changes

    HOW TO ACTIVATE THIS PLOT:
        Add to RouteMapBCPlanner.compute_planner_trajectory():
            import json
            goal_log_path = Path('/tmp/goal_log_{scenario_name}.jsonl')
            with open(goal_log_path, 'a') as f:
                json.dump({'t': float(ts), 'ego_x': float(x_g), 'ego_y': float(y_g),
                           'gx': float(dx_goal), 'gy': float(dy_goal),
                           'speed': float(np.sqrt(vx**2+vy**2))}, f)
                f.write('\n')
        Then load that JSONL file here and plot the arrows.

    For now, this function saves a placeholder PNG with instructions.

    Args:
        all_data:  Output of load_all_results() (not used by placeholder).
        out_dir:   Directory to save the PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Placeholder: schematic diagram explaining the intended visualization
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')

    # Draw a schematic route centerline
    route_x = np.linspace(1, 9, 30)
    route_y = 3.0 + 0.8 * np.sin(route_x * 0.5)
    ax.plot(route_x, route_y, color='lightgray', linewidth=4, zorder=1, label='Route centerline')

    # Schematic ego trajectory
    ego_x = np.linspace(1.5, 8.5, 20)
    ego_y = 3.0 + 0.7 * np.sin(ego_x * 0.5) + np.random.default_rng(0).normal(0, 0.1, 20)
    ax.plot(ego_x, ego_y, color='black', linewidth=1.5, zorder=2, label='Ego trajectory (schematic)')

    # Schematic goal arrows at every other sampled point
    arrow_configs = [
        ('GoalBCPlanner',                   PLANNER_COLOR['GoalBCPlanner'],                 0.00),   # exact direction
        ('RouteMapBCPlanner',               PLANNER_COLOR['RouteMapBCPlanner'],              0.30),   # forward offset
        ('SpeedAdaptiveRouteMapBCPlanner',  PLANNER_COLOR['SpeedAdaptiveRouteMapBCPlanner'], 0.15),   # closer to GoalBC
    ]
    for i in range(3, 18, 4):
        for planner_name, color, angular_offset in arrow_configs:
            base_dir = np.array([route_x[min(i+2, 29)] - route_x[i],
                                 route_y[min(i+2, 29)] - route_y[i]])
            base_dir /= (np.linalg.norm(base_dir) + 1e-8)
            angle = angular_offset
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            direction = rot @ base_dir * 0.8
            ax.annotate('', xy=(ego_x[i] + direction[0], ego_y[i] + direction[1]),
                        xytext=(ego_x[i], ego_y[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=PLANNER_COLOR['GoalBCPlanner'],                 label='GoalBC goal direction (expert T+0.8s)'),
        mpatches.Patch(color=PLANNER_COLOR['RouteMapBCPlanner'],             label='RouteMapBC goal direction (8m fixed)'),
        mpatches.Patch(color=PLANNER_COLOR['SpeedAdaptiveRouteMapBCPlanner'],label='SpeedAdaptive goal direction (speed×0.8s)'),
        mpatches.Patch(color='lightgray',                                    label='Route centerline'),
        mpatches.Patch(color='black',                                        label='Ego trajectory (schematic)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    ax.set_title(
        'Goal Vector Visualization — SCHEMATIC (actual data not in parquet)\n'
        'See docstring for how to activate with real goal logs',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('x position (m) — schematic', fontsize=10)
    ax.set_ylabel('y position (m) — schematic', fontsize=10)

    # Instruction box
    instruction = (
        "To generate with real data:\n"
        "1. Instrument planner to log (ego_x, ego_y, goal_dx, goal_dy) per step\n"
        "2. Load goal log JSONL here\n"
        "3. Plot arrows from ego pos at every 10th timestep\n"
        "   — GoalBC=blue, RouteMapBC=orange, SpeedAdaptive=green"
    )
    ax.text(0.02, 0.97, instruction, transform=ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

    ax.tick_params(labelsize=9)
    plt.tight_layout()
    out_path = out_dir / 'goal_vectors_representative.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path.name}  (schematic — see docstring to activate)')


# ─────────────────────────────────────────────────────────────────────────────
# Additional utility: per-planner summary bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_planner_summary_bar(
    all_data: Dict[str, Dict],
    out_dir: Path = EVAL_OUT,
) -> None:
    """
    Horizontal bar chart of mean avg-L2 per planner, sorted best to worst.

    Error bars show ±1 std across scenarios. Bars are colored by PLANNER_COLOR.
    Deployable planners (no expert DB required at inference) are annotated with [D].

    This is a companion to the trajectory plots — gives the one-glance leaderboard.

    Args:
        all_data:  Output of load_all_results().
        out_dir:   Directory to save the PNG.
    """
    DEPLOYABLE = {
        'IDMPlanner', 'RouteMapBCPlanner', 'SpeedAdaptiveRouteMapBCPlanner',
        'MapBCPlanner',
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    planners_sorted = sorted(
        all_data.keys(),
        key=lambda p: np.mean([s['avg_l2'] for s in all_data[p]['scenarios']]),
    )

    means = []
    stds  = []
    names = []
    colors= []

    for p in planners_sorted:
        vals = [s['avg_l2'] for s in all_data[p]['scenarios'] if not np.isnan(s['avg_l2'])]
        if not vals:
            continue
        label = p.replace('Planner', '').replace('RouteMap', 'RMap').replace('SpeedAdaptive', 'SpdAdp')
        if p in DEPLOYABLE:
            label = label + ' [D]'
        names.append(label)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        colors.append(PLANNER_COLOR.get(p, '#888888'))

    if not means:
        warnings.warn('[plot_planner_summary_bar] No data to plot.')
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(means) * 0.65)))
    plt.style.use('seaborn-v0_8-whitegrid')

    y_pos = range(len(means))
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
            error_kw=dict(elinewidth=1.5, capsize=4, capthick=1.5, ecolor='black'),
            height=0.65, zorder=3)

    # Annotate each bar with the mean value
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.5, i, f'{m:.1f} m', va='center', ha='left', fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(20.0, color='red',   linestyle='--', lw=1.2, alpha=0.6, label='20 m failure threshold')
    ax.axvline(5.0,  color='green', linestyle='--', lw=1.2, alpha=0.6, label='5 m good threshold')
    ax.set_xlabel('Mean avg L2 error (metres)', fontsize=12)
    ax.set_title('Planner Leaderboard — Mean Avg L2 Error (lower is better)\n[D] = deployable (no expert DB at inference)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(left=0)
    ax.tick_params(axis='x', labelsize=10)
    ax.invert_yaxis()   # WHY: best (lowest L2) at top

    plt.tight_layout()
    out_path = out_dir / 'planner_summary.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run all four visualizations.

    Loads data from sim_results/, saves all PNGs to eval_results/.
    Gracefully skips any experiment whose metrics directory is missing.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    print('\n=== trajectory_viz.py ===')
    print(f'  SIM_ROOT : {SIM_ROOT}')
    print(f'  EVAL_OUT : {EVAL_OUT}')
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    print('\n[1/5] Loading simulation results...')
    all_data = load_all_results(SIM_ROOT)

    if not all_data:
        print('[ERROR] No data loaded. Check that sim_results/ exists and contains parquets.')
        return

    print(f'  Loaded {len(all_data)} planner(s): {", ".join(all_data.keys())}')

    print('\n[2/5] Plot 1: Trajectory comparison grid (L2 time series per scenario × planner)...')
    plot_trajectory_comparison_grid(all_data, EVAL_OUT)

    print('\n[3/5] Plot 2: L2 error over time (mean ± std band across scenarios)...')
    plot_l2_over_time(all_data, EVAL_OUT)

    print('\n[4/5] Plot 3: Failure distribution (mean speed vs avg L2)...')
    plot_failure_vs_speed(all_data, EVAL_OUT)

    print('\n[5/5] Plot 4: Goal vector visualization...')
    plot_goal_vectors(all_data, EVAL_OUT)

    print('\n[bonus] Planner summary bar chart...')
    plot_planner_summary_bar(all_data, EVAL_OUT)

    print(f'\nDone. All plots saved to {EVAL_OUT}/')


if __name__ == '__main__':
    main()
