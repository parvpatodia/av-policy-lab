"""
Closed-loop simulation of BCPlanner and IDMPlanner on nuPlan mini dataset.
Runs a small set of scenarios and reports available metrics.

Usage:
    cd /Users/parvpatodia/nuplan-devkit
    python /Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/closed_loop_eval.py
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# ── Environment setup (must happen before nuplan imports) ─────────────────────
os.environ.setdefault('NUPLAN_DATA_ROOT', '/Users/parvpatodia/nuplan-devkit/data/cache')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/Users/parvpatodia/nuplan-devkit/maps')
os.environ.setdefault('NUPLAN_EXP_ROOT',  '/Users/parvpatodia/nuplan-devkit/exp')
os.environ.setdefault('NUPLAN_TUTORIAL_PATH', '/Users/parvpatodia/nuplan-devkit/tutorials')

sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')

import numpy as np
import torch
import torch.nn as nn
import hydra

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

CKPT_PATH    = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/checkpoints/bc_best.pt')
FUTURE_STEPS = 16

# ── BCPolicy model definition ─────────────────────────────────────────────────
class BCPolicy(nn.Module):
    def __init__(self, in_dim=6, hidden=256, out_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── BCPlanner ─────────────────────────────────────────────────────────────────
class BCPlanner(AbstractPlanner):
    """Behavior Cloning planner. Loads MLP checkpoint, runs forward pass each step."""

    def __init__(self, ckpt_path: str = str(CKPT_PATH)):
        self._ckpt_path = ckpt_path
        self._device    = torch.device('cpu')   # WHY: cpu for sim stability on macOS
        self._model     = None
        self._X_mean = self._X_std = self._Y_mean = self._Y_std = None
        self._dt = 0.1

    def name(self) -> str:
        return 'BCPlanner'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = BCPolicy().to(self._device)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']
        self._Y_std  = ckpt['Y_std']

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego     = current_input.history.current_state[0]
        heading = ego.rear_axle.heading
        vx      = ego.dynamic_car_state.rear_axle_velocity_2d.x
        vy      = ego.dynamic_car_state.rear_axle_velocity_2d.y
        ax      = ego.dynamic_car_state.rear_axle_acceleration_2d.x
        ay      = ego.dynamic_car_state.rear_axle_acceleration_2d.y
        cx, cy  = ego.rear_axle.x, ego.rear_axle.y
        t0      = ego.time_point.time_us
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        x_raw  = torch.tensor([np.sin(heading), np.cos(heading), vx, vy, ax, ay], dtype=torch.float32)
        x_norm = (x_raw - self._X_mean) / self._X_std
        with torch.no_grad():
            pred = (self._model(x_norm.unsqueeze(0)).squeeze(0).numpy()
                    * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)

        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            wx    = cx + cos_h * dx_e - sin_h * dy_e
            wy    = cy + sin_h * dx_e + cos_h * dy_e
            w_yaw = heading + d_yaw
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(wx, wy, w_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * self._dt * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)


# ── IDMPlanner ────────────────────────────────────────────────────────────────
class IDMPlanner(AbstractPlanner):
    """IDM car-following planner. Lead vehicle detection + free-road fallback."""

    V0 = 15.0; T = 1.5; A_MAX = 1.5; B = 2.0; S0 = 2.0; DELTA = 4; DT = 0.1

    def name(self) -> str: return 'IDMPlanner'
    def observation_type(self): return DetectionsTracks
    def initialize(self, initialization: PlannerInitialization) -> None: pass

    def _find_lead(self, ego, obs):
        heading = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        v_ego = ego.dynamic_car_state.rear_axle_velocity_2d.x
        best_gap, best_dv = np.inf, 0.0
        try:
            for obj in obs.tracked_objects.tracked_objects:
                dx = obj.box.center.x - ego.rear_axle.x
                dy = obj.box.center.y - ego.rear_axle.y
                x_e =  cos_h * dx + sin_h * dy
                y_e = -sin_h * dx + cos_h * dy
                if x_e < 1.0 or abs(y_e) > 3.0:
                    continue
                gap = x_e - 2.5 - getattr(obj.box, 'length', 4.0) / 2.0
                if gap < best_gap:
                    best_gap = gap
                    try: best_dv = v_ego - (obj.velocity.x * cos_h + obj.velocity.y * sin_h)
                    except: best_dv = v_ego
        except: pass
        return (max(best_gap, 0.01), best_dv) if best_gap < np.inf else None

    def _accel(self, v, lead):
        free = 1.0 - (v / max(self.V0, 0.1)) ** self.DELTA
        if lead is None:
            return self.A_MAX * free
        gap, dv = lead
        s_star = self.S0 + max(0.0, v * self.T + v * dv / (2.0 * np.sqrt(self.A_MAX * self.B)))
        return self.A_MAX * (free - (s_star / max(gap, 0.01)) ** 2)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego     = current_input.history.current_state[0]
        heading = ego.rear_axle.heading
        cx, cy  = ego.rear_axle.x, ego.rear_axle.y
        t0      = ego.time_point.time_us
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        # WHY: observations live in history buffer, not directly on PlannerInput
        observation = current_input.history.observation_buffer[-1]
        lead = self._find_lead(ego, observation)
        v, x, y = max(ego.dynamic_car_state.rear_axle_velocity_2d.x, 0.0), cx, cy
        states  = [ego]
        for j in range(FUTURE_STEPS):
            a  = np.clip(self._accel(v, lead), -4.0, 2.0)
            v  = max(v + a * self.DT, 0.0)
            ds = v * self.DT
            x += cos_h * ds; y += sin_h * ds
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(v, 0.0),
                rear_axle_acceleration_2d=StateVector2D(a, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * self.DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)


# ── Simulation runner ─────────────────────────────────────────────────────────
def run(planner, save_dir, n_scenarios=2):
    from nuplan.planning.script.run_simulation import run_simulation as main_sim

    BASE = '/Users/parvpatodia/nuplan-devkit/nuplan/planning/script'
    paths = construct_simulation_hydra_paths(BASE)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # WHY: initialize_config_dir accepts absolute paths; initialize() requires relative
    hydra.initialize_config_dir(config_dir=paths.config_path, version_base='1.1')

    cfg = hydra.compose(
        config_name=paths.config_name,
        overrides=[
            f'group={save_dir}',
            f'experiment_name=closed_loop_{planner.name()}',
            'job_name=eval',
            'experiment=${experiment_name}/${job_name}',
            'worker=sequential',
            'ego_controller=perfect_tracking_controller',
            'observation=box_observation',
            f'hydra.searchpath=[{paths.common_dir}, {paths.experiment_dir}]',
            'output_dir=${group}/${experiment}',
            'scenario_builder=nuplan_mini',
            # WHY: db_files overrides data_root — our mini data lives at data/cache/mini, not
            #      the default data/cache/nuplan-v1.1/splits/mini expected by the config
            'scenario_builder.db_files=/Users/parvpatodia/nuplan-devkit/data/cache/mini',
            'scenario_filter=one_continuous_log',
            "scenario_filter.log_names=['2021.05.12.22.00.38_veh-35_01008_01518']",
            f'scenario_filter.limit_total_scenarios={n_scenarios}',
        ],
    )

    print(f'\n>>> Running closed-loop simulation: {planner.name()} ({n_scenarios} scenarios)')
    main_sim(cfg, planner)
    print(f'>>> Done. Results in: {save_dir}')
    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == '__main__':
    save_dir = str(Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan/sim_results'))
    Path(save_dir).mkdir(exist_ok=True)

    run(BCPlanner(),  save_dir, n_scenarios=3)
    run(IDMPlanner(), save_dir, n_scenarios=3)

    print('\nSimulation complete. Check sim_results/ for .nuboard files and metric JSON.')
