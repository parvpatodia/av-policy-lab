"""
av-policy-lab planner module.
All planners are proper importable classes compatible with nuPlan's AbstractPlanner.
Import this module before running SimulationLog.load_data() so pickle finds the classes.

Classes
-------
BCPolicy          : MLP architecture (6 -> 256 -> 256 -> 256 -> 48)
BCPlanner         : Behavior-cloning AbstractPlanner wrapper
IDMPlanner        : Intelligent Driver Model AbstractPlanner wrapper
DAggerPlanner     : DAgger data-collection wrapper around BCPlanner
BEVPolicy         : CNN architecture (3x64x64 + 6) -> 48
BEVPlanner        : BEV CNN AbstractPlanner wrapper (ego-history rasterization)
MILEPolicy        : World model (encoder + GRU transition + policy), joint imitation+consistency loss
MILEPlanner       : MILE AbstractPlanner wrapper (inference: state -> encoder -> latent -> policy)
GoalBCPolicy      : Goal-conditioned MLP (8 -> 256 -> 256 -> 256 -> 48, state + T+8 expert waypoint)
GoalBCPlanner     : GoalBC wrapper -- expert DB lookup for goal at inference (Phase 3a, oracle)
MapBCPlanner      : GoalBC weights + road centerline goal at inference -- no expert required (Phase 3b)
RouteMapBCPlanner : GoalBC weights + pre-computed global route goal, fixed 8m look-ahead (Phase 3c)
TrainedRouteBCPlanner          : RouteMapBC loading route-goal-trained weights (Phase 3c')
SpeedAdaptiveRouteMapBCPlanner : RouteMapBC with look-ahead = speed x 0.8s (Phase 3c'', scale fix)
RoadblockRouteMapBCPlanner     : SpeedAdaptive + route_roadblock_ids junction selection (Phase 3c''')
DualHorizonRouteMapBCPlanner   : near + far dual-horizon goals (Phase 3c''''')
DiffusionPolicyPlanner         : DDPM generative policy, same 10-dim goal conditioning (Phase 3d)
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

FUTURE_STEPS = 16
DT           = 0.1   # Trajectory waypoint spacing (s). nuPlan sim calls planner at ~10 Hz.
# NOTE: The nuPlan mini SQLite DB is at 100 Hz (10 ms/row), not 10 Hz.
#       Training targets are raw consecutive rows (0.01 s apart). DT=0.1 stamps
#       them 10x further apart. perfect_tracking_controller executes by spatial
#       position, so this mismatch does not affect L2 in practice. See verify_pipeline.py.


# -- Model architecture -------------------------------------------------------

class BCPolicy(nn.Module):
    """
    MLP behavior-cloning policy.
    Input : [sin(yaw), cos(yaw), vx, vy, ax, ay]  (6-dim)
    Output: [(dx, dy, d_yaw) x 16]                (48-dim, ego-frame relative)
    """

    def __init__(self, in_dim: int = 6, hidden: int = 256, out_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -- BCPlanner ----------------------------------------------------------------

class BCPlanner(AbstractPlanner):
    """
    Behavior-cloning planner.
    Loads a trained BCPolicy checkpoint and runs one forward pass per planning step.
    """

    def __init__(self, ckpt_path: str):
        self._ckpt_path  = ckpt_path
        self._device     = torch.device('cpu')   # WHY: CPU for sim stability on macOS MPS
        self._model: Optional[BCPolicy]  = None
        self._X_mean = self._X_std = None
        self._Y_mean = self._Y_std = None

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

    def _ego_features(self, ego: EgoState) -> np.ndarray:
        h  = ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        return np.array([
            np.sin(h), np.cos(h),
            dcs.rear_axle_velocity_2d.x,
            dcs.rear_axle_velocity_2d.y,
            dcs.rear_axle_acceleration_2d.x,
            dcs.rear_axle_acceleration_2d.y,
        ], dtype=np.float32)

    def _predict(self, feat: np.ndarray) -> np.ndarray:
        """Raw features (6,) -> trajectory (16, 3) ego-frame."""
        xt = torch.tensor(feat, dtype=torch.float32)
        xt = (xt - self._X_mean) / self._X_std
        with torch.no_grad():
            pred = self._model(xt.unsqueeze(0)).squeeze(0).numpy()
        return (pred * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)

    def _build_trajectory(self, ego: EgoState, pred: np.ndarray) -> InterpolatedTrajectory:
        cx, cy  = ego.rear_axle.x, ego.rear_axle.y
        heading = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        vx = ego.dynamic_car_state.rear_axle_velocity_2d.x
        vy = ego.dynamic_car_state.rear_axle_velocity_2d.y
        ax = ego.dynamic_car_state.rear_axle_acceleration_2d.x
        ay = ego.dynamic_car_state.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us

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
                time_point=TimePoint(t0 + int((j + 1) * DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego  = current_input.history.current_state[0]
        feat = self._ego_features(ego)
        pred = self._predict(feat)
        return self._build_trajectory(ego, pred)


# -- IDMPlanner ---------------------------------------------------------------

class IDMPlanner(AbstractPlanner):
    """
    Intelligent Driver Model planner.
    Longitudinal: IDM acceleration (Treiber et al. 2000).
    Lateral: constant heading (straight-line lane-following).
    Lead vehicle detection via DetectionsTracks; free-road fallback.
    REF: Treiber et al. (2000) Phys. Rev. E 62(2).
    """

    V0    = 15.0   # desired free-road speed [m/s]
    T     = 1.5    # desired time headway [s]
    A_MAX = 1.5    # max acceleration [m/s^2]
    B     = 2.0    # comfortable deceleration [m/s^2]
    S0    = 2.0    # minimum bumper-to-bumper gap [m]
    DELTA = 4      # free-road exponent

    def name(self) -> str: return 'IDMPlanner'
    def observation_type(self):  return DetectionsTracks
    def initialize(self, _):     pass

    def _find_lead(self, ego: EgoState, obs) -> Optional[Tuple[float, float]]:
        heading   = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        v_ego     = ego.dynamic_car_state.rear_axle_velocity_2d.x
        best_gap  = np.inf
        best_dv   = 0.0
        try:
            for obj in obs.tracked_objects.tracked_objects:
                dx   = obj.box.center.x - ego.rear_axle.x
                dy   = obj.box.center.y - ego.rear_axle.y
                x_e  =  cos_h * dx + sin_h * dy
                y_e  = -sin_h * dx + cos_h * dy
                if x_e < 1.0 or abs(y_e) > 3.0:
                    continue
                gap = x_e - 2.5 - getattr(obj.box, 'length', 4.0) / 2.0
                if gap < best_gap:
                    best_gap = gap
                    try:
                        best_dv = v_ego - (obj.velocity.x * cos_h + obj.velocity.y * sin_h)
                    except AttributeError:
                        best_dv = v_ego
        except AttributeError:
            pass
        return (max(best_gap, 0.01), best_dv) if best_gap < np.inf else None

    def _idm_accel(self, v: float, lead: Optional[Tuple[float, float]]) -> float:
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
        obs     = current_input.history.observation_buffer[-1]
        lead    = self._find_lead(ego, obs)
        v, x, y = max(ego.dynamic_car_state.rear_axle_velocity_2d.x, 0.0), cx, cy
        states  = [ego]
        for j in range(FUTURE_STEPS):
            a = np.clip(self._idm_accel(v, lead), -4.0, 2.0)
            v = max(v + a * DT, 0.0)
            x += cos_h * v * DT
            y += sin_h * v * DT
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(v, 0.0),
                rear_axle_acceleration_2d=StateVector2D(a, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)


# -- DAggerPlanner ------------------------------------------------------------

class DAggerPlanner(AbstractPlanner):
    """
    DAgger (Dataset Aggregation) data-collection wrapper.
    REF: Ross et al. (2011) "A Reduction of Imitation Learning and Structured
         Prediction to No-Regret Online Learning." AISTATS 2011.

    At each planning step:
      1. Runs the current BC policy to produce the trajectory (controls the ego).
      2. Records the visited ego-state features.
      3. Queries the original DB for the expert's future trajectory at the same
         timestamp -- the label the policy SHOULD have produced.

    The aggregated dataset (visited states + expert labels) is stored in
    self.dagger_X and self.dagger_Y and can be retrieved after simulation.

    Why this approximation works:
      The nuPlan DB stores the full expert trajectory at every 100ms timestamp.
      Even when the BC ego drifts from the log ego position, the expert action
      at time t (= what the human would do starting at t) is still available.
      We use this as a supervision signal for the states BC actually visits,
      directly addressing the distributional mismatch.
    """

    def __init__(self, bc_planner: BCPlanner, db_path: str):
        self._bc      = bc_planner
        self._db_path = db_path
        self._expert: dict = {}
        self.dagger_X: List[np.ndarray] = []
        self.dagger_Y: List[np.ndarray] = []

    def name(self) -> str:
        return f'DAggerPlanner_iter{getattr(self, "_iter", 0)}'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._bc.initialize(initialization)
        self._build_expert_lookup()

    def _build_expert_lookup(self) -> None:
        con  = sqlite3.connect(self._db_path)
        rows = con.execute(
            "SELECT timestamp, x, y, qw, qx, qy, qz, vx, vy, acceleration_x, acceleration_y "
            "FROM ego_pose ORDER BY timestamp"
        ).fetchall()
        con.close()

        arr  = np.array(rows, dtype=np.float64)
        ts   = arr[:, 0].astype(np.int64)
        x_g  = arr[:, 1]; y_g = arr[:, 2]
        qw, qx_, qy_, qz_ = arr[:,3], arr[:,4], arr[:,5], arr[:,6]
        vx   = arr[:, 7]; vy = arr[:, 8]
        ax_  = arr[:, 9]; ay_ = arr[:,10]
        yaw  = np.arctan2(2*(qw*qz_ + qx_*qy_), 1 - 2*(qy_**2 + qz_**2))

        N = len(arr)
        for i in range(N - FUTURE_STEPS):
            feat = np.array([np.sin(yaw[i]), np.cos(yaw[i]),
                             vx[i], vy[i], ax_[i], ay_[i]], dtype=np.float32)
            cx, cy, cyaw = x_g[i], y_g[i], yaw[i]
            cos_h = np.cos(-cyaw); sin_h = np.sin(-cyaw)
            tgt = np.zeros(FUTURE_STEPS * 3, dtype=np.float32)
            for j in range(FUTURE_STEPS):
                fi = i + j + 1
                dx_w = x_g[fi] - cx; dy_w = y_g[fi] - cy
                tgt[j*3]   = cos_h * dx_w - sin_h * dy_w
                tgt[j*3+1] = sin_h * dx_w + cos_h * dy_w
                dyaw = yaw[fi] - cyaw
                tgt[j*3+2] = (dyaw + np.pi) % (2*np.pi) - np.pi
            self._expert[int(ts[i])] = (feat, tgt)

    def _nearest_expert(self, timestamp_us: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if timestamp_us in self._expert:
            return self._expert[timestamp_us]
        keys = np.array(list(self._expert.keys()), dtype=np.int64)
        idx  = np.argmin(np.abs(keys - timestamp_us))
        nearest = int(keys[idx])
        if abs(nearest - timestamp_us) < 500_000:
            return self._expert[nearest]
        return None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego       = current_input.history.current_state[0]
        timestamp = ego.time_point.time_us
        traj = self._bc.compute_planner_trajectory(current_input)
        result = self._nearest_expert(timestamp)
        if result is not None:
            visited_feat = self._bc._ego_features(ego)
            _, expert_traj = result
            self.dagger_X.append(visited_feat)
            self.dagger_Y.append(expert_traj)
        return traj

    @property
    def collected_samples(self) -> int:
        return len(self.dagger_X)

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.dagger_X, dtype=np.float32), np.array(self.dagger_Y, dtype=np.float32)


# -- BEV CNN ------------------------------------------------------------------

_HISTORY_STEPS = 10
_FUTURE_STEPS  = 16
_GRID_H = _GRID_W = 64
_M_PER_PIX      = 0.5
_BEV_CHANNELS   = 3
_V_MAX          = 20.0
_BEV_DT         = 0.1


def _rasterize_ego_bev(history_x, history_y, history_yaw, history_vx, history_vy):
    T   = len(history_x)
    img = np.zeros((_BEV_CHANNELS, _GRID_H, _GRID_W), dtype=np.float32)
    cx, cy, c_yaw = history_x[-1], history_y[-1], history_yaw[-1]
    cos_h = np.cos(-c_yaw); sin_h = np.sin(-c_yaw)
    for t in range(T):
        dx_e =  cos_h * (history_x[t] - cx) - sin_h * (history_y[t] - cy)
        dy_e =  sin_h * (history_x[t] - cx) + cos_h * (history_y[t] - cy)
        px   = int(_GRID_W / 2 + dx_e / _M_PER_PIX)
        py   = int(_GRID_H / 2 - dy_e / _M_PER_PIX)
        if not (0 <= px < _GRID_W and 0 <= py < _GRID_H):
            continue
        tw    = (t + 1) / T
        speed = np.sqrt(history_vx[t] ** 2 + history_vy[t] ** 2)
        d_yaw = history_yaw[t] - c_yaw
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
        img[0, py, px] = max(img[0, py, px], tw)
        img[1, py, px] = max(img[1, py, px], min(speed / _V_MAX, 1.0))
        img[2, py, px] = d_yaw / np.pi
    return img


class BEVPolicy(nn.Module):
    """BEV CNN + ego state MLP -> trajectory. REF: loosely follows VectorNet BEV encoder."""

    def __init__(self, bev_ch=_BEV_CHANNELS, state_dim=6, out_dim=_FUTURE_STEPS*3):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.encoder = nn.Sequential(
            conv_block(bev_ch, 32), conv_block(32, 64), conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.state_enc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(128+64, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, out_dim))

    def forward(self, bev, state):
        return self.head(torch.cat([self.encoder(bev), self.state_enc(state)], dim=-1))


class BEVPlanner(AbstractPlanner):
    def __init__(self, ckpt_path: str):
        self._ckpt_path = ckpt_path; self._device = torch.device('cpu')
        self._model: Optional[BEVPolicy] = None
        self._S_mean = self._S_std = self._T_mean = self._T_std = None
        self._history: List[Tuple[float, float, float, float, float]] = []

    def name(self) -> str: return 'BEVPlanner'
    def observation_type(self): return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = BEVPolicy().to(self._device); self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._S_mean = torch.tensor(ckpt['S_mean'], dtype=torch.float32)
        self._S_std  = torch.tensor(ckpt['S_std'],  dtype=torch.float32)
        self._T_mean = ckpt['T_mean']; self._T_std = ckpt['T_std']; self._history = []

    def _build_bev_tensor(self) -> torch.Tensor:
        buf = list(self._history)
        buf = ([buf[0]] * (_HISTORY_STEPS - len(buf)) + buf) if len(buf) < _HISTORY_STEPS else buf[-_HISTORY_STEPS:]
        h = np.array(buf, dtype=np.float32)
        return torch.from_numpy(_rasterize_ego_bev(h[:,0], h[:,1], h[:,2], h[:,3], h[:,4])).unsqueeze(0)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]; dcs = ego.dynamic_car_state
        self._history.append((ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading,
                               dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y))
        bev_t = self._build_bev_tensor().to(self._device)
        yaw = ego.rear_axle.heading
        state_np = np.array([np.sin(yaw), np.cos(yaw), dcs.rear_axle_velocity_2d.x,
                              dcs.rear_axle_velocity_2d.y, dcs.rear_axle_acceleration_2d.x,
                              dcs.rear_axle_acceleration_2d.y], dtype=np.float32)
        state_t = torch.tensor((state_np - self._S_mean.numpy()) / self._S_std.numpy(),
                                dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred_norm = self._model(bev_t, state_t).squeeze(0).numpy()
        pred = (pred_norm * self._T_std + self._T_mean).reshape(_FUTURE_STEPS, 3)
        cx, cy = ego.rear_axle.x, ego.rear_axle.y
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        vx = dcs.rear_axle_velocity_2d.x; vy = dcs.rear_axle_velocity_2d.y
        ax = dcs.rear_axle_acceleration_2d.x; ay = dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(cx + cos_h*dx_e - sin_h*dy_e, cy + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*_BEV_DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)


# -- MILE World Model ---------------------------------------------------------

_MILE_LATENT = 64
_MILE_FUTURE = 16


class MILEPolicy(nn.Module):
    """
    MILE-inspired world model policy.
    REF: Hu et al. (2022) "Model-Based Imitation Learning for Urban Driving." NeurIPS 2022.
    """
    def __init__(self, state_dim=6, latent_dim=_MILE_LATENT, act_dim=3, out_dim=_MILE_FUTURE*3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True),
                                     nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim))
        self.world_model = nn.GRUCell(latent_dim + act_dim, latent_dim)
        self.policy = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
                                    nn.Linear(128, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, out_dim))

    def encode(self, state): return self.encoder(state)
    def step_world(self, z, action): return self.world_model(torch.cat([z, action], dim=-1), z)
    def predict_trajectory(self, z): return self.policy(z)
    def forward(self, state): return self.predict_trajectory(self.encode(state))


class MILEPlanner(AbstractPlanner):
    def __init__(self, ckpt_path: str):
        self._ckpt_path = ckpt_path; self._device = torch.device('cpu')
        self._model: Optional[MILEPolicy] = None
        self._S_mean = self._S_std = self._T_mean = self._T_std = None

    def name(self) -> str: return 'MILEPlanner'
    def observation_type(self): return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = MILEPolicy().to(self._device); self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._S_mean = torch.tensor(ckpt['S_mean'], dtype=torch.float32)
        self._S_std  = torch.tensor(ckpt['S_std'],  dtype=torch.float32)
        self._T_mean = ckpt['T_mean']; self._T_std = ckpt['T_std']

    def _ego_features(self, ego: EgoState) -> np.ndarray:
        h = ego.rear_axle.heading; dcs = ego.dynamic_car_state
        return np.array([np.sin(h), np.cos(h), dcs.rear_axle_velocity_2d.x,
                         dcs.rear_axle_velocity_2d.y, dcs.rear_axle_acceleration_2d.x,
                         dcs.rear_axle_acceleration_2d.y], dtype=np.float32)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        feat = self._ego_features(ego)
        x_t = torch.tensor((feat - self._S_mean.numpy()) / self._S_std.numpy(),
                            dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self._model(x_t).squeeze(0).numpy()
        pred = (pred_norm * self._T_std + self._T_mean).reshape(_MILE_FUTURE, 3)
        cx, cy = ego.rear_axle.x, ego.rear_axle.y; heading = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading); dcs = ego.dynamic_car_state
        vx = dcs.rear_axle_velocity_2d.x; vy = dcs.rear_axle_velocity_2d.y
        ax = dcs.rear_axle_acceleration_2d.x; ay = dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(cx + cos_h*dx_e - sin_h*dy_e, cy + sin_h*dx_e + cos_h*dy_e, heading + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)


# -- Goal-conditioned BC ------------------------------------------------------

class GoalBCPolicy(nn.Module):
    """
    Goal-conditioned BC policy.
    Input : [sin(yaw), cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal]  (8-dim default)
    Output: [(dx, dy, d_yaw) x 16]  (48-dim, ego-frame)
    WHY in_dim param: dual-horizon variant (Phase 3c''''') uses 10 features.
    """
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 48),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalBCPlanner(AbstractPlanner):
    """
    Goal-conditioned BC planner. Wraps GoalBCPolicy with expert T+8 waypoint lookup.
    WHY: oracle upper-bound -- tests if policy can execute given the correct goal.
    """
    def __init__(self, checkpoint_path: str, db_path: str):
        self._ckpt_path = checkpoint_path; self._db_path = db_path
        self._device = torch.device('cpu')
        self._model: Optional[GoalBCPolicy] = None
        self._X_mean = self._X_std = self._Y_mean = self._Y_std = None
        self._expert: dict = {}; self._sorted_ts: List[int] = []

    def name(self) -> str: return 'GoalBCPlanner'
    def observation_type(self): return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = GoalBCPolicy().to(self._device)
        self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']; self._Y_std = ckpt['Y_std']
        self._build_expert_lookup()

    def _build_expert_lookup(self) -> None:
        con = sqlite3.connect(self._db_path)
        rows = con.execute('SELECT timestamp, x, y, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp').fetchall()
        con.close()
        for ts, x, y, qw, qx, qy, qz in rows:
            yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))
            self._expert[int(ts)] = (x, y, yaw)
        self._sorted_ts = sorted(self._expert.keys())

    def _get_expert_at_offset(self, current_ts_us: int, offset_steps: int = 8) -> Tuple[float, float, float]:
        target_ts = current_ts_us + offset_steps * 100_000
        idx = min(int(np.searchsorted(self._sorted_ts, target_ts)), len(self._sorted_ts) - 1)
        return self._expert[self._sorted_ts[idx]]

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g, y_g, yaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx, vy = dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y
        ax, ay = dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y
        ts = int(ego.time_point.time_us); t0 = ego.time_point.time_us
        gx, gy, _ = self._get_expert_at_offset(ts, offset_steps=8)
        cn, sn = np.cos(-yaw), np.sin(-yaw)
        dx_goal = cn*(gx-x_g) - sn*(gy-y_g); dy_goal = sn*(gx-x_g) + cn*(gy-y_g)
        feat = torch.tensor([np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal], dtype=torch.float32)
        feat_norm = (feat - self._X_mean) / self._X_std
        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x_g + cos_h*dx_e - sin_h*dy_e, y_g + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)


# ---------------------------------------------------------------------------
# Phase 3b -- MapBCPlanner
# ---------------------------------------------------------------------------

class MapBCPlanner(AbstractPlanner):
    """
    MapBC: GoalBC weights at inference with road-centerline goal (no expert).
    WHY reuse GoalBCPolicy weights: identical training; only inference goal SOURCE differs.
    """
    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        self._look_ahead_m = look_ahead_m; self._map_api = None
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model = GoalBCPolicy().to('cpu')
        self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']; self._Y_std = ckpt['Y_std']

    def name(self) -> str: return 'MapBCPlanner'

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        self._map_api = initialization.map_api

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g, y_g, yaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx, vy = dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y
        ax, ay = dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us
        dx_goal, dy_goal = self._get_map_goal(x_g, y_g, yaw)
        feat = torch.tensor([np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal], dtype=torch.float32)
        feat_norm = (feat - self._X_mean) / self._X_std
        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x_g + cos_h*dx_e - sin_h*dy_e, y_g + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)

    def _get_map_goal(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D
        FALLBACK = (float(self._look_ahead_m), 0.0)
        if self._map_api is None: return FALLBACK
        try:
            result = self._map_api.get_proximal_map_objects(Point2D(x, y), radius=30.0, layers=[SemanticMapLayer.LANE])
            lanes = result[SemanticMapLayer.LANE]
        except Exception: return FALLBACK
        if not lanes: return FALLBACK
        ego_dir = np.array([np.cos(yaw), np.sin(yaw)])
        best_lane, best_score = None, float('inf')
        for lane in lanes:
            try:
                cl_pts = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
                dists_l = np.linalg.norm(cl_pts - np.array([x, y]), axis=1)
                i0_l = int(np.argmin(dists_l)); d = float(dists_l[i0_l])
                i_next = min(i0_l+1, len(cl_pts)-1)
                tangent = cl_pts[i_next] - cl_pts[i0_l]; t_norm = float(np.linalg.norm(tangent))
                heading_penalty = max(0.0, -float(np.dot(tangent/t_norm, ego_dir))) if t_norm > 1e-6 else 0.0
                score = d + 30.0 * heading_penalty
                if score < best_score: best_score, best_lane = score, lane
            except Exception: continue
        if best_lane is None: return FALLBACK
        cl = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])
        dists = np.linalg.norm(cl - np.array([x, y]), axis=1); i0 = int(np.argmin(dists))
        cum = 0.0; i_goal = min(i0+1, len(cl)-1)
        for i in range(i0, len(cl)-1):
            cum += float(np.linalg.norm(cl[i+1] - cl[i]))
            if cum >= self._look_ahead_m: i_goal = i+1; break
        else: i_goal = len(cl)-1
        gx, gy = cl[i_goal]
        cn, sn = np.cos(-yaw), np.sin(-yaw)
        return (float(cn*(gx-x) - sn*(gy-y)), float(sn*(gx-x) + cn*(gy-y)))


# ---------------------------------------------------------------------------
# Phase 3c -- RouteMapBCPlanner
# ---------------------------------------------------------------------------

class RouteMapBCPlanner(AbstractPlanner):
    """
    RouteMapBC: GoalBC weights at inference with a globally-tracked pre-computed route.
    WHY: fixes MapBC drift-bootstrapping by pre-computing route once at scenario start.
    REF: IDM reference-path tracking concept. Treiber et al. (2000).
    """

    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0,
                 speed_adaptive: bool = False) -> None:
        self._look_ahead_m = look_ahead_m; self._speed_adaptive = speed_adaptive
        self._route_pts: Optional[np.ndarray] = None; self._map_api = None
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model = GoalBCPolicy().to('cpu')
        self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']; self._Y_std = ckpt['Y_std']

    def name(self) -> str: return 'RouteMapBCPlanner'

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        self._map_api = initialization.map_api; self._route_pts = None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g, y_g, yaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx, vy = dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y
        ax, ay = dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us
        if self._route_pts is None:
            self._route_pts = self._build_route(ego, self._map_api)
        if self._speed_adaptive:
            speed = float(np.sqrt(vx**2 + vy**2))
            look_ahead_m = max(0.05, speed * _GOAL_LOOKAHEAD_S)
        else:
            look_ahead_m = self._look_ahead_m
        dx_goal, dy_goal = self._get_route_goal(x_g, y_g, yaw, look_ahead_m)
        feat = torch.tensor([np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal], dtype=torch.float32)
        feat_norm = (feat - self._X_mean) / self._X_std
        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x_g + cos_h*dx_e - sin_h*dy_e, y_g + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)

    def _build_route(self, initial_ego_state, map_api) -> np.ndarray:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D
        x0, y0, yaw0 = initial_ego_state.rear_axle.x, initial_ego_state.rear_axle.y, initial_ego_state.rear_axle.heading
        try:
            result = map_api.get_proximal_map_objects(Point2D(x0, y0), radius=50.0, layers=[SemanticMapLayer.LANE])
            lanes = result[SemanticMapLayer.LANE]
        except Exception:
            return self._straight_route(x0, y0, yaw0)
        if not lanes: return self._straight_route(x0, y0, yaw0)
        ego_dir = np.array([np.cos(yaw0), np.sin(yaw0)])
        best_lane, best_score = None, float('inf')
        for lane in lanes:
            try:
                cl_pts = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
                dists = np.linalg.norm(cl_pts - np.array([x0, y0]), axis=1)
                i0 = int(np.argmin(dists)); d = float(dists[i0])
                i_next = min(i0+1, len(cl_pts)-1)
                tangent = cl_pts[i_next] - cl_pts[i0]; t_norm = np.linalg.norm(tangent)
                cos_a = float(np.dot(tangent/t_norm, ego_dir)) if t_norm > 1e-6 else 0.0
                score = d + 30.0 * max(0.0, -cos_a)
                if score < best_score: best_score, best_lane = score, lane
            except Exception: continue
        if best_lane is None: return self._straight_route(x0, y0, yaw0)
        cl = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])
        dists = np.linalg.norm(cl - np.array([x0, y0]), axis=1)
        i0 = int(np.argmin(dists)); route = cl[i0:]
        total_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1))) if len(route) > 1 else 0.0
        current_lane = best_lane
        for _ in range(8):
            if total_length >= 200.0: break
            try:
                successors = current_lane.outgoing_edges
                if not successors: break
                last_dir = route[-1] - route[-2] if len(route) > 1 else ego_dir
                last_dir_n = last_dir / (np.linalg.norm(last_dir) + 1e-8)
                best_succ = self._select_successor(successors, last_dir_n)
                if best_succ is None: break
                succ_pts = np.array([(s.x, s.y) for s in best_succ.baseline_path.discrete_path])
                route = np.vstack([route, succ_pts])
                total_length += float(np.sum(np.linalg.norm(np.diff(succ_pts, axis=0), axis=1)))
                current_lane = best_succ
            except Exception: break
        return route.astype(np.float64)

    def _select_successor(self, successors, last_dir_n):
        best_succ, best_cos = None, -1.0
        for succ in successors:
            try:
                succ_pts = np.array([(s.x, s.y) for s in succ.baseline_path.discrete_path])
                if len(succ_pts) < 2: continue
                tangent = succ_pts[1] - succ_pts[0]
                cos_a = float(np.dot(tangent / (np.linalg.norm(tangent) + 1e-8), last_dir_n))
                if cos_a > best_cos: best_cos, best_succ = cos_a, succ
            except Exception: continue
        return best_succ

    def _straight_route(self, x: float, y: float, yaw: float, length_m: float = 200.0, step_m: float = 2.0) -> np.ndarray:
        n = int(length_m / step_m)
        return np.array([[x + step_m*i*np.cos(yaw), y + step_m*i*np.sin(yaw)] for i in range(n)], dtype=np.float64)

    def _get_route_goal(self, x: float, y: float, yaw: float, look_ahead_m: Optional[float] = None) -> Tuple[float, float]:
        if look_ahead_m is None: look_ahead_m = self._look_ahead_m
        if self._route_pts is None or len(self._route_pts) == 0:
            return (float(look_ahead_m), 0.0)
        dists = np.linalg.norm(self._route_pts - np.array([x, y]), axis=1)
        i0 = int(np.argmin(dists))
        cum = 0.0; i_goal = min(i0+1, len(self._route_pts)-1)
        for i in range(i0, len(self._route_pts)-1):
            cum += float(np.linalg.norm(self._route_pts[i+1] - self._route_pts[i]))
            if cum >= look_ahead_m: i_goal = i+1; break
        else: i_goal = len(self._route_pts)-1
        gx, gy = self._route_pts[i_goal]
        cn, sn = np.cos(-yaw), np.sin(-yaw)
        return (float(cn*(gx-x) - sn*(gy-y)), float(sn*(gx-x) + cn*(gy-y)))


class TrainedRouteBCPlanner(RouteMapBCPlanner):
    """Phase 3c' -- retrained with arc-length-8m route goals at BOTH train and inference time."""
    def name(self) -> str: return 'TrainedRouteBCPlanner'


# T+8 time window in seconds (8 * 100ms sim steps)
_GOAL_LOOKAHEAD_S: float = 8 * 0.1   # = 0.8 s


class SpeedAdaptiveRouteMapBCPlanner(RouteMapBCPlanner):
    """Phase 3c'' -- RouteMapBC with speed-adaptive look-ahead: max(0.05, speed * 0.8s)."""
    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        super().__init__(checkpoint_path, look_ahead_m, speed_adaptive=True)
    def name(self) -> str: return 'SpeedAdaptiveRouteMapBCPlanner'


class RoadblockRouteMapBCPlanner(SpeedAdaptiveRouteMapBCPlanner):
    """
    Phase 3c''' -- RouteMapBC that follows route_roadblock_ids at intersections.
    Inherits speed_adaptive=True. ONLY change: junction-branch selection uses intended route.
    WHY: the 4 catastrophic tail failures of SpeedAdaptive were due to route taking the wrong branch.
    Liskov: when route_roadblock_ids is empty, behaviour is identical to parent.
    """
    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        super().__init__(checkpoint_path, look_ahead_m)
        self._route_roadblock_ids: frozenset = frozenset()

    def name(self) -> str: return 'RoadblockRouteMapBCPlanner'

    def initialize(self, initialization) -> None:
        super().initialize(initialization)
        ids = getattr(initialization, 'route_roadblock_ids', None) or []
        self._route_roadblock_ids = frozenset(str(i) for i in ids)

    def _select_successor(self, successors, last_dir_n):
        if self._route_roadblock_ids:
            on_route = [s for s in successors if self._on_route(s)]
            if on_route: return super()._select_successor(on_route, last_dir_n)
        return super()._select_successor(successors, last_dir_n)

    def _on_route(self, lane) -> bool:
        try:
            if str(lane.get_roadblock_id()) in self._route_roadblock_ids: return True
        except Exception: pass
        try:
            if str(lane.id) in self._route_roadblock_ids: return True
        except Exception: pass
        return False


# Far-preview look-ahead: at 20m ~80% of turning windows show goal deflection >15deg.
# At 3.5m near-horizon only ~6% do -- turns are invisible. REF: phase3_roadmap.md.
_FAR_LOOKAHEAD_M: float = 20.0


class DualHorizonRouteMapBCPlanner(RouteMapBCPlanner):
    """
    Phase 3c''''' -- dual-horizon goal: near (speed*0.8s) + far (20m fixed).
    WHY: near goal alone misses turns; far goal sees them. Same conditioning as Phase 3d.
    MODE SWAP FINDING: mean 27.55m -- MLP averages junction modes despite having info.
    This finding JUSTIFIES Phase 3d (DiffusionPolicyPlanner).
    """
    def __init__(self, checkpoint_path: str, far_m: float = _FAR_LOOKAHEAD_M) -> None:
        self._look_ahead_m = 8.0; self._speed_adaptive = True; self._far_m = far_m
        self._route_pts: Optional[np.ndarray] = None; self._map_api = None
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model = GoalBCPolicy(in_dim=10).to('cpu')
        self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']; self._Y_std = ckpt['Y_std']

    def name(self) -> str: return 'DualHorizonRouteMapBCPlanner'

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g, y_g, yaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx, vy = dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y
        ax, ay = dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us
        if self._route_pts is None:
            self._route_pts = self._build_route(ego, self._map_api)
        speed = float(np.sqrt(vx**2 + vy**2))
        near_la = max(0.05, speed * _GOAL_LOOKAHEAD_S)
        dxn, dyn = self._get_route_goal(x_g, y_g, yaw, near_la)
        dxf, dyf = self._get_route_goal(x_g, y_g, yaw, self._far_m)
        feat = torch.tensor([np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dxn, dyn, dxf, dyf], dtype=torch.float32)
        feat_norm = (feat - self._X_mean) / self._X_std
        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x_g + cos_h*dx_e - sin_h*dy_e, y_g + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)


# -- Phase 3d -- DiffusionPolicyPlanner ---------------------------------------

class DiffusionPolicyPlanner(AbstractPlanner):
    """
    Phase 3d -- Goal-Conditioned Diffusion Policy planner.

    Resolves the mode-swap failure of DualHorizonRouteMapBCPlanner by replacing
    the deterministic MLP head with a generative DDPM denoiser.

    WHY this is necessary (DualHorizon post-mortem):
      DualHorizon mean L2 = 27.55m -- WORSE than SpeedAdaptive (18.19m).
      The far-goal encodes the correct turn direction, so the information IS present.
      The mode-swap signature: the MLP averages "turn left" and "turn right" into a
      straight-line compromise. DDPM can sample ONE trajectory from the multi-modal
      distribution, naturally committing to one mode. K=8 candidates + goal-proximity
      scoring selects the candidate that commits to the correct mode.

    Architecture:
      Denoiser: GoalConditionedDenoiser (4-layer MLP, ~175K params)
      Training: DDPM, T=100 steps, cosine noise schedule, epsilon-prediction
      Inference: DDIM (10 steps, eta=0), K=8 candidates, scored by near-goal proximity
      Conditioning: same 10-dim dual-horizon goal as DualHorizonRouteMapBCPlanner

    Liskov safety:
      On any denoiser failure, falls back to straight-ahead at current speed.
      Never worse than IDM straight-line baseline.

    REF:
      Ho et al. (2020) arXiv:2006.11239 -- DDPM
      Song et al. (2020) arXiv:2010.02502 -- DDIM (eq. 12, eta=0)
      Nichol & Dhariwal (2021) arXiv:2102.09672 -- cosine schedule
      Chi et al. (2023) arXiv:2303.04137 -- Diffusion Policy
    """

    def __init__(
        self,
        checkpoint_path: str,
        far_m:      float = _FAR_LOOKAHEAD_M,
        k_samples:  int   = 8,
        ddim_steps: int   = 10,
    ):
        self._ckpt_path  = checkpoint_path
        self._far_m      = far_m
        self._k_samples  = k_samples
        self._ddim_steps = ddim_steps
        # WHY CPU for inference: nuPlan simulation calls the planner at ~10 Hz
        # synchronously. MPS has per-call warm-up overhead that makes it slower
        # than CPU for a 175K-param model called 10 times/second.
        self._device     = torch.device('cpu')
        self._model      = None
        self._schedule: Optional[dict] = None
        self._X_mean = self._X_std = None
        self._Y_mean = self._Y_std = None
        self._T: int = 100
        self._route_pts: Optional[np.ndarray] = None
        self._map_api = None

    def name(self) -> str:
        return 'DiffusionPolicyPlanner'

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        """Load DDPM checkpoint, restore noise schedule, store map_api."""
        import sys as _sys
        _sys.path.insert(0, '/Users/parvpatodia/Desktop/diffusion-policy-zoo/nuplan')
        from train_diffusion_policy import (
            GoalConditionedDenoiser,
            build_cosine_schedule,
            T_DIFFUSION,
        )
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = GoalConditionedDenoiser().to(self._device)
        self._model.load_state_dict(ckpt['model']); self._model.eval()
        self._X_mean = ckpt['X_mean']   # (10,) float32 numpy
        self._X_std  = ckpt['X_std']    # (10,) float32 numpy
        self._Y_mean = ckpt['Y_mean']   # (48,) float32 numpy
        self._Y_std  = ckpt['Y_std']    # (48,) float32 numpy
        self._T      = int(ckpt.get('T', T_DIFFUSION))
        # WHY prefer checkpoint schedule: ensures exact match with training run.
        if 'schedule' in ckpt:
            self._schedule = {k: v.to(self._device) for k, v in ckpt['schedule'].items()}
        else:
            sched_cpu = build_cosine_schedule(self._T)
            self._schedule = {k: v.to(self._device) for k, v in sched_cpu.items()}
        self._map_api   = initialization.map_api
        self._route_pts = None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g, y_g, yaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx, vy = dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y
        ax, ay = dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us

        if self._route_pts is None:
            self._route_pts = self._build_route_dp(ego, self._map_api)

        speed   = float(np.sqrt(vx**2 + vy**2))
        near_la = max(0.05, speed * _GOAL_LOOKAHEAD_S)
        dxn, dyn = self._get_route_goal_dp(x_g, y_g, yaw, near_la)
        dxf, dyf = self._get_route_goal_dp(x_g, y_g, yaw, self._far_m)

        feat_raw  = np.array([np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dxn, dyn, dxf, dyf], dtype=np.float32)
        feat_norm = (feat_raw - self._X_mean) / self._X_std

        try:
            pred = self._sample_best_trajectory(feat_norm, dxn, dyn)
        except Exception:
            # Liskov safety: straight-ahead at current speed on any denoiser failure
            pred = np.zeros((FUTURE_STEPS, 3), dtype=np.float32)
            for j in range(FUTURE_STEPS):
                pred[j, 0] = speed * DT * (j + 1)

        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x_g + cos_h*dx_e - sin_h*dy_e, y_g + sin_h*dx_e + cos_h*dy_e, yaw + d_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j+1)*DT*1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)

    def _sample_best_trajectory(
        self,
        feat_norm: np.ndarray,   # (10,) normalized conditioning
        dx_near:   float,        # near goal x in ego frame (unnormalized meters)
        dy_near:   float,        # near goal y in ego frame (unnormalized meters)
    ) -> np.ndarray:
        """
        Sample K DDIM trajectories; return best (unnormalized (16,3)) scored by
        step-8 proximity to near-goal.

        WHY step 8 for scoring:
          Step 8 (~0.8s) matches the near-goal look-ahead horizon. Junction branch
          decisions manifest here: left-turn has positive dy, right-turn negative dy.

        WHY K=8 independent samples:
          P(missing correct mode) = (1 - p_mode)^K. At p_mode=0.5 and K=8: 0.4%.
          Diversity comes from K different x_T ~ N(0,I) seeds inside ddim_sample.

        REF: Song et al. (2020) eq. 12, eta=0.
        """
        from train_diffusion_policy import ddim_sample

        c_batch = torch.from_numpy(
            np.tile(feat_norm, (self._k_samples, 1))
        ).float().to(self._device)  # (K, 10)

        trajs_norm = ddim_sample(
            self._model, c_batch, self._schedule,
            T=self._T, n_steps=self._ddim_steps, device=self._device,
        )  # (K, 48), normalized

        # Denormalize
        trajs_unnorm = (trajs_norm.numpy() * self._Y_std + self._Y_mean)  # (K, 48)
        trajs_steps  = trajs_unnorm.reshape(self._k_samples, FUTURE_STEPS, 3)  # (K, 16, 3)

        # Score: distance from step-8 (dx, dy) to near-goal
        step8_pos = trajs_steps[:, 7, :2]   # (K, 2)
        goal_vec  = np.array([dx_near, dy_near], dtype=np.float32)
        distances = np.sqrt(((step8_pos - goal_vec) ** 2).sum(axis=1))  # (K,)
        return trajs_steps[int(np.argmin(distances))]   # (16, 3)

    # -- Route helpers (mirrors RouteMapBCPlanner, _dp suffix = explicit copy) -----
    # WHY copy not inherit: DiffusionPolicyPlanner overrides compute_planner_trajectory
    # entirely. Inheriting from RouteMapBCPlanner would misleadingly imply the parent's
    # method is called. Direct inheritance from AbstractPlanner with copied helpers is cleaner.

    def _build_route_dp(self, initial_ego_state, map_api) -> np.ndarray:
        """Build 200m route. Mirrors RouteMapBCPlanner._build_route."""
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D
        x0, y0, yaw0 = (initial_ego_state.rear_axle.x,
                        initial_ego_state.rear_axle.y,
                        initial_ego_state.rear_axle.heading)
        fallback = self._straight_route_pts_dp(x0, y0, yaw0)
        try:
            result = map_api.get_proximal_map_objects(Point2D(x0, y0), radius=50.0, layers=[SemanticMapLayer.LANE])
            lanes = result[SemanticMapLayer.LANE]
        except Exception:
            return fallback
        if not lanes: return fallback
        ego_dir = np.array([np.cos(yaw0), np.sin(yaw0)])
        best_lane, best_score = None, float('inf')
        for lane in lanes:
            try:
                cl = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
                dists = np.linalg.norm(cl - np.array([x0, y0]), axis=1)
                i0 = int(np.argmin(dists)); d = float(dists[i0])
                tang = cl[min(i0+1, len(cl)-1)] - cl[i0]; tn = np.linalg.norm(tang)
                cos_a = float(np.dot(tang/tn, ego_dir)) if tn > 1e-6 else 0.0
                s = d + 30.0 * max(0.0, -cos_a)
                if s < best_score: best_score, best_lane = s, lane
            except Exception: continue
        if best_lane is None: return fallback
        cl = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])
        dists = np.linalg.norm(cl - np.array([x0, y0]), axis=1)
        route = cl[int(np.argmin(dists)):]
        total = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1))) if len(route) > 1 else 0.0
        cur = best_lane
        for _ in range(8):
            if total >= 200.0: break
            try:
                succs = cur.outgoing_edges
                if not succs: break
                ld = route[-1] - route[-2] if len(route) > 1 else ego_dir
                ldn = ld / (np.linalg.norm(ld) + 1e-8)
                best_s, best_c = None, -1.0
                for s in succs:
                    try:
                        sp = np.array([(p.x, p.y) for p in s.baseline_path.discrete_path])
                        if len(sp) < 2: continue
                        t = sp[1] - sp[0]
                        c = float(np.dot(t / (np.linalg.norm(t) + 1e-8), ldn))
                        if c > best_c: best_c, best_s = c, s
                    except Exception: continue
                if best_s is None: break
                sp = np.array([(p.x, p.y) for p in best_s.baseline_path.discrete_path])
                route = np.vstack([route, sp])
                total += float(np.sum(np.linalg.norm(np.diff(sp, axis=0), axis=1)))
                cur = best_s
            except Exception: break
        return route.astype(np.float64)

    def _get_route_goal_dp(self, x: float, y: float, yaw: float, look_ahead_m: float) -> Tuple[float, float]:
        """Return route point look_ahead_m ahead of ego in ego frame."""
        if self._route_pts is None or len(self._route_pts) == 0:
            return (float(look_ahead_m), 0.0)
        dists = np.linalg.norm(self._route_pts - np.array([x, y]), axis=1)
        i0 = int(np.argmin(dists))
        cum = 0.0; i_goal = min(i0+1, len(self._route_pts)-1)
        for i in range(i0, len(self._route_pts)-1):
            cum += float(np.linalg.norm(self._route_pts[i+1] - self._route_pts[i]))
            if cum >= look_ahead_m: i_goal = i+1; break
        else: i_goal = len(self._route_pts)-1
        gx, gy = self._route_pts[i_goal]
        cn, sn = np.cos(-yaw), np.sin(-yaw)
        return (float(cn*(gx-x) - sn*(gy-y)), float(sn*(gx-x) + cn*(gy-y)))

    def _straight_route_pts_dp(self, x: float, y: float, yaw: float,
                                length_m: float = 200.0, step_m: float = 2.0) -> np.ndarray:
        """Fallback straight-line route."""
        n = int(length_m / step_m)
        return np.array([[x + step_m*i*np.cos(yaw), y + step_m*i*np.sin(yaw)]
                         for i in range(n)], dtype=np.float64)
