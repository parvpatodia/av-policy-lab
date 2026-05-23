"""
av-policy-lab planner module.
All planners are proper importable classes compatible with nuPlan's AbstractPlanner.
Import this module before running SimulationLog.load_data() so pickle finds the classes.

Classes
-------
BCPolicy       : MLP architecture (6 -> 256 -> 256 -> 256 -> 48)
BCPlanner      : Behavior-cloning AbstractPlanner wrapper
IDMPlanner     : Intelligent Driver Model AbstractPlanner wrapper
DAggerPlanner  : DAgger data-collection wrapper around BCPlanner
BEVPolicy      : CNN architecture (3×64×64 + 6) -> 48
BEVPlanner     : BEV CNN AbstractPlanner wrapper (ego-history rasterization)
MILEPolicy     : World model (encoder + GRU transition + policy), joint imitation+consistency loss
MILEPlanner    : MILE AbstractPlanner wrapper (inference: state → encoder → latent → policy)
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
DT           = 0.1   # nuPlan 10 Hz


# ── Model architecture ────────────────────────────────────────────────────────

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


# ── BCPlanner ─────────────────────────────────────────────────────────────────

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


# ── IDMPlanner ────────────────────────────────────────────────────────────────

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


# ── DAggerPlanner ─────────────────────────────────────────────────────────────

class DAggerPlanner(AbstractPlanner):
    """
    DAgger (Dataset Aggregation) data-collection wrapper.
    REF: Ross et al. (2011) "A Reduction of Imitation Learning and Structured
         Prediction to No-Regret Online Learning." AISTATS 2011.

    At each planning step:
      1. Runs the current BC policy to produce the trajectory (controls the ego).
      2. Records the visited ego-state features.
      3. Queries the original DB for the expert's future trajectory at the same
         timestamp — the label the policy SHOULD have produced.

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

        # Populated at initialize() — maps timestamp_us -> (feat_6, traj_48)
        self._expert: dict = {}

        # Accumulated on-policy data (filled during simulation)
        self.dagger_X: List[np.ndarray] = []   # (6,) ego features at visited states
        self.dagger_Y: List[np.ndarray] = []   # (48,) expert trajectory at same timestamp

    def name(self) -> str:
        return f'DAggerPlanner_iter{getattr(self, "_iter", 0)}'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._bc.initialize(initialization)
        self._build_expert_lookup()

    def _build_expert_lookup(self) -> None:
        """
        Pre-load the full ego-pose table from the DB into a timestamp-keyed dict.
        At each step we do an O(1) lookup rather than a DB query.

        WHY: nuPlan DB has one ego_pose row per 100ms tick. We build a map
             timestamp_us -> (input_features_6, target_trajectory_48) covering
             the whole log so any drifted simulation state can be labeled.
        """
        con  = sqlite3.connect(self._db_path)
        rows = con.execute(
            "SELECT timestamp, x, y, qw, qx, qy, qz, vx, vy, acceleration_x, acceleration_y "
            "FROM ego_pose ORDER BY timestamp"
        ).fetchall()
        con.close()

        arr  = np.array(rows, dtype=np.float64)
        ts   = arr[:, 0].astype(np.int64)
        x_g  = arr[:, 1]
        y_g  = arr[:, 2]
        qw, qx_, qy_, qz_ = arr[:,3], arr[:,4], arr[:,5], arr[:,6]
        vx   = arr[:, 7]
        vy   = arr[:, 8]
        ax_  = arr[:, 9]
        ay_  = arr[:,10]
        yaw  = np.arctan2(2*(qw*qz_ + qx_*qy_), 1 - 2*(qy_**2 + qz_**2))

        N = len(arr)
        for i in range(N - FUTURE_STEPS):
            # Input features at timestep i
            feat = np.array([
                np.sin(yaw[i]), np.cos(yaw[i]),
                vx[i], vy[i], ax_[i], ay_[i],
            ], dtype=np.float32)

            # Expert trajectory: 16 future steps relative to current pose
            cx, cy, cyaw = x_g[i], y_g[i], yaw[i]
            cos_h  = np.cos(-cyaw)
            sin_h  = np.sin(-cyaw)
            tgt    = np.zeros(FUTURE_STEPS * 3, dtype=np.float32)
            for j in range(FUTURE_STEPS):
                fi = i + j + 1
                dx_w = x_g[fi] - cx
                dy_w = y_g[fi] - cy
                dx_e = cos_h * dx_w - sin_h * dy_w
                dy_e = sin_h * dx_w + cos_h * dy_w
                dyaw = yaw[fi] - cyaw
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                tgt[j * 3]     = dx_e
                tgt[j * 3 + 1] = dy_e
                tgt[j * 3 + 2] = dyaw

            self._expert[int(ts[i])] = (feat, tgt)

    def _nearest_expert(self, timestamp_us: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (feat, traj) for the closest DB timestamp to timestamp_us.
        Uses linear scan with early exit — DB timestamps are sorted and
        the simulation runs at the same 100ms rate, so exact matches are common.
        """
        if timestamp_us in self._expert:
            return self._expert[timestamp_us]

        # Nearest-neighbour fallback (simulation may drift in wall-clock time)
        keys = np.array(list(self._expert.keys()), dtype=np.int64)
        idx  = np.argmin(np.abs(keys - timestamp_us))
        nearest = int(keys[idx])
        # Only accept if within 500ms (5 ticks) to avoid bad labels
        if abs(nearest - timestamp_us) < 500_000:
            return self._expert[nearest]
        return None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego       = current_input.history.current_state[0]
        timestamp = ego.time_point.time_us

        # 1. BC policy controls the ego (what the simulation follows)
        traj = self._bc.compute_planner_trajectory(current_input)

        # 2. Record visited state + expert label for DAgger aggregation
        result = self._nearest_expert(timestamp)
        if result is not None:
            # WHY: we record the BC features (at the VISITED state, not log state)
            #      but use the expert trajectory as label — this is the DAgger correction
            visited_feat = self._bc._ego_features(ego)
            _, expert_traj = result
            self.dagger_X.append(visited_feat)
            self.dagger_Y.append(expert_traj)

        return traj

    @property
    def collected_samples(self) -> int:
        return len(self.dagger_X)

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return collected (X, Y) arrays after simulation finishes."""
        return np.array(self.dagger_X, dtype=np.float32), \
               np.array(self.dagger_Y, dtype=np.float32)


# ── BEV CNN ───────────────────────────────────────────────────────────────────

# Grid constants (must match bev_cnn.ipynb)
_HISTORY_STEPS = 10
_FUTURE_STEPS  = 16
_GRID_H = _GRID_W = 64
_M_PER_PIX      = 0.5
_BEV_CHANNELS   = 3
_V_MAX          = 20.0
_BEV_DT         = 0.1   # nuPlan 10 Hz


def _rasterize_ego_bev(
    history_x:   np.ndarray,
    history_y:   np.ndarray,
    history_yaw: np.ndarray,
    history_vx:  np.ndarray,
    history_vy:  np.ndarray,
) -> np.ndarray:
    """
    (T,) global-frame arrays → (3, 64, 64) float32 BEV image.
    Ch 0: temporal occupancy (0.1…1.0, oldest→newest)
    Ch 1: speed magnitude    (0…1, normalised by V_MAX)
    Ch 2: heading delta      (−1…1, normalised by π)
    WHY: see bev_cnn.ipynb Cell 2 for full rationale.
    """
    T   = len(history_x)
    img = np.zeros((_BEV_CHANNELS, _GRID_H, _GRID_W), dtype=np.float32)
    cx, cy, c_yaw = history_x[-1], history_y[-1], history_yaw[-1]
    cos_h = np.cos(-c_yaw)
    sin_h = np.sin(-c_yaw)
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
    """
    BEV CNN + ego state MLP → trajectory.
    Input:  bev   (B, 3, 64, 64)
            state (B, 6)  -- normalised [sin(yaw), cos(yaw), vx, vy, ax, ay]
    Output: traj  (B, 48) -- normalised (dx, dy, d_yaw) × 16

    Architecture:
        CNN: 3 conv blocks (32→64→128ch), AdaptiveAvgPool(1) → 128-dim
        MLP: 6 → 64 → 64
        Head: (128+64) → 256 → 48
    Parameters: ~370K  (BC MLP: ~260K)
    REF: loosely follows VectorNet BEV encoder (Gao et al. 2020).
    """

    def __init__(
        self,
        bev_ch:    int = _BEV_CHANNELS,
        state_dim: int = 6,
        out_dim:   int = _FUTURE_STEPS * 3,
    ):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.encoder = nn.Sequential(
            conv_block(bev_ch, 32),
            conv_block(32,     64),
            conv_block(64,     128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64),        nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, bev: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(bev), self.state_enc(state)], dim=-1))


class BEVPlanner(AbstractPlanner):
    """
    AbstractPlanner wrapper for BEVPolicy.

    Maintains a rolling HISTORY_STEPS buffer of ego states and rasterizes a
    BEV image at each planning step before calling the CNN policy.

    Padding: if fewer than HISTORY_STEPS states are in the buffer (start of
    scenario), the oldest available state is repeated (zero-order hold).
    WHY: zero-order hold is a neutral, static assumption. Padding with zeros
    would introduce artificial velocity / heading discontinuities.
    """

    def __init__(self, ckpt_path: str):
        self._ckpt_path = ckpt_path
        self._device    = torch.device('cpu')
        self._model: Optional[BEVPolicy] = None
        self._S_mean = self._S_std = None
        self._T_mean = self._T_std = None
        self._history: List[Tuple[float, float, float, float, float]] = []

    def name(self) -> str:
        return 'BEVPlanner'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = BEVPolicy().to(self._device)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._S_mean = torch.tensor(ckpt['S_mean'], dtype=torch.float32)
        self._S_std  = torch.tensor(ckpt['S_std'],  dtype=torch.float32)
        self._T_mean = ckpt['T_mean']
        self._T_std  = ckpt['T_std']
        self._history = []

    def _build_bev_tensor(self) -> torch.Tensor:
        buf = list(self._history)
        if len(buf) < _HISTORY_STEPS:
            buf = [buf[0]] * (_HISTORY_STEPS - len(buf)) + buf
        else:
            buf = buf[-_HISTORY_STEPS:]
        h   = np.array(buf, dtype=np.float32)
        bev = _rasterize_ego_bev(h[:,0], h[:,1], h[:,2], h[:,3], h[:,4])
        return torch.from_numpy(bev).unsqueeze(0)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        dcs = ego.dynamic_car_state
        self._history.append((
            ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading,
            dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y,
        ))

        bev_t   = self._build_bev_tensor().to(self._device)
        yaw     = ego.rear_axle.heading
        state_np = np.array([
            np.sin(yaw), np.cos(yaw),
            dcs.rear_axle_velocity_2d.x,
            dcs.rear_axle_velocity_2d.y,
            dcs.rear_axle_acceleration_2d.x,
            dcs.rear_axle_acceleration_2d.y,
        ], dtype=np.float32)
        state_t = torch.tensor(
            (state_np - self._S_mean.numpy()) / self._S_std.numpy(),
            dtype=torch.float32,
        ).unsqueeze(0).to(self._device)

        with torch.no_grad():
            pred_norm = self._model(bev_t, state_t).squeeze(0).numpy()
        pred = (pred_norm * self._T_std + self._T_mean).reshape(_FUTURE_STEPS, 3)

        cx, cy    = ego.rear_axle.x, ego.rear_axle.y
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        vx = dcs.rear_axle_velocity_2d.x
        vy = dcs.rear_axle_velocity_2d.y
        ax = dcs.rear_axle_acceleration_2d.x
        ay = dcs.rear_axle_acceleration_2d.y
        t0 = ego.time_point.time_us

        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            wx    = cx + cos_h * dx_e - sin_h * dy_e
            wy    = cy + sin_h * dx_e + cos_h * dy_e
            w_yaw = yaw + d_yaw
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(wx, wy, w_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * _BEV_DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)


# ── MILE World Model ──────────────────────────────────────────────────────────

_MILE_LATENT = 64
_MILE_FUTURE = 16


class MILEPolicy(nn.Module):
    """
    MILE-inspired world model policy.
    REF: Hu et al. (2022) "Model-Based Imitation Learning for Urban Driving."
         NeurIPS 2022. arXiv:2209.14430

    Components trained jointly:
      Encoder     : 6 → 128 → 64   (state → latent z, with LayerNorm)
      World model : GRUCell(64+3, 64)  (z_t + action_t → z_{t+1})
      Policy      : 64 → 128 → 256 → 48  (z_t → trajectory)

    Training objective:
      L = L_imitation + BETA * L_consistency
      L_imitation  = MSE(policy(z_t), traj_gt)
      L_consistency = mean_j MSE(world_model(z_j, a_j), encoder(state_{t+j+1}))
      — teacher forcing: a_j = GT action at step j (prevents circular dependency)

    Inference path:
      state → encode → z → policy → trajectory
      (world model not used at inference)

    Parameters: ~73K  (BC MLP: ~260K)
    """

    def __init__(
        self,
        state_dim:  int = 6,
        latent_dim: int = _MILE_LATENT,
        act_dim:    int = 3,
        out_dim:    int = _MILE_FUTURE * 3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim,  128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            # WHY LayerNorm: prevents consistency loss from collapsing all z → 0
        )
        # WHY GRUCell (not full GRU module): manual rollout needed to accumulate
        # per-step consistency loss with teacher-forcing actions.
        self.world_model = nn.GRUCell(latent_dim + act_dim, latent_dim)
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256),        nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def step_world(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.world_model(torch.cat([z, action], dim=-1), z)

    def predict_trajectory(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy(z)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Inference: state (B,6) → trajectory (B,48)."""
        return self.predict_trajectory(self.encode(state))


class MILEPlanner(AbstractPlanner):
    """
    AbstractPlanner wrapper for MILEPolicy.
    Inference path: state → encoder → latent → policy → trajectory.
    The world model (GRUCell) is used only during training — not at inference.

    This is structurally identical to BCPlanner; the difference is the
    internal architecture (encoder + latent + policy vs flat MLP) and the
    normalization stats (S_mean/S_std vs X_mean/X_std naming).
    """

    def __init__(self, ckpt_path: str):
        self._ckpt_path = ckpt_path
        self._device    = torch.device('cpu')
        self._model: Optional[MILEPolicy] = None
        self._S_mean = self._S_std = None
        self._T_mean = self._T_std = None

    def name(self) -> str:
        return 'MILEPlanner'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = MILEPolicy().to(self._device)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._S_mean = torch.tensor(ckpt['S_mean'], dtype=torch.float32)
        self._S_std  = torch.tensor(ckpt['S_std'],  dtype=torch.float32)
        self._T_mean = ckpt['T_mean']
        self._T_std  = ckpt['T_std']

    def _ego_features(self, ego: EgoState) -> np.ndarray:
        h   = ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        return np.array([
            np.sin(h), np.cos(h),
            dcs.rear_axle_velocity_2d.x,
            dcs.rear_axle_velocity_2d.y,
            dcs.rear_axle_acceleration_2d.x,
            dcs.rear_axle_acceleration_2d.y,
        ], dtype=np.float32)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego  = current_input.history.current_state[0]
        feat = self._ego_features(ego)
        x_t  = torch.tensor(
            (feat - self._S_mean.numpy()) / self._S_std.numpy(),
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self._model(x_t).squeeze(0).numpy()
        pred = (pred_norm * self._T_std + self._T_mean).reshape(_MILE_FUTURE, 3)

        cx, cy    = ego.rear_axle.x, ego.rear_axle.y
        heading   = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dcs = ego.dynamic_car_state
        vx  = dcs.rear_axle_velocity_2d.x
        vy  = dcs.rear_axle_velocity_2d.y
        ax  = dcs.rear_axle_acceleration_2d.x
        ay  = dcs.rear_axle_acceleration_2d.y
        t0  = ego.time_point.time_us

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
