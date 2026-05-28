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
BEVPolicy         : CNN architecture (3×64×64 + 6) -> 48
BEVPlanner        : BEV CNN AbstractPlanner wrapper (ego-history rasterization)
MILEPolicy        : World model (encoder + GRU transition + policy), joint imitation+consistency loss
MILEPlanner       : MILE AbstractPlanner wrapper (inference: state → encoder → latent → policy)
GoalBCPolicy      : Goal-conditioned MLP (8 -> 256 -> 256 -> 256 -> 48, state + T+8 expert waypoint)
GoalBCPlanner     : GoalBC wrapper — expert DB lookup for goal at inference (Phase 3a, oracle)
MapBCPlanner      : GoalBC weights + road centerline goal at inference — no expert required (Phase 3b)
RouteMapBCPlanner : GoalBC weights + pre-computed global route goal, fixed 8m look-ahead (Phase 3c)
TrainedRouteBCPlanner          : RouteMapBC loading route-goal-trained weights (Phase 3c')
SpeedAdaptiveRouteMapBCPlanner : RouteMapBC with look-ahead = speed × 0.8s (Phase 3c'', scale fix)
RoadblockRouteMapBCPlanner     : SpeedAdaptive + route_roadblock_ids junction selection (Phase 3c''')
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
#       them 10× further apart. perfect_tracking_controller executes by spatial
#       position, so this mismatch does not affect L2 in practice. See verify_pipeline.py.


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


# ── Goal-conditioned BC ───────────────────────────────────────────────────────

class GoalBCPolicy(nn.Module):
    """
    Goal-conditioned BC policy.
    Input : [sin(yaw), cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal]  (8-dim)
    Output: [(dx, dy, d_yaw) x 16]                                  (48-dim, ego-frame)

    Ablation over BCPolicy: everything identical except 2 additional goal features.
    WHY goal = T+8 waypoint: 0.8s horizon gives the policy enough road context
    to anticipate upcoming curves without leaking far-future expert trajectory.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 48),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalBCPlanner(AbstractPlanner):
    """
    Goal-conditioned BC planner.
    Wraps GoalBCPolicy with expert T+8 waypoint lookup from the nuPlan DB.

    WHY expert goal at inference: this is an honest upper-bound eval.
    We test "given correct goal, can the policy execute road-following?"
    If closed-loop L2 drops significantly vs BCPlanner, the root cause of
    Phase 2 plateau is confirmed: the policy can execute when given directional
    guidance. If L2 stays flat, the bottleneck is control precision.
    """

    def __init__(self, checkpoint_path: str, db_path: str):
        # WHY db_path: we look up the expert T+8 waypoint from the DB at each planning step.
        self._ckpt_path = checkpoint_path
        self._db_path   = db_path
        self._device    = torch.device('cpu')   # WHY: CPU for sim stability on macOS MPS
        self._model: Optional[GoalBCPolicy] = None
        self._X_mean = self._X_std = None
        self._Y_mean = self._Y_std = None
        self._expert: dict = {}       # timestamp_us -> (x, y, yaw)
        self._sorted_ts: List[int] = []

    def name(self) -> str:
        return 'GoalBCPlanner'

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        # Load model weights
        ckpt = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
        self._model = GoalBCPolicy().to(self._device)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']
        self._Y_std  = ckpt['Y_std']

        # Build expert timestamp lookup from DB
        # WHY: pre-load at initialize() not __init__ to avoid DB connection before simulation
        self._build_expert_lookup()

    def _build_expert_lookup(self) -> None:
        """Pre-load ego_pose table as {timestamp_us -> (x, y, yaw)}."""
        con  = sqlite3.connect(self._db_path)
        rows = con.execute(
            'SELECT timestamp, x, y, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp'
        ).fetchall()
        con.close()
        for ts, x, y, qw, qx, qy, qz in rows:
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))
            self._expert[int(ts)] = (x, y, yaw)
        self._sorted_ts = sorted(self._expert.keys())

    def _get_expert_at_offset(self, current_ts_us: int, offset_steps: int = 8) -> Tuple[float, float, float]:
        """Return expert (x, y, yaw) at current_ts + offset_steps * 100 ms.

        NOTE: The DB is 100 Hz (10 ms/row). offset_steps here is in 100 ms simulation
        steps (nuPlan calls the planner at ~10 Hz). offset_steps=8 → T+0.8 s at inference,
        which is 10× the training goal horizon (GOAL_OFFSET=8 raw rows × 10 ms = 0.08 s).
        See verify_pipeline.py Check 3.
        """
        target_ts = current_ts_us + offset_steps * 100_000   # 100_000 µs = 100 ms = 1 sim step
        idx = int(np.searchsorted(self._sorted_ts, target_ts))
        idx = min(idx, len(self._sorted_ts) - 1)
        return self._expert[self._sorted_ts[idx]]

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego    = current_input.history.current_state[0]
        x_g    = ego.rear_axle.x
        y_g    = ego.rear_axle.y
        yaw    = ego.rear_axle.heading
        dcs    = ego.dynamic_car_state
        vx     = dcs.rear_axle_velocity_2d.x
        vy     = dcs.rear_axle_velocity_2d.y
        ax     = dcs.rear_axle_acceleration_2d.x
        ay     = dcs.rear_axle_acceleration_2d.y
        ts     = int(ego.time_point.time_us)
        t0     = ego.time_point.time_us

        # Get T+8 goal position from expert DB, transform to ego-frame
        gx, gy, _ = self._get_expert_at_offset(ts, offset_steps=8)
        cos_neg_yaw = np.cos(-yaw)
        sin_neg_yaw = np.sin(-yaw)
        dx_w    = gx - x_g
        dy_w    = gy - y_g
        dx_goal = cos_neg_yaw * dx_w - sin_neg_yaw * dy_w
        dy_goal = sin_neg_yaw * dx_w + cos_neg_yaw * dy_w

        feat = torch.tensor(
            [np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal],
            dtype=torch.float32,
        )
        feat_norm = (feat - self._X_mean) / self._X_std

        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)

        cos_h, sin_h = np.cos(yaw), np.sin(yaw)

        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            wx    = x_g + cos_h * dx_e - sin_h * dy_e
            wy    = y_g + sin_h * dx_e + cos_h * dy_e
            w_yaw = yaw + d_yaw
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(wx, wy, w_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)


# ---------------------------------------------------------------------------
# Phase 3b — MapBCPlanner
# ---------------------------------------------------------------------------

class MapBCPlanner(AbstractPlanner):
    """
    MapBC: GoalBC weights at inference with road-centerline goal (no expert).

    WHY reuse GoalBCPolicy weights (goal_bc.pt):
      MapBC and GoalBC share identical training data and architecture. Both use
      [state(6) + goal(2)] = 8-dim input at training time with expert T+8 goals.
      The trained weights are therefore identical. The ONLY difference is inference:
        GoalBCPlanner  -> goal from expert DB lookup (T+8 expert position)
        MapBCPlanner   -> goal from road centerline (nuPlan map look-ahead)
      This cleanly isolates the effect of goal SOURCE on closed-loop performance.
      Any L2 gap between GoalBC (1.820m) and MapBC is purely due to how closely
      the map centerline approximates the expert T+8 position.

    At deployment: MapBCPlanner needs only a checkpoint + HD map. No expert data
    required at runtime -- this is a fully deployable policy.
    """

    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        self._look_ahead_m = look_ahead_m
        self._map_api      = None    # injected by nuPlan via initialize()

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model = GoalBCPolicy().to('cpu')
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']
        self._Y_std  = ckpt['Y_std']

    def name(self) -> str:
        return 'MapBCPlanner'

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        # WHY store map_api here: nuPlan injects the scenario map_api via initialize().
        # This gives real-time map access without a DB path and automatically uses
        # the correct city map for each scenario.
        self._map_api = initialization.map_api

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g = ego.rear_axle.x
        y_g = ego.rear_axle.y
        yaw = ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx  = dcs.rear_axle_velocity_2d.x
        vy  = dcs.rear_axle_velocity_2d.y
        ax  = dcs.rear_axle_acceleration_2d.x
        ay  = dcs.rear_axle_acceleration_2d.y
        t0  = ego.time_point.time_us

        # Get centerline goal from HD map -- no expert data required
        dx_goal, dy_goal = self._get_map_goal(x_g, y_g, yaw)

        feat = torch.tensor(
            [np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal],
            dtype=torch.float32,
        )
        feat_norm = (feat - self._X_mean) / self._X_std

        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)

        cos_h = np.cos(yaw)
        sin_h = np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            wx    = x_g + cos_h * dx_e - sin_h * dy_e
            wy    = y_g + sin_h * dx_e + cos_h * dy_e
            w_yaw = yaw + d_yaw
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(wx, wy, w_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)

    def _get_map_goal(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """Query nuPlan map_api for look-ahead centerline goal in ego-frame."""
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D

        FALLBACK = (float(self._look_ahead_m), 0.0)   # straight-ahead in ego-frame

        if self._map_api is None:
            return FALLBACK

        try:
            result = self._map_api.get_proximal_map_objects(
                Point2D(x, y), radius=30.0, layers=[SemanticMapLayer.LANE]
            )
            lanes = result[SemanticMapLayer.LANE]
        except Exception:
            # WHY broad except: ego may drift outside mapped region during
            # compounding error. Fallback keeps simulation running cleanly.
            return FALLBACK

        if not lanes:
            return FALLBACK

        # Select lane by heading-weighted distance (v2 fix).
        # WHY: naive nearest-centerline (v1) scored 56.3m — WORSE than BC_v0 (49.5m).
        # Root cause: at intersections / after drift the closest lane tangent was
        # anti-aligned with ego heading, so the look-ahead goal pointed BACKWARD.
        # Fix: penalise lanes whose tangent at the closest point opposes ego heading.
        # Score = dist + 30 * heading_penalty, where heading_penalty = max(0, -cos_angle).
        # A lane pointing directly backward gets +30m penalty (≈ the search radius),
        # effectively excluding it unless no forward-aligned lane exists within 30m.
        ego_dir = np.array([np.cos(yaw), np.sin(yaw)])
        best_lane, best_score = None, float('inf')
        for lane in lanes:
            try:
                cl_pts  = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
                dists_l = np.linalg.norm(cl_pts - np.array([x, y]), axis=1)
                i0_l    = int(np.argmin(dists_l))
                d       = float(dists_l[i0_l])
                i_next  = min(i0_l + 1, len(cl_pts) - 1)
                tangent = cl_pts[i_next] - cl_pts[i0_l]
                t_norm  = float(np.linalg.norm(tangent))
                if t_norm > 1e-6:
                    cos_a           = float(np.dot(tangent / t_norm, ego_dir))
                    heading_penalty = max(0.0, -cos_a)
                else:
                    heading_penalty = 0.0
                score = d + 30.0 * heading_penalty
                if score < best_score:
                    best_score, best_lane = score, lane
            except Exception:
                continue

        if best_lane is None:
            return FALLBACK

        cl    = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])
        dists = np.linalg.norm(cl - np.array([x, y]), axis=1)
        i0    = int(np.argmin(dists))

        # Walk look_ahead_m along centerline from closest point
        cum    = 0.0
        i_goal = min(i0 + 1, len(cl) - 1)
        for i in range(i0, len(cl) - 1):
            cum += float(np.linalg.norm(cl[i + 1] - cl[i]))
            if cum >= self._look_ahead_m:
                i_goal = i + 1
                break
        else:
            i_goal = len(cl) - 1

        gx, gy  = cl[i_goal]
        cos_neg = np.cos(-yaw)
        sin_neg = np.sin(-yaw)
        return (
            float(cos_neg * (gx - x) - sin_neg * (gy - y)),
            float(sin_neg * (gx - x) + cos_neg * (gy - y)),
        )


# ---------------------------------------------------------------------------
# Phase 3c — RouteMapBCPlanner
# ---------------------------------------------------------------------------

class RouteMapBCPlanner(AbstractPlanner):
    """
    RouteMapBC: GoalBC weights at inference with a globally-tracked pre-computed route.

    Key difference from MapBCPlanner:
      MapBC:      live map query at each step → fails when ego drifts off-road
                  (get_proximal_map_objects returns 0 lanes once ego is >30m from road)
      RouteMapBC: route pre-computed in initialize() → always valid, no live queries

    WHY this fixes Phase 3b:
      The drift-bootstrapping failure in MapBC happens because the point query
      `get_proximal_map_objects(radius=30m)` is a LOCAL reference. Once the ego
      compounding-drifts 2–3m off-road, the query returns zero lanes and the
      straight-ahead fallback fires every step — worse than BC's implicit prior.

      RouteMapBC mirrors how IDM works: compute a reference path AT SCENARIO START
      and track progress along it globally. The stored route_pts array is always
      valid regardless of how far the ego drifts — we just find the closest stored
      point using argmin over all N stored waypoints.

    Route construction (initialize()):
      1. Get initial ego position from initialization.initial_ego_state
      2. Query map for nearby lanes at initial position (radius=50m for robustness)
      3. Select most forward-aligned lane (same heading-weighted scoring as MapBC v2)
      4. Walk along centerline for 200m by chaining successor lanes
      5. Store as self._route_pts — (N, 2) float64 array in global UTM coordinates

    Goal computation (each planning step):
      1. Find closest route point to current ego position (argmin over all route_pts)
      2. Walk 8m forward from that point along the stored route
      3. Transform goal to ego-frame → (dx_goal, dy_goal)
      4. Always valid: route_pts are pre-stored, not live-queried

    WHY 200m route:
      nuPlan mini scenarios are 15–25 seconds at 5–15 m/s → 75–375m.
      200m covers the majority of scenarios. If route runs out, fall back to last
      waypoint direction (equivalent to "keep driving forward").

    WHY reuse goal_bc.pt:
      MapBC and GoalBC have identical training (same data, architecture, T+8 expert
      goals). Only inference differs — the goal SOURCE changes. RouteMapBC uses the
      same weights; the pre-computed route replaces both the expert DB and the live
      map query. This isolates the effect of global vs. local goal reference.

    REF: IDM reference-path tracking concept. Treiber et al. (2000).
    """

    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0,
                 speed_adaptive: bool = False) -> None:
        self._look_ahead_m   = look_ahead_m
        # WHY speed_adaptive flag: keeps all look-ahead logic in one place so
        # SpeedAdaptiveRouteMapBCPlanner is a trivial 3-line subclass with no duplication.
        self._speed_adaptive = speed_adaptive
        self._route_pts: Optional[np.ndarray] = None   # (N, 2) global UTM waypoints
        self._map_api = None   # injected via initialize()

        # WHY load weights in __init__ (same as MapBCPlanner): avoids repeated disk
        # reads if the planner is instantiated once and reused across scenarios.
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model = GoalBCPolicy().to('cpu')
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._X_mean = torch.tensor(ckpt['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(ckpt['X_std'],  dtype=torch.float32)
        self._Y_mean = ckpt['Y_mean']
        self._Y_std  = ckpt['Y_std']

    def name(self) -> str:
        return 'RouteMapBCPlanner'

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        """Store map_api and reset route. Route is built lazily on the first planning step.

        WHY lazy construction: PlannerInitialization does not expose an initial_ego_state
        field in this nuPlan version — only map_api, mission_goal, and route_roadblock_ids.
        The first call to compute_planner_trajectory() receives the t=0 ego state via
        PlannerInput, so we defer _build_route() to that point. The result is identical
        to "compute at scenario start" because step-0 IS the scenario start.
        """
        self._map_api   = initialization.map_api
        self._route_pts = None   # reset so each scenario gets a fresh route

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        ego = current_input.history.current_state[0]
        x_g = ego.rear_axle.x
        y_g = ego.rear_axle.y
        yaw = ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        vx  = dcs.rear_axle_velocity_2d.x
        vy  = dcs.rear_axle_velocity_2d.y
        ax  = dcs.rear_axle_acceleration_2d.x
        ay  = dcs.rear_axle_acceleration_2d.y
        t0  = ego.time_point.time_us

        # WHY lazy route construction: PlannerInitialization has no initial_ego_state.
        # We build the route once on the first planning call (which IS t=0) using the
        # actual ego state from PlannerInput. All subsequent steps reuse self._route_pts.
        if self._route_pts is None:
            self._route_pts = self._build_route(ego, self._map_api)

        # Look-ahead distance: fixed (default) or T+0.8 s equivalent (speed_adaptive).
        # WHY two modes in one method: avoids duplicating the 30-line trajectory-building
        # block in SpeedAdaptiveRouteMapBCPlanner. The flag is set once at __init__.
        if self._speed_adaptive:
            speed       = float(np.sqrt(vx ** 2 + vy ** 2))
            look_ahead_m = max(0.05, speed * _GOAL_LOOKAHEAD_S)
            # WHY 0.05 floor: at a full stop, 0m look-ahead collapses to a degenerate
            # goal equal to the ego position. 0.05 m keeps the goal just ahead.
        else:
            look_ahead_m = self._look_ahead_m

        # Goal from pre-computed route — always valid regardless of ego drift
        dx_goal, dy_goal = self._get_route_goal(x_g, y_g, yaw, look_ahead_m)

        feat = torch.tensor(
            [np.sin(yaw), np.cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal],
            dtype=torch.float32,
        )
        feat_norm = (feat - self._X_mean) / self._X_std

        with torch.no_grad():
            pred_norm = self._model(feat_norm.unsqueeze(0)).squeeze(0).numpy()
        pred = (pred_norm * self._Y_std + self._Y_mean).reshape(FUTURE_STEPS, 3)

        cos_h = np.cos(yaw)
        sin_h = np.sin(yaw)
        states = [ego]
        for j, (dx_e, dy_e, d_yaw) in enumerate(pred):
            wx    = x_g + cos_h * dx_e - sin_h * dy_e
            wy    = y_g + sin_h * dx_e + cos_h * dy_e
            w_yaw = yaw + d_yaw
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(wx, wy, w_yaw),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(ax, ay),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * DT * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)

    def _build_route(self, initial_ego_state, map_api) -> np.ndarray:
        """
        Build a 200m route from the initial ego position along the road centerline.

        Returns: (N, 2) float64 array of global UTM waypoints.

        Algorithm:
          1. Query lanes at initial position (radius=50m — wider than MapBC's 30m
             for robustness on scenarios where ego starts near intersection edges)
          2. Select most forward-aligned lane (heading-weighted score, same as MapBC v2)
          3. Collect centerline from closest point to end of that lane
          4. Chain successor lanes until 200m accumulated or MAX_CHAIN reached
          5. If no lanes found, fall back to a 200m straight in current heading direction
        """
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D

        x0  = initial_ego_state.rear_axle.x
        y0  = initial_ego_state.rear_axle.y
        yaw0 = initial_ego_state.rear_axle.heading

        # WHY radius=50m: at scenario start the ego is always on-road so 50m is
        # generous enough to find lanes even at wide intersections. MapBC used 30m
        # live per step — starting at 50m once is cheap and more reliable.
        try:
            result = map_api.get_proximal_map_objects(
                Point2D(x0, y0), radius=50.0, layers=[SemanticMapLayer.LANE]
            )
            lanes = result[SemanticMapLayer.LANE]
        except Exception:
            return self._straight_route(x0, y0, yaw0, length_m=200.0, step_m=2.0)

        if not lanes:
            # Some scenarios start in a parking lot or off-road. Straight is safest.
            return self._straight_route(x0, y0, yaw0, length_m=200.0, step_m=2.0)

        # Select the most forward-aligned lane using heading-weighted distance score
        # (identical to MapBC v2 selection — ensures the best lane is picked at start)
        ego_dir = np.array([np.cos(yaw0), np.sin(yaw0)])
        best_lane, best_score = None, float('inf')
        for lane in lanes:
            try:
                cl_pts = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
                dists  = np.linalg.norm(cl_pts - np.array([x0, y0]), axis=1)
                i0     = int(np.argmin(dists))
                d      = float(dists[i0])
                i_next = min(i0 + 1, len(cl_pts) - 1)
                tangent = cl_pts[i_next] - cl_pts[i0]
                t_norm  = np.linalg.norm(tangent)
                cos_a   = float(np.dot(tangent / t_norm, ego_dir)) if t_norm > 1e-6 else 0.0
                # WHY 30.0 penalty: same calibration as MapBC v2. Anti-aligned lane
                # (cos_a = -1) gets +30m added to its apparent distance → excluded
                # unless it is the only lane within 30m.
                score = d + 30.0 * max(0.0, -cos_a)
                if score < best_score:
                    best_score, best_lane = score, lane
            except Exception:
                continue

        if best_lane is None:
            return self._straight_route(x0, y0, yaw0, length_m=200.0, step_m=2.0)

        # Collect centerline from closest point to end of best lane
        cl = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])
        dists = np.linalg.norm(cl - np.array([x0, y0]), axis=1)
        i0    = int(np.argmin(dists))
        route = cl[i0:]   # WHY start from i0: skip the behind-ego portion of the lane

        # Measure how much route we already have
        total_length = 0.0
        if len(route) > 1:
            total_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))

        # Chain successor lanes until we reach 200m
        # WHY MAX_CHAIN=8: typical nuPlan lane segments are 20–50m → 8 segments = 160–400m.
        # Hard cap prevents infinite loop on roundabout topology (circular successors).
        current_lane = best_lane
        MAX_CHAIN = 8
        for _ in range(MAX_CHAIN):
            if total_length >= 200.0:
                break
            try:
                successors = current_lane.outgoing_edges
                if not successors:
                    break
                # Pick the successor to follow at this junction.
                # WHY delegate to _select_successor: keeps the heading-alignment rule
                # in one overridable place. RoadblockRouteMapBCPlanner overrides it to
                # prefer on-route successors (Open/Closed) — the chaining loop is shared.
                last_dir = route[-1] - route[-2] if len(route) > 1 else ego_dir
                last_dir_n = last_dir / (np.linalg.norm(last_dir) + 1e-8)
                best_succ = self._select_successor(successors, last_dir_n)
                if best_succ is None:
                    break
                succ_pts = np.array(
                    [(s.x, s.y) for s in best_succ.baseline_path.discrete_path]
                )
                route = np.vstack([route, succ_pts])
                total_length += float(
                    np.sum(np.linalg.norm(np.diff(succ_pts, axis=0), axis=1))
                )
                current_lane = best_succ
            except Exception:
                break

        return route.astype(np.float64)

    def _select_successor(self, successors, last_dir_n):
        """
        Pick the successor lane whose entry tangent best aligns with travel direction.

        Args:
            successors : list of outgoing lane/connector map objects.
            last_dir_n : unit vector of current travel direction (global frame).

        Returns the best-aligned successor, or None if none is usable.

        WHY a separate method: this is the junction-branch decision. Extracting it
        lets RoadblockRouteMapBCPlanner override only the branch choice (prefer the
        intended route) while reusing the parent's lane-chaining loop unchanged.
        """
        best_succ, best_cos = None, -1.0
        for succ in successors:
            try:
                succ_pts = np.array(
                    [(s.x, s.y) for s in succ.baseline_path.discrete_path]
                )
                if len(succ_pts) < 2:
                    continue
                tangent = succ_pts[1] - succ_pts[0]
                cos_a   = float(
                    np.dot(tangent / (np.linalg.norm(tangent) + 1e-8), last_dir_n)
                )
                if cos_a > best_cos:
                    best_cos, best_succ = cos_a, succ
            except Exception:
                continue
        return best_succ

    def _straight_route(
        self, x: float, y: float, yaw: float, length_m: float = 200.0, step_m: float = 2.0
    ) -> np.ndarray:
        """
        Fallback: straight-line route in current heading direction.
        WHY: some nuPlan scenarios start off-road or in parking areas where the map
        returns no lanes within 50m. A straight route is neutral — same as the
        MapBC fallback but pre-committed so it won't fire every step.
        """
        n   = int(length_m / step_m)
        pts = np.array([
            [x + step_m * i * np.cos(yaw), y + step_m * i * np.sin(yaw)]
            for i in range(n)
        ], dtype=np.float64)
        return pts

    def _get_route_goal(self, x: float, y: float, yaw: float,
                        look_ahead_m: Optional[float] = None) -> Tuple[float, float]:
        """
        Return the route point look_ahead_m ahead of the ego, in ego-frame.

        Args:
            x, y, yaw : current ego rear-axle pose (global frame)
            look_ahead_m : arc-length to walk along route from the nearest point.
                           Defaults to self._look_ahead_m (set in __init__).
                           Callers pass an explicit value for speed-adaptive mode.

        WHY argmin over ALL route_pts (not a sliding window):
          The ego may drift 50+ m off road. A local window around the last-tracked
          index could become stale and produce a goal behind the ego. Full argmin is
          O(N), N ≈ 100–1000, called at 10 Hz — negligible cost. Guarantees that
          the closest route point is always found regardless of drift magnitude.
        """
        if look_ahead_m is None:
            look_ahead_m = self._look_ahead_m

        if self._route_pts is None or len(self._route_pts) == 0:
            # No route available — project straight ahead as neutral fallback.
            return (float(look_ahead_m), 0.0)

        # Step 1: closest route point to current ego position
        dists = np.linalg.norm(self._route_pts - np.array([x, y]), axis=1)
        i0    = int(np.argmin(dists))

        # Step 2: walk look_ahead_m forward from i0 along stored route
        cum    = 0.0
        i_goal = min(i0 + 1, len(self._route_pts) - 1)
        for i in range(i0, len(self._route_pts) - 1):
            cum += float(np.linalg.norm(self._route_pts[i + 1] - self._route_pts[i]))
            if cum >= look_ahead_m:
                i_goal = i + 1
                break
        else:
            # Route ran out — use last stored point rather than falling back to
            # straight-ahead, which is what caused MapBC's drift-bootstrapping failure.
            i_goal = len(self._route_pts) - 1

        gx, gy  = self._route_pts[i_goal]
        cos_neg = np.cos(-yaw)
        sin_neg = np.sin(-yaw)
        return (
            float(cos_neg * (gx - x) - sin_neg * (gy - y)),
            float(sin_neg * (gx - x) + cos_neg * (gy - y)),
        )


class TrainedRouteBCPlanner(RouteMapBCPlanner):
    """
    Phase 3c' — TrainedRouteBC: retrained version of RouteMapBC.

    The ONLY difference from RouteMapBCPlanner is the checkpoint it loads:
      RouteMapBCPlanner     → goal_bc.pt           (trained on expert T+8 goals)
      TrainedRouteBCPlanner → trained_route_bc.pt  (trained on arc-length-8m route goals)

    WHY this matters (train/inference mismatch from Phase 3c):
      RouteMapBC achieved 32.085m vs GoalBC 1.820m (17.6× gap).
      goal_bc.pt learned: "goal offset = where expert will be in 0.8s."
      At inference, RouteMapBC feeds: "goal = road centerline 8m ahead."
      These have systematically different statistics — the policy decodes them incorrectly.

      Fix: retrain with arc-length-8m route goals at BOTH training and inference time.
      The training distribution NOW matches inference. Expected result: ≈ GoalBC (1.820m)
      without requiring expert data at inference → deployable policy claim.

    Inference code: 100% identical to RouteMapBCPlanner. Only checkpoint differs.
    REF: Phase 3c finding, binding insight 2, phase3_roadmap.md.
    """

    def name(self) -> str:
        return 'TrainedRouteBCPlanner'


# ─────────────────────────────────────────────────────────────────────────────
# WHY SpeedAdaptiveRouteMapBCPlanner exists — root-cause analysis
#
# The nuPlan mini SQLite DB is sampled at 100 Hz (10 ms between rows), confirmed
# by timestamp deltas: dt[0] = 9893 µs ≈ 10 ms. GoalBCPlanner._get_expert_at_offset
# explicitly uses offset_steps * 100_000 µs = 8 × 100 ms = T+0.8 s at inference.
#
# GoalBC INFERENCE goal magnitude at avg speed 4.33 m/s:
#   4.33 × 0.8 = 3.46 m
#
# GoalBC TRAINING goal magnitude (DB at 100 Hz, GOAL_OFFSET=8 raw rows × 10 ms):
#   4.33 × 0.08 = 0.35 m  (verified: measured mean=0.342 m across all DB files)
#
# GoalBC works despite the 10× training-inference scale gap because goal_bc.pt
# learned a goal-to-trajectory mapping that extrapolates via ReLU. The direction
# is correct at any scale; ReLU networks scale their output proportionally.
#
# Why RouteMapBC (fixed 8 m) got 32 m:
#   GoalBC inference scale: speed × 0.8
#   RouteMapBC fixed scale: 8.0 m (regardless of speed)
#
#   Speed-dependent mismatch:
#     at 10 m/s: GoalBC = 8.0 m, RouteMapBC = 8.0 m  → identical ✓
#     at  4 m/s: GoalBC = 3.2 m, RouteMapBC = 8.0 m  → 2.5× too large
#     at  0 m/s: GoalBC ≈ 0 m,   RouteMapBC = 8.0 m  → ∞ too large
#
#   nuPlan mini urban scenarios include significant stopped/low-speed time
#   (intersections, traffic). During these segments, RouteMapBC's 8 m goal is
#   badly out-of-scale for goal_bc.pt → policy mis-fires → drift accumulates.
#   At highway speeds, fixed 8 m ≈ GoalBC → those segments track well.
#
# Why TrainedRouteBCPlanner (retrained on 8 m goals) got 49 m:
#   Retraining fixed the scale mismatch at inference (8 m = 8 m). But during
#   training the 8 m arc-length goal is ~12× the prediction horizon (16 raw rows
#   at 100 Hz × 4.33 m/s = 0.69 m). The goal lies so far beyond the prediction
#   window that the MSE loss can reach near-zero without the policy attending to
#   the goal at all — kinematics alone predict 0.69 m accurately. The network
#   converges to ignore the goal feature. At inference: BC-like behaviour (49 m).
#
# The fix: speed_adaptive = True → look_ahead = max(0.05, speed × 0.8)
#   At every speed this matches GoalBC's T+0.8 s temporal horizon.
#   goal_bc.pt already learned to use goals at this scale (GoalBC inference works).
#   Uses existing goal_bc.pt — no retraining needed.
# ─────────────────────────────────────────────────────────────────────────────

# T+8 time window in seconds, matching GoalBCPlanner._get_expert_at_offset:
#   offset_steps=8, step_size=100_000 µs = 0.1 s  →  8 × 0.1 = 0.8 s
_GOAL_LOOKAHEAD_S: float = 8 * 0.1   # = 0.8 s


class SpeedAdaptiveRouteMapBCPlanner(RouteMapBCPlanner):
    """
    Phase 3c'' — RouteMapBC with speed-adaptive look-ahead.

    Thin wrapper: sets speed_adaptive=True in the parent constructor.
    All logic lives in RouteMapBCPlanner.compute_planner_trajectory (no duplication).

    look_ahead = max(0.05, speed × 0.8)   — T+0.8 s equivalent at every speed.

    At 10 m/s  → 8.0 m  (same as fixed RouteMapBC, no change at highway speed)
    At  4 m/s  → 3.2 m  (matches GoalBC inference scale)
    At  0 m/s  → 0.05 m (matches GoalBC's near-zero goal when stopped)

    Uses goal_bc.pt unchanged — no retraining needed.
    Deployable: pre-computed route from HD map replaces the expert DB.
    """

    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        super().__init__(checkpoint_path, look_ahead_m, speed_adaptive=True)

    def name(self) -> str:
        return 'SpeedAdaptiveRouteMapBCPlanner'


# ─────────────────────────────────────────────────────────────────────────────
# WHY RoadblockRouteMapBCPlanner exists — Phase 3c''' root-cause analysis
#
# SpeedAdaptiveRouteMapBC fixed the goal-scale mismatch and reached 7.50m MEDIAN
# L2 on 30 scenarios — but the MEAN is dragged up by 4 catastrophic tail failures
# (L2: 55.7, 80.3, 85.3, 121.2 m). Those 4 are all the same failure mode.
#
# Root cause — straight-through-at-junction:
#   _build_route() chains successor lanes by HEADING ALIGNMENT only. At an
#   intersection the most forward-aligned successor is the straight-ahead lane.
#   But on the 4 failing scenarios the EXPERT TURNS. The pre-computed route then
#   commits to the wrong arm of the intersection; every downstream goal points the
#   policy straight while the expert curves away → L2 diverges to 50–120 m.
#
#   Speed-adaptive look-ahead cannot fix this: the goal scale is correct, the goal
#   DIRECTION is wrong because the underlying route took the wrong branch.
#
# The fix — use the intended route:
#   PlannerInitialization.route_roadblock_ids lists the roadblock / lane-connector
#   IDs of the scenario's intended route (the same signal IDM and the nuPlan PDM
#   planners consume). At each junction we prefer a successor that lies on that
#   route over the straight-ahead one; we fall back to heading alignment only when
#   no successor is on-route (route ran out, or IDs unavailable on this map). This
#   is strictly safer than the parent: identical behaviour when no route info, the
#   correct turn when there is → removes the 4 tail failures without retraining.
#
# Uses goal_bc.pt unchanged and inherits speed_adaptive=True. Only the route
# CONSTRUCTION changes (which branch to take), not the goal/look-ahead machinery.
# REF: PlannerInitialization.route_roadblock_ids; nuPlan PDM route-tracking concept.
# ─────────────────────────────────────────────────────────────────────────────


class RoadblockRouteMapBCPlanner(SpeedAdaptiveRouteMapBCPlanner):
    """
    Phase 3c''' — RouteMapBC that follows route_roadblock_ids at intersections.

    Inherits everything from SpeedAdaptiveRouteMapBCPlanner (speed_adaptive=True,
    goal_bc.pt, speed × 0.8 s look-ahead, pre-computed 200m route). The ONLY change
    is junction-branch selection inside _build_route():

      Parent:  pick the most heading-aligned successor  → goes straight at junctions
      This:    pick a successor on the intended route    → turns where the expert turns
               (falls back to heading alignment if none on route)

    WHY this is the right fix (not retraining, not a bigger look-ahead):
      The 4 catastrophic tail failures of SpeedAdaptiveRouteMapBC are all the same
      failure: the route took the straight arm of an intersection where the expert
      turned. The goal magnitude was already correct after the speed-adaptive fix —
      only the route's chosen branch was wrong. route_roadblock_ids encodes the
      correct branch, so guiding lane chaining with it fixes the direction directly.

    Substitutability (Liskov): when route_roadblock_ids is empty or unavailable,
    behaviour is byte-for-byte identical to the parent — so this is never worse.
    """

    def __init__(self, checkpoint_path: str, look_ahead_m: float = 8.0) -> None:
        super().__init__(checkpoint_path, look_ahead_m)
        # WHY frozenset: membership is tested inside the successor loop; a set gives
        # O(1) lookups and dedups the raw ID list. Empty until initialize() fills it.
        self._route_roadblock_ids: frozenset = frozenset()

    def name(self) -> str:
        return 'RoadblockRouteMapBCPlanner'

    def initialize(self, initialization) -> None:
        """Store map_api (via parent) AND the intended-route roadblock IDs; reset route.

        WHY also store route_roadblock_ids: the parent keeps only map_api, so it has
        no way to know which branch the expert takes at a junction. These IDs are the
        scenario's intended route — the missing signal that drives _select_successor.
        """
        super().initialize(initialization)
        ids = getattr(initialization, 'route_roadblock_ids', None) or []
        # WHY str() on both sides: roadblock IDs are ints on some maps and strings on
        # others, and lane.get_roadblock_id() returns str — normalize to compare safely.
        self._route_roadblock_ids = frozenset(str(i) for i in ids)

    def _select_successor(self, successors, last_dir_n):
        """Prefer a successor on the intended route; else fall back to heading alignment.

        At an intersection several successors exist. The parent's heading-aligned
        choice is the straight-ahead lane, which is wrong whenever the expert turns.
        route_roadblock_ids encodes the correct arm, so we first restrict the candidate
        set to on-route successors and pick the best-aligned among THOSE (handles the
        rare case of two on-route lanes at one junction). With no on-route successor —
        route ran out, or no IDs available — we defer entirely to the parent.
        """
        if self._route_roadblock_ids:
            on_route = [s for s in successors if self._on_route(s)]
            if on_route:
                return super()._select_successor(on_route, last_dir_n)
        # No route IDs, or no successor on route → parent heading-alignment behaviour.
        return super()._select_successor(successors, last_dir_n)

    def _on_route(self, lane) -> bool:
        """True if a lane (or its parent roadblock) belongs to route_roadblock_ids.

        WHY check both roadblock id and lane id: route_roadblock_ids are roadblock /
        lane-connector IDs, but successors are lane objects. A lane is on-route if its
        parent roadblock is listed (the common case). We also check the lane's own id
        as a fallback for map versions that enumerate lane/connector IDs directly.
        """
        try:
            if str(lane.get_roadblock_id()) in self._route_roadblock_ids:
                return True
        except Exception:
            pass
        try:
            if str(lane.id) in self._route_roadblock_ids:
                return True
        except Exception:
            pass
        return False
