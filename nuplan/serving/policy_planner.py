"""Sim-time serving for the 2x2 heads: feature adapter + AbstractPlanner.

Design rules (each closes an audit finding):
- Features come from SceneFeatureExtractor.build_input_features, the SAME
  function the offline extractor uses. Parity by construction; verified
  numerically by nuplan/serving/parity_check.py.
- Both heads get IDENTICAL post-processing (heading-from-path, velocity
  profile, 2 Hz -> InterpolatedTrajectory). Any feasibility shaping that
  differed between heads would bias the comparison.
- Diffusion trajectory selection is pre-registered: medoid of K samples
  (closest to the ensemble in mean pairwise xy L2). A route-progress
  selector can be added later as a SEPARATE pre-registered variant; results
  are reported per selector, never cherry-picked.
- Serving uses EMA weights from the checkpoint (validation selected on EMA;
  serving anything else would be a train/serve mismatch).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_NUPLAN_DIR = Path(__file__).resolve().parent.parent
if str(_NUPLAN_DIR) not in sys.path:
    sys.path.append(str(_NUPLAN_DIR))

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2, StateVector2D, TimePoint,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

from features.scene_features import SceneFeatureExtractor
from models.f0_dataset import scale_future, unscale_future
from models.policy_heads import (
    CosineSchedule, DeterministicHead, DiffusionHead, HeadConfig, WTAHead,
)
from models.samplers import ddim_sample
from models.scene_encoder import SceneEncoder, SceneEncoderConfig

DT_S = 0.5          # model waypoint spacing (16 steps x 0.5 s = 8 s)
ENCODER_KEYS = (
    "ego", "agents", "agent_mask", "map_polylines", "map_mask",
    "crosswalks", "crosswalk_mask", "route_polyline", "route_mask",
    "traffic_lights",
)


def features_from_planner_input(
    extractor: SceneFeatureExtractor,
    current_input: PlannerInput,
    initialization: PlannerInitialization,
):
    """PlannerInput -> the exact tensor dict the offline extractor produces."""
    ego_states = list(current_input.history.ego_states)        # oldest -> newest
    observations = list(current_input.history.observations)
    tl = list(current_input.traffic_light_data or [])
    feats, transform, current_ego = extractor.build_input_features(
        ego_states, observations, tl,
        list(initialization.route_roadblock_ids or []),
        initialization.map_api,
    )
    return feats, transform, current_ego


def load_ema_into(encoder, head, ckpt: dict) -> bool:
    """Load checkpoint weights, preferring the EMA shadow. Returns True if EMA."""
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    ema = ckpt.get("ema")
    if not ema:
        return False
    shadow = ema["shadow"]
    for mod, prefix in ((encoder, "encoder."), (head, "head.")):
        sd = {k[len(prefix):]: v for k, v in shadow.items() if k.startswith(prefix)}
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        # WHY strict=False: EMA tracks parameters only; buffers (LayerNorm
        # running stats do not exist here, but e.g. registered buffers) come
        # from the raw state_dict loaded above.
        assert not unexpected, f"unexpected EMA keys: {unexpected[:3]}"
    return True


def select_medoid(samples: torch.Tensor) -> torch.Tensor:
    """(K, H, 3) -> (H, 3): sample minimizing mean pairwise xy L2 distance."""
    xy = samples[..., :2]                                       # (K, H, 2)
    d = (xy.unsqueeze(0) - xy.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # (K, K)
    return samples[d.sum(dim=1).argmin()]


class PolicyPlanner(AbstractPlanner):
    """Serves one trained cell of the 2x2 inside nuPlan closed-loop sim."""

    # WHY True: the devkit's planner builder passes scenario= to the
    # constructor only when this class attribute is set; the precise-goal
    # condition needs the log future, and route mode simply ignores it.
    requires_scenario: bool = True

    def __init__(
        self,
        ckpt_path: str,
        head_type: str,                 # "det" | "diff"
        goal_mode: str,                 # "route" | "precise"
        scenario=None,                  # required for goal_mode="precise"
        num_samples: int = 8,
        ddim_steps: int = 20,
        device: str = "cpu",
    ):
        assert head_type in ("det", "diff", "wta") and goal_mode in ("route", "precise")
        if goal_mode == "precise" and scenario is None:
            raise ValueError("precise goal conditioning needs the log scenario")
        self._ckpt_path = ckpt_path
        self._head_type = head_type
        self._goal_mode = goal_mode
        self._scenario = scenario
        self._K = num_samples
        self._ddim_steps = ddim_steps
        self._device = torch.device(device)
        self._extractor = SceneFeatureExtractor()
        self._encoder = self._head = self._schedule = None
        self._last_goal: Optional[torch.Tensor] = None

    def name(self) -> str:
        return f"PolicyPlanner_{self._head_type}_{self._goal_mode}"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization
        ck = torch.load(self._ckpt_path, map_location=self._device,
                        weights_only=False)
        self._encoder = SceneEncoder(SceneEncoderConfig()).to(self._device)
        cfg = HeadConfig()
        if self._head_type == "det":
            self._head = DeterministicHead(cfg).to(self._device)
        elif self._head_type == "wta":
            self._head = WTAHead(cfg).to(self._device)
        else:
            self._head = DiffusionHead(cfg).to(self._device)
        self._used_ema = load_ema_into(self._encoder, self._head, ck)
        self._encoder.eval(), self._head.eval()
        self._schedule = CosineSchedule(T=cfg.T).to(self._device)

    # ---- goal ----

    def _precise_goal(self, iteration: int, transform) -> Optional[torch.Tensor]:
        """Expert near/far future points from the LOG, in the frame of the
        CURRENT SIMULATED pose. After divergence the log future no longer
        matches the sim state; that degradation is inherent to log-goal
        conditioning and is reported, not hidden."""
        try:
            fut = list(self._scenario.get_ego_future_trajectory(
                iteration, time_horizon=8.0, num_samples=16))
            if len(fut) < 16:
                raise ValueError("short future")
            fut = fut[-16:]
            xy = transform.world_to_ego(
                np.array([[s.rear_axle.x, s.rear_axle.y] for s in fut]))
            goal_raw = np.concatenate([xy[7], xy[15]]).astype(np.float32)
            # heads were trained on goals built from SCALED futures (xy / 10)
            self._last_goal = torch.from_numpy(goal_raw / 10.0).unsqueeze(0)
        except Exception:
            pass  # near scenario end: reuse the previous goal
        return self._last_goal

    # ---- trajectory construction (identical for both heads) ----

    def _build_trajectory(self, ego: EgoState, pred: np.ndarray) -> InterpolatedTrajectory:
        cx, cy = ego.rear_axle.x, ego.rear_axle.y
        heading = ego.rear_axle.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        wx = cx + cos_h * pred[:, 0] - sin_h * pred[:, 1]
        wy = cy + sin_h * pred[:, 0] + cos_h * pred[:, 1]
        # WHY heading from path tangent, not the predicted heading channel:
        # the tracker needs heading consistent with the path; independently
        # regressed headings destabilize it. Low displacement -> keep current.
        px = np.concatenate([[cx], wx])
        py = np.concatenate([[cy], wy])
        dx, dy = np.diff(px), np.diff(py)
        disp = np.hypot(dx, dy)
        yaw = np.where(disp > 0.1, np.arctan2(dy, dx), heading)
        for j in range(1, len(yaw)):                # forward-fill slow segments
            if disp[j] <= 0.1:
                yaw[j] = yaw[j - 1]
        speeds = disp / DT_S
        t0 = ego.time_point.time_us
        states: List[EgoState] = [ego]
        for j in range(pred.shape[0]):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(wx[j]), float(wy[j]), float(yaw[j])),
                rear_axle_velocity_2d=StateVector2D(float(speeds[j]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + int((j + 1) * DT_S * 1e6)),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
            ))
        return InterpolatedTrajectory(states)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        feats, transform, current_ego = features_from_planner_input(
            self._extractor, current_input, self._initialization)
        batch = {}
        for k in ENCODER_KEYS:
            v = torch.from_numpy(np.ascontiguousarray(feats[k]))
            batch[k] = v.unsqueeze(0).to(self._device)
        goal = None
        if self._goal_mode == "precise":
            goal = self._precise_goal(current_input.iteration.index, transform)
        with torch.no_grad():
            memory = self._encoder(batch)
            if self._head_type == "det":
                traj_scaled = self._head(memory, goal=goal)[0]
            elif self._head_type == "wta":
                # mode-committing selector (ADR-029 #1c): execute the TOP-SCORED
                # mode (not the medoid) -- the point of a multimodal policy is to
                # commit to a chosen maneuver, not average the alternatives.
                trajs, scores = self._head(memory, goal=goal)
                # GATE-CL-2 oracle: WTA_MODE_INDEX forces execution of a chosen mode
                # (to measure best-of-modes closed-loop CLS); else the deployed top-scored mode.
                _mi = os.environ.get("WTA_MODE_INDEX", "")
                _m = int(_mi) if _mi != "" else int(scores[0].argmax())
                _m = min(max(_m, 0), trajs.shape[1] - 1)
                traj_scaled = trajs[0, _m]
            else:
                gen = torch.Generator(device=self._device).manual_seed(
                    current_input.iteration.index)
                samples = ddim_sample(
                    self._head, self._schedule, memory, goal=goal,
                    num_samples=self._K, num_steps=self._ddim_steps,
                    generator=gen)[0]
                traj_scaled = select_medoid(samples)
        pred = unscale_future(traj_scaled).cpu().numpy()        # (16, 3) meters
        return self._build_trajectory(current_ego, pred)
