"""scene_features.py — vectorized, ego-centric scene-feature extractor for nuPlan (F0).

Produces, per scenario sample, a dict of named tensors (ego / agents / map /
route / traffic-light) in an ego-centric frame (ego at origin facing +x) and
saves them as ``.pt`` shards. This is the F0 deliverable of the av-policy-lab
frontier build; the representation follows ``docs/frontier/STAGE_2_data.md``.

Design (SOLID, single-responsibility):
    Normalizer            — all per-feature scaling/encoding in ONE place.
    EgoFrameTransform     — world<->ego rigid transform (the load-bearing invariant).
    EgoFeatureBuilder     — ego history tensor.
    AgentFeatureBuilder   — nearest-N agents + validity mask.
    MapFeatureBuilder     — lane / lane-connector / crosswalk polylines + route + TL.
    SceneFeatureExtractor — orchestrator: walks a scenario, builds samples, saves shards.

WHY a separate module from planners.py: extraction is an offline, CPU-bound data
job (Interface Segregation) — it must not import the simulation runtime or torch
training stack. The geometry/normalization core is pure-numpy so it unit-tests
without the nuPlan devkit installed (see tests/test_scene_features.py).

devkit-call provenance:
    Every nuPlan-devkit call is tagged inline:
      [VERIFIED from devkit docs/source] — confirmed against the motional/nuplan-devkit
        master source (AbstractScenario, AbstractMap, AgentState, maps_datatypes) and
        the repo's own working planners.py / map_utils.py patterns.
      [UNVERIFIED — confirm in env] — plausible but version-sensitive; the brief
        requires these to be re-confirmed by running ``--smoke`` on nuPlan MINI.
    See docs/frontier/F0_IMPLEMENTATION.md for the full ledger.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # WHY guard: torch is required to SAVE shards but not to import the pure-numpy
    # builder core (Normalizer / EgoFrameTransform / resampling), which the unit
    # tests exercise without torch. Real extraction always has torch present.
    import torch
except Exception:  # pragma: no cover - exercised only in torch-less test envs
    torch = None  # type: ignore

logger = logging.getLogger("scene_features")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    """All fixed dimensions and normalization scales for the F0 representation.

    WHY a frozen dataclass: one immutable source of truth for every magic number,
    so the builders, the tensor shapes, the tests, and the docs cannot drift apart.
    Defaults follow the F0 brief; they are a subset/realization of STAGE_2_data.md.
    """

    # --- temporal ---
    hz: float = 10.0                 # nuPlan canonical replay rate
    history_seconds: float = 2.0     # last 2 s of context
    # WHY 20 (not 21): the F0 brief specifies "last 2 s @ 10 Hz (20 steps)". We
    # take 20 *past* steps ending at (and including) the current step. STAGE_2 uses
    # T_h=21 (20 past + current); F0 uses a 20-step window. Documented divergence.
    history_steps: int = 20

    # --- future target (the training label; F0 v1 shipped inputs only) ---
    # WHY 16 @ 2 Hz over 8 s: STAGE_3 horizon decision (E4); near/far point
    # goals (F2) and imitation supervision (F3/F4) all derive from this field.
    future_steps: int = 16
    future_horizon_s: float = 8.0
    # WHY: per-iteration wall-clock budget; a single scenario that hangs inside
    # the devkit is skipped rather than stalling the whole job (job 7626090
    # task 14 hung ~20h on one scenario). 0 disables (tests).
    iteration_timeout_s: int = 90

    # --- agents ---
    max_agents: int = 32             # nearest-N kept; pad/truncate to this
    agent_radius_m: float = 100.0    # consider agents within this radius of ego
    num_agent_types: int = 3         # one-hot: [vehicle, pedestrian, bicycle]

    # --- map ---
    max_map_polylines: int = 128     # P
    map_polyline_points: int = 20    # S — points each polyline is resampled to
    max_crosswalks: int = 16
    map_query_radius_m: float = 100.0
    num_map_types: int = 3           # one-hot: [lane, lane_connector, crosswalk]

    # --- route ---
    route_points: int = 40           # R — resampled route centerline points
    route_reach_m: float = 120.0     # cap on route arc-length (matches STAGE_2 §4)
    route_max_chain: int = 12        # max successor lanes to chain (loop guard)
    # WHY corridor mode (v3, ADR-017): the lane-centerline route picks ONE
    # successor at every fork, leaking the expert's branch choice into all
    # four cells and collapsing the ambiguity the route-region condition
    # exists to preserve. Corridor mode averages each roadblock's lane
    # baselines (roadblock-level turn-by-turn, what a navigator provides)
    # and never commits to a lane. "lane" mode kept for the leak ablation.
    route_mode: str = "corridor"     # "corridor" (v3) | "lane" (v2 ablation)
    # WHY perturbation (v3, ADR-017): pure-expert histories train policies
    # that have never seen an off-path state; closed-loop then measures
    # recovery brittleness, not policy quality (our own DAgger result).
    # Standard recovery augmentation: blend the last steps toward a perturbed
    # current pose; the label stays the TRUE expert future in the perturbed
    # frame, so the target is the recovery maneuver.
    perturb_prob: float = 0.0        # set 0.5 for v3 training extraction
    perturb_lat_std_m: float = 0.3
    perturb_lat_max_m: float = 1.0
    perturb_head_std_rad: float = 0.05
    perturb_head_max_rad: float = 0.2
    perturb_blend_steps: int = 5

    # --- normalization scales (STAGE_2 §1) ---
    pos_scale_m: float = 120.0       # positions / max-radius
    vel_scale_mps: float = 15.0      # velocities / 15 m/s
    acc_scale_mps2: float = 5.0      # accel / 5 m/s^2

    @property
    def dt(self) -> float:
        """Seconds per step."""
        return 1.0 / self.hz

    @property
    def ego_feature_dim(self) -> int:
        """[x, y, sin, cos, vx, vy, ax, ay]."""
        return 8

    @property
    def agent_feature_dim(self) -> int:
        """[x, y, sin, cos, vx, vy] + one-hot type."""
        return 6 + self.num_agent_types

    @property
    def map_feature_dim(self) -> int:
        """[x, y, dir_x, dir_y] + one-hot type."""
        return 4 + self.num_map_types


# ===========================================================================
# Normalization — single source of truth (DRY)
# ===========================================================================

class Normalizer:
    """All feature standardization in one place (STAGE_2_data.md §1).

    WHY one class: PlanTF's central empirical result is that perturbation
    augmentation only helps *with correct normalization* — normalization is
    load-bearing, not cosmetic (REF: arXiv:2309.10443). Keeping every scale here
    means the (later) augmentation re-normalization step cannot silently disagree
    with the extractor. Every method is the exact inverse of its counterpart so
    invertibility is unit-testable.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._cfg = config

    # -- positions --
    def norm_pos(self, xy: np.ndarray) -> np.ndarray:
        """Scale ego-frame metres -> ~unit by dividing by the max radius."""
        return xy / self._cfg.pos_scale_m

    def denorm_pos(self, xy: np.ndarray) -> np.ndarray:
        return xy * self._cfg.pos_scale_m

    # -- velocities --
    def norm_vel(self, v: np.ndarray) -> np.ndarray:
        return v / self._cfg.vel_scale_mps

    def denorm_vel(self, v: np.ndarray) -> np.ndarray:
        return v * self._cfg.vel_scale_mps

    # -- accelerations --
    def norm_acc(self, a: np.ndarray) -> np.ndarray:
        return a / self._cfg.acc_scale_mps2

    def denorm_acc(self, a: np.ndarray) -> np.ndarray:
        return a * self._cfg.acc_scale_mps2

    @staticmethod
    def encode_heading(yaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Heading -> (sin, cos). WHY: avoids the 2*pi wrap discontinuity a raw
        angle would feed the network; (sin, cos) is already ~unit scale."""
        return np.sin(yaw), np.cos(yaw)

    @staticmethod
    def decode_heading(sin_yaw: np.ndarray, cos_yaw: np.ndarray) -> np.ndarray:
        """(sin, cos) -> heading via atan2. Inverse of encode_heading."""
        return np.arctan2(sin_yaw, cos_yaw)


# ===========================================================================
# Ego-frame transform — the single most important invariant
# ===========================================================================

class EgoFrameTransform:
    """World<->ego rigid transform about a reference pose (ego at origin, +x fwd).

    Mirrors the EXACT convention proven in the repo's planners.py / map_utils.py:
      world->ego uses cos(-yaw)/sin(-yaw); ego->world uses cos(yaw)/sin(yaw).
    These are exact inverses, so world->ego->world is the identity (the invariant
    the smoke gate and unit tests assert). REF: tests/test_planner_geometry.py.
    """

    def __init__(self, ref_x: float, ref_y: float, ref_yaw: float) -> None:
        self._x = float(ref_x)
        self._y = float(ref_y)
        self._yaw = float(ref_yaw)

    @property
    def yaw(self) -> float:
        return self._yaw

    def world_to_ego(self, pts_world: np.ndarray) -> np.ndarray:
        """(...,2) world points -> ego frame. Translate then rotate by -yaw."""
        cos_n = math.cos(-self._yaw)
        sin_n = math.sin(-self._yaw)
        dx = pts_world[..., 0] - self._x
        dy = pts_world[..., 1] - self._y
        ex = cos_n * dx - sin_n * dy
        ey = sin_n * dx + cos_n * dy
        return np.stack([ex, ey], axis=-1)

    def ego_to_world(self, pts_ego: np.ndarray) -> np.ndarray:
        """(...,2) ego points -> world frame. Rotate by +yaw then translate."""
        cos_h = math.cos(self._yaw)
        sin_h = math.sin(self._yaw)
        ex = pts_ego[..., 0]
        ey = pts_ego[..., 1]
        wx = cos_h * ex - sin_h * ey + self._x
        wy = sin_h * ex + cos_h * ey + self._y
        return np.stack([wx, wy], axis=-1)

    def rotate_vector_to_ego(self, vec_world: np.ndarray) -> np.ndarray:
        """Rotate a free vector (velocity/direction) into ego frame — NO translation.

        WHY separate from world_to_ego: velocities and polyline tangents are
        direction vectors; translating them (as if they were positions) is wrong.
        """
        cos_n = math.cos(-self._yaw)
        sin_n = math.sin(-self._yaw)
        vx = vec_world[..., 0]
        vy = vec_world[..., 1]
        ex = cos_n * vx - sin_n * vy
        ey = sin_n * vx + cos_n * vy
        return np.stack([ex, ey], axis=-1)

    def heading_to_ego(self, yaw_world: np.ndarray) -> np.ndarray:
        """World heading -> ego-frame heading (subtract reference yaw)."""
        return yaw_world - self._yaw


# ===========================================================================
# Pure-numpy geometry helpers (no devkit) — unit-tested directly
# ===========================================================================

def resample_polyline(points: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a polyline to exactly ``n_out`` points at uniform arc-length.

    Args:
        points: (M, 2) ordered polyline vertices. M may be < or > n_out.
        n_out:  number of output points (>= 2).

    Returns:
        (n_out, 2) float64 polyline, arc-length-uniform, endpoints preserved.

    WHY uniform arc-length (not index) resampling: raw devkit centerlines have
    non-uniform vertex spacing; index-resampling would bunch points where the
    source is dense. Arc-length spacing gives the network a geometrically
    consistent stride (matches PLUTO's fixed n_p-point polylines, arXiv:2404.14327).
    Degenerate inputs (0/1 points, zero length) degrade gracefully to a repeat.
    """
    if n_out < 2:
        raise ValueError("n_out must be >= 2")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((n_out, 2), dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n_out, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        # Degenerate (all coincident) — return the single distinct point repeated.
        return np.repeat(pts[:1], n_out, axis=0)

    targets = np.linspace(0.0, total, n_out)
    out_x = np.interp(targets, cum, pts[:, 0])
    out_y = np.interp(targets, cum, pts[:, 1])
    return np.stack([out_x, out_y], axis=1)


def polyline_directions(points: np.ndarray) -> np.ndarray:
    """Per-point unit tangent (dir_x, dir_y) of a polyline.

    The last point reuses the previous segment's direction so the output length
    matches the input. Zero-length segments yield a zero vector (handled by callers).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.zeros((len(pts), 2), dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    dirs = diffs / norms
    # Repeat the final direction so output length == input length.
    return np.vstack([dirs, dirs[-1:]])


def pad_or_truncate(
    items: Sequence[np.ndarray], max_n: int, feature_shape: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack up to ``max_n`` feature arrays, zero-pad the rest, return (data, mask).

    Args:
        items: ordered list of (already nearest-first) per-item feature arrays,
               each of shape ``feature_shape``.
        max_n: fixed slot count to pad/truncate to.
        feature_shape: shape of one item's feature array.

    Returns:
        data: (max_n, *feature_shape) float32 — first ``min(len, max_n)`` slots are
              real, the rest zero.
        mask: (max_n,) bool — True where the slot holds a real item.

    WHY here (not in the builder): padding/masking is pure array bookkeeping, so it
    lives in the geometry layer and is unit-testable without the devkit.
    """
    data = np.zeros((max_n, *feature_shape), dtype=np.float32)
    mask = np.zeros((max_n,), dtype=bool)
    n = min(len(items), max_n)
    for i in range(n):
        data[i] = items[i]
        mask[i] = True
    return data, mask


# ===========================================================================
# Feature builders (single-responsibility)
# ===========================================================================

class EgoFeatureBuilder:
    """Builds the ego history tensor: (T, 8) = [x, y, sin, cos, vx, vy, ax, ay].

    Positions/headings are expressed in the ego frame anchored at the CURRENT
    step; velocities/accelerations are rotated (not translated) into that frame.
    All values are normalized via the shared Normalizer.
    """

    def __init__(self, config: FeatureConfig, normalizer: Normalizer) -> None:
        self._cfg = config
        self._norm = normalizer

    def build(
        self, ego_states: List["EgoState"], transform: EgoFrameTransform
    ) -> np.ndarray:
        """Build (T, 8) from oldest->newest EgoStates (length == history_steps).

        devkit field access (all [VERIFIED from devkit docs/source] — confirmed in
        the repo's planners.py L102-128 working code and EgoState source):
            ego.rear_axle.x / .y / .heading
            ego.dynamic_car_state.rear_axle_velocity_2d.x / .y
            ego.dynamic_car_state.rear_axle_acceleration_2d.x / .y
        """
        T = self._cfg.history_steps
        out = np.zeros((T, self._cfg.ego_feature_dim), dtype=np.float32)
        xy_world = np.array(
            [[e.rear_axle.x, e.rear_axle.y] for e in ego_states], dtype=np.float64
        )
        yaw_world = np.array([e.rear_axle.heading for e in ego_states], dtype=np.float64)
        vel_world = np.array(
            [
                [
                    e.dynamic_car_state.rear_axle_velocity_2d.x,
                    e.dynamic_car_state.rear_axle_velocity_2d.y,
                ]
                for e in ego_states
            ],
            dtype=np.float64,
        )
        acc_world = np.array(
            [
                [
                    e.dynamic_car_state.rear_axle_acceleration_2d.x,
                    e.dynamic_car_state.rear_axle_acceleration_2d.y,
                ]
                for e in ego_states
            ],
            dtype=np.float64,
        )

        # NOTE: velocity/accel are stored in the EACH STEP's own body frame by the
        # devkit. We rotate by the *delta* heading between that step and the ego
        # reference so the whole history lives in one consistent ego frame.
        xy_ego = self._norm.norm_pos(transform.world_to_ego(xy_world))
        yaw_ego = transform.heading_to_ego(yaw_world)
        sin_y, cos_y = self._norm.encode_heading(yaw_ego)

        # WHY rotate per-step body-frame vel/acc by that step's own heading first:
        # rear_axle_velocity_2d is in the vehicle body frame, so world = R(yaw_w)·body,
        # then ego = R(-yaw_ref)·world. We compose into R(yaw_w - yaw_ref) directly.
        d_yaw = yaw_world - transform.yaw
        c, s = np.cos(d_yaw), np.sin(d_yaw)
        vel_ego = np.stack(
            [c * vel_world[:, 0] - s * vel_world[:, 1],
             s * vel_world[:, 0] + c * vel_world[:, 1]],
            axis=1,
        )
        acc_ego = np.stack(
            [c * acc_world[:, 0] - s * acc_world[:, 1],
             s * acc_world[:, 0] + c * acc_world[:, 1]],
            axis=1,
        )
        vel_ego = self._norm.norm_vel(vel_ego)
        acc_ego = self._norm.norm_acc(acc_ego)

        out[:, 0:2] = xy_ego
        out[:, 2] = sin_y
        out[:, 3] = cos_y
        out[:, 4:6] = vel_ego
        out[:, 6:8] = acc_ego
        return out


class AgentFeatureBuilder:
    """Builds nearest-N agent histories + validity mask.

    Output:
        agents:     (N, T, agent_feature_dim) float32
        agent_mask: (N, T) bool  (per-step validity; a slot may appear mid-window)
    """

    def __init__(self, config: FeatureConfig, normalizer: Normalizer) -> None:
        self._cfg = config
        self._norm = normalizer
        # WHY only these three: F0 keeps vehicles/pedestrians/bicycles (the dynamic
        # actors a planner must react to). Static objects (cones/barriers) live in
        # the map/raster, not the agent track set.
        self._type_index: Dict[str, int] = {"VEHICLE": 0, "PEDESTRIAN": 1, "BICYCLE": 2}

    def build(
        self,
        per_step_tracks: List["DetectionsTracks"],
        transform: EgoFrameTransform,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build agent tensors from oldest->newest DetectionsTracks (len == T).

        devkit access:
            tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.X)
                [VERIFIED from devkit source] TrackedObjects.get_tracked_objects_of_type
            agent.track_token  [VERIFIED] SceneObject.track_token (temporal id)
            agent.box.center.x / .y / .heading  [VERIFIED] SceneObject.box -> OrientedBox.center (StateSE2)
            agent.box.length / .width           [VERIFIED] OrientedBox.length / .width
            agent.velocity.x / .y               [VERIFIED] AgentState.velocity (StateVector2D)
            agent.tracked_object_type           [VERIFIED] SceneObject.tracked_object_type
        """
        from nuplan.common.actor_state.tracked_objects_types import (  # local import: devkit-only
            TrackedObjectType,
        )

        T = self._cfg.history_steps
        type_enums = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]

        # Step 1: gather per-track time series keyed by track_token. A track may be
        # absent in some steps; we fill present steps and leave the mask False else.
        # value: dict[token] -> list of (step_idx, feat_world_dict)
        track_series: Dict[str, List[Tuple[int, dict]]] = {}
        for t, tracks in enumerate(per_step_tracks):
            objs = []
            for enum in type_enums:
                # [VERIFIED] get_tracked_objects_of_type returns [] for absent types.
                objs.extend(tracks.tracked_objects.get_tracked_objects_of_type(enum))
            for obj in objs:
                token = getattr(obj, "track_token", None) or id(obj)
                rec = {
                    "x": float(obj.box.center.x),
                    "y": float(obj.box.center.y),
                    "yaw": float(obj.box.center.heading),
                    "vx": float(obj.velocity.x),
                    "vy": float(obj.velocity.y),
                    "type": obj.tracked_object_type.name,
                }
                track_series.setdefault(str(token), []).append((t, rec))

        # Step 2: rank tracks by closest distance to ego (at any observed step) at
        # the CURRENT (last) step preferentially, nearest-first.
        ego_origin = np.array([0.0, 0.0])

        def min_dist(series: List[Tuple[int, dict]]) -> float:
            pts = np.array([[r["x"], r["y"]] for _, r in series], dtype=np.float64)
            ego_pts = transform.world_to_ego(pts)
            return float(np.linalg.norm(ego_pts - ego_origin, axis=1).min())

        ranked = sorted(track_series.items(), key=lambda kv: min_dist(kv[1]))

        # Step 3: keep only those whose nearest approach is within radius, then
        # build per-agent (T, D) arrays with per-step masks.
        agents_data: List[np.ndarray] = []
        masks_data: List[np.ndarray] = []
        for _token, series in ranked:
            if min_dist(series) > self._cfg.agent_radius_m:
                continue
            feat = np.zeros((T, self._cfg.agent_feature_dim), dtype=np.float32)
            mask = np.zeros((T,), dtype=bool)
            for t, r in series:
                if not (0 <= t < T):
                    continue
                xy_ego = self._norm.norm_pos(
                    transform.world_to_ego(np.array([r["x"], r["y"]]))
                )
                yaw_ego = transform.heading_to_ego(np.array(r["yaw"]))
                sin_y, cos_y = self._norm.encode_heading(yaw_ego)
                vel_ego = self._norm.norm_vel(
                    transform.rotate_vector_to_ego(np.array([r["vx"], r["vy"]]))
                )
                feat[t, 0:2] = xy_ego
                feat[t, 2] = sin_y
                feat[t, 3] = cos_y
                feat[t, 4:6] = vel_ego
                # one-hot type
                ti = self._type_index.get(r["type"])
                if ti is not None:
                    feat[t, 6 + ti] = 1.0
                mask[t] = True
            agents_data.append(feat)
            masks_data.append(mask)
            if len(agents_data) >= self._cfg.max_agents:
                break

        # Step 4: pad/truncate to N. We pad the per-step masks too.
        N, T_ = self._cfg.max_agents, T
        agents = np.zeros((N, T_, self._cfg.agent_feature_dim), dtype=np.float32)
        agent_mask = np.zeros((N, T_), dtype=bool)
        for i in range(min(len(agents_data), N)):
            agents[i] = agents_data[i]
            agent_mask[i] = masks_data[i]
        return agents, agent_mask


class MapFeatureBuilder:
    """Builds vectorized map polylines, the on-route polyline, and TL status.

    Outputs:
        map_polylines: (P, S, map_feature_dim) float32
        map_mask:      (P,) bool
        crosswalks:    (C, S, 2) float32 (ego-frame polygon boundary, resampled)
        crosswalk_mask:(C,) bool
        route_polyline:(R, 4) float32 [x, y, dir_x, dir_y] ego-frame, normalized pos
        route_mask:    (R,) bool
        traffic_lights:(P,) int64 — per map-polyline TL status id (0=green..3=unknown,
                       -1 = not a traffic-controlled lane / no signal)
    """

    # Static map type one-hot indices.
    _TYPE_LANE = 0
    _TYPE_LANE_CONNECTOR = 1
    _TYPE_CROSSWALK = 2

    def __init__(self, config: FeatureConfig, normalizer: Normalizer) -> None:
        self._cfg = config
        self._norm = normalizer

    # ---- public entry ----
    def build(
        self,
        map_api: "AbstractMap",
        ego_world_xy: Tuple[float, float],
        route_roadblock_ids: List[str],
        traffic_lights: List["TrafficLightStatusData"],
        transform: EgoFrameTransform,
    ) -> Dict[str, np.ndarray]:
        from nuplan.common.actor_state.state_representation import Point2D  # devkit-only
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # devkit-only

        point = Point2D(ego_world_xy[0], ego_world_xy[1])

        # [VERIFIED from devkit source] AbstractMap.get_proximal_map_objects(
        #   point, radius, layers) -> Dict[SemanticMapLayer, List[MapObject]].
        # Same call the repo's planners.py / map_utils.py use successfully.
        result = map_api.get_proximal_map_objects(
            point,
            self._cfg.map_query_radius_m,
            layers=[
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.CROSSWALK,
            ],
        )
        lanes = result.get(SemanticMapLayer.LANE, [])
        connectors = result.get(SemanticMapLayer.LANE_CONNECTOR, [])
        crosswalks = result.get(SemanticMapLayer.CROSSWALK, [])

        # --- traffic-light lookup: lane_connector_id -> status int ---
        tl_status = self._traffic_light_map(traffic_lights)

        # --- build lane + lane-connector polylines ---
        map_items: List[np.ndarray] = []
        tl_ids: List[int] = []
        ordered = (
            [(self._TYPE_LANE, o) for o in lanes]
            + [(self._TYPE_LANE_CONNECTOR, o) for o in connectors]
        )
        # Nearest-first ordering by polyline start distance to ego.
        ordered = self._sort_by_distance(ordered, transform)
        for type_idx, obj in ordered:
            poly = self._lane_polyline_feature(obj, type_idx, transform)
            if poly is None:
                continue
            map_items.append(poly)
            # TL status applies to lane connectors (traffic-controlled), keyed by id.
            tl_ids.append(self._lookup_tl(obj, type_idx, tl_status))
            if len(map_items) >= self._cfg.max_map_polylines:
                break

        map_polylines, map_mask = pad_or_truncate(
            map_items,
            self._cfg.max_map_polylines,
            (self._cfg.map_polyline_points, self._cfg.map_feature_dim),
        )
        traffic_light_arr = np.full((self._cfg.max_map_polylines,), -1, dtype=np.int64)
        for i, v in enumerate(tl_ids[: self._cfg.max_map_polylines]):
            traffic_light_arr[i] = v

        # --- crosswalk polygons ---
        cw_items = self._crosswalk_features(crosswalks, transform)
        crosswalk_arr, crosswalk_mask = pad_or_truncate(
            cw_items, self._cfg.max_crosswalks, (self._cfg.map_polyline_points, 2)
        )

        # --- route polyline ---
        route_poly, route_mask = self._build_route_polyline(
            map_api, ego_world_xy, route_roadblock_ids, lanes, transform
        )

        return {
            "map_polylines": map_polylines,
            "map_mask": map_mask,
            "crosswalks": crosswalk_arr,
            "crosswalk_mask": crosswalk_mask,
            "route_polyline": route_poly,
            "route_mask": route_mask,
            "traffic_lights": traffic_light_arr,
        }

    # ---- helpers ----
    def _traffic_light_map(
        self, traffic_lights: List["TrafficLightStatusData"]
    ) -> Dict[int, int]:
        """lane_connector_id -> status int (0=green,1=yellow,2=red,3=unknown).

        devkit: [VERIFIED from devkit source] TrafficLightStatusData has fields
        `.status: TrafficLightStatusType` (IntEnum GREEN=0,YELLOW=1,RED=2,UNKNOWN=3)
        and `.lane_connector_id: int` (maps_datatypes.py).
        """
        out: Dict[int, int] = {}
        for tl in traffic_lights:
            try:
                out[int(tl.lane_connector_id)] = int(tl.status.value)
            except Exception:  # WHY broad: malformed/empty TL records must not crash extraction.
                continue
        return out

    @staticmethod
    def _discrete_path_xy(obj: "MapObject") -> Optional[np.ndarray]:
        """Extract a lane/connector centerline as (M, 2).

        devkit: [VERIFIED from devkit source + repo planners.py] Lane/LaneConnector
        expose `.baseline_path.discrete_path -> List[StateSE2]`, each with `.x/.y`.
        """
        try:
            path = obj.baseline_path.discrete_path
            return np.array([(s.x, s.y) for s in path], dtype=np.float64)
        except Exception:
            return None

    def _lane_polyline_feature(
        self, obj: "MapObject", type_idx: int, transform: EgoFrameTransform
    ) -> Optional[np.ndarray]:
        xy = self._discrete_path_xy(obj)
        if xy is None or len(xy) == 0:
            return None
        xy_ego = transform.world_to_ego(xy)
        xy_res = resample_polyline(xy_ego, self._cfg.map_polyline_points)
        dirs = polyline_directions(xy_res)  # tangents already in ego frame
        feat = np.zeros(
            (self._cfg.map_polyline_points, self._cfg.map_feature_dim), dtype=np.float32
        )
        feat[:, 0:2] = self._norm.norm_pos(xy_res)
        feat[:, 2:4] = dirs
        feat[:, 4 + type_idx] = 1.0
        return feat

    def _crosswalk_features(
        self, crosswalks: List["MapObject"], transform: EgoFrameTransform
    ) -> List[np.ndarray]:
        """Crosswalk polygon exteriors -> ego-frame resampled (S, 2) polylines.

        devkit: [VERIFIED from devkit source] CROSSWALK objects are PolygonMapObject
        with `.polygon` (a shapely Polygon); `.polygon.exterior.coords` gives the ring.
        """
        out: List[np.ndarray] = []
        ordered = self._sort_by_distance(
            [(self._TYPE_CROSSWALK, c) for c in crosswalks], transform
        )
        for _type_idx, cw in ordered:
            try:
                coords = np.array(cw.polygon.exterior.coords, dtype=np.float64)[:, :2]
            except Exception:
                continue
            if len(coords) == 0:
                continue
            xy_ego = transform.world_to_ego(coords)
            xy_res = resample_polyline(xy_ego, self._cfg.map_polyline_points)
            out.append(self._norm.norm_pos(xy_res).astype(np.float32))
            if len(out) >= self._cfg.max_crosswalks:
                break
        return out

    def _sort_by_distance(
        self, typed_objs: List[Tuple[int, "MapObject"]], transform: EgoFrameTransform
    ) -> List[Tuple[int, "MapObject"]]:
        """Order map objects nearest-first by min vertex distance to ego origin."""
        def key(item: Tuple[int, "MapObject"]) -> float:
            type_idx, obj = item
            if type_idx == self._TYPE_CROSSWALK:
                try:
                    xy = np.array(obj.polygon.exterior.coords, dtype=np.float64)[:, :2]
                except Exception:
                    return float("inf")
            else:
                xy = self._discrete_path_xy(obj)
                if xy is None:
                    return float("inf")
            if len(xy) == 0:
                return float("inf")
            ego_pts = transform.world_to_ego(xy)
            return float(np.linalg.norm(ego_pts, axis=1).min())

        return sorted(typed_objs, key=key)

    def _lookup_tl(
        self, obj: "MapObject", type_idx: int, tl_status: Dict[int, int]
    ) -> int:
        """Per-polyline traffic-light status id, or -1 if none applies.

        Traffic lights are keyed by lane_connector_id, so only LANE_CONNECTOR
        polylines can be controlled. We match the connector's own id.
        """
        if type_idx != self._TYPE_LANE_CONNECTOR:
            return -1
        try:
            return tl_status.get(int(obj.id), -1)
        except Exception:
            return -1

    def _build_route_polyline(
        self,
        map_api: "AbstractMap",
        ego_world_xy: Tuple[float, float],
        route_roadblock_ids: List[str],
        proximal_lanes: List["MapObject"],
        transform: EgoFrameTransform,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """On-route lane sequence -> (R, 4) [x, y, dir_x, dir_y] ego-frame.

        Algorithm (reuses planners.py RouteMapBC._build_route patterns):
          1. From route_roadblock_ids, resolve roadblock objects and take their
             interior lanes. [UNVERIFIED — confirm in env] get_map_object(rb_id,
             SemanticMapLayer.ROADBLOCK).interior_edges — roadblock vs roadblock-
             connector layer choice is the most likely version-skew point.
          2. Pick the lane in the FIRST on-route roadblock nearest the ego, then
             chain successor lanes (outgoing_edges) staying on-route, capped by
             route_reach_m / route_max_chain. [VERIFIED] outgoing_edges & baseline_path.
          3. Resample the concatenated centerline to R points; compute tangents.
        If no route ids resolve, fall back to the nearest proximal lane's centerline
        chained forward (so the route tensor is never empty for an on-road ego).
        """
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # devkit-only

        if self._cfg.route_mode == "corridor":
            route_world = self._resolve_route_corridor(
                map_api, route_roadblock_ids, ego_world_xy, SemanticMapLayer
            )
            if route_world is None:  # corridor unresolved: old fallback path
                route_world = self._resolve_route_centerline(
                    map_api, route_roadblock_ids, ego_world_xy, proximal_lanes,
                    SemanticMapLayer,
                )
        else:
            route_world = self._resolve_route_centerline(
                map_api, route_roadblock_ids, ego_world_xy, proximal_lanes,
                SemanticMapLayer,
            )
        R = self._cfg.route_points
        if route_world is None or len(route_world) < 2:
            return (
                np.zeros((R, 4), dtype=np.float32),
                np.zeros((R,), dtype=bool),
            )
        route_ego = transform.world_to_ego(route_world)
        route_res = resample_polyline(route_ego, R)
        dirs = polyline_directions(route_res)
        out = np.zeros((R, 4), dtype=np.float32)
        out[:, 0:2] = self._norm.norm_pos(route_res)
        out[:, 2:4] = dirs
        return out, np.ones((R,), dtype=bool)

    def _resolve_route_corridor(
        self,
        map_api: "AbstractMap",
        route_roadblock_ids: List[str],
        ego_world_xy: Tuple[float, float],
        SemanticMapLayer,
    ) -> Optional[np.ndarray]:
        """Roadblock-level corridor sweep (M, 2) world frame, or None.

        Per route roadblock (IN ORDER), resample every interior lane baseline
        to a fixed count and average them: the sweep follows the corridor and
        the turn-by-turn intent without committing to any lane. The arm of a
        junction is revealed by the roadblock SEQUENCE, which is legitimate
        navigation input; the lane-level path is not.
        """
        pts = []
        for rb_id in [str(i) for i in (route_roadblock_ids or [])]:
            lanes = self._roadblock_interior_lanes(map_api, rb_id, SemanticMapLayer)
            paths = [self._discrete_path_xy(l) for l in lanes]
            paths = [q for q in paths if q is not None and len(q) >= 2]
            if not paths:
                continue
            res = np.stack([resample_polyline(q, 5) for q in paths])
            pts.append(res.mean(axis=0))
        if not pts:
            return None
        corridor = np.vstack(pts)
        if len(corridor) < 2:
            return None
        ego_xy = np.array(ego_world_xy, dtype=np.float64)
        i0 = int(np.argmin(np.linalg.norm(corridor - ego_xy, axis=1)))
        out = corridor[max(0, i0 - 1):]
        if len(out) < 2:
            out = corridor
        # WHY trim to route_reach_m: untrimmed, the sweep spans the whole
        # route (km scale), blowing past pos_scale_m and starving the next
        # 100 m of resolution after the 40-point resample.
        seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        keep = int(np.searchsorted(cum, self._cfg.route_reach_m)) + 1
        out = out[: max(keep, 2)]
        return out

    def _resolve_route_centerline(
        self,
        map_api: "AbstractMap",
        route_roadblock_ids: List[str],
        ego_world_xy: Tuple[float, float],
        proximal_lanes: List["MapObject"],
        SemanticMapLayer,
    ) -> Optional[np.ndarray]:
        """Return concatenated world-frame route centerline (M, 2), or None."""
        on_route = frozenset(str(i) for i in (route_roadblock_ids or []))
        start_lane = self._select_start_lane(
            map_api, on_route, ego_world_xy, proximal_lanes, SemanticMapLayer
        )
        if start_lane is None:
            return None

        ego_xy = np.array(ego_world_xy, dtype=np.float64)
        cl = self._discrete_path_xy(start_lane)
        if cl is None or len(cl) == 0:
            return None
        # Trim the behind-ego portion (start at nearest centerline index).
        i0 = int(np.argmin(np.linalg.norm(cl - ego_xy, axis=1)))
        route = cl[i0:]
        total = self._arc_length(route)

        current = start_lane
        for _ in range(self._cfg.route_max_chain):
            if total >= self._cfg.route_reach_m:
                break
            succ = self._select_route_successor(current, on_route)
            if succ is None:
                break
            spts = self._discrete_path_xy(succ)
            if spts is None or len(spts) == 0:
                break
            route = np.vstack([route, spts])
            total += self._arc_length(spts)
            current = succ
        return route

    def _select_start_lane(
        self, map_api, on_route, ego_world_xy, proximal_lanes, SemanticMapLayer
    ) -> Optional["MapObject"]:
        """Pick the on-route lane nearest the ego, else nearest proximal lane.

        [UNVERIFIED — confirm in env]: resolving a roadblock id to its interior lanes
        via get_map_object(rb_id, SemanticMapLayer.ROADBLOCK).interior_edges. Some
        map versions store route ids as ROADBLOCK_CONNECTOR; the smoke test prints
        how many route ids resolve so this can be confirmed on mini in <15 min.
        """
        ego_xy = np.array(ego_world_xy, dtype=np.float64)
        candidates: List["MapObject"] = []
        for rb_id in on_route:
            lanes = self._roadblock_interior_lanes(map_api, rb_id, SemanticMapLayer)
            candidates.extend(lanes)
        # Fall back to proximal lanes if the route ids did not resolve to lanes.
        pool = candidates if candidates else list(proximal_lanes)
        best, best_d = None, float("inf")
        for lane in pool:
            cl = self._discrete_path_xy(lane)
            if cl is None or len(cl) == 0:
                continue
            d = float(np.linalg.norm(cl - ego_xy, axis=1).min())
            if d < best_d:
                best_d, best = d, lane
        return best

    @staticmethod
    def _roadblock_interior_lanes(map_api, rb_id, SemanticMapLayer) -> List["MapObject"]:
        """Resolve a roadblock id to its interior lane objects (best-effort).

        [UNVERIFIED — confirm in env] get_map_object + .interior_edges.
        """
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                rb = map_api.get_map_object(str(rb_id), layer)
                if rb is not None:
                    return list(rb.interior_edges)
            except Exception:
                continue
        return []

    def _select_route_successor(
        self, lane: "MapObject", on_route: frozenset
    ) -> Optional["MapObject"]:
        """Choose the successor staying on-route; else the straightest successor.

        Mirrors RoadblockRouteMapBCPlanner._select_successor (planners.py): prefer an
        on-route outgoing edge, else fall back to heading alignment with last tangent.
        [VERIFIED] outgoing_edges, get_roadblock_id(), id, baseline_path.discrete_path.
        """
        try:
            successors = list(lane.outgoing_edges)
        except Exception:
            return None
        if not successors:
            return None

        def is_on_route(s: "MapObject") -> bool:
            if not on_route:
                return False
            try:
                if str(s.get_roadblock_id()) in on_route:
                    return True
            except Exception:
                pass
            try:
                return str(s.id) in on_route
            except Exception:
                return False

        on = [s for s in successors if is_on_route(s)]
        pool = on if on else successors

        # straightest: align entry tangent with the lane's own exit tangent.
        try:
            cl = self._discrete_path_xy(lane)
            last_dir = cl[-1] - cl[-2] if cl is not None and len(cl) > 1 else np.array([1.0, 0.0])
            last_dir = last_dir / (np.linalg.norm(last_dir) + 1e-8)
        except Exception:
            last_dir = np.array([1.0, 0.0])

        best, best_cos = None, -2.0
        for s in pool:
            spts = self._discrete_path_xy(s)
            if spts is None or len(spts) < 2:
                continue
            tan = spts[1] - spts[0]
            tan = tan / (np.linalg.norm(tan) + 1e-8)
            cos_a = float(np.dot(tan, last_dir))
            if cos_a > best_cos:
                best_cos, best = cos_a, s
        return best

    @staticmethod
    def _arc_length(pts: np.ndarray) -> float:
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ===========================================================================
# Orchestrator
# ===========================================================================


class _IterationTimeout(Exception):
    """Raised when a single scenario iteration exceeds its wall-clock budget."""


def _timeout_guard(seconds: int):
    """Context-manager-ish via signal.alarm; main-thread only (Sequential
    worker). WHY: a single pathological scenario can hang inside the devkit
    (map queries deadlock / infinite loop), and a hang is NOT an exception, so
    try/except cannot catch it. SIGALRM converts the hang into a catchable
    TimeoutError so extract_scenario skips it instead of stalling an 8h job
    (observed: job 7626090 task 14 hung ~20h on one scenario after 15 shards)."""
    import signal

    class _G:
        def __enter__(self):
            self._old = signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(seconds)
            return self

        def __exit__(self, *a):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)

        @staticmethod
        def _raise(signum, frame):
            raise _IterationTimeout(f"iteration exceeded {seconds}s")

    return _G()


class SceneFeatureExtractor:
    """Walks a nuPlan scenario, builds per-sample feature dicts, saves .pt shards.

    Single responsibility: ORCHESTRATION. It owns no feature math — it delegates to
    the builders and the Normalizer (Dependency Inversion: it depends on the builder
    interfaces, not their internals).
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self._cfg = config or FeatureConfig()
        self._norm = Normalizer(self._cfg)
        self._ego_builder = EgoFeatureBuilder(self._cfg, self._norm)
        self._agent_builder = AgentFeatureBuilder(self._cfg, self._norm)
        self._map_builder = MapFeatureBuilder(self._cfg, self._norm)

    @property
    def config(self) -> FeatureConfig:
        return self._cfg

    # ---- per-sample extraction ----
    def extract_sample(self, scenario: "AbstractScenario", iteration: int) -> Dict[str, np.ndarray]:
        """Build the full feature dict for one (scenario, iteration) sample.

        Requires iteration >= history_steps-1 so a full 2 s history exists.

        devkit access (all [VERIFIED from devkit source] AbstractScenario unless noted):
            scenario.get_ego_state_at_iteration(i) -> EgoState
            scenario.get_ego_past_trajectory(i, time_horizon, num_samples) -> Gen[EgoState]
            scenario.get_tracked_objects_at_iteration(i) -> DetectionsTracks
            scenario.get_past_tracked_objects(i, time_horizon, num_samples) -> Gen[DetectionsTracks]
            scenario.get_traffic_light_status_at_iteration(i) -> Gen[TrafficLightStatusData]
            scenario.get_route_roadblock_ids() -> List[str]
            scenario.map_api -> AbstractMap
        """
        cfg = self._cfg
        n_hist = cfg.history_steps
        horizon_s = (n_hist - 1) * cfg.dt  # past window excluding current step

        # --- gather raw histories from the scenario (offline adapter) ---
        # WHY per-iteration queries, not get_*_past_trajectory: the simulator
        # hands the planner exact consecutive per-iteration states; the
        # devkit's batched past-sampling resamples timestamps differently and
        # broke the train/serve parity gate (max feature diff 1.9). Offline
        # MUST query the same way serving receives.
        ego_states, per_step_tracks = [], []
        for it in range(max(0, iteration - n_hist + 1), iteration + 1):
            ego_states.append(scenario.get_ego_state_at_iteration(it))
            per_step_tracks.append(scenario.get_tracked_objects_at_iteration(it))
        if cfg.perturb_prob > 0.0:
            ego_states = self._maybe_perturb_history(
                ego_states, str(scenario.token), int(iteration)
            )
        traffic_lights = list(
            scenario.get_traffic_light_status_at_iteration(iteration)  # [VERIFIED]
        )
        try:
            route_ids = list(scenario.get_route_roadblock_ids())  # [VERIFIED]
        except Exception:
            route_ids = []

        feats, transform, current_ego = self.build_input_features(
            ego_states,
            per_step_tracks,
            traffic_lights,
            route_ids,
            scenario.map_api,
        )
        ego, agents, agent_mask = feats["ego"], feats["agents"], feats["agent_mask"]

        # --- ego future (training target), ego frame at t=0 ---
        # WHY raise on short futures: extract_scenario catches per-iteration
        # exceptions and logs a skip, so samples near scenario end are dropped
        # instead of carrying a padded/garbage label.
        future_states = list(
            scenario.get_ego_future_trajectory(  # [VERIFIED abstract_scenario.py:318]
                iteration,
                time_horizon=cfg.future_horizon_s,
                num_samples=cfg.future_steps,
            )
        )
        if len(future_states) < cfg.future_steps:
            raise ValueError(
                f"incomplete future: {len(future_states)}/{cfg.future_steps} states"
            )
        # WHY [-future_steps:]: defensive against devkit variants that prepend
        # the current state; the horizon endpoint is what must be preserved.
        future_states = future_states[-cfg.future_steps :]
        fut_xy = transform.world_to_ego(
            np.array([[st.rear_axle.x, st.rear_axle.y] for st in future_states])
        )
        fut_h = (
            np.array([st.rear_axle.heading for st in future_states])
            - current_ego.rear_axle.heading
        )
        fut_h = np.arctan2(np.sin(fut_h), np.cos(fut_h))  # wrap to [-pi, pi]
        ego_future = np.concatenate(
            [fut_xy, fut_h[:, None]], axis=1
        ).astype(np.float32)  # (future_steps, 3)

        sample = {**feats, "ego_future": ego_future}
        self._assert_sample_consistency(sample)
        # WHY identifiers: F4 scoring, per-type validation gates, and joining
        # offline scores to closed-loop runs all need to know which scenario a
        # sample came from; anonymous tensors cannot be audited or stratified.
        sample["scenario_token"] = str(scenario.token)
        sample["scenario_type"] = str(scenario.scenario_type)
        sample["log_name"] = str(scenario.log_name)
        sample["iteration"] = int(iteration)
        return sample

    def build_input_features(
        self,
        ego_states: list,
        per_step_tracks: list,
        traffic_lights: list,
        route_ids: list,
        map_api,
    ):
        """Shared feature core: raw histories -> model input tensors.

        WHY one code path: train/serve feature skew is the classic silent
        killer in learned planning stacks. The offline extractor (scenario
        adapter, extract_sample) and the sim-time planner (PlannerInput
        adapter, nuplan/serving/policy_planner.py) BOTH call this function,
        so parity holds by construction; nuplan/serving/parity_check.py
        verifies it numerically on real scenarios.

        Args are oldest->newest; the last entry of each history is "now".
        Returns (feature dict WITHOUT label/identifiers, transform, current_ego).
        """
        cfg = self._cfg
        n_hist = cfg.history_steps
        current_ego = ego_states[-1]
        transform = EgoFrameTransform(
            current_ego.rear_axle.x, current_ego.rear_axle.y, current_ego.rear_axle.heading
        )
        ego_states = self._normalize_history_length(list(ego_states), n_hist)
        ego = self._ego_builder.build(ego_states, transform)
        per_step_tracks = self._normalize_history_length(list(per_step_tracks), n_hist)
        agents, agent_mask = self._agent_builder.build(per_step_tracks, transform)
        map_feats = self._map_builder.build(
            map_api,
            (current_ego.rear_axle.x, current_ego.rear_axle.y),
            list(route_ids),
            list(traffic_lights),
            transform,
        )
        feats = {"ego": ego, "agents": agents, "agent_mask": agent_mask, **map_feats}
        return feats, transform, current_ego

    def _maybe_perturb_history(self, ego_states: list, token: str, iteration: int) -> list:
        """Recovery augmentation: blend the tail of the ego history toward a
        laterally/heading-perturbed current pose. Deterministic per
        (token, iteration) so extraction is resume-safe and reproducible.
        Serving NEVER perturbs; this runs only in the offline adapter."""
        cfg = self._cfg
        rng = np.random.default_rng(abs(hash((token, iteration))) % (2**31))
        if rng.random() >= cfg.perturb_prob:
            return ego_states
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2

        lat = float(np.clip(rng.normal(0.0, cfg.perturb_lat_std_m),
                            -cfg.perturb_lat_max_m, cfg.perturb_lat_max_m))
        dh = float(np.clip(rng.normal(0.0, cfg.perturb_head_std_rad),
                           -cfg.perturb_head_max_rad, cfg.perturb_head_max_rad))
        h_cur = ego_states[-1].rear_axle.heading
        ox, oy = -np.sin(h_cur) * lat, np.cos(h_cur) * lat
        m = min(cfg.perturb_blend_steps, len(ego_states))
        out = list(ego_states)
        for j in range(m):
            idx = len(out) - m + j
            frac = (j + 1) / m              # 1/m ... 1.0 (current pose fully shifted)
            st = out[idx]
            pose = StateSE2(
                st.rear_axle.x + ox * frac,
                st.rear_axle.y + oy * frac,
                st.rear_axle.heading + dh * frac,
            )
            out[idx] = EgoState.build_from_rear_axle(
                rear_axle_pose=pose,
                rear_axle_velocity_2d=st.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=st.dynamic_car_state.rear_axle_acceleration_2d,
                tire_steering_angle=st.tire_steering_angle,
                time_point=st.time_point,
                vehicle_parameters=st.car_footprint.vehicle_parameters,
            )
        return out

    @staticmethod
    def _normalize_history_length(items: list, n: int) -> list:
        """Pad (repeat-oldest) or truncate a history list to exactly n, oldest->newest.

        WHY repeat-oldest at scenario start: early iterations have < n_hist of past;
        repeating the oldest available frame keeps shapes fixed without injecting a
        fake jump. The ego/agent masks/values still reflect the real frame content.
        """
        if len(items) >= n:
            return items[-n:]
        if not items:
            return items
        pad = [items[0]] * (n - len(items))
        return pad + items

    # ---- shape/mask self-consistency (used by smoke gate too) ----
    def _assert_sample_consistency(self, sample: Dict[str, np.ndarray]) -> None:
        cfg = self._cfg
        T, N = cfg.history_steps, cfg.max_agents
        assert sample["ego"].shape == (T, cfg.ego_feature_dim), sample["ego"].shape
        assert sample["agents"].shape == (N, T, cfg.agent_feature_dim), sample["agents"].shape
        assert sample["agent_mask"].shape == (N, T), sample["agent_mask"].shape
        assert sample["map_polylines"].shape == (
            cfg.max_map_polylines, cfg.map_polyline_points, cfg.map_feature_dim,
        )
        assert sample["map_mask"].shape == (cfg.max_map_polylines,)
        assert sample["route_polyline"].shape == (cfg.route_points, 4)
        assert sample["traffic_lights"].shape == (cfg.max_map_polylines,)
        assert sample["ego_future"].shape == (cfg.future_steps, 3), sample["ego_future"].shape
        # finiteness: NaNs/Infs here would silently poison training.
        for key in ("ego", "agents", "map_polylines", "route_polyline", "crosswalks", "ego_future"):
            assert np.all(np.isfinite(sample[key])), f"non-finite values in {key}"
        # masked agent slots must be all-zero feature rows (no leaked data).
        invalid = ~sample["agent_mask"]
        assert not np.any(sample["agents"][invalid].ravel()), "data leaked into masked agent slot"

    # ---- scenario -> shard ----
    def extract_scenario(
        self, scenario: "AbstractScenario", stride: int = 1, max_samples: Optional[int] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Extract every valid iteration of a scenario (with full history)."""
        n_iter = scenario.get_number_of_iterations()  # [VERIFIED]
        start = self._cfg.history_steps - 1
        budget = self._cfg.iteration_timeout_s
        samples: List[Dict[str, np.ndarray]] = []
        for it in range(start, n_iter, stride):
            try:
                if budget > 0:
                    with _timeout_guard(budget):
                        samples.append(self.extract_sample(scenario, it))
                else:
                    samples.append(self.extract_sample(scenario, it))
            except (Exception, _IterationTimeout) as exc:  # bad/hung frame -> skip
                logger.warning("skip %s it=%d: %s", getattr(scenario, "token", "?"), it, exc)
            if max_samples is not None and len(samples) >= max_samples:
                break
        return samples

    def save_shard(self, samples: List[Dict[str, np.ndarray]], out_path: Path) -> None:
        """Serialize a list of samples to a single .pt shard (tensors)."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to save shards")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tensorized = [
            {
                # WHY isinstance gate: identifier fields are str/int, not arrays.
                k: torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v
                for k, v in s.items()
            }
            for s in samples
        ]
        # WHY include config: shards are self-describing — a downstream loader can
        # validate the representation it was built with rather than guessing.
        torch.save(
            {"samples": tensorized, "config": dataclasses.asdict(self._cfg)},
            str(out_path),
        )

    def run(
        self,
        scenarios: List["AbstractScenario"],
        out_dir: Path,
        scenarios_per_shard: int = 8,
        stride: int = 1,
    ) -> List[Path]:
        """Extract a list of scenarios into .pt shards under out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        buf: List[Dict[str, np.ndarray]] = []
        for i, scn in enumerate(scenarios):
            # WHY resume: derive the shard index from i (i // scenarios_per_shard)
            # instead of a running counter, so a re-submitted job SKIPS scenarios whose
            # shard already exists. The old running counter advanced only on flush, so a
            # `continue` would mislabel/overwrite earlier shards. Scenario order is
            # deterministic (sorted DBs at the CLI, ScenarioFilter shuffle=False,
            # Sequential worker), so shard_for_i maps to the same scenarios every run.
            shard_for_i = i // scenarios_per_shard
            shard_path = out_dir / f"scene_shard_{shard_for_i:05d}.pt"
            if shard_path.exists():
                continue
            buf.extend(self.extract_scenario(scn, stride=stride))
            if (i + 1) % scenarios_per_shard == 0 and buf:
                path = out_dir / f"scene_shard_{shard_for_i:05d}.pt"
                self.save_shard(buf, path)
                written.append(path)
                logger.info("wrote %s (%d samples)", path, len(buf))
                buf = []
        if buf:
            path = out_dir / f"scene_shard_{shard_for_i:05d}.pt"
            self.save_shard(buf, path)
            written.append(path)
            logger.info("wrote %s (%d samples)", path, len(buf))
        return written


# ===========================================================================
# Smoke mode + CLI
# ===========================================================================

def _build_mini_scenarios(
    data_root: str,
    map_root: str,
    limit: int,
    log_names: Optional[List[str]] = None,
    num_scenarios_per_type: Optional[int] = None,
    scenario_types: Optional[List[str]] = None,
) -> List["AbstractScenario"]:
    """Construct nuPlan scenarios from a data root (used by --smoke on MINI and the full run).

    [UNVERIFIED — confirm in env] The scenario-builder construction below uses the
    NuPlanScenarioBuilder + ScenarioFilter Hydra-free path. Builder/filter kwargs are
    the single most version-sensitive part of the devkit and MUST be confirmed by
    running --smoke on mini (see docs/frontier/F0_IMPLEMENTATION.md). We keep this in
    its own function so a version fix is a one-spot edit.

    Args:
        data_root:  parent directory containing the .db files (nuPlan logs).
        map_root:   nuPlan maps directory.
        limit:      hard cap on total scenarios returned (use 10_000_000 for "all").
        log_names:  if not None, restrict to these log names (DB stem, no .db extension).
                    Used by the SLURM array path to fan work across tasks.
    """
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )
    scenario_filter = ScenarioFilter(
        scenario_types=scenario_types,
        scenario_tokens=None,
        # WHY log_names: ScenarioFilter already supports per-log filtering.
        # Passing a list here restricts the builder to those specific DB files,
        # which is how the SLURM array path assigns disjoint work to each task.
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=num_scenarios_per_type,
        limit_total_scenarios=limit,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def run_smoke(data_root: str, map_root: str, n_scenarios: int = 5) -> None:
    """Smoke gate: extract a few MINI scenarios, print shapes, assert consistency,
    and confirm the ego-frame transform round-trips. Proves the pipeline produces
    correctly-shaped, finite tensors before any large run."""
    extractor = SceneFeatureExtractor()
    cfg = extractor.config

    # 1) Pure-geometry round-trip check (no devkit) — the load-bearing invariant.
    _assert_transform_roundtrip()
    print("[smoke] ego-frame transform round-trip OK")

    # 2) Build mini scenarios and extract one sample from each.
    scenarios = _build_mini_scenarios(data_root, map_root, n_scenarios)
    print(f"[smoke] loaded {len(scenarios)} MINI scenarios")
    if not scenarios:
        raise RuntimeError("no scenarios built — check data_root/map_root and devkit version")

    n_route_resolved = 0
    for si, scn in enumerate(scenarios[:n_scenarios]):
        it = cfg.history_steps - 1  # first iteration with a full history
        sample = extractor.extract_sample(scn, it)
        print(f"\n[smoke] scenario {si} token={getattr(scn, 'token', '?')} iter={it}")
        for k, v in sample.items():
            print(f"    {k:16s} shape={tuple(v.shape)} dtype={v.dtype}")
        # internal consistency (raises on failure)
        extractor._assert_sample_consistency(sample)
        if bool(sample["route_mask"].any()):
            n_route_resolved += 1
        n_agents = int(sample["agent_mask"].any(axis=1).sum())
        n_map = int(sample["map_mask"].sum())
        print(f"    valid agents={n_agents}/{cfg.max_agents}  map polylines={n_map}/{cfg.max_map_polylines}")

    print(
        f"\n[smoke] route resolved for {n_route_resolved}/{len(scenarios[:n_scenarios])} "
        f"scenarios (if 0, confirm roadblock-id layer — see F0_IMPLEMENTATION.md)"
    )
    print("[smoke] PASS — shapes/masks consistent, tensors finite")


def _assert_transform_roundtrip() -> None:
    """world->ego->world identity over a grid of poses/points (numpy only)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        ref = rng.uniform(-100, 100, size=3)
        ref[2] = rng.uniform(-np.pi, np.pi)
        tf = EgoFrameTransform(*ref)
        pts = rng.uniform(-100, 100, size=(10, 2))
        back = tf.ego_to_world(tf.world_to_ego(pts))
        assert np.allclose(back, pts, atol=1e-9), "ego-frame transform not invertible"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="nuPlan vectorized scene-feature extractor (F0)")
    parser.add_argument("--smoke", action="store_true", help="run smoke test on nuPlan MINI")
    parser.add_argument("--n-scenarios", type=int, default=5, help="smoke: # scenarios to probe")
    parser.add_argument("--data-root", type=str, default=None, help="nuPlan DB root (/scratch/...)")
    parser.add_argument("--map-root", type=str, default=None, help="nuPlan maps root")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/scratch/$USER/av-policy-lab/features/f0_balanced",
        help="output dir for .pt shards (default a /scratch path)",
    )
    parser.add_argument("--scenarios-per-shard", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="iteration stride within a scenario")
    parser.add_argument("--limit", type=int, default=None, help="full run: cap total scenarios")
    parser.add_argument("--num-scenarios-per-type", type=int, default=None, help="balanced sampling: max scenarios per nuPlan scenario type")
    # ── SLURM array parallelisation ─────────────────────────────────────────
    # WHY: the nuPlan map-query bottleneck makes single-job extraction take
    # ~90 h for mini on one CPU.  Splitting across N SLURM array tasks (one
    # chunk of DB files per task) drops wall-clock to ~1-2 h on the short
    # partition.  Each task writes to its own sub-directory; a DataLoader can
    # glob all sub-dirs, or run nuplan/slurm/merge_shards.py post-extraction.
    parser.add_argument(
        "--scenario-types",
        type=str,
        default=None,
        help="Comma-separated scenario types (enrichment task for rare types).",
    )
    parser.add_argument(
        "--perturb-prob",
        type=float,
        default=None,
        help="Recovery-augmentation probability (v3 training extraction: 0.5).",
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help=(
            "SLURM array task ID (0-indexed). Set automatically when invoked via "
            "extract_features_array.sbatch ($SLURM_ARRAY_TASK_ID). "
            "Must be combined with --num-array-tasks."
        ),
    )
    parser.add_argument(
        "--num-array-tasks",
        type=int,
        default=None,
        help="Total number of SLURM array tasks. DB files are divided into equal chunks.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke:
        if not (args.data_root and args.map_root):
            parser.error("--smoke requires --data-root and --map-root pointing at nuPlan MINI")
        run_smoke(args.data_root, args.map_root, args.n_scenarios)
        return

    # Full extraction path.
    if not (args.data_root and args.map_root):
        parser.error("full run requires --data-root and --map-root")

    # ── Resolve which DB files this task processes ───────────────────────────
    import math as _math
    data_root_path = Path(args.data_root)
    all_dbs = sorted(data_root_path.glob("*.db"))
    if not all_dbs:
        raise FileNotFoundError(
            f"No .db files found under {args.data_root}. "
            "Check --data-root points at the directory containing nuPlan .db log files."
        )

    log_names: Optional[List[str]] = None  # None = all logs (single-job path)

    if args.array_task_id is not None and args.num_array_tasks is not None:
        # SLURM array path: assign a disjoint chunk of DB files to this task.
        if not (0 <= args.array_task_id < args.num_array_tasks):
            raise ValueError(
                f"--array-task-id {args.array_task_id} must be in "
                f"[0, {args.num_array_tasks - 1}]"
            )
        chunk_size = _math.ceil(len(all_dbs) / args.num_array_tasks)
        start = args.array_task_id * chunk_size
        end = min(start + chunk_size, len(all_dbs))
        task_dbs = all_dbs[start:end]
        # ScenarioFilter accepts log names WITHOUT the .db extension.
        log_names = [p.stem for p in task_dbs]
        logger.info(
            "Array task %d/%d — processing %d logs: %s … %s",
            args.array_task_id, args.num_array_tasks,
            len(task_dbs), task_dbs[0].name, task_dbs[-1].name,
        )
    elif (args.array_task_id is None) != (args.num_array_tasks is None):
        parser.error("--array-task-id and --num-array-tasks must be used together")
    else:
        logger.info("Single-job path — processing all %d DB files", len(all_dbs))

    # ── Output directory: task-specific sub-dir to avoid shard-name collisions ──
    out_dir = Path(args.out_dir.replace("$USER", _safe_user()))
    if args.array_task_id is not None:
        # e.g.  .../f0/task_0003/  — keeps shards from each task separate.
        # WHY sub-dir not flat: concurrent writes to the same directory with
        # identical shard indices (scene_shard_00000.pt) would silently clobber
        # each other.  Sub-dirs are trivially merged by a glob at training time.
        out_dir = out_dir / f"task_{args.array_task_id:04d}"

    # WHY dataclasses.replace: FeatureConfig is frozen by design (a config
    # mutated mid-run cannot be trusted in the shard header).
    feat_cfg = FeatureConfig()
    if args.perturb_prob is not None:
        feat_cfg = dataclasses.replace(feat_cfg, perturb_prob=args.perturb_prob)
    extractor = SceneFeatureExtractor(feat_cfg)
    scenarios = _build_mini_scenarios(
        args.data_root,
        args.map_root,
        args.limit or 10_000_000,
        log_names=log_names,
        num_scenarios_per_type=args.num_scenarios_per_type,
        scenario_types=(args.scenario_types.split(",") if args.scenario_types else None),
    )
    if not scenarios:
        logger.warning("No scenarios found for task %s — exiting cleanly.", args.array_task_id)
        return

    logger.info("Extracting %d scenarios → %s", len(scenarios), out_dir)
    paths = extractor.run(
        scenarios, out_dir,
        scenarios_per_shard=args.scenarios_per_shard, stride=args.stride,
    )
    logger.info("done: %d shards under %s", len(paths), out_dir)


def _safe_user() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return "user"


if __name__ == "__main__":
    main()
