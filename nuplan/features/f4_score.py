"""F4: per-scenario interaction-multimodality score (shard-side components).

Spec: docs/frontier/F4_SPEC.md (checker-verified 2026-06-10). This module
computes everything derivable from F0 v2 shards: S_inter, G_stop, the
shard-geometry S_branch/S_lane sanity variant, and the combination rule.
The PRIMARY S_branch/S_lane come from the nuPlan map API (lane connectivity
is not in shards) and are joined in by nuplan/slurm/score_f4.py.

Scoring granularity: one score per scenario, computed on the sample with the
lowest iteration (closed-loop starts there; later iterations are the same
scene partially resolved).

WHY ego's nominal path is the route polyline, never ego_future: ego_future
encodes how the expert RESOLVED each yield-or-go ambiguity; using it leaks
the imitation label into the moderator (F4_SPEC sec. 1). The route is an
input both heads condition on and is symmetric across all four cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# fixed constants (F4_SPEC; every one of these enters the sensitivity grid)
PRET_CENTER_S = 2.5
PRET_WIDTH_S = 1.5
TAU_CENTER_S = 1.5
TAU_WIDTH_S = 1.0
ALONG_PATH_CUTOFF_M = 30.0   # ~ center + 3 sigma at v0 >= 6 m/s
LATERAL_GATE_M = 3.0         # agent must be on the corridor to count as lead
PED_SPEED_MS = 0.5
PED_CROSSWALK_DIST_M = 3.0
PED_OVERRIDE = 0.5
EGO_ROLLOUT_S = 5.0
AGENT_ROLLOUT_S = 5.0
ROLLOUT_DT_S = 0.25
TOP_K_AGENTS = 3
RED_STATE = 2                # GREEN=0 YELLOW=1 RED=2 UNKNOWN=3, -1 none
STOP_LOOKAHEAD_M = 20.0
STOP_SPEED_MS = 1.0
G_STOP_VALUE = 0.25
S_LANE_WEIGHT = 0.5
BRANCH_CAP = 3
LANE_CAP = 2
HIST_DT_S = 0.1              # agents history is 10 Hz


@dataclass
class DenormConfig:
    """Scale factors the shard config dict must confirm (F4_SPEC units shim)."""
    pos_scale_m: float = 120.0
    vel_scale_mps: float = 15.0

    @classmethod
    def from_shard_config(cls, cfg: dict) -> "DenormConfig":
        out = cls(float(cfg["pos_scale_m"]), float(cfg["vel_scale_mps"]))
        # WHY assert: a silently changed normalization upstream would make
        # every meter-threshold below wrong by that factor.
        assert out.pos_scale_m > 1.0 and out.vel_scale_mps > 1.0, cfg
        return out


# ---------- geometry helpers (pure numpy, meters) ----------

def _seg_intersect(p1, p2, p3, p4) -> Optional[float]:
    """Intersection of segments p1p2 and p3p4 -> (u, v) params in [0,1] or None."""
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    diff = p3 - p1
    u = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    v = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
        return (u, v)
    return None


def _path_crossing_times(path_a: np.ndarray, t_a: np.ndarray,
                         path_b: np.ndarray, t_b: np.ndarray) -> Optional[tuple]:
    """First crossing of two timed polylines -> (t at crossing on a, on b)."""
    for i in range(len(path_a) - 1):
        for j in range(len(path_b) - 1):
            uv = _seg_intersect(path_a[i], path_a[i + 1], path_b[j], path_b[j + 1])
            if uv is not None:
                u, v = uv
                ta = t_a[i] + u * (t_a[i + 1] - t_a[i])
                tb = t_b[j] + v * (t_b[j + 1] - t_b[j])
                return (ta, tb)
    return None


def _arclengths(path: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _roll_along(path: np.ndarray, speed: float, horizon_s: float,
                dt: float) -> tuple:
    """Constant-speed rollout along a polyline from its start -> (pts, times)."""
    s = _arclengths(path)
    t = np.arange(dt, horizon_s + 1e-9, dt)
    want = np.minimum(speed * t, s[-1])
    pts = np.stack([np.interp(want, s, path[:, 0]), np.interp(want, s, path[:, 1])], axis=1)
    return pts, t


def _project_along(path: np.ndarray, point: np.ndarray) -> tuple:
    """(along-path arclength, lateral distance) of point's projection."""
    best = (math.inf, math.inf)
    s = _arclengths(path)
    for i in range(len(path) - 1):
        d = path[i + 1] - path[i]
        L2 = float(d @ d)
        if L2 < 1e-12:
            continue
        u = float(np.clip((point - path[i]) @ d / L2, 0.0, 1.0))
        proj = path[i] + u * d
        lat = float(np.linalg.norm(point - proj))
        if lat < best[1]:
            best = (float(s[i] + u * math.sqrt(L2)), lat)
    return best


# ---------- sample denormalization ----------

def denorm_sample(sample: Dict, dn: DenormConfig) -> Dict[str, np.ndarray]:
    """Extract meter-space views of the fields F4 needs. Tensors -> numpy."""
    g = lambda k: np.asarray(sample[k], dtype=np.float64)
    out = {
        "ego": g("ego"),
        "agents": g("agents"),
        "agent_mask": np.asarray(sample["agent_mask"], dtype=bool),
        "route": g("route_polyline")[:, :2] * dn.pos_scale_m,
        "route_mask": np.asarray(sample["route_mask"], dtype=bool),
        "crosswalks": g("crosswalks") * dn.pos_scale_m,
        "crosswalk_mask": np.asarray(sample["crosswalk_mask"], dtype=bool),
        "map_polylines": g("map_polylines"),
        "map_mask": np.asarray(sample["map_mask"], dtype=bool),
        "traffic_lights": np.asarray(sample["traffic_lights"], dtype=np.int64),
    }
    out["v0"] = float(np.hypot(out["ego"][-1, 4], out["ego"][-1, 5]) * dn.vel_scale_mps)
    return out


def _agent_state(agents_row: np.ndarray, mask_row: np.ndarray,
                 dn: DenormConfig) -> Optional[dict]:
    """Last valid state + heading rate for one agent, in meters/rad/s."""
    valid = np.flatnonzero(mask_row)
    if len(valid) == 0:
        return None
    k = valid[-1]
    x, y = agents_row[k, 0] * dn.pos_scale_m, agents_row[k, 1] * dn.pos_scale_m
    h = math.atan2(agents_row[k, 2], agents_row[k, 3])
    v = float(np.hypot(agents_row[k, 4], agents_row[k, 5]) * dn.vel_scale_mps)
    hr = 0.0
    if len(valid) >= 2:
        k2 = valid[-2]
        h2 = math.atan2(agents_row[k2, 2], agents_row[k2, 3])
        dh = math.atan2(math.sin(h - h2), math.cos(h - h2))
        hr = dh / (HIST_DT_S * (k - k2))
    onehot = agents_row[k, 6:9]
    return {"xy": np.array([x, y]), "h": h, "v": v, "hr": hr,
            "is_ped": bool(np.argmax(onehot) == 1) if onehot.any() else False}


def _agent_rollout(st: dict, horizon_s: float, dt: float) -> tuple:
    """Constant turn-rate rollout (F4_SPEC: CV over 8 s is indefensible)."""
    t = np.arange(dt, horizon_s + 1e-9, dt)
    h = st["h"] + st["hr"] * t
    if abs(st["hr"]) < 1e-6:
        xs = st["xy"][0] + st["v"] * t * math.cos(st["h"])
        ys = st["xy"][1] + st["v"] * t * math.sin(st["h"])
    else:
        # closed-form unicycle integral
        xs = st["xy"][0] + st["v"] / st["hr"] * (np.sin(h) - math.sin(st["h"]))
        ys = st["xy"][1] - st["v"] / st["hr"] * (np.cos(h) - math.cos(st["h"]))
    return np.stack([xs, ys], axis=1), t


# ---------- subscores ----------

def ego_nominal_path(d: Dict) -> Optional[tuple]:
    """Roll ego at v0 along the route polyline. None if route is unusable."""
    if not d["route_mask"].any():
        return None
    route = d["route"][d["route_mask"]]
    if len(route) < 2:
        return None
    speed = max(d["v0"], 1.0)  # a stopped ego still has a nominal forward path
    return _roll_along(route, speed, EGO_ROLLOUT_S, ROLLOUT_DT_S)


def s_inter(d: Dict, dn: DenormConfig) -> float:
    nominal = ego_nominal_path(d)
    if nominal is None:
        return float("nan")
    ego_path, ego_t = nominal
    full_path = np.concatenate([np.zeros((1, 2)), ego_path], axis=0)
    scores = []
    for j in range(d["agents"].shape[0]):
        st = _agent_state(d["agents"][j], d["agent_mask"][j], dn)
        if st is None or np.linalg.norm(st["xy"]) < 1e-6:
            continue
        a_path, a_t = _agent_rollout(st, AGENT_ROLLOUT_S, ROLLOUT_DT_S)
        a_full = np.concatenate([st["xy"][None], a_path], axis=0)
        a_tf = np.concatenate([[0.0], a_t])
        e_tf = np.concatenate([[0.0], ego_t])
        I_j = 0.0
        cross = _path_crossing_times(full_path, e_tf, a_full, a_tf)
        if cross is not None:
            gap = abs(cross[0] - cross[1])  # PrET / time advantage, NOT PET
            I_j = math.exp(-(((gap - PRET_CENTER_S) / PRET_WIDTH_S) ** 2))
        else:
            along, lat = _project_along(full_path, st["xy"])
            if lat < LATERAL_GATE_M and 0.0 < along < ALONG_PATH_CUTOFF_M:
                tau = along / max(d["v0"], 1.0)
                I_j = math.exp(-(((tau - TAU_CENTER_S) / TAU_WIDTH_S) ** 2))
        if st["is_ped"] and st["v"] < PED_SPEED_MS:
            if _ped_near_crossing_crosswalk(st["xy"], d, full_path):
                I_j = max(I_j, PED_OVERRIDE)
        if I_j > 0.0:
            scores.append(I_j)
    # WHY top-k noisy-OR: full noisy-OR saturates under platoons/groups of
    # correlated agents and compresses the upper range the regression needs.
    top = sorted(scores, reverse=True)[:TOP_K_AGENTS]
    out = 1.0
    for i in top:
        out *= (1.0 - i)
    return 1.0 - out


def _ped_near_crossing_crosswalk(xy: np.ndarray, d: Dict,
                                 ego_path: np.ndarray) -> bool:
    dummy_t = np.arange(len(ego_path), dtype=float)
    for c in range(d["crosswalks"].shape[0]):
        if not d["crosswalk_mask"][c]:
            continue
        poly = d["crosswalks"][c]
        if np.min(np.linalg.norm(poly - xy, axis=1)) > PED_CROSSWALK_DIST_M:
            continue
        ct = np.arange(len(poly), dtype=float)
        if _path_crossing_times(ego_path, dummy_t, poly, ct) is not None:
            return True
    return False


def g_stop(d: Dict, dn: DenormConfig) -> float:
    if d["v0"] >= STOP_SPEED_MS:
        return 1.0
    nominal = ego_nominal_path(d)
    if nominal is None:
        return 1.0
    ego_path = np.concatenate([np.zeros((1, 2)), nominal[0]], axis=0)
    s = _arclengths(ego_path)
    near = ego_path[s <= STOP_LOOKAHEAD_M]
    for l in np.flatnonzero(d["map_mask"]):
        if d["traffic_lights"][l] != RED_STATE:
            continue
        pts = d["map_polylines"][l, :, :2] * dn.pos_scale_m
        dmin = np.min(np.linalg.norm(pts[:, None, :] - near[None, :, :], axis=-1))
        if dmin < LATERAL_GATE_M + 0.5:
            return G_STOP_VALUE
    return 1.0


def combine(s_branch: float, s_lane: float, si: float, gs: float) -> float:
    return gs * (1.0 - (1.0 - s_branch) * (1.0 - S_LANE_WEIGHT * s_lane) * (1.0 - si))


def score_sample(sample: Dict, shard_config: dict,
                 s_branch: float = 0.0, s_lane: float = 0.0) -> Dict[str, float]:
    """Shard-side F4. Primary s_branch/s_lane are joined in by the scoring
    script (map API); defaults of 0 give the interaction-only partial score."""
    dn = DenormConfig.from_shard_config(shard_config)
    d = denorm_sample(sample, dn)
    if not d["route_mask"].any():
        # F4_SPEC: route-resolution failures are excluded, never scored 0
        return {"excluded": 1.0, "f4": float("nan"), "s_inter": float("nan"),
                "g_stop": float("nan"), "v0": d["v0"]}
    si = s_inter(d, dn)
    gs = g_stop(d, dn)
    return {"excluded": 0.0, "f4": combine(s_branch, s_lane, si, gs),
            "s_inter": si, "g_stop": gs, "v0": d["v0"]}
