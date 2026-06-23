"""Signal A — ground-truth-conflict convergent validity for F4.

Re-derives interaction conflict at iteration 19 using the SAME geometry as
f4_score.s_inter (ego route rollout at v0, PrET Gaussian, top-k noisy-OR) but
replaces F4's constant-turn-rate agent rollout with each agent's LOGGED future
trajectory (nuPlan get_future_tracked_objects). This isolates one variable:
F4's agent motion model. A positive F4<->a_gt relationship shows F4's ambiguity
is not an artifact of the constant-turn-rate assumption.

REF: geometry ported from av-policy-lab/nuplan/features/f4_score.py (v1.1).
No ego future is used (route is an input, never the imitation label).

Self-test (one high-F4 scene):
    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/signal_a_gt_conflict.py
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from scene_loader import (F4_ITERATION, build_scenarios, index_by_token,
                          load_f4_scores)

# ---- constants identical to f4_score.py v1.1 so a_gt is comparable to s_inter ----
PRET_CENTER_S = 2.5
PRET_WIDTH_S = 1.5
EGO_ROLLOUT_S = 5.0
AGENT_ROLLOUT_S = 5.0
ROLLOUT_DT_S = 0.25
TOP_K_AGENTS = 3


# ---------- geometry (ported from f4_score.py) ----------

def _seg_intersect(p1, p2, p3, p4) -> Optional[Tuple[float, float]]:
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


def _path_crossing_times(path_a, t_a, path_b, t_b) -> Optional[Tuple[float, float]]:
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


def _roll_along(path: np.ndarray, speed: float, horizon_s: float, dt: float):
    s = _arclengths(path)
    t = np.arange(dt, horizon_s + 1e-9, dt)
    want = np.minimum(speed * t, s[-1])
    pts = np.stack([np.interp(want, s, path[:, 0]), np.interp(want, s, path[:, 1])], axis=1)
    return pts, t


# ---------- nuPlan extraction ----------

def reconstruct_route(scenario, max_len_m: float = 120.0) -> Optional[np.ndarray]:
    """Ego nominal route polyline from route_roadblock_ids + lane centerlines.

    WHY this matches F4: F4's route_polyline was built from the scenario route
    (roadblocks -> lane baseline paths). We chain roadblock interior-edge
    centerlines in order, in global UTM, up to max_len_m. None if no usable route.
    """
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    try:
        rb_ids = list(scenario.get_route_roadblock_ids())
    except Exception:
        rb_ids = []
    if not rb_ids:
        return None

    map_api = scenario.map_api
    pts: List[np.ndarray] = []
    for rid in rb_ids:
        rb = None
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                rb = map_api.get_map_object(str(rid), layer)
            except Exception:
                rb = None
            if rb is not None:
                break
        if rb is None:
            continue
        # pick the longest interior lane edge as the roadblock's representative path
        best = None
        best_len = -1.0
        for edge in getattr(rb, 'interior_edges', []):
            try:
                dp = edge.baseline_path.discrete_path
            except Exception:
                continue
            arr = np.array([[s.x, s.y] for s in dp], dtype=np.float64)
            if len(arr) >= 2:
                L = float(_arclengths(arr)[-1])
                if L > best_len:
                    best_len, best = L, arr
        if best is not None:
            pts.append(best)

    if not pts:
        return None
    route = np.concatenate(pts, axis=0)
    # dedup consecutive duplicates
    keep = np.concatenate([[True], np.any(np.abs(np.diff(route, axis=0)) > 1e-6, axis=1)])
    route = route[keep]
    if len(route) < 2:
        return None
    s = _arclengths(route)
    return route[s <= max_len_m] if s[-1] > max_len_m else route


def ego_xy(ego) -> np.ndarray:
    return np.array([ego.rear_axle.x, ego.rear_axle.y], dtype=np.float64)


def agent_future_paths(scenario, iteration: int, horizon_s: float = AGENT_ROLLOUT_S
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each agent's LOGGED future path (global UTM) + sample times, from iteration.

    Matches agents across frames by track_token. Returns [(path (T,2), t (T,)), ...].
    """
    n = int(round(horizon_s / ROLLOUT_DT_S))
    try:
        future = list(scenario.get_future_tracked_objects(
            iteration=iteration, time_horizon=horizon_s, num_samples=n))
    except Exception:
        return []
    # collect per-track positions over the future frames
    by_track: Dict[str, List[Tuple[float, float, float]]] = {}
    dt = horizon_s / max(len(future), 1)
    for k, det in enumerate(future):
        t = (k + 1) * dt
        for obj in det.tracked_objects.tracked_objects:
            tok = getattr(obj, 'track_token', None) or id(obj)
            by_track.setdefault(tok, []).append((t, obj.box.center.x, obj.box.center.y))
    out = []
    for tok, rows in by_track.items():
        if len(rows) < 2:
            continue
        rows.sort()
        t = np.array([r[0] for r in rows])
        xy = np.array([[r[1], r[2]] for r in rows], dtype=np.float64)
        out.append((xy, t))
    return out


def a_gt(scenario, iteration: int = F4_ITERATION) -> Dict[str, float]:
    """Ground-truth-conflict score for one scene. Same kernel as s_inter, logged
    agent futures instead of constant-turn-rate. Returns dict with a_gt + diagnostics."""
    route = reconstruct_route(scenario)
    if route is None:
        return {'a_gt': float('nan'), 'reason': 'no_route', 'n_conflict': 0}

    ego = scenario.get_ego_state_at_iteration(iteration)
    v0 = float(ego.dynamic_car_state.speed)
    speed = max(v0, 1.0)  # WHY floor 1.0: identical to f4_score.ego_nominal_path
    e0 = ego_xy(ego)
    h = ego.rear_axle.heading
    hv = np.array([math.cos(h), math.sin(h)])
    # roll ego forward along route from its current position. The route is built in
    # roadblock order; orient it so index increases in the ego's heading direction,
    # then take the forward tail. Fall back to a straight heading rollout if the
    # forward tail is degenerate (keeps every scene scorable; a stopped/edge ego
    # then simply finds few crossings -> low a_gt, which is the correct reading).
    d2 = np.sum((route - e0) ** 2, axis=1)
    i0 = int(np.argmin(d2))
    j = min(i0 + 1, len(route) - 1)
    if j == i0 or np.dot(route[j] - route[i0], hv) < 0:
        route = route[::-1]
        d2 = np.sum((route - e0) ** 2, axis=1)
        i0 = int(np.argmin(d2))
    fwd = route[i0:]
    if len(fwd) < 2:
        fwd = np.stack([e0, e0 + 60.0 * hv], axis=0)
    ego_path, ego_t = _roll_along(fwd, speed, EGO_ROLLOUT_S, ROLLOUT_DT_S)
    full_ego = np.concatenate([e0[None], ego_path], axis=0)
    e_tf = np.concatenate([[0.0], ego_t])

    scores = []
    for a_path, a_t in agent_future_paths(scenario, iteration):
        a_full = a_path
        a_tf = a_t
        cross = _path_crossing_times(full_ego, e_tf, a_full, a_tf)
        if cross is not None:
            gap = abs(cross[0] - cross[1])  # PrET
            scores.append(math.exp(-(((gap - PRET_CENTER_S) / PRET_WIDTH_S) ** 2)))
    top = sorted(scores, reverse=True)[:TOP_K_AGENTS]
    out = 1.0
    for i in top:
        out *= (1.0 - i)
    return {'a_gt': 1.0 - out, 'reason': 'ok', 'n_conflict': len(scores), 'v0': v0,
            'route_len': float(_arclengths(route)[-1])}


def _self_test():
    f4 = load_f4_scores()
    toks = list(f4.keys())
    print('building scenarios...')
    by_tok = index_by_token(build_scenarios(tokens=toks))
    # one high-F4 + one zero-F4 scene
    hi = max(toks, key=lambda t: f4[t].get('f4') or 0.0)
    lo = min(toks, key=lambda t: (f4[t].get('f4') or 0.0, -(f4[t].get('v0') or 0.0)))
    for label, tok in [('HI', hi), ('LO', lo)]:
        if tok not in by_tok:
            print(f'  {label} {tok}: not built'); continue
        res = a_gt(by_tok[tok])
        print(f'  {label} tok={tok} F4={f4[tok]["f4"]:.3f} s_inter={f4[tok]["s_inter"]:.3f} '
              f'-> {res}')


if __name__ == '__main__':
    _self_test()
