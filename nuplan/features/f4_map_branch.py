"""F4 primary S_branch / S_lane: lane-graph corridor branching (map-API side).

Pure logic over a plain lane-graph dict so it is unit-testable without the
nuPlan devkit; nuplan/slurm/score_f4.py adapts devkit map objects into
LaneInfo records (route-corridor lanes only, so graphs stay small).

WHY corridor-restricted BFS (F4_SPEC sec. 1): route-region conditioning pins
down the roadblock sequence but not the lane or junction arm. Counting exits
reachable inside that corridor measures exactly the lateral ambiguity the
conditioning leaves. The single-successor route_polyline is never used here;
it resolves forks and would erase the ambiguity being measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MERGE_DIST_M = 4.5     # single-linkage merge: adjacent parallel lanes fuse,
                       # junction arms (typ. > 6 m apart at 20+ m) stay apart
BRANCH_CAP = 3
LANE_CAP = 2
MAX_EXPANSIONS = 200   # hard cap; corridors are short, runaway = bad graph


@dataclass
class LaneInfo:
    """Minimal lane record: baseline points (meters), successors, roadblock."""
    id: str
    rb_id: str
    pts: np.ndarray                  # (N, 2) baseline polyline
    succ: List[str] = field(default_factory=list)


def lookahead_arc_m(v0: float) -> float:
    """F4_SPEC: s* = clip(4.0 * max(v0, 3.0), 20, 60)."""
    return float(np.clip(4.0 * max(v0, 3.0), 20.0, 60.0))


def _arclengths(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at_arc(pts: np.ndarray, s: float) -> np.ndarray:
    a = _arclengths(pts)
    s = float(np.clip(s, 0.0, a[-1]))
    return np.array([np.interp(s, a, pts[:, 0]), np.interp(s, a, pts[:, 1])])


def _closest_arc(pts: np.ndarray, xy: np.ndarray) -> Tuple[float, float]:
    """(arc position of projection, distance) of xy onto the polyline."""
    a = _arclengths(pts)
    best = (0.0, math.inf)
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        L2 = float(d @ d)
        if L2 < 1e-12:
            continue
        u = float(np.clip((xy - pts[i]) @ d / L2, 0.0, 1.0))
        proj = pts[i] + u * d
        dist = float(np.linalg.norm(xy - proj))
        if dist < best[1]:
            best = (float(a[i] + u * math.sqrt(L2)), dist)
    return best


def _single_linkage(points: List[np.ndarray], merge_dist: float) -> int:
    """Number of clusters under single-linkage with the given merge radius."""
    n = len(points)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= merge_dist:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def find_start_roadblock(graph: Dict[str, LaneInfo], route_rb_ids: List[str],
                         ego_xy: np.ndarray) -> Optional[str]:
    """Route roadblock whose lanes pass closest to ego (route order breaks ties)."""
    best: Tuple[float, Optional[str]] = (math.inf, None)
    for rb in route_rb_ids:
        lanes = [l for l in graph.values() if l.rb_id == rb]
        if not lanes:
            continue
        d = min(_closest_arc(l.pts, ego_xy)[1] for l in lanes)
        if d < best[0] - 1e-9:
            best = (d, rb)
    return best[1]


def corridor_branching(graph: Dict[str, LaneInfo], route_rb_ids: List[str],
                       ego_xy: np.ndarray, v0: float) -> dict:
    """B_R / N_par / S_branch / S_lane for one scenario.

    Returns excluded=True when ego cannot be localized on the corridor
    (off-route start, empty graph) — F4_SPEC: exclude, never score 0.
    """
    s_star = lookahead_arc_m(v0)
    start_rb = find_start_roadblock(graph, route_rb_ids, ego_xy)
    if start_rb is None:
        return {"excluded": True}
    start_lanes = [l for l in graph.values() if l.rb_id == start_rb]
    n_par = len(start_lanes)

    # BFS from ego's projection on EVERY start-roadblock lane: under
    # route-region conditioning the lane within the roadblock is not pinned.
    on_route = {l.id for l in graph.values() if l.rb_id in set(route_rb_ids)}
    terminals: List[np.ndarray] = []
    seen = set()
    queue: List[Tuple[str, float, float]] = []  # (lane_id, arc_into_lane, cum)
    for l in start_lanes:
        arc0, dist = _closest_arc(l.pts, ego_xy)
        if dist < 25.0:  # sanity: ignore start lanes nowhere near ego
            queue.append((l.id, arc0, 0.0))
    expansions = 0
    while queue and expansions < MAX_EXPANSIONS:
        lane_id, arc0, cum = queue.pop(0)
        key = (lane_id, round(cum / 5.0))
        if key in seen:
            continue
        seen.add(key)
        expansions += 1
        lane = graph[lane_id]
        length = _arclengths(lane.pts)[-1]
        remaining = length - arc0
        if cum + remaining >= s_star:
            terminals.append(_point_at_arc(lane.pts, arc0 + (s_star - cum)))
            continue
        nxt = [s for s in lane.succ if s in on_route and s in graph]
        if not nxt:
            # corridor ends before s*: the end point is still a distinct exit
            terminals.append(_point_at_arc(lane.pts, length))
            continue
        for s in nxt:
            queue.append((s, 0.0, cum + remaining))

    b_r = _single_linkage(terminals, MERGE_DIST_M)
    return {
        "excluded": False,
        "b_r": b_r,
        "n_par": n_par,
        "s_star": s_star,
        "s_branch": min(max(b_r - 1, 0), BRANCH_CAP) / BRANCH_CAP,
        "s_lane": min(max(n_par - 1, 0), LANE_CAP) / LANE_CAP,
    }
