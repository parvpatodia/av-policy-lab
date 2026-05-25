"""
map_utils.py — nuPlan map helper utilities for MapBC (Phase 3b)

Two public functions:
  get_map_for_db(db_path, maps_root)   → NuPlanMap for the log's city
  get_centerline_goal(map, x, y, yaw)  → (dx_goal, dy_goal) in ego-frame

WHY a separate module: planners.py imports are loaded by the nuPlan simulation
runner (via pickle). Keeping map utilities separate avoids polluting the
planner namespace and makes unit-testing map logic easier.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np

# --- Map version constants ---
# WHY hardcoded: nuPlan mini only ships one map bundle ('nuplan-maps-v1.0').
# If a newer bundle is present this will need updating.
_MAP_VERSION = 'nuplan-maps-v1.0'
_LOOK_AHEAD_M = 8.0        # ~0.8s at 10 m/s — matches GoalBC T+8 training horizon
_FALLBACK_AHEAD_M = 8.0    # straight-ahead fallback distance when no lane found


def get_map_name_for_db(db_path: str) -> str:
    """Return the nuPlan map name stored in the DB's log table.

    The nuPlan SQLite log table has a 'map_version' column that stores the
    map name string (e.g. 'us-nv-las-vegas-strip').
    WHY read from DB: the log filename alone doesn't reliably encode the city.
    """
    conn = sqlite3.connect(db_path)
    row  = conn.execute('SELECT map_version FROM log LIMIT 1').fetchone()
    conn.close()
    if row is None:
        raise ValueError(f'No log entry in {db_path}')
    return row[0]


def get_map_for_db(db_path: str, maps_root: str) -> 'NuPlanMap':
    """Load the NuPlanMap corresponding to a log DB file.

    Args:
        db_path:   Path to a nuPlan SQLite DB file.
        maps_root: Root directory containing nuplan-maps-v1.0.json and
                   the per-city GPKG subdirectories.

    Returns:
        A NuPlanMap instance ready for lane queries.
    """
    import sys
    import os
    sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')

    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
    from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap

    map_name = get_map_name_for_db(db_path)
    maps_db  = GPKGMapsDB(map_version=_MAP_VERSION, map_root=maps_root)
    return NuPlanMap(maps_db=maps_db, map_name=map_name)


def get_centerline_goal(
    nuplan_map,
    x: float,
    y: float,
    yaw: float,
    look_ahead_m: float = _LOOK_AHEAD_M,
    search_radius_m: float = 30.0,
) -> Tuple[float, float]:
    """Return the road centerline look-ahead point in ego-frame.

    Algorithm:
      1. Query nearest lanes within search_radius_m.
      2. Among returned lanes, find the one whose centerline has the minimum
         distance to the current ego position.
      3. Walk along that centerline for look_ahead_m from the closest point.
      4. Transform the resulting world-frame waypoint to ego-frame.

    Returns:
        (dx_goal, dy_goal): 2D offset in ego-frame (metres).
        Fallback: (look_ahead_m, 0.0) if no lane found (straight-ahead).

    WHY closest-lane selection: 'closest centerline point' is more robust than
    'closest lane object centroid' at intersections where multiple lanes share
    a region. Minimising point-wise distance picks the lane the ego is actually
    on or nearest to.
    """
    import sys
    sys.path.insert(0, '/Users/parvpatodia/nuplan-devkit')

    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.state_representation import Point2D

    try:
        result = nuplan_map.get_proximal_map_objects(
            Point2D(x, y), radius=search_radius_m, layers=[SemanticMapLayer.LANE]
        )
        lanes = result[SemanticMapLayer.LANE]
    except Exception:
        # WHY broad except: map query can fail if ego is outside mapped region.
        # Fallback to straight-ahead rather than crashing the simulation.
        lanes = []

    if not lanes:
        # Fallback: project straight ahead in ego-frame
        # WHY (look_ahead_m, 0.0): forward direction in ego-frame is +x.
        return float(look_ahead_m), 0.0

    # Find closest lane by minimum centerline point distance
    best_lane, best_dist = None, float('inf')
    for lane in lanes:
        try:
            cl_pts = np.array([(s.x, s.y) for s in lane.baseline_path.discrete_path])
            d = float(np.linalg.norm(cl_pts - np.array([x, y]), axis=1).min())
            if d < best_dist:
                best_dist, best_lane = d, lane
        except Exception:
            continue

    if best_lane is None:
        return float(look_ahead_m), 0.0

    cl = np.array([(s.x, s.y) for s in best_lane.baseline_path.discrete_path])

    # Walk look_ahead_m along centerline from the closest point
    dists = np.linalg.norm(cl - np.array([x, y]), axis=1)
    i0    = int(np.argmin(dists))
    cum   = 0.0
    i_goal = min(i0 + 1, len(cl) - 1)   # at least one step forward
    for i in range(i0, len(cl) - 1):
        seg = float(np.linalg.norm(cl[i + 1] - cl[i]))
        cum += seg
        if cum >= look_ahead_m:
            i_goal = i + 1
            break
    else:
        # Ran out of centerline — use last point
        i_goal = len(cl) - 1

    gx, gy = cl[i_goal]

    # Transform world-frame goal to ego-frame
    # WHY ego-frame: GoalBC was trained on ego-frame displacements.
    cos_neg = np.cos(-yaw)
    sin_neg = np.sin(-yaw)
    dx_w = gx - x
    dy_w = gy - y
    dx_goal = float(cos_neg * dx_w - sin_neg * dy_w)
    dy_goal = float(sin_neg * dx_w + cos_neg * dy_w)

    return dx_goal, dy_goal
