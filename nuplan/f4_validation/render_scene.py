"""Top-down scene renderer for the Signal B rater panel. Ego-centered, heading-up,
no metadata in the image (nothing that could leak F4). Draws lanes, crosswalks,
traffic-light-controlled connectors, ego, and agents with velocity arrows.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/render_scene.py
"""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle, FancyArrow
import numpy as np

from scene_loader import F4_ITERATION

RADIUS_M = 55.0
OUT_DIR = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders')


def _rot(pts, c, s, origin):
    """Rotate (N,2) by [c,-s; s,c] after translating by -origin (ego-up frame)."""
    p = np.asarray(pts, dtype=float) - origin
    x = c * p[..., 0] + s * p[..., 1]
    y = -s * p[..., 0] + c * p[..., 1]
    return np.stack([x, y], axis=-1)


def render(scenario, out_path: Path, iteration: int = F4_ITERATION, show_futures: bool = False):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType

    ego = scenario.get_ego_state_at_iteration(iteration)
    ex, ey = ego.rear_axle.x, ego.rear_axle.y
    eh = ego.rear_axle.heading
    origin = np.array([ex, ey])
    # rotate so ego heading points +y (up): angle to rotate = (pi/2 - eh)
    c, s = math.cos(math.pi / 2 - eh), math.sin(math.pi / 2 - eh)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_facecolor('#0e1116')

    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
              SemanticMapLayer.CROSSWALK]
    near = scenario.map_api.get_proximal_map_objects(Point2D(ex, ey), RADIUS_M, layers)

    # red/green connector ids
    tl = {}
    try:
        for st in scenario.get_traffic_light_status_at_iteration(iteration):
            tl[str(st.lane_connector_id)] = st.status
    except Exception:
        pass

    def draw_centerline(obj, color, lw, z):
        try:
            dp = obj.baseline_path.discrete_path
        except Exception:
            return
        arr = np.array([[p.x, p.y] for p in dp])
        if len(arr) < 2:
            return
        r = _rot(arr, c, s, origin)
        ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, zorder=z, alpha=0.9)

    for ln in near.get(SemanticMapLayer.LANE, []):
        draw_centerline(ln, '#3a4250', 1.2, 1)
    for cn in near.get(SemanticMapLayer.LANE_CONNECTOR, []):
        col = '#3a4250'
        stt = tl.get(str(cn.id))
        if stt is not None:
            name = getattr(stt, 'name', str(stt))
            col = {'RED': '#e2554f', 'GREEN': '#4fb477', 'YELLOW': '#e0a64f'}.get(name, '#5a6270')
        draw_centerline(cn, col, 1.6, 2)
    for cw in near.get(SemanticMapLayer.CROSSWALK, []):
        try:
            poly = np.array(cw.polygon.exterior.coords)
            r = _rot(poly, c, s, origin)
            ax.add_patch(MplPoly(r, closed=True, facecolor='#c9b46a', alpha=0.25,
                                 edgecolor='#c9b46a', lw=1.0, hatch='//', zorder=1))
        except Exception:
            pass

    def draw_box(cx, cy, hd, L, W, color, z, alpha=0.95):
        rc = _rot([[cx, cy]], c, s, origin)[0]
        ang = math.degrees(hd - eh)  # heading relative to ego, ego is up
        rect = Rectangle((rc[0] - L / 2, rc[1] - W / 2), L, W, angle=ang,
                         rotation_point='center', facecolor=color, edgecolor='white',
                         lw=0.8, alpha=alpha, zorder=z)
        ax.add_patch(rect)
        # velocity arrow in ego frame
        return rc, ang

    # agents
    tracks = scenario.get_tracked_objects_at_iteration(iteration)
    for obj in tracks.tracked_objects.tracked_objects:
        b = obj.box
        tname = type(obj).__name__
        onehot = getattr(obj, 'tracked_object_type', None)
        is_ped = 'PEDESTRIAN' in str(onehot).upper()
        color = '#4fb477' if is_ped else '#e08a3c'
        L = getattr(b, 'length', 4.0); W = getattr(b, 'width', 2.0)
        rc, ang = draw_box(b.center.x, b.center.y, b.center.heading, L, W, color, 4)
        try:
            v = np.array([obj.velocity.x, obj.velocity.y])
            if np.linalg.norm(v) > 0.5:
                vr = _rot([origin + v], c, s, origin)[0]  # rotate velocity vector
                ax.add_patch(FancyArrow(rc[0], rc[1], vr[0] * 0.8, vr[1] * 0.8,
                             width=0.25, head_width=1.2, color=color, alpha=0.8, zorder=5))
        except Exception:
            pass

    # temporal overlay: each agent's ACTUAL logged 5s future path (ground truth,
    # independent of F4's constant-turn-rate formula -> no leak) + ego nominal route.
    # WHY: a static frame cannot show the space-time crossing that s_inter scores;
    # drawing real future paths lets a judge SEE whether/when paths conflict.
    if show_futures:
        from signal_a_gt_conflict import reconstruct_route, agent_future_paths
        for a_path, a_t in agent_future_paths(scenario, iteration):
            r = _rot(a_path, c, s, origin)
            ax.plot(r[:, 0], r[:, 1], color='#ffd24d', lw=1.0, alpha=0.55, zorder=3)
            ax.plot(r[-1, 0], r[-1, 1], 'o', color='#ffd24d', ms=2.5, alpha=0.7, zorder=3)
        route = reconstruct_route(scenario)
        if route is not None:
            rr = _rot(route, c, s, origin)
            ax.plot(rr[:, 0], rr[:, 1], color='#9ec5ff', lw=1.2, alpha=0.5, ls='--', zorder=3)

    # ego (blue, pointing up)
    L = ego.car_footprint.length; W = ego.car_footprint.width
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, angle=90, rotation_point='center',
                 facecolor='#3d7bd9', edgecolor='white', lw=1.2, zorder=6))
    ev = ego.dynamic_car_state.speed
    if ev > 0.3:
        ax.add_patch(FancyArrow(0, 0, 0, min(ev * 0.8, 12), width=0.3, head_width=1.6,
                     color='#9ec5ff', alpha=0.9, zorder=7))

    ax.set_xlim(-RADIUS_M, RADIUS_M)
    ax.set_ylim(-RADIUS_M, RADIUS_M)
    ax.set_aspect('equal')
    ax.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05, facecolor='#0e1116')
    plt.close(fig)
    return out_path


def _self_test():
    from scene_loader import build_scenarios, index_by_token, load_f4_scores
    f4 = load_f4_scores()
    toks = list(f4.keys())
    # pick: high s_inter, high s_branch, and a zero scene
    hi_int = max(toks, key=lambda t: f4[t]['s_inter'] or 0)
    hi_br = max(toks, key=lambda t: f4[t].get('s_branch') or 0)
    zero = min(toks, key=lambda t: f4[t]['f4'] or 0)
    pick = {'hi_inter': hi_int, 'hi_branch': hi_br, 'zero': zero}
    by_tok = index_by_token(build_scenarios(tokens=list(pick.values())))
    for label, t in pick.items():
        if t in by_tok:
            p = render(by_tok[t], OUT_DIR / f'{label}_{t}.png')
            v = f4[t]
            print(f'  {label} {t}: f4={v["f4"]:.2f} s_inter={v["s_inter"]:.2f} '
                  f's_branch={v.get("s_branch")} -> {p}')


if __name__ == '__main__':
    _self_test()
