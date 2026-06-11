"""Tests for the lane-graph branching score with synthetic corridors."""
import numpy as np

from features.f4_map_branch import (
    LaneInfo,
    corridor_branching,
    lookahead_arc_m,
)


def _lane(id, rb, pts, succ=()):
    return LaneInfo(id=id, rb_id=rb, pts=np.asarray(pts, dtype=float),
                    succ=list(succ))


def _straight(x0, x1, y, n=10):
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=1)


EGO = np.array([0.0, 0.0])


def test_lookahead_clip():
    assert lookahead_arc_m(0.0) == 20.0
    assert lookahead_arc_m(8.0) == 32.0
    assert lookahead_arc_m(50.0) == 60.0


def test_single_lane_corridor_no_ambiguity():
    g = {"a": _lane("a", "rb1", _straight(-5, 100, 0))}
    r = corridor_branching(g, ["rb1"], EGO, v0=8.0)
    assert not r["excluded"]
    assert r["b_r"] == 1 and r["s_branch"] == 0.0
    assert r["n_par"] == 1 and r["s_lane"] == 0.0


def test_fork_two_arms():
    # one lane forks into two connectors that end 30 m apart laterally
    g = {
        "a": _lane("a", "rb1", _straight(-5, 15, 0), succ=("L", "R")),
        "L": _lane("L", "rb2", np.stack([np.linspace(15, 45, 10),
                                         np.linspace(0, 25, 10)], axis=1)),
        "R": _lane("R", "rb2", np.stack([np.linspace(15, 45, 10),
                                         np.linspace(0, -25, 10)], axis=1)),
    }
    r = corridor_branching(g, ["rb1", "rb2"], EGO, v0=8.0)
    assert r["b_r"] == 2
    assert r["s_branch"] == 1 / 3


def test_parallel_lanes_merge_into_one_exit_but_count_in_s_lane():
    # two parallel same-direction lanes 3 m apart: exits merge (single
    # corridor), but lane choice shows up in S_lane
    g = {
        "a": _lane("a", "rb1", _straight(-5, 100, 0)),
        "b": _lane("b", "rb1", _straight(-5, 100, 3.0)),
    }
    r = corridor_branching(g, ["rb1"], EGO, v0=8.0)
    assert r["b_r"] == 1          # 3 m < 4.5 m merge radius
    assert r["n_par"] == 2 and r["s_lane"] == 0.5


def test_off_route_ego_excluded():
    g = {"a": _lane("a", "rb1", _straight(500, 600, 500))}
    r = corridor_branching(g, ["rb1"], EGO, v0=8.0)
    # start roadblock found but ego projects > 25 m away -> no terminals
    assert r["excluded"] or r["b_r"] == 0


def test_empty_graph_excluded():
    assert corridor_branching({}, ["rb1"], EGO, v0=8.0)["excluded"]


def test_short_corridor_end_counts_as_exit():
    g = {"a": _lane("a", "rb1", _straight(-5, 12, 0))}  # ends well before s*
    r = corridor_branching(g, ["rb1"], EGO, v0=8.0)
    assert r["b_r"] == 1


def test_three_arm_junction_caps_at_full_score():
    arms = {
        f"arm{i}": _lane(f"arm{i}", "rb2",
                         np.stack([np.linspace(15, 45, 10),
                                   np.linspace(0, (i - 1) * 30, 10)], axis=1))
        for i in range(3)
    }
    g = {"a": _lane("a", "rb1", _straight(-5, 15, 0), succ=tuple(arms)), **arms}
    r = corridor_branching(g, ["rb1", "rb2"], EGO, v0=8.0)
    assert r["b_r"] == 3
    assert r["s_branch"] == 2 / 3


def test_diamond_reconverging_paths_one_exit():
    # fork that reconverges before s*: both chains end at the same point,
    # so it is NOT a lasting decision and must merge back to one exit
    top = np.stack([np.linspace(10, 25, 8), np.linspace(0, 6, 8)], axis=1)
    bot = np.stack([np.linspace(10, 25, 8), np.linspace(0, -6, 8)], axis=1)
    tail_pts = _straight(25, 80, 0)
    g = {
        "a": _lane("a", "rb1", _straight(-5, 10, 0), succ=("t", "b")),
        "t": _lane("t", "rb2", top, succ=("tail",)),
        "b": _lane("b", "rb2", bot, succ=("tail",)),
        "tail": _lane("tail", "rb3", tail_pts),
    }
    r = corridor_branching(g, ["rb1", "rb2", "rb3"], EGO, v0=8.0)
    assert r["b_r"] == 1, r
