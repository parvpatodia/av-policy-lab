"""Tests for shard-side F4 components. Scenes built in meters, then
normalized into shard format so the denorm shim is exercised everywhere."""
import math

import numpy as np
import pytest

from features.f4_score import (
    DenormConfig,
    G_STOP_VALUE,
    PED_OVERRIDE,
    PRET_CENTER_S,
    combine,
    score_sample,
)

POS, VEL = 120.0, 15.0
CFG = {"pos_scale_m": POS, "vel_scale_mps": VEL}


def _empty_scene(v0=8.0, route_len_m=100.0):
    """Ego at origin heading +x at v0; straight route; no agents; no lights."""
    s = {
        "ego": np.zeros((20, 8), dtype=np.float32),
        "agents": np.zeros((32, 20, 9), dtype=np.float32),
        "agent_mask": np.zeros((32, 20), dtype=bool),
        "map_polylines": np.zeros((128, 20, 7), dtype=np.float32),
        "map_mask": np.zeros(128, dtype=bool),
        "crosswalks": np.zeros((16, 20, 2), dtype=np.float32),
        "crosswalk_mask": np.zeros(16, dtype=bool),
        "route_polyline": np.zeros((40, 4), dtype=np.float32),
        "route_mask": np.ones(40, dtype=bool),
        "traffic_lights": np.full(128, -1, dtype=np.int64),
    }
    s["ego"][:, 3] = 1.0                      # cos(h)=1
    s["ego"][-1, 4] = v0 / VEL                # vx normalized
    s["route_polyline"][:, 0] = np.linspace(0, route_len_m, 40) / POS
    s["route_polyline"][:, 2] = 1.0           # dir +x
    return s


def _add_agent(s, idx, xy_m, heading, speed_ms, kind="vehicle"):
    onehot = {"vehicle": 6, "ped": 7, "bicycle": 8}[kind]
    s["agent_mask"][idx, :] = True
    for t in range(20):
        s["agents"][idx, t, 0] = xy_m[0] / POS
        s["agents"][idx, t, 1] = xy_m[1] / POS
        s["agents"][idx, t, 2] = math.sin(heading)
        s["agents"][idx, t, 3] = math.cos(heading)
        s["agents"][idx, t, 4] = speed_ms * math.cos(heading) / VEL
        s["agents"][idx, t, 5] = speed_ms * math.sin(heading) / VEL
        s["agents"][idx, t, onehot] = 1.0


def test_empty_scene_zero_interaction():
    r = score_sample(_empty_scene(), CFG)
    assert r["excluded"] == 0.0
    assert r["s_inter"] == 0.0 and r["g_stop"] == 1.0 and r["f4"] == 0.0
    assert r["v0"] == pytest.approx(8.0, abs=1e-6)


def test_route_failure_excluded():
    s = _empty_scene()
    s["route_mask"][:] = False
    r = score_sample(s, CFG)
    assert r["excluded"] == 1.0 and math.isnan(r["f4"])


def test_crossing_agent_at_ambiguous_gap_scores_high():
    # ego 8 m/s on +x; agent crossing from the right hits ego's path where
    # ego arrives at t=3 s (24 m): set agent to arrive PRET_CENTER_S early
    s = _empty_scene(v0=8.0)
    t_ego = 3.0
    cross = np.array([8.0 * t_ego, 0.0])
    t_agent = t_ego - PRET_CENTER_S          # gap exactly at band-pass peak
    speed = 10.0
    start = cross + np.array([0.0, speed * t_agent])  # coming from +y, heading -y
    _add_agent(s, 0, start, -math.pi / 2, speed)
    r = score_sample(s, CFG)
    assert r["s_inter"] > 0.95, r


def test_crossing_agent_at_extreme_gap_scores_low():
    # same geometry but the agent crosses ~7.5 s before ego: forced "go"
    s = _empty_scene(v0=8.0)
    cross = np.array([24.0, 0.0])
    speed = 10.0
    start = cross + np.array([0.0, speed * (-4.5)])   # crosses at t=-4.5? no:
    # place agent so it reaches cross at t=0.5 while ego reaches at t=3 ->
    # gap 2.5... instead use slow far agent: reaches at t=8 (beyond rollout)
    s = _empty_scene(v0=8.0)
    start = cross + np.array([0.0, 40.0])
    _add_agent(s, 0, start, -math.pi / 2, 40.0 / 8.0)  # arrives t=8 > 5 s cap
    r = score_sample(s, CFG)
    assert r["s_inter"] < 0.2, r


def test_lead_vehicle_band_pass():
    # lead car 12 m ahead at v0=8 -> tau=1.5 s = band center
    s = _empty_scene(v0=8.0)
    _add_agent(s, 0, (12.0, 0.5), 0.0, 8.0)
    high = score_sample(s, CFG)["s_inter"]
    # lead car 2 m ahead -> tau=0.25 s, forced behavior, low ambiguity
    s2 = _empty_scene(v0=8.0)
    _add_agent(s2, 0, (2.0, 0.5), 0.0, 8.0)
    low = score_sample(s2, CFG)["s_inter"]
    assert high > 0.9 and low < high * 0.5, (high, low)


def test_off_corridor_agent_ignored():
    s = _empty_scene(v0=8.0)
    _add_agent(s, 0, (12.0, 8.0), 0.0, 8.0)   # parallel lane, 8 m lateral
    assert score_sample(s, CFG)["s_inter"] == 0.0


def test_platoon_saturation_capped():
    # 8 near-identical off-peak leads (tau ~1.9 s, I_j < 1); top-3 noisy-OR
    # must cap at the 3-agent bound instead of saturating over all 8
    s = _empty_scene(v0=8.0)
    for i in range(8):
        _add_agent(s, i, (15.0 + 0.1 * i, 0.5), 0.0, 8.0)
    one = _empty_scene(v0=8.0)
    _add_agent(one, 0, (15.0, 0.5), 0.0, 8.0)
    i1 = score_sample(one, CFG)["s_inter"]
    i8 = score_sample(s, CFG)["s_inter"]
    assert 0.0 < i1 < 1.0
    assert i8 <= 1.0 - (1.0 - i1) ** 3 + 1e-9
    assert i8 < 1.0


def test_stationary_ped_at_crosswalk_override():
    s = _empty_scene(v0=8.0)
    # crosswalk polygon crossing the route at x=15
    s["crosswalk_mask"][0] = True
    xs = np.linspace(14.0, 16.0, 20) / POS
    ys = np.linspace(-4.0, 4.0, 20) / POS
    s["crosswalks"][0, :, 0] = xs
    s["crosswalks"][0, :, 1] = ys
    _add_agent(s, 0, (15.0, 4.5), math.pi, 0.0, kind="ped")
    r = score_sample(s, CFG)
    assert r["s_inter"] >= PED_OVERRIDE - 1e-9
    # same agent as a stopped vehicle: no override
    s2 = _empty_scene(v0=8.0)
    s2["crosswalk_mask"][0] = True
    s2["crosswalks"][0, :, 0] = xs
    s2["crosswalks"][0, :, 1] = ys
    _add_agent(s2, 0, (15.0, 4.5), math.pi, 0.0, kind="vehicle")
    assert score_sample(s2, CFG)["s_inter"] < PED_OVERRIDE


def test_g_stop_red_light_standstill():
    s = _empty_scene(v0=0.0)
    s["map_mask"][0] = True
    s["traffic_lights"][0] = 2                # RED
    s["map_polylines"][0, :, 0] = np.linspace(5, 15, 20) / POS
    s["map_polylines"][0, :, 1] = 0.0
    r = score_sample(s, CFG)
    assert r["g_stop"] == G_STOP_VALUE
    # moving ego: suppressor off
    s["ego"][-1, 4] = 8.0 / VEL
    assert score_sample(s, CFG)["g_stop"] == 1.0
    # green light: suppressor off
    s["ego"][-1, 4] = 0.0
    s["traffic_lights"][0] = 0
    assert score_sample(s, CFG)["g_stop"] == 1.0


def test_combine_rule():
    assert combine(0, 0, 0, 1.0) == 0.0
    assert combine(1, 0, 0, 1.0) == 1.0
    assert combine(0, 1, 0, 1.0) == pytest.approx(0.5)
    assert combine(0.5, 0.0, 0.5, 1.0) == pytest.approx(0.75)
    assert combine(1, 1, 1, 0.25) == pytest.approx(0.25)
    f_lo = combine(0.3, 0.2, 0.1, 1.0)
    f_hi = combine(0.6, 0.2, 0.1, 1.0)
    assert f_hi > f_lo  # monotone in each subscore


def test_denorm_config_guard():
    with pytest.raises((AssertionError, KeyError)):
        DenormConfig.from_shard_config({"pos_scale_m": 1.0, "vel_scale_mps": 1.0})
    with pytest.raises((AssertionError, KeyError)):
        DenormConfig.from_shard_config({})


def test_curved_agent_rollout_differs_from_straight():
    """An agent with yaw rate must not be rolled out straight: build a scene
    where only the curved path crosses the route."""
    s = _empty_scene(v0=8.0)
    # agent left of route, heading +x, curving right toward the route;
    # crossing lands at x~32 m (t_agent~3.3 s), inside ego's 40 m rollout
    _add_agent(s, 0, (10.0, 12.0), 0.0, 8.0)
    # inject heading rate via history: rotate heading across last two steps
    hr = -0.3  # rad/s, curving toward the road (turn radius ~27 m)
    h_prev = 0.0 - hr * 0.1
    s["agents"][0, 18, 2] = math.sin(h_prev)
    s["agents"][0, 18, 3] = math.cos(h_prev)
    curved = score_sample(s, CFG)["s_inter"]
    s["agents"][0, 18, 2] = 0.0
    s["agents"][0, 18, 3] = 1.0
    straight = score_sample(s, CFG)["s_inter"]
    assert curved > straight, (curved, straight)
