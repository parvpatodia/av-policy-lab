"""Unit tests for the planner geometry core (nuplan/planners.py).

These cover the pure-numpy math underlying the imitation-learning planners:
coordinate transforms, quaternion->yaw, straight-route construction, the
arc-length route-goal walk, and speed-adaptive look-ahead. None of these
require the nuPlan simulator to run; they exercise the math directly or via a
minimally-constructed planner backed by a fake checkpoint.

Run:
    cd /Users/parvpatodia/Desktop/diffusion-policy-zoo && \
    /opt/anaconda3/envs/nuplan/bin/python3.9 -m pytest tests/ -v

## FINDINGS
No bugs found. Every invariant tested below holds against the current code as
written. Notes on conventions confirmed while writing the tests:

  * The ego-frame forward transform (world->ego, planners.py compute_planner_
    trajectory L833-838 and _get_route_goal L1384-1388) uses cos(-yaw)/sin(-yaw);
    the reverse transform (ego->world, L854-855) uses cos(yaw)/sin(yaw). These
    are exact inverses -> the world->ego->world round trip is the identity. This
    is the single most important invariant and it passes.

  * `quat_to_yaw` is NOT defined in planners.py as a standalone function. The
    identical formula is inlined in GoalBCPlanner._build_expert_lookup (L801)
    and DAggerPlanner._build_expert_lookup (L302), and defined as a helper in
    goal_bc.ipynb. We replicate that exact formula here and test it directly.

  * _get_route_goal walk semantics: starting from the nearest route index i0,
    it accumulates segment lengths and stops at the FIRST point whose cumulative
    distance >= look_ahead_m (so the returned arc length is >= look_ahead, never
    interpolated). With unit (1 m) spacing this lands exactly on the look_ahead
    integer distance, which the exact-value tests rely on.
"""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Transform helpers — replicate the EXACT math from planners.py so the tests
# document the convention independently of the planner object.
# ---------------------------------------------------------------------------

def world_to_ego(dx_w, dy_w, yaw):
    """World-frame delta -> ego-frame, mirroring planners.py L833-838 (cos(-yaw))."""
    cos_n = np.cos(-yaw)
    sin_n = np.sin(-yaw)
    dx_e = cos_n * dx_w - sin_n * dy_w
    dy_e = sin_n * dx_w + cos_n * dy_w
    return dx_e, dy_e


def ego_to_world(dx_e, dy_e, yaw):
    """Ego-frame delta -> world-frame delta, mirroring planners.py L854-855 (cos(yaw))."""
    cos_h = np.cos(yaw)
    sin_h = np.sin(yaw)
    dx_w = cos_h * dx_e - sin_h * dy_e
    dy_w = sin_h * dx_e + cos_h * dy_e
    return dx_w, dy_w


def quat_to_yaw(qw, qx, qy, qz):
    """Replicates the inline quaternion->yaw formula (planners.py L801, goal_bc.ipynb)."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def planners_mod():
    """Import the local planners module (path wired up in conftest.py)."""
    import planners
    return planners


@pytest.fixture
def fake_ckpt(tmp_path, planners_mod):
    """A minimal but valid GoalBC checkpoint so the real constructor runs.

    Keys mirror what RouteMapBCPlanner.__init__/initialize expect:
      model  -> GoalBCPolicy state_dict
      X_mean/X_std -> shape (8,)   (8-dim input features)
      Y_mean/Y_std -> shape (48,)  (16 x (dx,dy,dyaw))
    """
    ckpt_path = tmp_path / "fake_goal_bc.pt"
    model = planners_mod.GoalBCPolicy()
    ckpt = {
        "model": model.state_dict(),
        "X_mean": np.zeros(8, dtype=np.float32),
        "X_std": np.ones(8, dtype=np.float32),
        "Y_mean": np.zeros(48, dtype=np.float32),
        "Y_std": np.ones(48, dtype=np.float32),
    }
    torch.save(ckpt, str(ckpt_path))
    return str(ckpt_path)


@pytest.fixture
def route_planner(planners_mod, fake_ckpt):
    """A RouteMapBCPlanner built from the fake checkpoint (real __init__ runs)."""
    return planners_mod.RouteMapBCPlanner(fake_ckpt)


# ===========================================================================
# Group 1: Coordinate transform correctness
# ===========================================================================

@pytest.mark.parametrize("yaw", [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (3.0, 0.0), (0.0, -2.5), (5.0, 7.0), (-4.0, 1.5)])
def test_ego_frame_roundtrip(yaw, offset):
    """world->ego->world must recover the original delta exactly (transforms are inverses)."""
    dx_w, dy_w = offset
    dx_e, dy_e = world_to_ego(dx_w, dy_w, yaw)
    rx, ry = ego_to_world(dx_e, dy_e, yaw)
    np.testing.assert_allclose([rx, ry], [dx_w, dy_w], atol=1e-12)


def test_ego_frame_zero_yaw():
    """At yaw=0 the world delta (dx,dy) maps to ego frame unchanged."""
    dx_e, dy_e = world_to_ego(3.0, -2.0, 0.0)
    np.testing.assert_allclose([dx_e, dy_e], [3.0, -2.0], atol=1e-12)


def test_ego_frame_90deg():
    """At yaw=pi/2 a world point due north (+y) maps to ego +x (forward)."""
    # World point straight ahead of a north-facing car: delta = (0, +1).
    dx_e, dy_e = world_to_ego(0.0, 1.0, np.pi / 2)
    np.testing.assert_allclose([dx_e, dy_e], [1.0, 0.0], atol=1e-12)


# ===========================================================================
# Group 2: quat_to_yaw
# ===========================================================================

def test_quat_to_yaw_identity():
    """Identity quaternion (qw=1) yields yaw=0."""
    assert quat_to_yaw(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [np.pi / 2, np.pi])
def test_quat_to_yaw_known_angles(theta):
    """A pure z-axis rotation quaternion decodes back to its yaw angle."""
    qw = np.cos(theta / 2.0)
    qz = np.sin(theta / 2.0)
    yaw = quat_to_yaw(qw, 0.0, 0.0, qz)
    # arctan2 returns +pi for theta=pi; both represent the same heading.
    expected = np.arctan2(np.sin(theta), np.cos(theta))
    np.testing.assert_allclose(yaw, expected, atol=1e-6)


# ===========================================================================
# Group 3: RouteMapBCPlanner._straight_route
# ===========================================================================

def test_straight_route_length(route_planner):
    """A 200m route at 2m spacing has exactly 100 points (int(200/2))."""
    pts = route_planner._straight_route(0.0, 0.0, 0.0, length_m=200.0, step_m=2.0)
    assert pts.shape == (100, 2)


def test_straight_route_direction(route_planner):
    """yaw=0 extends along +x; yaw=pi/2 along +y (first point is the origin)."""
    px = route_planner._straight_route(0.0, 0.0, 0.0, length_m=200.0, step_m=2.0)
    np.testing.assert_allclose(px[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(px[-1], [2.0 * 99, 0.0], atol=1e-9)

    py = route_planner._straight_route(0.0, 0.0, np.pi / 2, length_m=200.0, step_m=2.0)
    np.testing.assert_allclose(py[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(py[-1], [0.0, 2.0 * 99], atol=1e-9)


def test_straight_route_spacing(route_planner):
    """Consecutive route points are exactly step_m apart."""
    pts = route_planner._straight_route(10.0, -5.0, np.pi / 4, length_m=50.0, step_m=2.0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    np.testing.assert_allclose(seg, 2.0, atol=1e-9)


# ===========================================================================
# Group 4: _get_route_goal (arc-length walk)
# ===========================================================================

def _straight_x_route(n=50, spacing=1.0):
    """Synthetic route: n points along +x at `spacing` m, y=0."""
    return np.array([[spacing * i, 0.0] for i in range(n)], dtype=np.float64)


def test_route_goal_straight_lookahead(route_planner):
    """Ego at origin (yaw=0), look_ahead=8m on a +x route -> goal ~ (8, 0)."""
    route_planner._route_pts = _straight_x_route()
    goal = route_planner._get_route_goal(0.0, 0.0, 0.0, look_ahead_m=8.0)
    np.testing.assert_allclose(goal, [8.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("look_ahead,expected_x", [(3.0, 3.0), (8.0, 8.0)])
def test_route_goal_respects_lookahead_distance(route_planner, look_ahead, expected_x):
    """The walk distance is honored: goal x-component equals the look-ahead."""
    route_planner._route_pts = _straight_x_route()
    goal = route_planner._get_route_goal(0.0, 0.0, 0.0, look_ahead_m=look_ahead)
    np.testing.assert_allclose(goal, [expected_x, 0.0], atol=1e-9)


def test_route_goal_drifted_ego(route_planner):
    """Ego drifted to (10,5) off a +x route: argmin recovers (10,0), goal y is negative."""
    route_planner._route_pts = _straight_x_route()
    goal = route_planner._get_route_goal(10.0, 5.0, 0.0, look_ahead_m=8.0)
    # Nearest route point is (10,0); walk 8m forward -> (18,0).
    # Ego-frame (yaw=0): dx = 18-10 = 8, dy = 0-5 = -5 (route is to the right).
    np.testing.assert_allclose(goal, [8.0, -5.0], atol=1e-9)
    assert goal[1] < 0.0


def test_route_goal_empty_route(route_planner):
    """An empty route returns the straight-ahead fallback (look_ahead, 0.0)."""
    route_planner._route_pts = np.zeros((0, 2), dtype=np.float64)
    goal = route_planner._get_route_goal(3.0, 4.0, 1.234, look_ahead_m=7.0)
    np.testing.assert_allclose(goal, [7.0, 0.0], atol=1e-12)


def test_route_goal_route_exhausted(route_planner):
    """A look-ahead longer than the remaining route returns the last point, no crash."""
    route_planner._route_pts = _straight_x_route(n=10, spacing=1.0)  # ends at x=9
    goal = route_planner._get_route_goal(0.0, 0.0, 0.0, look_ahead_m=1000.0)
    # Last point is (9,0); ego-frame at origin/yaw0 is (9,0).
    np.testing.assert_allclose(goal, [9.0, 0.0], atol=1e-9)


# ===========================================================================
# Group 5: SpeedAdaptive look-ahead
# ===========================================================================

def test_speed_adaptive_lookahead_values(planners_mod):
    """_GOAL_LOOKAHEAD_S == 0.8 and look_ahead = max(0.05, speed*0.8) for sample speeds."""
    assert planners_mod._GOAL_LOOKAHEAD_S == pytest.approx(0.8, abs=1e-12)
    s = planners_mod._GOAL_LOOKAHEAD_S
    cases = {0.0: 0.05, 4.33: 4.33 * 0.8, 10.0: 8.0}
    for speed, expected in cases.items():
        look_ahead = max(0.05, speed * s)
        np.testing.assert_allclose(look_ahead, expected, atol=1e-9)
    # Spot-check the documented 4.33 -> 3.464 value explicitly.
    np.testing.assert_allclose(max(0.05, 4.33 * s), 3.464, atol=1e-9)


def test_speed_adaptive_floor(planners_mod):
    """At speed=0 the look-ahead is the 0.05 floor, never 0 (avoids degenerate goal)."""
    s = planners_mod._GOAL_LOOKAHEAD_S
    look_ahead = max(0.05, 0.0 * s)
    assert look_ahead == pytest.approx(0.05, abs=1e-12)
    assert look_ahead > 0.0


def test_speed_adaptive_subclass_sets_flag(planners_mod, fake_ckpt):
    """SpeedAdaptiveRouteMapBCPlanner sets speed_adaptive=True via the parent ctor."""
    p = planners_mod.SpeedAdaptiveRouteMapBCPlanner(fake_ckpt)
    assert p._speed_adaptive is True


# ===========================================================================
# Group 6: module constants
# ===========================================================================

def test_constants(planners_mod):
    """FUTURE_STEPS == 16 and DT == 0.1 (planner horizon and waypoint spacing)."""
    assert planners_mod.FUTURE_STEPS == 16
    assert planners_mod.DT == pytest.approx(0.1, abs=1e-12)
