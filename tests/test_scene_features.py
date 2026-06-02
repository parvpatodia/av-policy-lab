"""Pure-numpy unit tests for the F0 scene-feature core (nuplan/features/scene_features.py).

These cover ONLY the parts that do not need the nuPlan devkit:
  * ego-frame transform round-trip (the load-bearing invariant)
  * agent padding/masking to N=32
  * polyline resampling to fixed length (arc-length uniform, endpoints preserved)
  * normalization invertibility (every norm_* has an exact inverse)
  * polyline direction (unit tangents)

They import `scene_features` directly. That module only imports numpy at top level
(every devkit import is function-local), so this suite runs WITHOUT nuPlan or torch
installed. The devkit-touching paths (extract_sample / map builder) are NOT covered
here by design — they are gated by `--smoke` on nuPlan MINI (see F0_IMPLEMENTATION.md).

Run IN-ENV (the agent cannot execute it here — these are runnable-by-inspection):
    cd <repo-root> && python -m pytest tests/test_scene_features.py -q
"""
import numpy as np
import pytest

# conftest.py adds nuplan/ to sys.path; the module lives in nuplan/features/.
from features.scene_features import (  # noqa: E402
    AgentFeatureBuilder,
    EgoFrameTransform,
    FeatureConfig,
    Normalizer,
    pad_or_truncate,
    polyline_directions,
    resample_polyline,
)


# ===========================================================================
# Group 1: EgoFrameTransform round-trip (matches test_planner_geometry.py style)
# ===========================================================================

@pytest.mark.parametrize("yaw", [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2, 2.345])
@pytest.mark.parametrize("ref", [(0.0, 0.0), (10.0, -5.0), (-123.4, 88.8)])
def test_world_ego_roundtrip(yaw, ref):
    """world->ego->world recovers the original points exactly (transforms are inverses)."""
    tf = EgoFrameTransform(ref[0], ref[1], yaw)
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, -2.5], [5.0, 7.0], [-4.0, 1.5]])
    back = tf.ego_to_world(tf.world_to_ego(pts))
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_ego_origin_maps_to_ref():
    """The reference pose itself maps to the ego origin (0, 0)."""
    tf = EgoFrameTransform(10.0, -5.0, 1.2)
    np.testing.assert_allclose(tf.world_to_ego(np.array([10.0, -5.0])), [0.0, 0.0], atol=1e-12)


def test_forward_axis_is_plus_x():
    """A point one metre ahead of a yaw=pi/2 (north-facing) ego is ego-frame (1, 0)."""
    tf = EgoFrameTransform(0.0, 0.0, np.pi / 2)
    ego = tf.world_to_ego(np.array([0.0, 1.0]))  # world +y == ego forward
    np.testing.assert_allclose(ego, [1.0, 0.0], atol=1e-12)


def test_rotate_vector_has_no_translation():
    """Free vectors rotate but do NOT translate (velocity must not pick up ego offset)."""
    tf = EgoFrameTransform(100.0, 200.0, 0.0)  # large translation, zero rotation
    v = tf.rotate_vector_to_ego(np.array([3.0, -2.0]))
    np.testing.assert_allclose(v, [3.0, -2.0], atol=1e-12)


def test_heading_to_ego_subtracts_reference():
    tf = EgoFrameTransform(0.0, 0.0, 0.5)
    np.testing.assert_allclose(tf.heading_to_ego(np.array([0.5, 1.5])), [0.0, 1.0], atol=1e-12)


# ===========================================================================
# Group 2: polyline resampling
# ===========================================================================

def test_resample_fixed_length():
    """Resampling always yields exactly n_out points regardless of input length."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    for n_out in (2, 5, 20, 40):
        out = resample_polyline(pts, n_out)
        assert out.shape == (n_out, 2)


def test_resample_preserves_endpoints():
    """First and last output points equal the input endpoints (arc-length anchored)."""
    pts = np.array([[1.0, 2.0], [4.0, 6.0], [10.0, -1.0]])
    out = resample_polyline(pts, 17)
    np.testing.assert_allclose(out[0], pts[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], pts[-1], atol=1e-9)


def test_resample_uniform_arclength_on_straight_line():
    """On a straight line, resampled points are equally spaced in arc length."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    out = resample_polyline(pts, 11)  # 0,1,2,...,10
    np.testing.assert_allclose(out[:, 0], np.linspace(0.0, 10.0, 11), atol=1e-9)
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-9)


def test_resample_handles_single_point():
    """A single-point polyline is repeated to n_out (no crash, no NaN)."""
    out = resample_polyline(np.array([[5.0, 6.0]]), 8)
    assert out.shape == (8, 2)
    np.testing.assert_allclose(out, np.tile([5.0, 6.0], (8, 1)))


def test_resample_handles_empty():
    """An empty polyline yields a finite zero array of the right shape."""
    out = resample_polyline(np.zeros((0, 2)), 6)
    assert out.shape == (6, 2)
    assert np.all(np.isfinite(out))


def test_resample_handles_coincident_points():
    """All-coincident vertices (zero total length) degrade to a repeat, not div-by-zero."""
    out = resample_polyline(np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]), 5)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out, np.tile([2.0, 2.0], (5, 1)))


def test_resample_rejects_bad_n_out():
    with pytest.raises(ValueError):
        resample_polyline(np.array([[0.0, 0.0], [1.0, 0.0]]), 1)


# ===========================================================================
# Group 3: polyline directions
# ===========================================================================

def test_polyline_directions_unit_and_length():
    """Directions are unit vectors and the output length matches the input."""
    pts = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]])
    dirs = polyline_directions(pts)
    assert dirs.shape == (3, 2)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)
    # first segment points +y, second points +x
    np.testing.assert_allclose(dirs[0], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(dirs[1], [1.0, 0.0], atol=1e-9)
    # last point reuses previous direction
    np.testing.assert_allclose(dirs[2], dirs[1], atol=1e-9)


def test_polyline_directions_degenerate():
    """A <2-point polyline yields zero directions of matching length (no crash)."""
    assert polyline_directions(np.array([[1.0, 1.0]])).shape == (1, 2)
    assert polyline_directions(np.zeros((0, 2))).shape == (0, 2)


# ===========================================================================
# Group 4: pad/truncate + masking
# ===========================================================================

def test_pad_to_max():
    """Fewer items than max_n -> first slots real, rest zero, mask matches."""
    items = [np.full((4, 9), float(i + 1), dtype=np.float32) for i in range(5)]
    data, mask = pad_or_truncate(items, max_n=32, feature_shape=(4, 9))
    assert data.shape == (32, 4, 9)
    assert mask.shape == (32,)
    assert mask[:5].all() and not mask[5:].any()
    assert np.all(data[5:] == 0.0)             # padded slots are zero
    assert np.all(data[0] == 1.0)              # first real item preserved


def test_truncate_to_max():
    """More items than max_n -> truncated to the first max_n, all masked True."""
    items = [np.ones((4, 9), dtype=np.float32) * i for i in range(50)]
    data, mask = pad_or_truncate(items, max_n=32, feature_shape=(4, 9))
    assert data.shape == (32, 4, 9)
    assert mask.all()
    np.testing.assert_allclose(data[31], np.ones((4, 9)) * 31)


def test_pad_empty():
    """No items -> all zeros, all-False mask."""
    data, mask = pad_or_truncate([], max_n=32, feature_shape=(20, 8))
    assert data.shape == (32, 20, 8)
    assert not mask.any()
    assert np.all(data == 0.0)


def test_agent_builder_type_index_covers_three_classes():
    """The agent builder maps exactly vehicle/pedestrian/bicycle to one-hot slots 0/1/2."""
    cfg = FeatureConfig()
    builder = AgentFeatureBuilder(cfg, Normalizer(cfg))
    assert builder._type_index == {"VEHICLE": 0, "PEDESTRIAN": 1, "BICYCLE": 2}
    assert cfg.agent_feature_dim == 6 + 3  # [x,y,sin,cos,vx,vy] + 3 one-hot


# ===========================================================================
# Group 5: normalization invertibility
# ===========================================================================

def test_norm_pos_invertible():
    cfg = FeatureConfig()
    nz = Normalizer(cfg)
    xy = np.array([[12.0, -34.0], [0.0, 119.0]])
    np.testing.assert_allclose(nz.denorm_pos(nz.norm_pos(xy)), xy, atol=1e-9)


def test_norm_pos_scale_is_unit_at_max_radius():
    """A point at the max radius normalizes to magnitude ~1 (the design intent)."""
    cfg = FeatureConfig()
    nz = Normalizer(cfg)
    out = nz.norm_pos(np.array([cfg.pos_scale_m, 0.0]))
    np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-9)


def test_norm_vel_acc_invertible():
    cfg = FeatureConfig()
    nz = Normalizer(cfg)
    v = np.array([3.3, -7.7])
    a = np.array([1.1, -2.2])
    np.testing.assert_allclose(nz.denorm_vel(nz.norm_vel(v)), v, atol=1e-9)
    np.testing.assert_allclose(nz.denorm_acc(nz.norm_acc(a)), a, atol=1e-9)


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, np.pi - 0.01, -np.pi + 0.01])
def test_heading_encode_decode_invertible(yaw):
    """(sin, cos) encoding decodes back to the original angle within (-pi, pi)."""
    s, c = Normalizer.encode_heading(np.array(yaw))
    back = Normalizer.decode_heading(s, c)
    np.testing.assert_allclose(back, yaw, atol=1e-9)


def test_heading_encode_is_unit_circle():
    """sin^2 + cos^2 == 1 for any heading (the encoding stays on the unit circle)."""
    yaws = np.linspace(-np.pi, np.pi, 37)
    s, c = Normalizer.encode_heading(yaws)
    np.testing.assert_allclose(s**2 + c**2, 1.0, atol=1e-12)


# ===========================================================================
# Group 6: FeatureConfig dimensions (lock the contract the docs/tensors rely on)
# ===========================================================================

def test_config_dimensions():
    cfg = FeatureConfig()
    assert cfg.history_steps == 20
    assert cfg.max_agents == 32
    assert cfg.ego_feature_dim == 8
    assert cfg.agent_feature_dim == 9
    assert cfg.map_feature_dim == 7
    assert cfg.route_points == 40
    assert cfg.dt == pytest.approx(0.1, abs=1e-12)
