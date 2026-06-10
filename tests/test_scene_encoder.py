"""Tests for the F1 scene encoder (nuplan/models/scene_encoder.py).

Pure torch — no nuPlan devkit needed. Shapes match the REAL f0_balanced
on-disk data (verified 2026-06-10), not the stale STAGE_2 doc figures.
"""
import torch
import pytest

from models.scene_encoder import SceneEncoder, SceneEncoderConfig


B = 3


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    batch = {
        "ego": rnd(B, 20, 8),
        "agents": rnd(B, 32, 20, 9),
        "agent_mask": torch.rand(B, 32, 20, generator=g) > 0.4,
        "map_polylines": rnd(B, 128, 20, 7),
        "map_mask": torch.rand(B, 128, generator=g) > 0.3,
        "crosswalks": rnd(B, 16, 20, 2),
        "crosswalk_mask": torch.rand(B, 16, generator=g) > 0.5,
        "route_polyline": rnd(B, 40, 4),
        "route_mask": torch.rand(B, 40, generator=g) > 0.2,
        "traffic_lights": torch.randint(-1, 4, (B, 128), generator=g),
    }
    # guarantee at least one valid entity per modality per sample
    batch["agent_mask"][:, 0, :] = True
    batch["map_mask"][:, 0] = True
    batch["crosswalk_mask"][:, 0] = True
    batch["route_mask"][:, 0] = True
    return batch


@pytest.fixture(scope="module")
def model():
    m = SceneEncoder(SceneEncoderConfig())
    m.eval()
    return m


def test_output_shape_and_finite(model):
    cfg = model.cfg
    with torch.no_grad():
        mem = model(_batch())
    assert mem.shape == (B, cfg.num_latents, cfg.d_model)
    assert torch.isfinite(mem).all()


def test_padding_invariance(model):
    """Garbage written into fully-masked slots must not change the output."""
    a, b = _batch(), _batch()
    # corrupt padded agents (rows whose mask is all-False) and padded map rows
    pad_agents = ~b["agent_mask"].any(-1)            # (B, 32)
    b["agents"][pad_agents] = 1e6
    b["map_polylines"][~b["map_mask"]] = -1e6
    b["crosswalks"][~b["crosswalk_mask"]] = 1e6
    b["route_polyline"][~b["route_mask"]] = -1e6
    with torch.no_grad():
        ma, mb = model(a), model(b)
    assert torch.allclose(ma, mb, atol=1e-5), "padding leaked into scene memory"


def test_determinism_eval(model):
    with torch.no_grad():
        m1, m2 = model(_batch()), model(_batch())
    assert torch.equal(m1, m2)


def test_param_budget(model):
    n = sum(p.numel() for p in model.parameters())
    assert n < 2_500_000, f"{n} params exceeds the 2.5M budget (STAGE_0_CONTRIBUTION)"
    assert n > 200_000, f"{n} params is suspiciously small — wiring bug?"


def test_gradients_flow(model):
    model.train()
    mem = model(_batch())
    mem.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "some parameters got no gradient"
    model.zero_grad(set_to_none=True)
    model.eval()


def test_handles_minimal_scene(model):
    """A scene with only the guaranteed-valid entities must not NaN."""
    batch = _batch()
    batch["agent_mask"][:, 1:, :] = False
    batch["map_mask"][:, 1:] = False
    batch["crosswalk_mask"][:, 1:] = False
    batch["route_mask"][:, 1:] = False
    with torch.no_grad():
        mem = model(batch)
    assert torch.isfinite(mem).all()
