"""Tests for F2 goal building + F3 capacity-matched twin policy heads."""
import torch
import pytest

from models.policy_heads import (
    CosineSchedule,
    DeterministicHead,
    DiffusionHead,
    HeadConfig,
    build_precise_goal,
)
from models.scene_encoder import SceneEncoder, SceneEncoderConfig

B, H = 4, 16


def _memory(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 32, 128, generator=g)


def _future(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, 3, generator=g)


# ---------- F2: goal building ----------

def test_precise_goal_shape_and_source():
    fut = _future()
    goal = build_precise_goal(fut)
    assert goal.shape == (B, 4)
    # WHY these indices: 2Hz x 8s future -> idx 7 = 4s (near), idx 15 = 8s (far)
    assert torch.equal(goal[:, :2], fut[:, 7, :2])
    assert torch.equal(goal[:, 2:], fut[:, 15, :2])


# ---------- F3: heads ----------

@pytest.fixture(scope="module")
def heads():
    cfg = HeadConfig()
    det, diff = DeterministicHead(cfg), DiffusionHead(cfg)
    det.eval(), diff.eval()
    return det, diff


def test_det_head_shapes_both_conditions(heads):
    det, _ = heads
    mem = _memory()
    with torch.no_grad():
        out_r = det(mem, goal=None)                       # route-region condition
        out_p = det(mem, goal=build_precise_goal(_future()))  # precise condition
    assert out_r.shape == out_p.shape == (B, H, 3)
    assert torch.isfinite(out_r).all() and torch.isfinite(out_p).all()
    # the goal must actually flow: conditions must differ
    assert not torch.allclose(out_r, out_p)


def test_diff_head_shapes_both_conditions(heads):
    _, diff = heads
    mem, x0 = _memory(), _future()
    t = torch.randint(0, 100, (B,))
    x_t = torch.randn_like(x0)
    with torch.no_grad():
        x0_r = diff(x_t, t, mem, goal=None)
        x0_p = diff(x_t, t, mem, goal=build_precise_goal(x0))
    assert x0_r.shape == x0_p.shape == (B, H, 3)
    assert not torch.allclose(x0_r, x0_p)


def test_capacity_matched(heads):
    det, diff = heads
    n_det = sum(p.numel() for p in det.parameters())
    n_diff = sum(p.numel() for p in diff.parameters())
    rel = abs(n_det - n_diff) / max(n_det, n_diff)
    print(f"det={n_det} diff={n_diff} rel_gap={rel:.3%}")
    # WHY <10% not exact: the denoiser needs timestep machinery the regressor
    # lacks; matching the shared trunk is the scientific requirement.
    assert rel < 0.10, f"capacity gap {rel:.1%} breaks the twin comparison"


def test_schedule_endpoints():
    sch = CosineSchedule(T=100)
    x0 = _future()
    eps = torch.randn_like(x0)
    near0 = sch.q_sample(x0, torch.zeros(B, dtype=torch.long), eps)
    nearT = sch.q_sample(x0, torch.full((B,), 99, dtype=torch.long), eps)
    assert (near0 - x0).abs().mean() < 0.2 * (nearT - x0).abs().mean()
    # at T the sample should be mostly noise
    corr_T = torch.corrcoef(torch.stack([nearT.flatten(), eps.flatten()]))[0, 1]
    assert corr_T > 0.9


def test_gradients_flow(heads):
    det, diff = heads
    det.train(), diff.train()
    mem, x0 = _memory(), _future()
    # WHY goal supplied on both: with goal=None the goal_proj branch is
    # (correctly) unused, so its params get no grad — that's the R condition
    # working as designed, not a wiring bug.
    det(mem, goal=build_precise_goal(x0)).sum().backward()
    assert all(p.grad is not None for p in det.parameters())
    t = torch.randint(0, 100, (B,))
    diff(torch.randn_like(x0), t, mem, goal=build_precise_goal(x0)).sum().backward()
    assert all(p.grad is not None for p in diff.parameters())
    det.zero_grad(set_to_none=True), diff.zero_grad(set_to_none=True)
    det.eval(), diff.eval()


def test_overfit_one_batch():
    """The practical can-it-learn gate: both heads must crush a single batch."""
    torch.manual_seed(0)
    cfg = HeadConfig(n_blocks=1)
    mem, x0 = _memory(), _future()
    # deterministic: plain regression
    det = DeterministicHead(cfg)
    opt = torch.optim.Adam(det.parameters(), lr=3e-3)
    first = last = None
    for i in range(150):
        loss = (det(mem, goal=None) - x0).pow(2).mean()
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    assert last < first / 10, f"det head failed to overfit: {first:.4f}->{last:.4f}"
    # diffusion: x0-prediction at fixed t
    diff = DiffusionHead(cfg)
    sch = CosineSchedule(T=100)
    opt = torch.optim.Adam(diff.parameters(), lr=3e-3)
    t = torch.full((B,), 50, dtype=torch.long)
    eps = torch.randn_like(x0)
    x_t = sch.q_sample(x0, t, eps)
    first = last = None
    for i in range(150):
        loss = (diff(x_t, t, mem, goal=None) - x0).pow(2).mean()
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    assert last < first / 10, f"diff head failed to overfit: {first:.4f}->{last:.4f}"


def test_end_to_end_with_encoder():
    """Encoder memory -> both heads: the full F1->F3 pipe is shape-compatible."""
    enc = SceneEncoder(SceneEncoderConfig())
    enc.eval()
    g = torch.Generator().manual_seed(7)
    rnd = lambda *s: torch.randn(*s, generator=g)
    batch = {
        "ego": rnd(B, 20, 8), "agents": rnd(B, 32, 20, 9),
        "agent_mask": torch.ones(B, 32, 20, dtype=torch.bool),
        "map_polylines": rnd(B, 128, 20, 7),
        "map_mask": torch.ones(B, 128, dtype=torch.bool),
        "crosswalks": rnd(B, 16, 20, 2),
        "crosswalk_mask": torch.ones(B, 16, dtype=torch.bool),
        "route_polyline": rnd(B, 40, 4),
        "route_mask": torch.ones(B, 40, dtype=torch.bool),
        "traffic_lights": torch.randint(-1, 4, (B, 128), generator=g),
    }
    cfg = HeadConfig()
    det, diff = DeterministicHead(cfg).eval(), DiffusionHead(cfg).eval()
    with torch.no_grad():
        mem = enc(batch)
        traj = det(mem, goal=None)
        x0h = diff(torch.randn(B, H, 3), torch.randint(0, 100, (B,)), mem, goal=None)
    assert traj.shape == x0h.shape == (B, H, 3)
    assert torch.isfinite(traj).all() and torch.isfinite(x0h).all()
