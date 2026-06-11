"""Tests for the DDIM sampler over the x0-predicting diffusion head."""
import torch

from models.policy_heads import CosineSchedule, DiffusionHead, HeadConfig
from models.samplers import ddim_sample

B, H = 3, 16


def _setup():
    torch.manual_seed(0)
    head = DiffusionHead(HeadConfig(n_blocks=1)).eval()
    sch = CosineSchedule(T=100)
    mem = torch.randn(B, 32, 128)
    return head, sch, mem


def test_shapes_and_finite():
    head, sch, mem = _setup()
    out = ddim_sample(head, sch, mem, num_samples=4, num_steps=5)
    assert out.shape == (B, 4, H, 3)
    assert torch.isfinite(out).all()


def test_deterministic_given_seed():
    head, sch, mem = _setup()
    a = ddim_sample(head, sch, mem, num_samples=2, num_steps=5,
                    generator=torch.Generator().manual_seed(7))
    b = ddim_sample(head, sch, mem, num_samples=2, num_steps=5,
                    generator=torch.Generator().manual_seed(7))
    c = ddim_sample(head, sch, mem, num_samples=2, num_steps=5,
                    generator=torch.Generator().manual_seed(8))
    # eta=0: all diversity comes from x_T, so same seed -> same samples
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_k_candidates_differ():
    head, sch, mem = _setup()
    out = ddim_sample(head, sch, mem, num_samples=2, num_steps=5,
                      generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_goal_conditioning_changes_samples():
    head, sch, mem = _setup()
    g = torch.Generator().manual_seed(0)
    no_goal = ddim_sample(head, sch, mem, goal=None, num_samples=1, num_steps=5,
                          generator=g)
    g = torch.Generator().manual_seed(0)
    goal = torch.randn(B, 4)
    with_goal = ddim_sample(head, sch, mem, goal=goal, num_samples=1, num_steps=5,
                            generator=g)
    assert not torch.allclose(no_goal, with_goal)


def test_recovers_overfit_target():
    """End-to-end sanity: train the head to denoise one trajectory, then the
    sampler must land near it. This is the only test that proves the x0-form
    DDIM update is algebraically right, not just shape-correct."""
    torch.manual_seed(0)
    head = DiffusionHead(HeadConfig(n_blocks=1))
    sch = CosineSchedule(T=100)
    mem = torch.randn(1, 32, 128)
    x0 = torch.randn(1, H, 3) * 0.5
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(800):
        t = torch.randint(0, 100, (1,))
        eps = torch.randn_like(x0)
        loss = (head(sch.q_sample(x0, t, eps), t, mem) - x0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    out = ddim_sample(head, sch, mem, num_samples=4, num_steps=20,
                      generator=torch.Generator().manual_seed(1))
    err = (out - x0.unsqueeze(1)).norm(dim=-1).mean(dim=-1).min().item()
    # measured 0.019 at 800 steps; 0.1 leaves margin without hiding regressions
    assert err < 0.1, f"sampler did not recover overfit target: {err:.3f}"
