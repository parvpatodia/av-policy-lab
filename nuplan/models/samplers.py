"""DDIM sampling for the diffusion policy head.

REF: arXiv:2010.02502 (Song et al.). With eta=0 the reverse process is
deterministic given x_T, so trajectory diversity comes only from the K
initial noise draws — exactly the property the 2x2 experiment measures.

The head predicts x0 (not eps), so the DDIM update is written in x0 form:
    eps_t = (x_t - sqrt(abar_t) * x0_pred) / sqrt(1 - abar_t)
    x_{t-1} = sqrt(abar_{t-1}) * x0_pred + sqrt(1 - abar_{t-1}) * eps_t
"""
from __future__ import annotations

import torch

from models.policy_heads import CosineSchedule, DiffusionHead


@torch.no_grad()
def ddim_sample(
    head: DiffusionHead,
    schedule: CosineSchedule,
    memory: torch.Tensor,
    goal: torch.Tensor | None = None,
    num_samples: int = 1,
    num_steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample trajectories in the SCALED space the head was trained in.

    Returns (B, K, H, 3); caller applies unscale_future for meter-space metrics.
    """
    B = memory.shape[0]
    K, H, C = num_samples, head.cfg.horizon, head.cfg.out_ch
    T = schedule.alphas_cumprod.shape[0]
    device = memory.device

    # WHY expand memory/goal over K here: one batched forward per step scores
    # all K candidates; looping K times would multiply sampler latency by K.
    mem_k = memory.repeat_interleave(K, dim=0)
    goal_k = goal.repeat_interleave(K, dim=0) if goal is not None else None

    x = torch.randn(B * K, H, C, device=device, generator=generator)
    # evenly spaced timesteps T-1 .. 0, e.g. T=100, steps=20 -> 99,94,...,4
    ts = torch.linspace(T - 1, 0, num_steps, device=device).round().long()

    for i, t in enumerate(ts):
        t_b = torch.full((B * K,), int(t), device=device, dtype=torch.long)
        x0_pred = head(x, t_b, mem_k, goal=goal_k)
        ab_t = schedule.alphas_cumprod[t]
        if i + 1 < num_steps:
            t_prev = ts[i + 1]
            ab_prev = schedule.alphas_cumprod[t_prev]
            eps = (x - ab_t.sqrt() * x0_pred) / (1 - ab_t).sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        else:
            x = x0_pred
    return x.view(B, K, H, C)
