"""F2 goal conditioning + F3 capacity-matched twin policy heads.

The 2x2 experiment's two controlled axes live here:

  Goal axis (F2) — the ONLY difference between conditions is whether the
  expert's answer leaks into the conditioning:
    route-region (R): heads see scene memory only; the route is already in
        the encoder's map/route tokens. p(traj | cond) stays multimodal at
        junctions — multiple legal continuations.
    precise (P):      heads additionally receive pinned expert-future points
        (4 s and 8 s waypoints from ego_future) as ONE extra cross-attention
        token. Two pinned points nearly determine the trajectory — the
        Phase-3d unimodal-collapse conditioning, reproduced deliberately.

  Head axis (F3) — one shared TrajectoryTrunk (temporal self-attention over
  H=16 waypoint tokens + cross-attention to scene memory), two thin wrappers:
    DeterministicHead: learned waypoint queries -> trunk -> (B,H,3) regression.
    DiffusionHead:     noisy traj + timestep embedding -> trunk -> x0-pred.
  Capacity is matched structurally (same trunk); the denoiser's timestep
  machinery makes exact equality impossible — tests enforce <10% gap and the
  honest number is reported, not hidden.

Diffusion: cosine schedule (Nichol & Dhariwal, arXiv:2102.09672), T=100,
x0-parameterization (arXiv:2501.15564 — stable for low-dim structured
trajectories and lets trajectory-space aux losses apply directly to the
prediction). Samplers (DDIM / DPM-Solver++) are inference-time concerns and
live with the evaluation code, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# WHY 7 and 15: ego_future is 16 waypoints @ 2 Hz over 8 s; index 7 = 4.0 s
# ("near"), index 15 = 8.0 s ("far"). Pinning two future points is what made
# the Phase-3d conditioning effectively unimodal — condition P reproduces it.
NEAR_IDX, FAR_IDX = 7, 15


def build_precise_goal(ego_future: torch.Tensor) -> torch.Tensor:
    """(B, 16, 3) ego_future -> (B, 4) [near_xy, far_xy] precise goal."""
    return torch.cat(
        [ego_future[:, NEAR_IDX, :2], ego_future[:, FAR_IDX, :2]], dim=-1
    )


@dataclass
class HeadConfig:
    d_model: int = 128
    n_heads: int = 8
    n_blocks: int = 2
    ffn_mult: int = 4
    horizon: int = 16
    out_ch: int = 3          # (x, y, heading)
    goal_ch: int = 4         # [near_xy, far_xy]
    dropout: float = 0.0
    T: int = 100             # diffusion training steps


class CosineSchedule(nn.Module):
    """Cosine alpha-bar schedule. REF: arXiv:2102.09672 eq. 17."""

    def __init__(self, T: int = 100, s: float = 0.008):
        super().__init__()
        t = torch.linspace(0, T, T + 1) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(1e-5, 1.0)[1:]  # (T,) at integer steps 1..T
        self.register_buffer("alphas_cumprod", abar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Closed-form forward noising: sqrt(abar_t) x0 + sqrt(1-abar_t) eps."""
        ab = self.alphas_cumprod[t].view(-1, *[1] * (x0.dim() - 1))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


class _Block(nn.Module):
    """Pre-LN: temporal self-attn over waypoints, cross-attn to memory, FFN."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d), nn.GELU(), nn.Linear(cfg.ffn_mult * d, d)
        )
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.ln2(x)
        x = x + self.cross_attn(h, kv, kv, need_weights=False)[0]
        return x + self.ffn(self.ln3(x))


class TrajectoryTrunk(nn.Module):
    """Shared by both heads — capacity matching lives here.

    WHY cross-attention (not concat): each waypoint token can attend to
    different scene entities (lead car near-term, turn lane far-term), which
    concatenation structurally cannot express (STAGE_3 §3).
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.goal_proj = nn.Sequential(nn.Linear(cfg.goal_ch, d), nn.ReLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_blocks))
        self.ln_out = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cfg.out_ch)
        self.time_pe = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, goal: torch.Tensor | None) -> torch.Tensor:
        # WHY goal as an extra KV token: conditions R (goal=None) and P share
        # every parameter and differ only in what cross-attention may read.
        kv = memory
        if goal is not None:
            kv = torch.cat([memory, self.goal_proj(goal).unsqueeze(1)], dim=1)
        x = x + self.time_pe
        for blk in self.blocks:
            x = blk(x, kv)
        return self.out_proj(self.ln_out(x))


class DeterministicHead(nn.Module):
    """Regressor twin: learned waypoint queries -> trunk -> trajectory."""

    def __init__(self, cfg: HeadConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or HeadConfig()
        self.queries = nn.Parameter(torch.randn(cfg.horizon, cfg.d_model) * 0.02)
        self.trunk = TrajectoryTrunk(cfg)

    def forward(self, memory: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        B = memory.shape[0]
        x = self.queries.unsqueeze(0).expand(B, -1, -1)
        return self.trunk(x, memory, goal)


class DiffusionHead(nn.Module):
    """Denoiser twin: x0-prediction from (x_t, t, memory, goal)."""

    def __init__(self, cfg: HeadConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or HeadConfig()
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.out_ch, d)
        # sinusoidal t -> MLP, added to every waypoint token (FiLM-lite)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.trunk = TrajectoryTrunk(cfg)

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d_model
        half = d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.t_mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.in_proj(x_t) + self._t_embed(t).unsqueeze(1)
        return self.trunk(x, memory, goal)
