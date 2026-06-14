"""F1 — Wayformer-style scene encoder for av-policy-lab.

Consumes the F0 vectorized scene tensors (REAL on-disk shapes, verified
against f0_balanced on 2026-06-10 — the STAGE_2 doc figures are stale):

    ego            (B, 20, 8)        agents         (B, 32, 20, 9)
    agent_mask     (B, 32, 20) bool  map_polylines  (B, 128, 20, 7)
    map_mask       (B, 128) bool     crosswalks     (B, 16, 20, 2)
    crosswalk_mask (B, 16) bool      route_polyline (B, 40, 4)
    route_mask     (B, 40) bool      traffic_lights (B, 128) int64

Produces scene memory M of shape (B, num_latents=32, d_model=128) — the
conditioning interface the capacity-matched twin heads (MLP / diffusion)
cross-attend to.

Architecture (STAGE_3 recipe, sized per STAGE_0_CONTRIBUTION to <2.5M
params — bigger encoders overfit closed-loop at the ~435K-sample scale):
per-entity tokenizers -> early-fusion transformer over all tokens ->
learned latent queries cross-attend -> M.
REF: Wayformer arXiv:2207.05844 (early fusion + latent queries),
     VectorNet arXiv:2005.04259 (polyline subgraph pooling).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# WHY +1/clamp: nuPlan traffic-light ids include -1 ("unknown/none"); shift
# into [0, NUM_TL_STATES-1] so they index a learnable embedding safely.
NUM_TL_STATES = 8


@dataclass
class SceneEncoderConfig:
    d_model: int = 128
    n_heads: int = 8
    n_fusion_layers: int = 4
    ffn_mult: int = 4
    num_latents: int = 32
    dropout: float = 0.0  # WHY 0: encoder is small vs data; add only if val overfits
    n_agents: int = 32
    agent_hist: int = 20
    agent_ch: int = 9
    ego_hist: int = 20
    ego_ch: int = 8
    map_pts: int = 20
    map_ch: int = 7
    xwalk_ch: int = 2
    route_ch: int = 4


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))


class PolylineTokenizer(nn.Module):
    """VectorNet-style subgraph: per-point MLP then masked max-pool over points.

    WHY max-pool over points (not flatten): a polyline is an unordered-ish set
    of resampled points; pooling is permutation-tolerant and parameter-cheap.
    """

    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.point_mlp = _mlp(in_ch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, P, C) -> (B, N, D); pool over the P point axis
        return self.point_mlp(x).amax(dim=2)


class SceneEncoder(nn.Module):
    def __init__(self, cfg: SceneEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SceneEncoderConfig()
        d = cfg.d_model

        # ── tokenizers: every entity becomes ONE d-dim token ──────────────
        # WHY flatten history here but never in the decoder: the encoder
        # SUMMARIZES observed context (a fixed input), while the decoder
        # GENERATES the future time axis — collapsing time is only a sin on
        # the generated side (root cause #5 of the Phase-3d analysis).
        self.ego_tok = _mlp(cfg.ego_hist * cfg.ego_ch, d)
        self.agent_tok = _mlp(cfg.agent_hist * cfg.agent_ch, d)
        self.map_tok = PolylineTokenizer(cfg.map_ch, d)
        self.xwalk_tok = PolylineTokenizer(cfg.xwalk_ch, d)
        self.route_tok = _mlp(cfg.route_ch, d)  # per-point: route stays 40 tokens
        self.tl_embed = nn.Embedding(NUM_TL_STATES, d)
        # WHY type embeddings: after fusion-concat, attention has no other way
        # to know an agent token from a lane token. 5 modalities.
        self.type_embed = nn.Parameter(torch.zeros(5, d))

        # ── early-fusion transformer over the full token set ──────────────
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_mult * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=cfg.n_fusion_layers)

        # ── latent-query bottleneck -> scene memory ────────────────────────
        # WHY: compresses ~217 tokens to 32 so the diffusion head's per-step,
        # per-candidate cross-attention stays cheap (Wayformer latent queries).
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, d) * 0.02)
        self.latent_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.latent_ffn = _mlp(d, d)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, batch: dict) -> torch.Tensor:
        cfg = self.cfg
        B = batch["ego"].shape[0]

        # tokenize each modality
        ego = self.ego_tok(batch["ego"].flatten(1)).unsqueeze(1)        # (B,1,D)
        agents = self.agent_tok(batch["agents"].flatten(2))             # (B,32,D)
        maps = self.map_tok(batch["map_polylines"])                     # (B,128,D)
        maps = maps + self.tl_embed((batch["traffic_lights"] + 1).clamp(0, NUM_TL_STATES - 1))
        xwalks = self.xwalk_tok(batch["crosswalks"])                    # (B,16,D)
        route = self.route_tok(batch["route_polyline"])                 # (B,40,D)

        tokens = torch.cat(
            [
                ego + self.type_embed[0],
                agents + self.type_embed[1],
                maps + self.type_embed[2],
                xwalks + self.type_embed[3],
                route + self.type_embed[4],
            ],
            dim=1,
        )  # (B, 217, D)

        # padding mask: True = IGNORE (torch convention). Ego always valid.
        pad = torch.cat(
            [
                torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
                ~batch["agent_mask"].any(-1),   # agent valid if any timestep valid
                ~batch["map_mask"],
                ~batch["crosswalk_mask"],
                ~batch["route_mask"],
            ],
            dim=1,
        )  # (B, 217)

        # WHY zero padded tokens BEFORE fusion: key_padding_mask stops attention
        # from READING them, but garbage values would still flow through the
        # residual stream of their own positions; zeroing makes padding inert.
        tokens = tokens.masked_fill(pad.unsqueeze(-1), 0.0)

        fused = self.fusion(tokens, src_key_padding_mask=pad)           # (B,217,D)

        q = self.ln_q(self.latents.unsqueeze(0).expand(B, -1, -1))      # (B,32,D)
        kv = self.ln_kv(fused)
        mem, _ = self.latent_attn(q, kv, kv, key_padding_mask=pad, need_weights=False)
        mem = mem + self.latent_ffn(mem)
        return self.ln_out(mem)                                          # (B,32,D)
