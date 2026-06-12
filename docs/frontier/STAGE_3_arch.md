# STAGE 3 — ARCHITECTURE
> NOTE (2026-06-11): tensor shapes in this document may lag the code. The canonical shapes are the asserts in nuplan/features/scene_features.py (_assert_sample_consistency) plus the F0 v2 additions: ego_future (16,3) and per-sample scenario identifiers.


> Status: DESIGN ONLY. Dims/param counts are target specifications. Builds on Stage 2 tensors and the multimodal route-region goal.

## Overview (data flow)

```
vectorized inputs (ego, agents, map, route)  [Stage 2 §2]
        │  per-entity MLP / MLP-Mixer tokenizer
        ▼
   entity tokens  ──►  Early-fusion Transformer encoder (Wayformer)
        │                       │
        │              Latent-query bottleneck  ──►  scene memory  M ∈ [L, D]
        ▼
 route-region goal tokens  (concatenated into scene memory as extra tokens)
        ▼
 TEMPORAL diffusion decoder:  x_t ∈ [H, 3]  --transformer w/ self-attn over time
        │                                       + CROSS-ATTN to scene memory M
        ▼
 x0-prediction  x̂0 ∈ [H, 3]   (DPM-Solver++ sampling, K candidates)
```

Total target size ≈ **8–15M params** (Diffusion-Planner / PLUTO class; ~100× the current 175K MLP, but still single-A100-trainable). WHY this size: SOTA planners are in the 5–20M range; the current 175K MLP is far below the capacity floor where scene reasoning is possible, but we deliberately avoid LLM-scale — the bottleneck was representation, not parameters.

## 1. Tokenizer (per-entity embedding)

- Each agent's `[T_h,8]` history → **1D-conv + MLP** → token `∈ R^D`. Map polyline `[S,20,8]` → **PointNet-style max-pool MLP** (VectorNet subgraph) → token. Ego → MLP token. Route polyline → MLP tokens.
- Add **learnable type embeddings** (agent class, lane type, TL state) and **Fourier positional embedding** of `(p,θ)`. REF: arXiv:2404.14327 (PLUTO), arXiv:2005.04259 (VectorNet).
- `D = 128`.

## 2. Scene encoder (Wayformer early-fusion + latent queries)

- **Early fusion:** concatenate all entity tokens (ego + 32 agents + 128 map + 40 route ≈ 201 tokens) into one set; run **L_enc = 6** standard transformer encoder layers (8 heads, FFN 4×D, pre-LN). WHY early fusion: Wayformer shows it is modality-agnostic and SOTA, and is the simplest correct choice. REF: arXiv:2207.05844
- **Latent-query bottleneck:** **L = 32 learnable latent queries** cross-attend the full token set → compressed scene memory `M ∈ [32, 128]`. WHY: Wayformer's latent-query attention gives 2–16× speedup with no quality loss; keeps the cross-attention in the diffusion decoder cheap (denoiser runs K×T times per scene, so a small memory matters). REF: arXiv:2207.05844
- Encoder param count ≈ 3–5M.

## 3. Conditioning embedding & WHY CROSS-ATTENTION (fixes root cause #1 mechanism)

- The conditioning is the **scene memory M (32 tokens)** plus **route-region goal tokens**, NOT a single concatenated vector.
- **Cross-attention, not concatenation.** WHY this matters specifically when conditioning is rich and multimodal:
  1. Concatenation forces the denoiser to compress the entire scene into a fixed vector that is *summed into every timestep identically* — it cannot attend to *different* scene entities for *different* parts of the trajectory (e.g., attend to the lead vehicle for the near future, the turn lane for the far future).
  2. With cross-attention, each denoised timestep query can **selectively attend** to the relevant agents/lanes, which is exactly what multimodal junction behavior requires.
  3. Empirically this is the Diffusion Planner design (multi-head cross-attention to encoded scene tokens). REF: arXiv:2501.15564.
  - The old 10-dim concat-MLP could not represent "which of several legal paths" — it averaged them (the conditional-mean collapse). Cross-attention + a multimodal goal is the joint fix.

## 4. Temporal diffusion decoder (fixes root cause #5)

**Choice: transformer denoiser over the time axis** (not 1D-conv-UNet).

- **Horizon decided (E4 fix): H = 16 waypoints @ 2 Hz over 8 s.** WHY: keeps the attention sequence short/cheap, matches the planning horizon of the SOTA planners, and the nuPlan tracker upsamples the 2 Hz plan to the 10 Hz control rate at execution — so 16 waypoints is sufficient resolution for closed-loop and far cheaper than 80.
- Input: noisy trajectory `x_t ∈ [H=16,3]` → linear to `[H, D]`; add **temporal positional embedding** + **diffusion-timestep embedding** (sinusoidal, FiLM-modulated into each block).
- **N_dec = 4 decoder blocks**, each: (a) **self-attention over the H time tokens** (models temporal smoothness/kinematics), (b) **cross-attention to scene memory M** (§3), (c) FFN. 8 heads, D=128.
- Output head: linear `[H,D] → [H,3]` predicting **x0** (clean trajectory).
- WHY transformer over conv-UNet: with H≤80 the sequence is short; self-attention over time captures long-range temporal structure (turn commitment) better than a fixed conv receptive field, and it composes naturally with the cross-attention conditioning. (1D-conv-UNet à la Diffusion Policy / Janner Diffuser is the documented fallback if attention is unstable at small data — keep as ablation. REF: Diffuser arXiv:2205.09991, Diffusion Policy arXiv:2303.04137.)
- Decoder param count ≈ 3–5M.

## 5. Diffusion formulation

- **x0-parameterization** (predict clean trajectory), matching Diffusion Planner. WHY: x0-pred is more stable for low-dimensional structured trajectory data and makes the auxiliary collision/drivable losses (Stage 4) directly applicable to the predicted trajectory each step. REF: arXiv:2501.15564.
- **Noise schedule:** keep **cosine** (already in repo, Nichol & Dhariwal) OR VP as in Diffusion Planner — ablate. T_train = 100.
- **Sampler: DPM-Solver++, ~10–15 steps** (replaces the current DDIM-10). WHY: higher-order solver, fewer steps for equal quality, the Diffusion-Planner choice. REF: arXiv:2501.15564, DPM-Solver++ arXiv:2211.01095.
- **K candidates** sampled per scene (K=8 kept); a lightweight **scorer head** (or the PDM proxy in Stage 4) selects the executed trajectory. WHY: this is how multimodality is *used* at inference — sample diverse legal trajectories, pick the best-scoring one.
- **Diversity-by-construction (E5 fix):** sampling stochasticity alone can yield K near-identical trajectories (mode collapse). Anchor the K candidates on the **route-goal lane set** (Stage 2 §4) — one initialization seed per candidate goal lane — so candidates are diverse *by construction* across the legal modes, then refined by the denoiser. This guarantees the multimodality is exercised rather than hoped for.

## 6. Optional: joint ego + agent prediction (stretch)

Diffusion Planner jointly denoises ego plan + neighbor futures, which improves interaction modeling. We mark this **stretch**: it roughly doubles decoder cost and complicates the loss. Core model denoises ego only with agents as *conditioning*; joint denoising is an ablation/extension if time permits.

## 7. Parameter budget summary

| Module | Params (target) |
|---|---|
| Tokenizers | ~1M |
| Encoder (6 layers) + latent queries | ~3–5M |
| Temporal diffusion decoder (4 blocks) | ~3–5M |
| Heads (x0, scorer) | <0.5M |
| **Total** | **~8–12M** |

Trainable on a single A100 (40/80GB) at batch 256–512 with AMP; multi-GPU DDP used mainly for throughput/scale (Stage 4).

## HANDOFF TO NEXT STAGE (Stage 4 — Training)

- Loss must combine: diffusion x0/noise loss + **imitation smooth-L1** + **auxiliary drivable-area & collision losses** (PLUTO) applied to x̂0 each step + optional mode/score CE + (stretch) contrastive CIL.
- Ablations the architecture is explicitly designed to support: **scene encoder on/off**, **cross-attn vs concat**, **temporal-transformer vs flat-MLP decoder**, **multimodal route-goal vs precise near/far goal**, **x0 vs ε-pred**, **DPM-Solver++ vs DDIM**, **perturbation on/off**. Each isolates one root-cause fix so gains are attributable.
- Encoder/decoder are separable modules → DDP-friendly; scene memory M is the clean interface between them.
- K-candidate sampling + scorer requires the PDM proxy from Stage 4/5 for trajectory selection.
