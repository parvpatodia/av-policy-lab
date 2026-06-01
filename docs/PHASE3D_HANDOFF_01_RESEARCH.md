# Phase 3d Research Handoff — Literature Survey

**Date:** 2026-06-01  
**Author:** Claude Sonnet 4.6 (orchestrator)  
**Stage:** 1 of 5

---

## 1. Papers Surveyed

### 1.1 Diffusion Planner (ICLR 2025 Oral, arXiv 2501.15564)

**Citation:** Zheng et al. "Diffusion-Based Planning for Autonomous Driving with Flexible Guidance." ICLR 2025.

**What it does:**
- Transformer-based DiT (Diffusion Transformer) denoiser applied to nuPlan closed-loop planning
- Jointly models ego future trajectories AND key agent predictions under one architecture
- Uses flexible classifier guidance: gradient of a learned trajectory score function steers generation
  toward safe, on-route trajectories at inference time (no retraining for different cost functions)
- Action space: ego trajectory in token form, joint with surrounding agent futures
- Achieves ~20 Hz inference via few-step DDIM

**How they condition:** Ego state + HD map tokens + surrounding agent tokens are encoded by a transformer.
The noised future trajectory is also tokenized and fed to the DiT denoiser. Conditioning is via
cross-attention between the noised trajectory tokens and the context tokens — not simple concatenation.

**What they achieve:** State-of-the-art closed-loop on nuPlan mini and a 200-hour proprietary dataset.
The classifier guidance mechanism post-hoc refines trajectories for safety without retraining.

### 1.2 Diffusion Policy (RSS 2023 / IJRR 2025, arXiv 2303.04137)

**Citation:** Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023.

**What it does:**
- Represents a robot's visuomotor policy as a conditional DDPM over action sequences
- Two denoiser variants: (a) 1D U-Net CNN over the action time dimension, (b) Transformer
- For the CNN variant: observation features are injected via FiLM conditioning (Feature-wise Linear
  Modulation — scale and shift the hidden features at each layer) — NOT simple concatenation
- For the Transformer variant: observation tokens are prepended and processed via causal self-attention
- T=100 diffusion steps during training; DDIM with 10 steps at inference (10x speedup)
- Noise schedule: cosine (Nichol & Dhariwal 2021), better than linear for trajectory prediction
- Loss: predict noise ε (standard DDPM formulation, not x_0 prediction)
- Demonstrated on Push-T, Block Pushing, robomimic tasks — strong multi-modality handling

**DDIM formula used (Song et al. 2020, eq. 12):**
```
x_{t-1} = sqrt(alpha_{t-1}) * (x_t - sqrt(1-alpha_t) * epsilon_theta(x_t, t)) / sqrt(alpha_t)
           + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon_theta(x_t, t)
           + sigma_t * z
```
where sigma_t=0 for deterministic DDIM (eta=0). This is the equation we implement.

### 1.3 BESO (RSS 2023, arXiv 2304.02532)

**Citation:** Reuss et al. "Goal-Conditioned Imitation Learning using Score-based Diffusion Policies." RSS 2023.

**What it does:**
- Applies score-based diffusion to goal-conditioned IL (robot arm manipulation)
- Goal is concatenated with current state in a pre-conditioning wrapper (GCDenoiser class)
- Uses score transformer (GPT-style) as the denoiser — goal/state as context tokens
- Fast inference: as few as 3 denoising steps via classifier-free guidance
- Relevance to Phase 3d: confirms that goal concatenation into the denoiser context is valid and
  sufficient for goal-conditioned diffusion IL. Full cross-attention is not required.

### 1.4 Diffusion for Autonomous Driving (2025-2026 survey)

**Key 2025 papers found:**
- **Diffusion Planner** (arXiv 2501.15564) — nuPlan SOTA, DiT with classifier guidance (above)
- **TAT (Trajectory Aggregation Tree)** (arXiv 2405.17879) — addresses stochastic risk in diffusion
  planners via ensemble aggregation; relevant risk: a single sample from a diffusion planner may be
  inconsistent. We mitigate by sampling K=8 candidates and scoring.
- **DITA** (ICCV 2025) — Scaling DiT for VLA policy; confirms transformer denoiser is the 2025 SOTA

**nuPlan-specific context:**
The Diffusion Planner (2501.15564) is the only paper we found that applies diffusion directly to
nuPlan closed-loop. It uses the full nuPlan scenario context (map, agents). Our Phase 3d is
intentionally simpler: we use only the pre-computed route dual-horizon goal (same 10-dim input as
DualHorizon) to keep the ablation clean — same input, different generative model.

---

## 2. How We Differ from Diffusion Planner (2501.15564)

| Axis | Diffusion Planner (2501.15564) | Phase 3d (ours) |
|---|---|---|
| Goal | Full HD map + agent context | Dual-horizon route goal (10-dim) |
| Architecture | DiT transformer, large | 4-layer MLP denoiser, small (~200K params) |
| Conditioning | Cross-attention (map/agent tokens) | Concatenation (goal appended to noised traj) |
| Inference | Classifier guidance refinement | DDIM, score by near-goal proximity |
| Scope | Full nuPlan benchmark | 30-scenario closed-loop ablation |
| Purpose | Production SOTA system | Ablation: does generative model fix mode-swap? |

**Why we are simpler and that is correct:**
Phase 3d is a controlled ablation. The input is held IDENTICAL to DualHorizonRouteMapBC (10-dim).
The ONLY change is the policy head: deterministic MLP → generative DDPM. If performance improves,
the cause is the generative model's ability to represent multi-modal junction distributions. A
DiT with map tokens would confound the ablation.

---

## 3. Chosen Architecture: GoalConditionedDenoiser

**Rationale:** Chi et al. show that for low-dimensional action spaces (ours: 48-dim trajectory),
a simple MLP denoiser with FiLM conditioning matches or exceeds 1D CNN and transformer variants
while being significantly faster to train. BESO confirms concatenation is sufficient for goal
conditioning. We use a hybrid: MLP denoiser with the conditioning (10-dim state+goal) concatenated
directly to the noised trajectory + diffusion timestep embedding.

**Architecture:** `ε_θ(x_t, t, c)` where:
- `x_t`: noised 48-dim trajectory (float32)
- `t`: diffusion timestep → sinusoidal embedding → 64-dim (float32)
- `c`: 10-dim normalized conditioning (state + near goal + far goal)
- Input to MLP: [x_t (48) ‖ t_emb (64) ‖ c_norm (10)] = 122-dim
- Hidden: 256 → 256 → 256 (ReLU, same depth as GoalBCPolicy for fair comparison)
- Output: 48-dim predicted noise ε

**Why concatenation not FiLM:**
FiLM requires separate scale/shift networks per layer (adds ~100K params and complexity). For a
10-dim conditioning signal (10 numbers vs. image features), concatenation is sufficient — the
network can learn to gate on the conditioning via the linear layers. FiLM is needed when the
conditioning is high-dimensional (images, map tokens) and needs spatial modulation.

---

## 4. Action Space Representation

**Decision:** Same as all prior planners — [(dx, dy, d_yaw) × 16] = 48-dim in ego frame.

**Why not change:** Every prior planner uses this representation. Changing it would break the
ability to reuse _build_trajectory() and confound the ablation. The 48-dim is low enough that
a simple MLP denoiser is appropriate (Chi et al. use MLPs for <100-dim action spaces).

**Normalization:** Trajectory targets normalized by training dataset mean/std (same as DualHorizon).
The diffusion model operates in NORMALIZED space — noising and denoising happen on normalized
trajectories, then denormalize at inference. This is critical: the noise schedule assumes unit
variance, so training data must be ~N(0,1).

---

## 5. Conditioning Approach

**Decision:** Concatenate normalized 10-dim conditioning to every denoiser input.

**Pipeline:**
1. At training: extract same dual-horizon goal as DualHorizonRouteMapBC (reuse extract_from_db)
2. Normalize X with X_mean/X_std (same as DualHorizon training)
3. For each training window: sample t ~ Uniform[1, T], add noise to normalized trajectory,
   feed [x_t ‖ t_emb ‖ c_norm] to denoiser, predict ε, minimize MSE(ε_pred, ε_true)
4. At inference: build dual-horizon goal via DualHorizonRouteMapBCPlanner logic,
   normalize with saved X_mean/X_std, run DDIM for K=8 samples, score by near-goal proximity

---

## 6. Key Risks and Potential Failure Modes

| Risk | Probability | Mitigation |
|---|---|---|
| Mode collapse (denoiser always outputs same trajectory) | Medium | Check that K=8 samples have >0.5m spread at step 8; if not, increase noise schedule variance |
| Diffusion mode is not the mode-swap fix (underlying bimodality is data-sparse) | Low-medium | Compute per-scenario L2 vs DualHorizon; if same 4 failures remain, hypothesis B is confirmed |
| Training divergence on MPS (float16 precision in MPS mixed-mode) | Low | Force float32 throughout; cosine schedule is numerically stable |
| Scoring function selects wrong candidate (near-goal proximity is noisy) | Medium | Try both near-goal score and trajectory smoothness score; log K candidates per scenario |
| Val MSE plateau above DualHorizon (0.1159) | Low-medium | DualHorizon is a deterministic regressor; DDPM val MSE is noise prediction MSE — different scale. Use ADE as the comparable metric. |

---

## 7. Decision Summary

| Decision | Choice | Justification |
|---|---|---|
| Denoiser | 4-layer MLP (122→256→256→256→48) | Fast on M1 MPS; sufficient for 48-dim; Chi et al. validated |
| Timestep T | 100 (train), 10 (DDIM inference) | Standard from Chi et al.; 10x inference speedup |
| Noise schedule | Cosine (Nichol & Dhariwal 2021) | Better tail behavior than linear for trajectory sequences |
| Loss | Predict noise ε (not x_0) | More stable training; standard DDPM formulation |
| Conditioning | Concatenation | Sufficient for 10-dim signal; simplest correct approach |
| K samples | 8 | Balance between diversity and inference speed; 8 DDIM runs at 10 steps each |
| Scoring | Near-goal proximity at step 8 | Directly tests if diffusion can commit to the correct mode |
