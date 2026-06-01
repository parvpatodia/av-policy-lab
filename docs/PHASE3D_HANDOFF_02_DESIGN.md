# Phase 3d Architecture Design Spec

**Date:** 2026-06-01  
**Author:** Claude Sonnet 4.6 (orchestrator)  
**Stage:** 2 of 5

---

## 1. System Overview

```
TRAINING:
  DB files (64) → extract_from_db() → (X: 10-dim, Y: 48-dim) pairs
                → normalize X, Y → DDPM training loop
                → GoalConditionedDenoiser ε_θ
                → checkpoint: trained_diffusion_policy.pt

INFERENCE:
  ego_state + route → dual-horizon goals (10-dim, same as DualHorizon)
                    → normalize with saved X_mean/X_std
                    → DDIM loop (K=8 samples, 10 steps each)
                    → score by near-goal proximity at step 8
                    → best trajectory → denormalize → InterpolatedTrajectory
```

---

## 2. Input/Output Specification

### 2.1 Conditioning vector c (10-dim)

| Index | Feature | Description |
|---|---|---|
| 0 | sin(yaw) | Ego heading sin |
| 1 | cos(yaw) | Ego heading cos |
| 2 | vx | Rear axle velocity x [m/s] |
| 3 | vy | Rear axle velocity y [m/s] |
| 4 | ax | Rear axle acceleration x [m/s^2] |
| 5 | ay | Rear axle acceleration y [m/s^2] |
| 6 | dx_near | Near goal x in ego frame [m] (speed × 0.8s arc-length) |
| 7 | dy_near | Near goal y in ego frame [m] |
| 8 | dx_far | Far goal x in ego frame [m] (20m arc-length fixed) |
| 9 | dy_far | Far goal y in ego frame [m] |

**Identical to DualHorizonRouteMapBC** — this is the control condition. Goal extraction uses
`extract_from_db()` from `train_dual_horizon.py` unchanged.

### 2.2 Action/output trajectory a (48-dim)

[(dx_0, dy_0, d_yaw_0), (dx_1, dy_1, d_yaw_1), ..., (dx_15, dy_15, d_yaw_15)]
= 16 steps × 3 dims = 48-dim ego-frame relative trajectory.

Identical representation to all prior planners. Step j is the ego rear-axle displacement
from the current position at time j+1, expressed in the current ego coordinate frame.

### 2.3 Diffusion variable x_t (48-dim)

At training step t: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε
where x_0 is the normalized ground-truth trajectory and ε ~ N(0, I).

At inference: x_T ~ N(0, I), then DDIM denoises to x_0.

---

## 3. Denoiser Architecture: GoalConditionedDenoiser

### 3.1 Sinusoidal timestep embedding

REF: Vaswani et al. (2017) "Attention Is All You Need", position encoding.

For diffusion timestep t ∈ {1, ..., T}:
```
embed_dim = 64
freqs = exp(-log(10000) * arange(0, embed_dim//2) / (embed_dim//2))
emb = [sin(t * freqs), cos(t * freqs)]   # shape: (embed_dim,) = (64,)
```

WHY sinusoidal over learned: same reason as transformers — extrapolates gracefully to timesteps
not seen during training, avoids overfitting on the schedule. 64-dim gives 32 frequency bands,
sufficient to distinguish 100 timestep levels.

### 3.2 MLP denoiser

```
Input: [x_t (48) ‖ t_emb (64) ‖ c_norm (10)] = 122-dim concatenation

Layer 1: Linear(122, 256), ReLU
Layer 2: Linear(256, 256), ReLU
Layer 3: Linear(256, 256), ReLU
Layer 4: Linear(256, 48)         <- predicts noise ε

Total parameters: 122×256 + 256 + 256×256 + 256 + 256×256 + 256 + 256×48 + 48
                = 31,232 + 65,792 + 65,792 + 12,336 = ~175,172 params
```

WHY same depth as GoalBCPolicy (3 hidden layers): fair comparison. GoalBCPolicy has
in_dim×256 + 256×256 + 256×256 + 256×48 ≈ 198K params. Our denoiser has 175K — slightly
smaller despite the timestep embedding because the trajectory input is 48-dim not the 10-dim
conditioning; the architecture is comparable.

WHY not U-Net 1D: U-Net requires the action to be a temporal sequence with skip connections
between encoder and decoder levels. Our 48-dim action is flat (not sequentially structured
at the denoiser level). Chi et al. use U-Net for their Transformer variant where the
trajectory has explicit time-step structure. For a flat MLP, concatenation is simpler and
produces equivalent results on low-dim actions.

### 3.3 Complete forward signature

```python
def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    x_t : (B, 48) noised trajectory
    t   : (B,)    diffusion timestep integer ∈ [1, T]
    c   : (B, 10) normalized conditioning vector
    returns: (B, 48) predicted noise ε
    """
```

---

## 4. DDPM Noise Schedule

REF: Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models", arXiv 2102.09672.

**Cosine schedule** (not linear):
```python
# Computed in float64 for numerical stability, cast to float32 for training
T = 100
s = 0.008   # offset to prevent beta_t from being too small near t=0
t_steps = np.arange(T + 1, dtype=np.float64)
f_t = np.cos((t_steps / T + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar = f_t / f_t[0]                   # shape: (T+1,), alpha_bar[0] = 1.0
alpha_bar = np.clip(alpha_bar, 1e-5, 1.0)  # numerical safety
beta = 1 - alpha_bar[1:] / alpha_bar[:-1]  # shape: (T,)
beta = np.clip(beta, 1e-4, 0.999)          # clip as in Nichol & Dhariwal
```

**WHY cosine over linear:**
- Linear schedule's alpha_bar near t=100 is very close to 0, causing the noised trajectory
  to be nearly pure noise for many timesteps (wasted computation)
- Cosine has a smoother transition; alpha_bar stays above 1e-5 throughout
- For trajectory prediction (continuous, smooth signal), cosine gives better reconstruction
  at low noise levels (the regime where mode decisions happen)

**Sanity check on schedule:**
- alpha_bar[0] should be ≈ 1.0 (no noise at t=0)
- alpha_bar[T] should be ≈ 0.0 (pure noise at t=T)
- beta values should be in [1e-4, 0.02] range (not too large, not zero)

---

## 5. Training Procedure

### 5.1 Dataset

Reuse `extract_from_db()` from `train_dual_horizon.py` (already imports correctly as
`extract_from_db_with_dual_horizon_goal` alias). Same 260K+ windows from 64 DB files.

### 5.2 Normalization

**Critical:** Diffusion requires the training data to have unit variance because the noise
schedule assumes N(0, I) as the prior.

```python
# Normalize trajectories to approx N(0,1)
Ym, Ysd = Y.mean(0), Y.std(0) + 1e-6
Yn = (Y - Ym) / Ysd
# After normalization, verify: Yn.std() ≈ 1.0, Yn.mean() ≈ 0.0
```

If Yn.std() is significantly != 1.0, the cosine schedule's alpha_bar values will be miscalibrated
(the "pure noise" level won't actually overwhelm the signal). This is the most common training
failure for DDPM on structured data.

### 5.3 DDPM Training Loop

```
for each batch (x_0, c):
    1. Sample t ~ Uniform(1, T) for each sample in batch
    2. Sample ε ~ N(0, I), same shape as x_0
    3. Compute x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * ε
       (the forward process, closed-form at arbitrary t)
    4. ε_pred = denoiser(x_t, t, c)
    5. loss = MSE(ε_pred, ε)
    6. loss.backward(); optimizer.step()
```

WHY predict ε (noise) not x_0: Ho et al. (2020) showed noise prediction is equivalent to
minimizing a reweighted variational lower bound. Empirically, ε-prediction has more stable
gradients for action/trajectory prediction than x_0 prediction (which can diverge when x_0
has large dynamic range).

### 5.4 Hyperparameters

| Param | Value | Justification |
|---|---|---|
| T (train steps) | 100 | Standard; Chi et al., Ho et al. use 100 for continuous actions |
| Batch size | 512 | Same as DualHorizon for fair training time comparison |
| LR | 1e-4 | Lower than DualHorizon's 1e-3 because DDPM loss has higher variance per step |
| Epochs | 100 | 2× DualHorizon; DDPM needs more epochs due to T random noise levels per sample |
| Optimizer | Adam | Standard; no weight decay (trajectory prediction benefits from zero weight decay) |
| Schedule | ReduceLROnPlateau(patience=10) | Same as prior planners |

**Expected training time:** ~25 minutes on M1 MPS (260K samples × 100 epochs / 512 batch
= ~51K batches; each batch: 1 denoiser forward + backward ≈ 0.03s on MPS → ~25 min).

### 5.5 Validation Metric

**DDPM val loss** (MSE on noise prediction): tracked for convergence. Target: < 0.5 
(pure chance baseline is ≈ 1.0 since ε ~ N(0,1) and a zero predictor gives MSE ≈ 1.0).

**Open-loop ADE** on val split: compute by running DDIM(1 sample) and measuring L2 between
predicted trajectory centroid and ground truth at step 8. This is the comparable metric to
DualHorizon's val MSE. Target: < 0.12 (DualHorizon best val 0.1159).

---

## 6. Inference Procedure: DDIM Sampling

REF: Song et al. (2020) "Denoising Diffusion Implicit Models", arXiv 2010.02502.

### 6.1 DDIM formula (deterministic, eta=0)

For each reverse step from t to t_prev:
```
ε_pred   = denoiser(x_t, t, c)
x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * ε_pred) / sqrt(alpha_bar_t)
x_t_prev = sqrt(alpha_bar_{t_prev}) * x_0_pred
           + sqrt(1 - alpha_bar_{t_prev}) * ε_pred
```

This is Song et al. (2020) eq. 12 with eta=0 (deterministic). No stochastic term.

WHY eta=0: at inference we want deterministic, reproducible trajectories for scoring.
Stochastic DDIM (eta>0) would require averaging more samples to get a stable near-goal score.

### 6.2 DDIM timestep sub-sequence

Training uses T=100 steps. DDIM can use any subset. We use 10 steps:
```
timesteps = [100, 89, 78, 67, 56, 45, 34, 23, 12, 1]
# evenly spaced, 10 values from T down to 1
```

### 6.3 K=8 candidate sampling

For junction scenarios the denoiser can produce different modes by starting from different
x_T ~ N(0, I) samples. We draw K=8 independent x_T and run DDIM for each.

**Scoring function:**
```python
# For each candidate trajectory, extract step-8 position (dx_8, dy_8)
# Compare to near-goal (dx_near, dy_near)
# Select the candidate with smallest distance to near-goal
scores = [sqrt((traj[7,0] - dx_near)**2 + (traj[7,1] - dy_near)**2) for traj in candidates]
best = candidates[argmin(scores)]
```

WHY near-goal at step 8 (not final step 16): step 8 is ~0.8s horizon, matching the near-goal
look-ahead. This tests if the diffusion policy is committing to the correct mode at a physically
meaningful horizon. Final step 16 (1.6s) is too far for reliable scoring in tight intersections.

---

## 7. DiffusionPolicyPlanner Class Design

### 7.1 Inheritance

```python
class DiffusionPolicyPlanner(AbstractPlanner):
```

WHY inherit from AbstractPlanner directly (not RouteMapBCPlanner):
- The route-building and goal-extraction code is IDENTICAL to DualHorizonRouteMapBCPlanner
- But the inference policy head is fundamentally different (generative vs deterministic)
- If we inherit from RouteMapBCPlanner, we'd override compute_planner_trajectory() completely,
  making the inheritance misleading — the parent's method is never called
- Directly inheriting AbstractPlanner with the route machinery copied is cleaner (not reused,
  reimplemented for clarity and type-safety)
- EXCEPTION: _build_route(), _get_route_goal(), _straight_route() ARE reused via composition
  (we instantiate a DualHorizonRouteMapBCPlanner internally and delegate route logic to it)

### 7.2 Liskov safety guarantee

DiffusionPolicyPlanner can never be worse than returning a straight trajectory:
1. If checkpoint is missing → instantiation raises FileNotFoundError → caught in build_planners()
2. If DDIM fails → fallback to straight-ahead trajectory (pure Python, no model call)
3. If scoring produces NaN → fallback to first candidate (never returns None)

---

## 8. File Layout

```
nuplan/
  train_diffusion_policy.py   <- NEW: DDPM training + sanity gate
  planners.py                 <- MODIFIED: + DiffusionPolicyPlanner class
  eval_production.py          <- MODIFIED: + 'diffusion' key, CKPT_DIFFUSION, DEPLOYABLE set

docs/
  PHASE3D_HANDOFF_01_RESEARCH.md   <- Stage 1 (this file's predecessor)
  PHASE3D_HANDOFF_02_DESIGN.md     <- Stage 2 (this file)
  PHASE3D_HANDOFF_03_IMPLEMENTATION.md <- Stage 3
  PHASE3D_HANDOFF_04_SELFREVIEW.md     <- Stage 4
```
