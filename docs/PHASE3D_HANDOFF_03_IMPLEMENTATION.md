# Phase 3d Implementation Handoff

**Date:** 2026-06-01  
**Author:** Claude Sonnet 4.6 (orchestrator)  
**Stage:** 3 of 5

---

## 1. Files Created / Modified

### New files
- `nuplan/train_diffusion_policy.py` — DDPM training script (862 lines)
- `docs/PHASE3D_HANDOFF_01_RESEARCH.md` — literature survey
- `docs/PHASE3D_HANDOFF_02_DESIGN.md` — architecture spec
- `docs/PHASE3D_HANDOFF_03_IMPLEMENTATION.md` — this file
- `docs/PHASE3D_HANDOFF_04_SELFREVIEW.md` — self-evaluation

### Modified files
- `nuplan/planners.py` — added `DiffusionPolicyPlanner` class at the bottom (~320 new lines)
- `nuplan/eval_production.py` — added `CKPT_DIFFUSION`, `'DiffusionPolicyPlanner'` to `DEPLOYABLE_PLANNERS`, `'diffusion'` key to `build_planners()`, and import

---

## 2. Architecture Decisions and WHY Comments

### 2.1 GoalConditionedDenoiser (train_diffusion_policy.py)

**Input:** [x_t(48) concat t_emb(64) concat c_norm(10)] = 122-dim

**Key decisions:**
- Concatenation not FiLM: 10-dim conditioning is too small to benefit from FiLM's scale/shift networks. Concatenation is sufficient and matches BESO (Reuss et al. 2023).
- Same 3-hidden-layer depth as GoalBCPolicy: ensures fair parameter comparison (~175K vs ~198K). Any performance difference is due to generative vs discriminative head, not capacity.
- Sinusoidal timestep embedding (64-dim): extrapolates to unseen timesteps, no trainable params. 32 frequency bands are more than sufficient to distinguish 100 levels.

### 2.2 Noise schedule (build_cosine_schedule)

Cosine schedule from Nichol & Dhariwal (2021). Computed in float64, cast to float32.

**Critical numbers:**
- alpha_bar[0] = 1.0 (no noise at t=0)
- alpha_bar[100] < 0.01 (pure noise at t=T)
- beta range: [1e-4, ~0.02] (same as Nichol & Dhariwal clip)

### 2.3 DDIM sampling (ddim_sample)

Implements Song et al. (2020) eq. 12 with eta=0 (deterministic):
```
x0_pred = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t)
x_{t-1} = sqrt(ab_{t-1}) * x0_pred + sqrt(1 - ab_{t-1}) * eps
```

**Key guard:** `x0_pred.clamp(-5.0, 5.0)` — prevents NaN at early denoising steps when x0_pred can fly off to extremes. The 5.0 bound is 5 standard deviations from the normalized trajectory mean.

**WHY deterministic (eta=0):** Scoring K=8 candidates by near-goal proximity requires stable, reproducible trajectories from a given x_T seed. Stochastic DDIM would give different scores on repeated calls.

### 2.4 DiffusionPolicyPlanner (planners.py)

**Inheritance:** Directly from `AbstractPlanner` (not RouteMapBCPlanner).

**WHY not inherit from RouteMapBCPlanner:**
RouteMapBCPlanner's `compute_planner_trajectory` calls the MLP model. DiffusionPolicyPlanner overrides the entire method. Inheriting from it would be misleading — the parent's core method is never called. Direct inheritance from AbstractPlanner with copied route helpers is cleaner.

**Route helpers:** `_build_route_dp`, `_get_route_goal_dp`, `_straight_route_pts_dp` — copies of RouteMapBCPlanner's methods with `_dp` suffix to make the duplication explicit. A future refactor could extract a `RouteMapMixin`.

**Liskov safety:** On any denoiser failure (NaN, shape mismatch, OOM), `compute_planner_trajectory` catches the exception and falls back to a straight-ahead trajectory at current speed. This is never worse than IDM's straight-line mode.

**Inference device:** CPU, not MPS. nuPlan simulation calls the planner at ~10 Hz synchronously. MPS has per-call warm-up overhead that makes it slower than CPU for a 175K-param model called 10 times/second.

### 2.5 Checkpoint format

The training script saves:
```python
{
    'model': state_dict,          # GoalConditionedDenoiser weights
    'X_mean': float32 (10,),      # conditioning normalization
    'X_std':  float32 (10,),
    'Y_mean': float32 (48,),      # trajectory normalization
    'Y_std':  float32 (48,),
    'T': int,                     # = 100
    'schedule': dict of tensors,  # alpha_bar, sqrt_ab, sqrt_1mab, beta
}
```

Saving the schedule in the checkpoint ensures exact reproducibility of DDIM at inference: if T is ever changed between training runs, the planner uses the correct schedule.

---

## 3. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No route_roadblock_ids guidance in DiffusionPolicyPlanner | Route may take wrong junction branch (same as SpeedAdaptive) | Near-goal scoring partially compensates — the correct-branch sample will score better |
| K=8 samples may all be the same mode | If the model collapses, scoring won't help | Monitor sample spread in sanity gate test 4 |
| Inference at CPU | ~80ms per planning step at K=8, 10 DDIM steps | Acceptable at 10 Hz; can reduce K or DDIM steps if needed |
| DDPM val MSE is on noise, not trajectory | Can't directly compare to DualHorizon val MSE (0.1159) | Use open-loop ADE (meters) as the comparable metric — reported after training |

---

## 4. Expected Behavior

**If diffusion works (hypothesis B confirmed — mode-swap is the root cause):**
- The 4 catastrophic tail failures from DualHorizon (L2: 55-120m) should drop to <20m
- Mean L2 should be < 18.19m (SpeedAdaptive baseline) and ideally < 15m
- Sample spread at test 4 of sanity gate should be > 0.1 (model generates diverse trajectories)
- The pattern: same 26 easy scenarios should stay easy; the 4 junction scenarios improve

**If diffusion does NOT work:**
- The 4 tail failures persist at similar L2 values
- This would mean the DDPM's learned distribution does not separate the modes well
- Possible cause: 260K training windows have very few junction scenarios (maybe only ~5-10% are at intersections); the model learns the dominant straight-driving mode and never samples the turn mode at test time
- Next step in that case: Phase 3e — data-augmented training or flow-matching

---

## 5. How to Run

### Step 1: Sanity gate (~60 seconds)
```bash
conda activate nuplan
python nuplan/train_diffusion_policy.py --sanity
```

Expected output:
```
[1] Noise schedule stats:
    alpha_bar[0]   = 1.000000  (should be ~1.0)
    alpha_bar[T/2] = 0.XXXXXX  (should be ~0.5)
    alpha_bar[T]   = 0.00XXXX  (should be ~0.0)
    beta range     = [0.0001XX, 0.0XXX]  (should be [1e-4, ~0.02])
    PASS: schedule goes from 1.0 to 0.0, beta in valid range
[2] Loading 512 samples from first DB file ...
[3] Testing: 3 forward+backward passes — does loss decrease?
    step 1: loss = X.XXXX
    step 2: loss = X.XXXX
    step 3: loss = X.XXXX
[4] Testing DDIM sampling shape and value range ...
    DDIM output shape: (8, 48)  PASS
    Sample spread: X.XXXX
SANITY PASSED — ready to train. Run without --sanity to start.
```

### Step 2: Full training (~25 minutes on M1 MPS)
```bash
python nuplan/train_diffusion_policy.py
```

Checkpoint saved to: `nuplan/checkpoints/trained_diffusion_policy.pt`

### Step 3: Eval (30 scenarios, ~90 minutes total)
```bash
python nuplan/eval_production.py --n_scenarios 30 --planners idm,speedadaptive,dualhorizon,diffusion
```

### Step 4: Statistical analysis
```bash
python nuplan/statistical_analysis.py --a DiffusionPolicyPlanner --b DualHorizonRouteMapBCPlanner
python nuplan/statistical_analysis.py --a DiffusionPolicyPlanner --b SpeedAdaptiveRouteMapBCPlanner
```

The Wilcoxon signed-rank test (paired, same 30 scenarios) will tell if the improvement is statistically significant.
