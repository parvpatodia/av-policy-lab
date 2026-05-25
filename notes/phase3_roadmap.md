# AV-Policy-Lab Phase 3 Roadmap

## Central question

Phase 2 showed all imitation policies plateau at ~49.5m closed-loop L2. IDM wins by 8x (6.285m).
Root cause: all three policies receive only `[sin(yaw), cos(yaw), vx, vy, ax, ay]` — 6 kinematic
scalars with NO road geometry. When compounding drift puts the ego off-track, all 6 features look
identical to a normal on-road state. The policy cannot perceive it is in trouble.

Phase 3 asks: can we close this gap with architectural improvements that add road context?

---

## Phase 3a — GoalBC (goal_bc.ipynb, May 24)

**Hypothesis:** If the policy knew where the road leads (T+8 waypoint = 0.8s ahead), would
covariate shift collapse?

**Input:** `[sin(yaw), cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal]` = 8-dim
**Architecture:** 8→256→256→256→48 (identical depth to BC, pure ablation over 2 goal features)
**Eval mode:** Expert T+8 lookup from DB at inference (honest upper-bound: "given correct goal, can the policy execute?")

### Results (fill after training)
- Open-loop ADE: TBD | FDE: TBD  (BC_v0 baseline: 0.058m / 0.063m)
- Closed-loop avg L2: TBD  (BC_v0: 49.449m | IDM: 6.285m)

---

## Phase 3b — MapBC: centerline-conditioned policy

**Motivation:** T+8 goal is a single waypoint. The full road geometry (upcoming curve shape)
requires more context. Phase 3b adds 10 centerline points = 2–5m look-ahead.

**Input:** state(6) + local centerline (10 points × 2 = 20) + goal(2) = 28-dim
**Architecture:** 28→512→512→512→48
**Data extraction:** `scenario.map_api.get_available_map_objects()` via nuPlan map API.
Extract centerline for the current lane (or nearest lane if off-road), sample 10 equidistant
points ahead of ego, transform to ego-frame.

**Expected:** larger improvement than GoalBC. Road geometry (full centerline shape) provides
recovery signal even after covariate drift. A single goal waypoint only tells direction, not curvature.

---

## Phase 3c — Diffusion Policy Planner

**Motivation:** Deterministic MLP regression may underfit multi-modal trajectory distributions
(e.g., lane change vs. straight ahead at intersection). DDPM denoising models multi-modality natively.

**Conditioning:** map(20) + goal(2) + state(6) = 28-dim condition vector `c`
**Score network:** small U-Net or transformer, T=100 denoising steps
**Inference:** DDIM (10 steps) for fast online planning; sample 8 trajectories, score by goal
proximity + collision avoidance, take best.

**REF:** Chi et al. 2023 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."
RSS 2023. arXiv:2303.04137

**REF:** Jiang et al. 2023 "MotionDiffuser: Controllable Multi-Agent Motion Prediction using
Diffusion." CVPR 2023.

---

## Phase 3d — PDM-Score evaluation

**Motivation:** L2 error measures trajectory deviation from expert but not driving quality.
PDM-Score (nuPlan's official metric) measures: comfort (jerk), progress (did ego reach goal?),
no_collision (contact with other agents).

**Plan:**
- Implement PDM-Score via nuPlan's built-in metrics pipeline
- Evaluate all Phase 3 planners on 20+ scenarios (not just 3)
- Qualitative failure analysis: plot ego trajectory vs expert for each failure case
- Compare against nuPlan leaderboard benchmarks for reference

---

## Hypothesis chain

```
GoalBC     → tests if any directional signal breaks the plateau
MapBC      → tests if full road geometry recovers trajectory
DiffPolicy → tests if multi-modal prediction handles intersections/merges
PDM-Score  → honest leaderboard-comparable metric (replaces L2-only eval)
```

Each phase is a single-factor ablation. The performance trajectory tells us exactly which
signal type (goal point → centerline → distribution modeling) is the binding constraint.
