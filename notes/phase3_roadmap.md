# AV-Policy-Lab Phase 3 Roadmap

## Central question

Phase 2 showed all imitation policies plateau at ~49.5m closed-loop L2. IDM wins by 8× (6.285m).
Root cause: all three policies receive only `[sin(yaw), cos(yaw), vx, vy, ax, ay]` — 6 kinematic
scalars with NO road geometry. When compounding drift puts the ego off-track, all 6 features look
identical to a normal on-road state. The policy cannot perceive it is in trouble.

Phase 3 asks: what kind of spatial reference is sufficient to fix this?

---

## Phase 3a — GoalBC (goal_bc.ipynb) ✅ COMPLETE

**Hypothesis:** If the policy knew where the road leads (T+8 expert waypoint), would covariate shift collapse?

**Input:** `[sin(yaw), cos(yaw), vx, vy, ax, ay, dx_goal, dy_goal]` = 8-dim  
**Architecture:** 8→256→256→256→48 (identical depth to BC — pure ablation over 2 goal features)  
**Goal source at inference:** Expert T+8 position from DB (honest upper-bound)

### Results
- Open-loop ADE: **0.004m** (–92.7% vs BC_v0 0.058m)
- **Closed-loop avg L2: 1.820m — 96.3% reduction vs BC_v0, 3.5× better than IDM**

### Finding
The MLP architecture was never the bottleneck. 2 goal dimensions completely break the 49.5m plateau.
The policy CAN execute complex trajectories — it just needed to know where to go.

---

## Phase 3b — MapBC (planners.py: MapBCPlanner) ✅ COMPLETE

**Hypothesis:** Replace expert T+8 goal with road centerline from HD map — no expert at inference.

**Goal source:** `nuplan_map.get_proximal_map_objects(ego_pos, radius=30m)` → walk 8m along centerline

### Results
- MapBC v1 (nearest-lane): **56.326m** — worse than BC_v0 (49.5m)
- MapBC v2 (heading-aligned): **56.326m** — identical, fix never fires

### Finding: the drift bootstrapping problem
Point-based map queries (`get_proximal_map_objects`) only work *on the road*. Once the ego
accumulates 2–3m of compounding drift, the query returns zero lanes (all > 30m away) and the
planner falls back to straight-ahead `(8, 0)`. The fallback is worse than BC's implicit prior.

GoalBC succeeds because the expert's T+8 position is a global reference — it's valid no matter
where the ego currently is. The map query is a local reference that fails under the very drift it
is supposed to fix.

### Why IDM doesn't have this problem
IDM pre-computes a reference path at scenario start and tracks the ego's progress along it globally.
It always knows "you are 2m to the left of waypoint 47" regardless of drift magnitude. A deployable
learned policy needs the same: a route tracker, not a point query.

---

## Phase 3c — RouteMapBC (planners.py: RouteMapBCPlanner) ✅ COMPLETE

**Fix for Phase 3b:** Pre-compute a 200m route at `initialize()` from map centerlines chained
via `outgoing_edges`. Lazy init (route built on first `compute_planner_trajectory` call using
step-0 ego state, since `PlannerInitialization` doesn't expose `initial_ego_state`). At each
step: argmin over all stored route_pts + walk 8m forward. Always valid — no live map queries.

**Implementation:** `RouteMapBCPlanner` in `nuplan/planners.py`:
- `initialize()`: store `map_api`, reset `_route_pts = None`
- First `compute_planner_trajectory()`: call `_build_route(ego, map_api)` → `self._route_pts`
- `_build_route()`: query lanes at t=0 (radius=50m), select heading-aligned lane, chain successors
- `_get_route_goal()`: argmin over all route_pts → walk 8m forward → ego-frame transform
- `compute_planner_trajectory()`: identical to MapBC except goal source = `_get_route_goal()`

### Results
- **Avg L2: 32.085m** (max 77.952m, p90 48.596m)
- vs MapBC: **43% better** (56.326 → 32.085m) — global route fixes drift bootstrapping ✅
- vs BC_v0: **35% better** (49.449 → 32.085m)
- vs GoalBC: **17.6× worse** (1.820 → 32.085m) ← key finding (see below)
- vs IDM:   **5.1× worse** (6.285 → 32.085m)

### Finding: train/inference distribution mismatch
The 32.085m result confirms global route construction works (gap from MapBC closed), but
reveals a new bottleneck: **the GoalBC weights were never trained with route-based goals**.

GoalBC training: `goal = expert_pos_T+8 − ego_pos` (in ego-frame) — encodes actual intended
trajectory including speed profile, turns, lane changes.

RouteMapBC inference: `goal = route_centerline_8m_ahead − ego_pos` — always forward, ignores
traffic intent, never encodes turn geometry.

The policy learned to decode "goal offset" as "where the expert intended to be in 800ms."
Route centerline goals violate this learned mapping → policy still deviates from road.

**The fix is NOT architectural — it is data:** retrain with route-based goals at training time.

---

## Phase 3c' — TrainedRouteBC

**Hypothesis:** Route-based goals work at inference — the gap vs GoalBC is purely a training
distribution mismatch. If we replace the expert T+8 lookup in the *training data pipeline*
with a route-based goal (same source as inference), the policy learns to decode route goals
correctly and should approach GoalBC performance without expert data at inference.

**Changes from GoalBC training:**
- `extract_from_db_with_goal()` → `extract_from_db_with_route_goal()`: for each training
  window, compute route goal using pre-built per-scenario route (same `_build_route()` logic)
  instead of expert T+8 lookup. Goal = route_centerline_8m_ahead in ego-frame.
- Everything else identical: architecture, training loop, normalization, closed-loop harness.

**Expected:** trained-route ≈ GoalBC (1.82m). This proves the gap was goal-source mismatch,
not route quality. If TrainedRouteBC ≈ GoalBC: the full 96% gain is achievable without expert
data at inference → deployable policy claim.

**Status:** planned — Phase 3d (DiffusionPlanner) first, TrainedRouteBC after data pipeline work.

---

## Phase 3d — Diffusion Policy Planner

**Motivation:** Deterministic MLP regression underfits multi-modal trajectory distributions
(lane change vs. straight ahead at intersection). DDPM denoising handles multi-modality natively.

**Conditioning:** route_goal(2) + state(6) = 8-dim condition (same as GoalBC/MapBC, reusable)  
**Score network:** small transformer denoiser, T=100 DDPM steps, DDIM sampling at 10 steps  
**Inference:** sample 8 trajectories, score by goal proximity, take best

**REF:** Chi et al. 2023. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023.  
**REF:** Jiang et al. 2023. "MotionDiffuser: Controllable Multi-Agent Motion Prediction." CVPR 2023.

---

## Phase 3e — PDM-Score evaluation

**Motivation:** L2 error measures trajectory deviation from expert but not driving quality.
PDM-Score (nuPlan's official metric) measures: comfort (jerk), progress (did ego reach goal?),
no_collision (contact with other agents). L2=1.82m is great but does GoalBC actually drive well?

**Plan:**
- Implement PDM-Score via nuPlan's built-in metrics
- Evaluate all Phase 3 planners on 20+ scenarios (not just 3)
- Qualitative failure analysis: plot ego vs expert for each failure case

---

## Results summary (all phases)

Single-log 3-scenario results (goal_bc.ipynb, eval_routemapbc.py, eval_speed_adaptive.py):

| Policy | 3-scen Avg L2 | Key signal |
|---|---|---|
| BC_v0 | 49.449m | none (kinematic only) |
| BEV CNN | 49.410m | ego history raster |
| MILE world model | 49.565m | latent consistency |
| DAgger iter 2 | 49.486m | on-policy data |
| IDM | **6.285m** | rule-based road following |
| **GoalBC** | **1.820m** | **expert T+8 goal (oracle — not deployable)** |
| MapBC (point query) | 56.326m | nearest lane → fails off-road |
| RouteMapBC (fixed 8m) | 32.085m | global route, wrong scale → 35% better than BC |
| TrainedRouteBC (8m) | 49.034m | retrained on 8m goals — network ignores 12×-horizon goal |
| **SpeedAdaptiveRouteMapBC** | **13.697m** | speed×0.8 look-ahead — correct scale, 57% vs RouteMapBC |

**30-scenario diverse eval** (eval_production.py, 30 scenarios × 64 logs, May 28 2026):

| Policy | Mean L2 | Median | Std | Fail>20m | Good<5m | vs IDM (per-scenario) |
|---|---|---|---|---|---|---|
| IDM | 13.97m | 8.50m | 16.26 | 8/30 | 12/30 | — |
| **SpeedAdaptive** | **18.19m** | **7.50m** | **28.57** | **6/30** | **12/30** | **wins 17/30** |
| BC | 27.18m | 16.99m | 28.19 | 14/30 | 12/30 | — |
| RouteMapBC | 47.36m | 53.57m | 25.43 | 26/30 | 0/30 | — |

**Critical finding from 30-scenario eval (statistically honest):**
SpeedAdaptive is **statistically TIED with IDM** — exact binomial on the 17/30 win rate p=0.585,
paired Wilcoxon p=0.761, median-difference 95% bootstrap CI [−10.3, +6.4] includes zero
(`statistical_analysis.py`). We do NOT claim it beats IDM at n=30.
The result is in the **distribution**: 4 of 30 scenarios carry **63% of total L2 mass**
(L2: 55.7, 80.3, 85.3, 121.2m), all intersection-turn scenarios where the centerline route
goes straight while the expert turns. Trimmed-4 mean: SA 7.81m vs IDM 9.08m.
**Defensible claim: a deploy-time-only policy reaches parity with tuned IDM, with one
localized, fixable failure mode (intersection topology → Phase 3c''').**

The distribution is bimodal:
- 12/30 scenarios: L2 < 5m (excellent, comparable to GoalBC oracle)
- 12/30 scenarios: L2 5–20m (moderate tracking)
- 4/30 scenarios: L2 > 55m (catastrophic — intersection topology failure)

---

## Phase 3c''' — RouteMapBC with route_roadblock_ids (planned)

**Hypothesis:** The 4 catastrophic failures are intersection-type scenarios where the pre-computed
centerline route goes straight while the expert turns. nuPlan's `PlannerInitialization` provides
`route_roadblock_ids` — the INTENDED roadblock sequence for the scenario. Using these roadblocks
to guide route construction would give the correct turn direction at intersections.

**Change:** In `_build_route()`, use `initialization.route_roadblock_ids` to filter which lanes
to follow when chaining successors. Currently the planner picks the most forward-aligned successor
regardless of the intended route — at a T-intersection this means straight, not the correct turn.

**Expected:** Eliminate the 4 intersection failures. Mean L2 drops from ~18m to ~8–9m, matching IDM.

---

## Hypothesis chain (complete, as of Phase 3c'')

```
GoalBC (1.82m, 3-scen)         → GLOBAL oracle: expert T+8 goal breaks plateau completely
MapBC  (56.3m)                 → LOCAL query fails: map queries don't work off-road
RouteMapBC (32.1m, 3-scen)     → GLOBAL route fixes drift; but 8m goal = wrong scale (23× training scale)
TrainedRouteBC (49m)           → retraining with 8m goals fails: 8m is 12× prediction horizon (0.69m),
                                  MSE trains without goal → BC behaviour
SpeedAdaptiveRouteMapBC:
  3-scen:  13.7m               → speed×0.8 = correct T+0.8s scale → 57% improvement over RouteMapBC
  30-scen: 18.2m mean / 7.5m median → wins 17/30 vs IDM; 4 intersection failures drive mean
Route_roadblock_ids (planned)  → use intended route at intersections → eliminate tail failures
DAgger + route goal (Phase 3d) → on-policy data for intersection recovery → close remaining gap
DiffusionPlanner (Phase 3d)    → multi-modal DDPM for intersection decisions
PDM-Score eval (Phase 3e)      → honest driving quality metric (comfort + progress + collision)
```

**Binding insight 1:** The reference must be GLOBAL (valid everywhere), not LOCAL (only near the road).
**Binding insight 2:** Goal-source at training time MUST match goal-source at inference time.
**Binding insight 3:** Architecture is not the bottleneck — goal representation is.
**Binding insight 4:** Intersection topology is the failure mode of centerline-following route goals.
**Binding insight 5:** Mean L2 hides bimodal distributions — always report median and per-scenario win rate.
