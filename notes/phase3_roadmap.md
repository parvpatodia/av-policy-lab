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

| Policy | Closed-loop Avg L2 | Key signal |
|---|---|---|
| BC_v0 | 49.449m | none (kinematic only) |
| BEV CNN | 49.410m | ego history raster |
| MILE world model | 49.565m | latent consistency |
| DAgger iter 2 | 49.486m | on-policy data |
| IDM | 6.285m | rule-based road following |
| **GoalBC** | **1.820m** | **expert T+8 goal** |
| MapBC (point query) | 56.326m | nearest lane → fails off-road |
| RouteMapBC | **32.085m** | pre-computed 200m route (Phase 3c) — 35% better than BC, but 17.6× worse than GoalBC |
| TrainedRouteBC | pending | route goals at train + inference time — expects ≈ GoalBC (Phase 3c') |
| DiffusionPlanner | TBD | multi-modal DDPM (Phase 3d) |

## Hypothesis chain (revised after Phase 3c)

```
GoalBC (1.82m)       → a GLOBAL reference breaks the plateau completely
MapBC  (56.3m)       → a LOCAL reference fails: map queries don't work off-road
RouteMapBC (32.1m)   → global route fixes drift bootstrapping (43% vs MapBC)
                        but 17.6× gap vs GoalBC reveals: weights trained on expert goals ≠ route goals
TrainedRouteBC       → retrain with route goals → eliminates train/inference mismatch
                        if ≈ GoalBC: deployable policy without expert data at inference
DiffusionPlanner     → tests if multi-modal prediction adds value over deterministic MLP
PDM-Score eval       → honest quality metric beyond L2
```

**Binding insight 1:** The reference must be GLOBAL (valid everywhere), not LOCAL (only near the road).
**Binding insight 2:** Goal-source at training time MUST match goal-source at inference time.
**Binding insight 3:** Architecture is not the bottleneck — goal representation is.
