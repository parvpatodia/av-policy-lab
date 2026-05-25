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

## Phase 3c — Route-tracked MapBC (next)

**Fix for Phase 3b:** Pre-compute the reference route at scenario start (first 5 ego poses →
project onto nearest centerline → store as ordered waypoints). At each step, find the closest
route waypoint + walk 8m forward. This never loses the route even at 50m off-road drift.

**Implementation:**
```python
class RouteMapBCPlanner(AbstractPlanner):
    def initialize(self, initialization):
        # pre-compute route from first few ego states + map
        self._route_pts = self._extract_route(initialization)
    
    def compute_planner_trajectory(self, current_input):
        # find closest route point, walk 8m forward
        # always valid because route_pts are pre-stored, not queried live
```

**Expected:** close the gap between MapBC (56m) and GoalBC (1.82m). If route-tracked MapBC
approaches GoalBC, the entire gain is recoverable without expert data.

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
| RouteMapBC | TBD | pre-computed route (Phase 3c) |
| DiffusionPlanner | TBD | multi-modal DDPM (Phase 3d) |

## Hypothesis chain (revised)

```
GoalBC (1.82m)     → a GLOBAL reference breaks the plateau completely
MapBC  (56.3m)     → a LOCAL reference fails: map queries don't work off-road
RouteMapBC         → tests if pre-computed global route closes the gap
DiffusionPlanner   → tests if multi-modal prediction adds value over deterministic MLP
PDM-Score eval     → honest quality metric beyond L2
```

The binding insight: the reference must be GLOBAL (valid everywhere) not LOCAL (only near the road).
