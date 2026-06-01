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

## Phase 3c''' — RoadblockRouteMapBC ✅ COMPLETE (negative-leaning, highly informative)

**Hypothesis:** SpeedAdaptive's 4 catastrophic tail failures are intersection scenarios where
the heading-aligned route goes straight while the expert turns. Using `route_roadblock_ids`
(the scenario's intended route, available at deploy time) to pick the junction branch should
give the correct turn direction and eliminate the tail.

**Implementation:** `RoadblockRouteMapBCPlanner(SpeedAdaptiveRouteMapBCPlanner)` overrides only
`_select_successor()` to prefer on-route successors; identical to parent when no route id matches
(Liskov-safe). Confirmed populated on 30/30 mini scenarios (mean 27.8 ids).

**Result (30 scenarios, `eval_production.py` + `statistical_analysis.py`):**
- Mean 17.00m vs parent 18.19m; **p90 58.16 → 31.71m** (moderate failures cleaned up).
- **Statistically TIED with parent** — Wilcoxon p=0.808, median-diff CI [−5.5, +5.6] includes 0.
- Route **changed on 28/30 scenarios** → the mechanism fires (not silent fallback).
- **15 improved, 13 regressed.** Dramatic fix: scen_0000 55.7→4.1m (a catastrophic failure
  eliminated). Dramatic regression: scen_0022 1.1→15.5m (a near-perfect scenario broken).
- Catastrophic (>50m): fixed {0}; {11,18,20} persist; no new catastrophic.
- PDM-Score unchanged: 0.526 (= parent). Comfort 0.833, collisions 0.667, TTC 0.600 all identical.

**Finding — a correct goal is necessary but NOT sufficient.** The route *direction* is now correct
(roadblock-guided), but the deterministic MLP **cannot execute the turn it is now correctly pointed
toward**. It was trained on expert goals that are overwhelmingly near-straight (the expert tracks
its lane smoothly); a sharp turn-goal at a junction is out-of-distribution, so the policy regresses
toward the straight-ahead mean. Correcting the route therefore fixes scenarios where the turn is
gentle and breaks scenarios where it is sharp — a wash.

**The bottleneck has moved:** goal REPRESENTATION (global route → speed-matched scale → correct
branch) is solved; the residual failure is policy EXECUTION of multi-modal junction trajectories.
This is the data-isolated motivation for Phase 3d (Diffusion Policy) and for a *speed-adaptive
route goal at TRAINING time* (3c' done wrong with fixed 8m; redo with speed×0.8 route goals so the
policy actually sees turn-goals during training).

---

## Phase 3c''''' — DualHorizonRouteMapBC (the look-ahead-horizon finding)

**The discovery that redirected the project.** Before building 3d, a controlled experiment
(retrain with speed-adaptive route goals) was proposed. Its *rigor gate* — a goal-angle analysis
run before any training — overturned the plan and saved a misdirected Diffusion build.

**Goal-angle vs look-ahead horizon** (expert paths, 20 DBs, turning windows = heading changes
>20° over next 20m, 15% of all windows):

| look-ahead | % of turning windows where the turn is visible (goal >15°) |
|---|---|
| 2 m | 3% |
| 4 m | 7% |
| **3.5 m (what SpeedAdaptive/Roadblock use)** | **~6%** |
| 8 m | 24% |
| 16 m | 64% |
| 24 m | 88% |

**At the 3.5 m horizon the planners actually use, the turn is invisible in 94% of turning
windows.** It only enters a single goal point at 16–24 m. So SpeedAdaptive and Roadblock did not
*mis-execute* turns — the turn was never in their input. **Corollary: Diffusion Policy alone would
NOT fix this** — a multi-modal head cannot commit to a turn it was never told about. This is an
input-information problem first, a policy-capacity problem only after.

**The fix — dual-horizon goal (like a real reference-path controller):**
- near goal = speed × 0.8 s arc-length (≈3.5 m) — precise local tracking (what already works)
- far  goal = fixed 20 m arc-length — turn anticipation (the missing information)
- 10-dim input; identical architecture/optimizer/targets; near goal held fixed → clean ablation
  that ADDS the far preview. `DualHorizonRouteMapBCPlanner` (inference) + `train_dual_horizon.py`.

**Rigor gate PASSED:** on turning windows the 20 m far goal exposes the turn 81% of the time vs
8% for the near goal (10× more). The dual-horizon input genuinely carries the missing turn info.

**RESULT (trained, 30 scenarios) — the decisive finding:**
Mean 27.55m (worse than SpeedAdaptive 18.19m), median 16.55m, 14 improved / 16 regressed.
The aggregate hides a clean MODE SWAP:
- FIXED the intersection turns that broke SpeedAdaptive: scen_0018 85.3→**0.0m**,
  scen_0000 55.7→**0.4m**, scen_0011 80.3→33.7m, scen_0024 16.0→0.5m, scen_0005 11.9→0.1m.
  → the turn information IS usable; the MLP executed turns it previously failed.
- BROKE the easy straights SpeedAdaptive nailed: scen_0013 3.1→**74m**, scen_0025 0.8→**47m**,
  scen_0006 1.3→61m, scen_0010 8.5→73m, scen_0014 6.9→70m.
- PDM fingerprint of hedging: comfort **1.00** (best), no-collision **0.90**, drivable **0.90**,
  direction **1.00** — but progress **0.40** (worst). Drives smoothly/safely, won't commit.

**Conclusion (resolves the (A) vs (B) question — BOTH, complementary):**
(A) information was necessary — with the far goal the MLP CAN do the turns (85→0). ✓
(B) a single deterministic regressor is insufficient — conditioned to anticipate the far turn,
    it over-applies on straights and under-commits, averaging the turn/straight modes. ✓
A single deterministic policy cannot serve both regimes from the same conditioning. This is the
airtight, data-isolated motivation for a MULTI-MODAL action policy → Phase 3d Diffusion Policy.

**Documented cheaper alternative control (for completeness):** a far-horizon sweep (e.g. 10/14m)
could probe a sweet spot, but the breaks are on near-perfect straights (0.8→47m), so the far goal
injects steering error on straights at any horizon that also reveals turns — mode competition, not
a horizon-tuning issue.

---

## Phase 3d — Diffusion Policy Planner ✅ NOW JUSTIFIED BY DATA

**The gate is cleared.** 3c''''' showed that with the turn information present (dual-horizon, far
goal verified to expose turns 81% of turning windows), the deterministic MLP fixes turns but breaks
straights — it cannot conditionally serve both regimes and instead averages them (progress 0.40,
comfort 1.0 — the hedging fingerprint). This is exactly the multi-modal trajectory distribution a
deterministic regressor collapses. Diffusion Policy is the principled, literature-grounded fix.

**Design:**
- Conditioning: the SAME 10-dim dual-horizon context [state(6) + near(2) + far(2)] — we keep the
  information that proved usable; we change only the policy class (deterministic → generative).
- Score network: small conditional MLP/transformer denoiser ε_θ(traj_t, t, cond); T=100 DDPM
  train steps; DDIM ~10-step sampling at inference.
- Inference: sample K trajectories, score by goal-consistency, take best (handles turn-vs-straight
  as distinct modes instead of averaging).
- **Pre-registered hypothesis:** Diffusion fixes the turns (like DualHorizon) WITHOUT breaking the
  straights (unlike DualHorizon) → mean L2 below SpeedAdaptive AND below IDM, progress recovers
  toward 0.8. Success = beats IDM (13.97m) on mean with progress ≥ 0.75 and comfort ≥ 0.8.
- **Falsifier:** if Diffusion also can't get both regimes, the limit is the conditioning/horizon,
  not the policy class — fall back to the far-horizon sweep / curvature-gated conditioning.

**REF:** Chi et al. 2023 (Diffusion Policy, RSS); Janner et al. 2022 (Diffuser, ICML).

**Conditioning:** route_goal(2) + state(6) = 8-dim condition (same as GoalBC/MapBC, reusable)  
**Score network:** small transformer denoiser, T=100 DDPM steps, DDIM sampling at 10 steps  
**Inference:** sample 8 trajectories, score by goal proximity, take best

**REF:** Chi et al. 2023. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023.  
**REF:** Jiang et al. 2023. "MotionDiffuser: Controllable Multi-Agent Motion Prediction." CVPR 2023.

---

## Phase 3e — PDM-Score evaluation ✅ PIPELINE COMPLETE

**Motivation:** L2 error measures trajectory deviation from expert but not driving quality.
PDM-Score (nuPlan's official metric, Dauner et al. 2023) measures: comfort (jerk), progress,
no_collision, drivable-area, TTC, speed-limit, direction. Does the policy actually *drive well*?

**Done:** `pdm_score.py` parses + aggregates the 7 components; `eval_production.py` enables
the `simulation_closed_loop_nonreactive_agents` metric set. 30-scenario results:

| Planner | PDM-Score | no-collision | drivable | TTC | comfort |
|---|---|---|---|---|---|
| IDM | **0.656** | 0.850 | 0.833 | 0.767 | 0.633 |
| SpeedAdaptiveRouteMapBC | 0.526 | 0.667 | 0.767 | 0.600 | **0.833** |
| RouteMapBC | 0.342 | 0.600 | 0.533 | 0.567 | 0.467 |

**Finding — comfort/safety trade-off:** IDM wins the composite, but SpeedAdaptive is
**more comfortable** (0.833 vs 0.633) and trades away safety (collisions, TTC, drivable-area).
The safety deficit co-locates with the L2 intersection tail. This is the sharpest framing
of the contribution: learned policy = smoother; rule-based = safer at junctions.

**Confirmed (`check_roadblock_availability.py`):** route_roadblock_ids populated on 30/30
scenarios (mean 27.8 ids, string-typed) → Phase 3c''' fix has real route data to act on.

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

| Policy | Mean L2 | Median | p90 | Std | Fail>20m | Good<5m |
|---|---|---|---|---|---|---|
| IDM | 13.97m | 8.50m | 28.68 | 16.26 | 8/30 | 12/30 |
| **RoadblockRouteMapBC** (3c''') | 17.00m | 7.50m | **31.71** | 27.71 | 6/30 | 12/30 |
| SpeedAdaptive (3c'') | 18.19m | 7.50m | 58.16 | 28.57 | 6/30 | 12/30 |
| BC | 27.18m | 16.99m | 71.84 | 28.19 | 14/30 | 12/30 |
| RouteMapBC | 47.36m | 53.57m | 70.07 | 25.43 | 26/30 | 0/30 |

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

## Hypothesis chain (complete, as of Phase 3c''')

```
GoalBC (1.82m, 3-scen)         → GLOBAL oracle: expert T+8 goal breaks plateau completely
MapBC  (56.3m)                 → LOCAL query fails: map queries don't work off-road
RouteMapBC (32.1m, 3-scen)     → GLOBAL route fixes drift; but 8m goal = wrong scale (23× training scale)
TrainedRouteBC (49m)           → retraining with 8m goals fails: 8m is 12× prediction horizon (0.69m),
                                  MSE trains without goal → BC behaviour
SpeedAdaptiveRouteMapBC:
  3-scen:  13.7m               → speed×0.8 = correct T+0.8s scale → 57% improvement over RouteMapBC
  30-scen: 18.2m mean / 7.5m median → TIED with IDM (Wilcoxon p=0.76); 4 intersection failures drive mean
RoadblockRouteMapBC (17.0m)    → intended route gives the CORRECT turn direction (28/30 routes changed,
                                  scen_0000 55.7→4.1m). But TIED with parent (p=0.81): 15 fixed / 13 broken.
                                  Correct goal ≠ executable goal — MLP can't track turn-goals it never trained on.
DiffusionPlanner (Phase 3d)    → multi-modal DDPM for the turn/straight bimodality the MLP averages away
SpeedAdaptive route goal @train→ retrain so the policy actually SEES turn-goals (3c' redone correctly)
PDM-Score eval (Phase 3e) ✅    → IDM 0.656 vs SpeedAdaptive/Roadblock 0.526; learned policy more COMFORTABLE
                                  (0.833 vs 0.633) but less safe (collisions 0.667, TTC 0.600)
```

**Binding insight 1:** The reference must be GLOBAL (valid everywhere), not LOCAL (only near the road).
**Binding insight 2:** Goal-source at training time MUST match goal-source at inference time.
**Binding insight 3:** Architecture is not the bottleneck — goal representation is... (until 3c''').
**Binding insight 4:** Intersection topology is the failure mode of centerline-following route goals.
**Binding insight 5:** Mean L2 hides bimodal distributions — always report median, CI, and a paired test.
**Binding insight 6 (Phase 3c'''):** A correct goal is necessary but NOT sufficient. Once the route
direction is right, the deterministic MLP still cannot *execute* a turn-goal that is out-of-distribution
relative to its near-straight expert training. The bottleneck moves from goal REPRESENTATION (solved)
to policy EXECUTION of multi-modal junction trajectories → motivates Phase 3d (Diffusion Policy).
