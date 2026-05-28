# av-policy-lab

Closed-loop driving policy learning in simulation. Three baselines trained on the [nuPlan](https://nuplan.org) dataset, evaluated with a reproducible open-loop and closed-loop harness.

**Status:** BC + DAgger + BEV CNN + MILE world model built. Closed-loop eval for DAgger iter 2, BEV, MILE pending (run notebooks).

---

## What this is

This repo benchmarks three imitation learning approaches on the nuPlan mini dataset:

1. **Behavior Cloning (BC)** — MLP that maps ego state to a 16-step future trajectory. Pure imitation, no world model.
2. **DAgger (Ross et al. 2011)** — dataset aggregation fix for BC covariate shift. Runs policy closed-loop, collects (visited state, expert label) pairs, retrains iteratively.
3. **BEV CNN** — replaces 6-dim scalar state with a top-down rasterized ego-history image (3×64×64). CNN encoder + state MLP + trajectory head. Adds spatial temporal context.
4. **MILE-style world model** — encoder (6→64 latent) + GRU world model + policy, trained with joint imitation + consistency loss. Based on [MILE (Hu et al., 2022)](https://arxiv.org/abs/2209.14430).

The goal is not to beat SOTA. The goal is to execute three baselines cleanly, document where each one fails, and produce an evaluation framework an AV team would actually find useful.

---

## Why nuPlan

nuPlan uses 1200+ hours of real Motional driving logs across Las Vegas, Boston, Pittsburgh, and Singapore. It has reactive agents built in, a standardized closed-loop simulation API, and a published scoring metric (PDM-Score) that accounts for comfort, progress, and collision avoidance simultaneously. CARLA requires building all of that infrastructure from scratch.

---

## Results

### Open-loop (ADE / FDE on nuPlan mini val split)

Evaluated on 2,000 randomly sampled windows from the held-out val split (80/10/10 split, seed 42). All planners receive the same initial ego state.

| Policy | ADE (m) | FDE (m) | Notes |
|---|---|---|---|
| BC MLP | 0.058 | 0.063 | 6→256→256→256→48, 260K windows |
| IDM (free-road) | 3.898 | 7.871 | Treiber 2000, V0=15 m/s |
| Constant velocity | 3.205 | 6.030 | vx/vy extrapolated |
| BEV CNN | 0.051 | 0.059 | 3×64×64 ego-history + 6-dim state, ~370K params; LR decayed 1e-3→2.5e-4 |
| MILE world model | 0.060 | 0.068 | encoder+GRU+policy, ~73K params; L_cons=0.006 (converged) |

### Closed-loop — single-log 3-scenario eval (Phase 1–3c)

Controller: `perfect_tracking_controller`. Observation: `box_observation`. Same 3 scenarios, 1 log.

| Policy | Avg L2 (m) | Max L2 (m) | p90 L2 (m) | Notes |
|---|---|---|---|---|
| BCPlanner (v0) | 49.449 | 104.614 | 91.526 | pure imitation |
| BCPlanner (v2, DAgger iter 2) | 49.486 | 104.689 | 91.593 | 12,678 samples — **no improvement** |
| BEVPlanner | 49.410 | 104.543 | 91.416 | –0.08% vs BC_v0 — spatial history irrelevant at drift scale |
| MILEPlanner | 49.565 | 104.834 | 91.723 | world model adds no recovery |
| IDMPlanner | **6.285** | **24.308** | **15.733** | reactive, no learning |
| **GoalBCPlanner** | **1.820** | **2.944** | **2.646** | **–96.3% vs BC_v0** — oracle (requires expert DB at inference) |
| MapBCPlanner | 56.326 | 108.124 | 98.831 | local map query fails off-road |
| RouteMapBCPlanner (8m fixed) | 32.085 | 77.952 | 48.596 | global route — 35% better than BC; wrong goal scale |
| TrainedRouteBCPlanner | 49.034 | 101.902 | 89.021 | retrained on 8m goals — goal ignored (12× horizon) |
| **SpeedAdaptiveRouteMapBC** | **13.697** | **23.725** | **19.842** | **speed×0.8 look-ahead — 57% better than RouteMapBC** |

### Closed-loop — 30-scenario diverse eval (Phase 3c'', `eval_production.py`)

30 scenarios sampled from all 64 nuPlan mini logs. `eval_production.py`. Run: May 28 2026.

| Policy | Mean | Median | Std | Fail >20m | Good <5m | vs IDM (per-scen) |
|---|---|---|---|---|---|---|
| **SpeedAdaptiveRouteMapBC** | **18.19m** | **7.50m** | 28.57 | 6/30 | 12/30 | **wins 17/30** |
| IDMPlanner | 13.97m | 8.50m | 16.26 | 8/30 | 12/30 | — |
| BCPlanner | 27.18m | 16.99m | 28.19 | 14/30 | 12/30 | — |
| RouteMapBCPlanner | 47.36m | 53.57m | 25.43 | 26/30 | 0/30 | — |

**GoalBC oracle: 1.820m** (3-scenario single-log — not re-run in production eval to avoid per-scenario DB mismatch).

**Key insight from 30-scenario distribution:** SpeedAdaptive wins 17/30 scenarios over IDM and has a better median (7.50m vs 8.50m). The worse mean (18.19 vs 13.97m) is caused by **4 catastrophic outliers** (L2: 55.7, 80.3, 85.3, 121.2m) where IDM scores 2.8–7.8m. Root cause: route centerline follows straight at intersections where the expert turns. Without these 4 tail failures, SpeedAdaptive mean ≈ **8.5m — beating IDM**. The policy is bimodal: excellent on straight-road scenarios, catastrophic at intersection topology.

**Key finding — covariate shift:** BC achieves 0.058m open-loop ADE (predicting from ground-truth states) but 49.4m closed-loop L2 (850x worse). Error compounds at every step because the model was never trained on states it caused itself.

**BEV CNN closed-loop:** 49.410m avg L2 vs BC_v0 49.449m — 0.08% improvement, essentially zero. The ego-history rasterization captures where the ego *has been*, not where the road *is*. Once the ego drifts 50m off-track, the 64×64 ego-centered window shows the ego's own off-road trajectory history — no road geometry, no recovery signal. Open-loop ADE improved (0.051m vs 0.058m for BC) because BEV adds useful short-horizon context from ground-truth states. Closed-loop that advantage evaporates immediately when off-distribution states begin.

**MILE world model closed-loop:** 49.565m avg L2 — 0.2% *worse* than BC_v0. The GRU world model trained to minimize consistency loss between adjacent latent states. In distribution this works; in severe compounding drift the latent state encodes nonsense (no training examples for 50m off-track states) and the policy head produces arbitrary outputs. The consistency loss did not act as a regularizer sufficient to prevent off-distribution collapse.

**DAgger iter 2 failure (architectural limit):** 12,678 on-policy samples (4.6%) — BC_v2 val loss improved (0.245→0.243) but closed-loop L2 unchanged (49.449→49.486m, ~0%). Root cause: the MLP policy (6-dim state) cannot perceive where it is relative to the road. More data doesn't fix perception.

**Central finding (Phase 2):** all three architecture variants (BC MLP, BEV CNN, MILE world model) plateau at ~49.4–49.6m closed-loop L2. IDM (6.285m) wins by 8×. Lesson: representation does not fix perception absence.

**Central finding (Phase 3a — GoalBC):** Adding a 2D goal waypoint (T+8 expert position in ego-frame) to the 6-dim input reduces closed-loop L2 from 49.486m → **1.820m — a 96.3% reduction**. GoalBC (1.820m) is **3.5× better than IDM** (6.285m). The MLP policy was never the bottleneck — it was operating without any spatial reference to the road. The 6-dim kinematic state looks identical whether the ego is on-road or 50m off-track. Two extra dimensions of goal information completely breaks the plateau.

**Phase 3b (MapBC) finding — the drift bootstrapping problem:** MapBC replaces the expert T+8 goal with a road centerline look-ahead from the HD map. Both naive (v1, nearest-lane) and heading-aligned (v2) selection score 56.326m — worse than BC_v0. Root cause: once the ego accumulates 2–3m of compounding drift, `get_proximal_map_objects(radius=30m)` returns zero lanes and the planner falls back to straight-ahead. The map query is only useful *on the road*; it fails for recovery. GoalBC succeeds because the expert's T+8 position is valid regardless of where the ego currently is — it's a global reference. The map query is a local reference that breaks under the very drift it's supposed to fix.

**Phase 3c (RouteMapBC) finding — train/inference distribution mismatch:** Pre-computing a 200m route at scenario start and tracking it globally (IDM-style) fixes the drift bootstrapping problem: RouteMapBC (32.085m) is 35% better than BC_v0 and 43% better than MapBC. However, it is still 17.6× worse than GoalBC (1.820m). The gap is not a route construction failure — it is a **train/inference mismatch**. GoalBC weights were trained exclusively on expert T+8 goals, where the goal encodes actual intended trajectory (speed, lane change, turn geometry). A route centerline 8m ahead has systematically different statistics: it always points forward, ignores traffic intent, and never encodes turn geometry. The policy learned to interpret goal offsets as "where the expert will be in 0.8s" — feeding it a road centerline violates that learned mapping. Fix: retrain with route-based goals at training time (TrainedRouteBC, Phase 3c').

**Implication for Phase 3d:** a deployable goal source needs to be used at BOTH training time and inference time. GoalBC proves the policy capacity is sufficient — the bottleneck is now goal-source consistency. TrainedRouteBC: replace expert T+8 lookup with route-goal lookup in the training data pipeline, retrain, redeploy.

**Phase 3c'' (SpeedAdaptiveRouteMapBC) — scale fix confirmed:** `GoalBCPlanner._get_expert_at_offset` uses `offset_steps × 100ms = T+0.8s`. The nuPlan mini DB is at 100Hz (10ms/row), so training T+8 goals ≈ 0.35m average. RouteMapBC's fixed 8m look-ahead was 23× the training scale. `SpeedAdaptiveRouteMapBCPlanner` sets `look_ahead = max(0.05, speed × 0.8)`, matching the GoalBC inference temporal horizon at every speed. 3-scenario result: 32.085m → 13.697m (57% reduction). 30-scenario median: 7.50m (beats IDM 8.50m). **Tail failures** (4/30 scenarios, L2 > 55m) reveal the intersection topology problem: route centerline follows straight while expert turns. Fix: use `route_roadblock_ids` from `PlannerInitialization` to guide lane selection at intersections.

---

## Repo layout

```
av-policy-lab/
├── nuplan/
│   ├── bc_pipeline.ipynb     # BC MLP training, ADE/FDE eval, IDM baseline
│   ├── dagger.ipynb          # DAgger iter 1 (failed) + iter 2 fix (multi-log collection)
│   ├── bev_cnn.ipynb         # BEV CNN: ego-history rasterizer, CNN encoder, BEVPlanner
│   ├── mile_policy.ipynb     # MILE world model: encoder + GRU + joint imitation+consistency
│   ├── closed_loop_eval.py   # Hydra sim harness (BC, IDM, BEV, MILE)
│   ├── planners.py           # All 8 planner classes (BC, IDM, DAgger, BEV, MILE × Policy+Planner)
│   └── checkpoints/
│       ├── bc_best.pt        # BC_v0 (pure imitation, 260K windows)
│       ├── bc_dagger_v1.pt   # BC_v1 (iter 1, 745 samples — no improvement)
│       └── bc_dagger_v2.pt   # BC_v2 (iter 2, ~15K samples — run dagger.ipynb Cell 4)
├── experiments/
│   └── week0_ddpm_scratch.py # DDPM noise schedule (preliminary)
├── notes/
│   └── research-sota-2026-05-01.md
├── DECISIONS.md              # Architecture and tooling decisions with rationale
└── README.md
```

---

## Reproduce the BC baseline

**Requirements:** conda, Python 3.9, nuplan-devkit installed

```bash
# 1. Clone and set up
git clone https://github.com/parvpatodia/av-policy-lab.git
cd av-policy-lab

# 2. Activate the nuplan environment
conda activate nuplan

# 3. Download nuPlan mini dataset
#    Register at https://nuplan.org, download mini split (~5 GB)
#    Place DB files at: /path/to/nuplan-devkit/data/cache/mini/

# 4. Open the BC notebook
jupyter notebook nuplan/bc_pipeline.ipynb
```

Update `DB_DIR` and `CKPT_DIR` in Cell 2 to match your local paths. Run all cells top to bottom. Training takes ~20 minutes on Apple M-series (MPS).

**What it does:**
- Extracts ~327K sliding-window samples from 64 SQLite DB files (stride=10, 16-step future horizon)
- Input: `[sin(yaw), cos(yaw), vx, vy, ax, ay]` — 6 features
- Output: `(dx, dy, d_yaw) x 16` — ego-frame relative trajectory
- MLP: 6 → 256 → 256 → 256 → 48, ReLU activations, Adam + ReduceLROnPlateau
- Eval: ADE / FDE vs. constant-velocity baseline
- `BCPlanner` class wraps the trained model as a drop-in `AbstractPlanner` for nuPlan simulation

---

## Timeline

| Phase | Weeks | Goal |
|---|---|---|
| Foundation | 1–4 (May 11 – Jun 7) | nuPlan setup, BC baseline, Karpathy lectures 1–4 |
| Three baselines | 5–10 (Jun 8 – Jul 19) | BC complete, MILE, VLA, first metrics |
| Eval + writeup | 11–16 (Jul 20 – Aug 30) | Eval harness, failure analysis, HuggingFace post |

---

## Papers

- Behavior Cloning in AV: [Urban Driver (Scheel et al., 2022)](https://arxiv.org/abs/2109.14480)
- World-model imitation: [MILE (Hu et al., 2022)](https://arxiv.org/abs/2209.14430)
- nuPlan benchmark: [Caesar et al., 2021](https://arxiv.org/abs/2106.11810)
- PDM-Score / closed-loop eval: [Dauner et al., 2023](https://arxiv.org/abs/2306.07962)

---

## Author

Parv Patodia — MS AI, Northeastern University Silicon Valley  
Prior work: AV validation at Venti Technologies (LiDAR, RViz), diffusion model research  
GitHub: [parvpatodia](https://github.com/parvpatodia)
