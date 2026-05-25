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

### Closed-loop (nuPlan L2 error, ego-vs-expert, 3 scenarios)

Controller: `perfect_tracking_controller`. Observation: `box_observation`.

| Policy | Avg L2 (m) | Max L2 (m) | p90 L2 (m) | Notes |
|---|---|---|---|---|
| BCPlanner (v0) | 49.449 | 104.614 | 91.526 | pure imitation |
| BCPlanner (v1, DAgger iter 1) | 49.470 | 104.656 | 91.564 | 745 samples (0.3%) — no improvement |
| BCPlanner (v2, DAgger iter 2) | 49.486 | 104.689 | 91.593 | 12,678 samples (4.6%) — **no improvement** |
| IDMPlanner | **6.285** | **24.308** | **15.733** | reactive, no learning |
| BEVPlanner | 49.410 | 104.543 | 91.416 | **–0.08% vs BC_v0** — spatial history negligible at this drift scale |
| MILEPlanner | 49.565 | 104.834 | 91.723 | +0.2% vs BC_v0 — world model adds no recovery |
| **GoalBCPlanner** | **1.820** | **2.944** | **2.646** | **–96.3% vs BC_v0, 3.5× better than IDM** — goal waypoint (T+8, expert) |

**Key finding — covariate shift:** BC achieves 0.058m open-loop ADE (predicting from ground-truth states) but 49.4m closed-loop L2 (850x worse). Error compounds at every step because the model was never trained on states it caused itself.

**BEV CNN closed-loop:** 49.410m avg L2 vs BC_v0 49.449m — 0.08% improvement, essentially zero. The ego-history rasterization captures where the ego *has been*, not where the road *is*. Once the ego drifts 50m off-track, the 64×64 ego-centered window shows the ego's own off-road trajectory history — no road geometry, no recovery signal. Open-loop ADE improved (0.051m vs 0.058m for BC) because BEV adds useful short-horizon context from ground-truth states. Closed-loop that advantage evaporates immediately when off-distribution states begin.

**MILE world model closed-loop:** 49.565m avg L2 — 0.2% *worse* than BC_v0. The GRU world model trained to minimize consistency loss between adjacent latent states. In distribution this works; in severe compounding drift the latent state encodes nonsense (no training examples for 50m off-track states) and the policy head produces arbitrary outputs. The consistency loss did not act as a regularizer sufficient to prevent off-distribution collapse.

**DAgger iter 2 failure (architectural limit):** 12,678 on-policy samples (4.6%) — BC_v2 val loss improved (0.245→0.243) but closed-loop L2 unchanged (49.449→49.486m, ~0%). Root cause: the MLP policy (6-dim state) cannot perceive where it is relative to the road. More data doesn't fix perception.

**Central finding (Phase 2):** all three architecture variants (BC MLP, BEV CNN, MILE world model) plateau at ~49.4–49.6m closed-loop L2. IDM (6.285m) wins by 8×. Lesson: representation does not fix perception absence.

**Central finding (Phase 3a — GoalBC):** Adding a 2D goal waypoint (T+8 expert position in ego-frame) to the 6-dim input reduces closed-loop L2 from 49.486m → **1.820m — a 96.3% reduction**. GoalBC (1.820m) is **3.5× better than IDM** (6.285m). The MLP policy was never the bottleneck — it was operating without any spatial reference to the road. The 6-dim kinematic state looks identical whether the ego is on-road or 50m off-track. Two extra dimensions of goal information completely breaks the plateau. Phase 3b (MapBC) tests whether the same gain holds when the goal comes from road centerline (no expert required at inference).

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
