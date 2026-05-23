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
| BCPlanner (v2, DAgger iter 2) | (run dagger.ipynb) | — | — | ~15K samples (5.7%) — expected 30-60% reduction |
| IDMPlanner | **6.285** | **24.308** | **15.733** | reactive, no learning |
| BEVPlanner | (run closed_loop_eval.py) | — | — | spatial history; checkpoint ready |
| MILEPlanner | (run closed_loop_eval.py) | — | — | world model consistency; checkpoint ready |

**Key finding — covariate shift:** BC achieves 0.058m open-loop ADE (predicting from ground-truth states) but 49.4m closed-loop L2 (850x worse). Error compounds at every step because the model was never trained on states it caused itself.

**BEV CNN open-loop:** Matches BC exactly (0.051m ADE). Spatial history adds marginal FDE improvement (-6.3%, 0.059 vs 0.063m). Open-loop ADE is not where BEV wins — the BC 6-dim state already captures velocity well for short-horizon prediction. The BEV advantage is expected in closed-loop, where spatial history aids recovery from compounding drift.

**MILE open-loop:** 0.060m ADE — 3% worse than BC, as expected (smaller policy head: 64-dim latent vs BC's 256-dim hidden). World model converged cleanly: L_cons=0.006, step-1 latent error=0.0014 growing monotonically to 0.0084 at step-16. No overfitting (val/train=1.01x). Closed-loop advantage pending.

**DAgger iter 1 failure:** 745 on-policy samples (0.3% of dataset) — the expert gradient overwhelmed the on-policy correction signal. Fix: iter 2 collects from 20 logs × 5 scenarios ≈ 15K samples (5.7%).

**MILE hypothesis:** joint encoder + GRU world model training forces the latent space to encode ego dynamics, not just mimic outputs. This should reduce covariate shift in closed-loop without requiring iterative data collection (DAgger).

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
