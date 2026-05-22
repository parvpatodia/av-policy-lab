# av-policy-lab

Closed-loop driving policy learning in simulation. Three baselines trained on the [nuPlan](https://nuplan.org) dataset, evaluated with a reproducible open-loop and closed-loop harness.

**Status:** BC baseline complete. World-model and VLA baselines in progress.

---

## What this is

This repo benchmarks three imitation learning approaches on the nuPlan mini dataset:

1. **Behavior Cloning (BC)** — MLP that maps ego state to a 16-step future trajectory. Pure imitation, no world model.
2. **MILE-style world model** — model-based imitation: predict next latent state, then decode trajectory. Based on [MILE (Hu et al., 2022)](https://arxiv.org/abs/2209.14430).
3. **Language-conditioned policy (VLA)** — small transformer with a CLIP text encoder. Accepts high-level commands ("change lanes left", "yield to pedestrian") as conditioning.

The goal is not to beat SOTA. The goal is to execute three baselines cleanly, document where each one fails, and produce an evaluation framework an AV team would actually find useful.

---

## Why nuPlan

nuPlan uses 1200+ hours of real Motional driving logs across Las Vegas, Boston, Pittsburgh, and Singapore. It has reactive agents built in, a standardized closed-loop simulation API, and a published scoring metric (PDM-Score) that accounts for comfort, progress, and collision avoidance simultaneously. CARLA requires building all of that infrastructure from scratch.

---

## Results

### Open-loop (ADE / FDE on nuPlan mini val split)

| Policy | ADE (m) | FDE (m) |
|---|---|---|
| BC MLP | — | — |
| Constant velocity | — | — |
| MILE world model | in progress | in progress |
| VLA (language-conditioned) | in progress | in progress |

*BC numbers will be filled in after closed-loop eval is complete.*

### Closed-loop (PDM-Score)

Coming in Phase 2 (June 2026).

---

## Repo layout

```
av-policy-lab/
├── nuplan/
│   └── bc_pipeline.ipynb     # Data extraction, MLP training, ADE/FDE eval, AbstractPlanner wrapper
├── experiments/
│   └── week0_ddpm_scratch.py # DDPM noise schedule (preliminary)
├── notes/
│   └── research-sota-2026-05-01.md
├── docs/
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
