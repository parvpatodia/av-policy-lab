# Phase 3d: Goal-Conditioned Diffusion Policy Planner

**Status:** Implementation complete, ready to train  
**Date:** 2026-06-01

---

## What Was Built

A DDPM-based trajectory planner for nuPlan closed-loop AV evaluation that resolves the **mode-swap failure** identified in DualHorizonRouteMapBC (mean L2 = 27.55m).

**The problem it solves:**  
DualHorizon proved the far-goal gives the MLP complete junction information — yet performance degraded. This is the mode-swap signature: a deterministic MLP averages "turn left" and "turn right" into a straight-line compromise that is wrong for both modes. A generative policy (DDPM) samples ONE trajectory from the learned multi-modal distribution and commits to one mode.

**Architecture:**
- Same 10-dim dual-horizon conditioning as DualHorizonRouteMapBC (`[state(6) + near_goal(2) + far_goal(2)]`)
- GoalConditionedDenoiser: 4-layer MLP (~175K params), [noised_traj(48) + timestep_emb(64) + conditioning(10)] = 122-dim input
- DDPM training: T=100 steps, cosine noise schedule (Nichol & Dhariwal 2021)
- DDIM inference: 10 steps, K=8 candidates, scored by near-goal proximity at step 8

**Why the same conditioning?** This is the controlled ablation. The ONLY change from DualHorizon is the policy head (MLP → DDPM). If L2 improves, the cause is the generative model's ability to represent junction bimodality.

---

## How to Run

**Step 1: Verify the implementation (60 seconds)**
```bash
conda activate nuplan
python nuplan/train_diffusion_policy.py --sanity
```
Expected: prints noise schedule stats, confirms 3-step loss decrease, confirms DDIM output shape (8, 48). Final line: `SANITY PASSED — ready to train.`

**Step 2: Train (~25 minutes on M1 MPS)**
```bash
python nuplan/train_diffusion_policy.py
```
Checkpoint saved to `nuplan/checkpoints/trained_diffusion_policy.pt`  
Watch for: val noise-MSE decreasing below 0.5, final open-loop ADE < 1.0m

**Step 3: Evaluate (30 scenarios, ~90 minutes)**
```bash
python nuplan/eval_production.py --n_scenarios 30 \
    --planners idm,speedadaptive,dualhorizon,diffusion
```

**Step 4: Statistical test**
```bash
python nuplan/statistical_analysis.py \
    --a DiffusionPolicyPlanner \
    --b DualHorizonRouteMapBCPlanner
```

---

## What to Expect

| Scenario | Expected behavior |
|---|---|
| Straight roads | Similar to SpeedAdaptive — the 26 easy scenarios should stay easy |
| Junction (left/right turn) | K=8 DDIM samples will include a turn candidate; near-goal scoring selects it |
| Best case | Mean L2 < 18.19m (SpeedAdaptive), 2-4 tail failures eliminated |
| Worst case | Mean L2 similar to DualHorizon (27.55m); mode-collapse or route-branch issue |

---

## What the Evaluation Tells Us

**The experiment tests Hypothesis (A) vs (B) from the DualHorizon post-mortem:**

**Hypothesis A (information was missing):** Refuted by DualHorizon — the far goal provides the information. Phase 3d does NOT test this further.

**Hypothesis B (deterministic MLP cannot represent junction bimodality):** This is what Phase 3d tests.

| Result | Interpretation |
|---|---|
| DiffusionPolicyPlanner mean < 18m AND tail failures drop | **Hypothesis B confirmed.** Mode-swap was the root cause. DDPM fixes it. Path forward: add `route_roadblock_ids` guidance for remaining failures (Phase 3e). |
| DiffusionPolicyPlanner mean < 27.55m (improvement) but tail failures persist | **Partial B confirmation.** DDPM helps on easy junctions but the route itself takes the wrong branch on the 4 hard scenarios. Fix: Phase 3e with roadblock guidance. |
| DiffusionPolicyPlanner mean >= 27.55m (no improvement) | **Hypothesis B inconclusive.** Possible causes: training data too sparse at junctions, model collapsed to the dominant straight-driving mode, or K=8 insufficient. Next: check sample diversity, augment junction data. |

**The decisive metric:**  
Compare the 4 worst-case scenarios from DualHorizon (L2: 55.7, 80.3, 85.3, 121.2m).  
If 2+ of these drop below 25m with DiffusionPolicyPlanner, Hypothesis B is supported.

---

## Files Created

| File | Purpose |
|---|---|
| `nuplan/train_diffusion_policy.py` | DDPM training + sanity gate |
| `nuplan/planners.py` | + `DiffusionPolicyPlanner` class (appended) |
| `nuplan/eval_production.py` | + `diffusion` key, `CKPT_DIFFUSION`, deployable set |
| `docs/PHASE3D_HANDOFF_01_RESEARCH.md` | Literature: Diffusion Planner, Chi et al., BESO |
| `docs/PHASE3D_HANDOFF_02_DESIGN.md` | Architecture spec with equations |
| `docs/PHASE3D_HANDOFF_03_IMPLEMENTATION.md` | Design decisions, known limitations |
| `docs/PHASE3D_HANDOFF_04_SELFREVIEW.md` | Self-review at PI engineering standards |
| `PHASE3D_SUMMARY.md` | This file |

---

## References

- Ho et al. (2020) DDPM — arXiv:2006.11239
- Song et al. (2020) DDIM — arXiv:2010.02502 (eq. 12, eta=0)
- Nichol & Dhariwal (2021) cosine schedule — arXiv:2102.09672
- Chi et al. (2023) Diffusion Policy — arXiv:2303.04137
- Reuss et al. (2023) BESO — arXiv:2304.02532
- Zheng et al. (2025) Diffusion Planner — arXiv:2501.15564 (ICLR 2025 Oral)
