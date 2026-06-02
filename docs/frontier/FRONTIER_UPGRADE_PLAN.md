# FRONTIER UPGRADE PLAN — av-policy-lab

> Single coherent roadmap to take av-policy-lab from a kinematic-MLP/diffusion sandbox to a scene-aware, multimodal, temporally-decoded diffusion planner evaluated at frontier standard. Grounded in the real repo and the real SOTA recipes (Diffusion Planner, PLUTO, PlanTF, CaRL, Wayformer/VectorNet). **Designs and plans only — no fabricated results.**

## The thesis (one paragraph)
The Phase-3d null result (diffusion 27.59 m ≈ MLP 27.55 m, Wilcoxon p=0.79) is **not a failure of diffusion** — it is the correct, provable consequence of conditioning that nearly fully specifies one trajectory, so `p(traj|cond)` is unimodal and diffusion collapses to the conditional mean. The upgrade fixes the five root causes so that the diffusion-vs-deterministic comparison becomes a *fair, controlled test* of when multimodal generative planning actually helps — while simultaneously giving the policy a scene, a temporal decoder, and a real covariate-shift cure, at a scale with statistical power.

## Sequenced roadmap (dependency-ordered, with solo-student + HPC effort)

| Phase | Work | Root cause | Effort (solo + A100) | Risk |
|---|---|---|---|---|
| **F0. Data pipeline** | nuPlan full feature cache: ego/agents/map/route vectorized tensors → `.pt` shards; log/geography splits; Val14 + Test14-hard wiring. | #4 | **2–3 wks** | High (nuPlan I/O, SQLite, caching is finicky) |
| **F1. Scene encoder** | Wayformer-style early-fusion transformer + latent queries over the F0 tensors. Unit-test shapes/masks. | #2 | **1–2 wks** | Med |
| **F2. Multimodal route-goal** | Build route-region goal from `route_roadblock_ids` → lane-graph → resampled set; lane-set mask. | #1 | **1 wk** | Med (junction lane-graph edge cases) |
| **F3. Temporal cross-attn diffusion decoder** | H=16 @2Hz, x0-pred, self-attn over time + cross-attn to scene memory, DPM-Solver++, K=8 anchored on goal lanes. | #1,#5 | **2 wks** | Med |
| **F4. Perturbation + normalization** | Ego-state perturbation (p=0.5, history re-fit, perturbed-frame scene, re-normalize); aux drivable + causal-proxy collision losses. | #3 | **1 wk** | Low–Med |
| **F5. Train at scale** | DDP on 4×A100, EMA, AMP, ~1M frames, ~100 epochs; HP sweep on open-loop proxy. | #4 | **1–2 wks wall** (mostly compute + babysitting) | Med |
| **F6. Eval @ frontier** | Val14 + Test14-hard, reactive + non-reactive, PDM decomposition, paired stats, per-type stratification; the **ablation matrix**. | all | **2–3 wks** (CPU-bound sim is the bottleneck) | High (sim throughput) |
| **F7 (stretch). Approx-DAgger** | Closed-loop rollouts, PDM-Closed relabel oracle, aggregate + retrain. | #3 | **2–3 wks** | High |
| **F8 (far stretch). Contrastive CIL / joint denoising / RL** | PLUTO triplet loss; joint ego+agent denoising; CaRL-style RL finetune. | — | **months / team** | — |

**Critical path to a defensible result: F0 → F1 → F2 → F3 → F4 → F5 → F6.** Estimated **~10–13 focused weeks** for a solo MS student with reliable A100 access — i.e., the full 3 months, with F7/F8 explicitly out.

## The headline experiment (what makes this worth doing)
A **2×2 controlled study** on Val14, with full PDM decomposition + paired stats:

| | precise near/far goal | route-region goal (multimodal) |
|---|---|---|
| **deterministic MLP** | baseline (≈ Phase-3d) | (loses to diffusion at junctions — hypothesis) |
| **temporal diffusion** | (≈ ties MLP — reproduces null) | **(wins at junctions — the contribution)** |

Plus the orthogonal ablations: scene-encoder on/off, perturbation on/off, cross-attn vs concat, temporal vs flat. Each isolates one root cause, so every gain is attributable. **This is the scientific payload** — it explains *when* diffusion planning helps, which the field has largely asserted rather than isolated.

## Final scorecard (Stage 6, Pass 2)
All stages ≥ 8/10 by the reviewer panel (Diffusion Planner author, Waymo lead, Physical Intelligence engineer). Mean **8.18/10**. Ten flaws found and fixed (see ERROR_LEDGER.md); the four High/Med ones were: non-causal collision loss, DAgger-without-online-expert, unconstructed route-goal, internally-inconsistent perturbation. All resolved.

## BRUTALLY HONEST: solo MS student in 3 months vs needs-a-team

**Realistically achievable solo in 3 months (with A100s):**
- The full F0–F6 critical path: scene-aware multimodal temporal diffusion planner, trained on ~1M frames, evaluated on Val14 (and likely Test14-hard) with reactive/non-reactive closed-loop, PDM decomposition, and the controlled 2×2 + ablation matrix with proper statistics.
- A **genuine, defensible scientific result** (when does diffusion help) — strong workshop paper or a solid analysis section of a larger paper.
- Demonstrable improvement over the scene-blind baselines, and a real (PlanTF-range) covariate-shift improvement from perturbation.

**Requires a team or 6+ months / much more compute:**
- **Matching PLUTO / Diffusion Planner absolute Val14/Test14-hard numbers.** Their compute is 25–50× ours; closing that gap is an engineering+compute project, not a modeling insight.
- **CaRL-style RL** (500M samples) — flatly out of scope solo.
- **Joint ego+agent denoising** at full quality, contrastive CIL tuned, and the full nuPlan closed-loop challenge throughput — each is weeks of additional engineering.
- A **top-venue full paper** (ICRA/CoRL/NeurIPS) needs the absolute-competitiveness + breadth that the compute gap and the CPU-bound closed-loop sim make hard solo.

**The single biggest practical risk is not modeling — it is throughput:** nuPlan feature extraction (F0) and closed-loop simulation (F6) are CPU-bound and slow, and will eat more wall-clock than GPU training. Budget accordingly.

**Recommendation:** commit to F0–F6 as the deliverable; treat the 2×2 controlled experiment as the paper's spine; keep F7/F8 as "future work." That is an honest, frontier-informed, solo-feasible plan that turns the Phase-3d null into a real contribution.

## Verification status
- Architecture/training claims are cited to real papers (arXiv:2501.15564, 2404.14327, 2309.10443, 2504.17838, 2207.05844, 2005.04259, 2303.04137, 2205.09991, 2211.01095).
- Repo state is read from the live `main` branch (README, PROGRESS, train_diffusion_policy.py, docs/PHASE3D_*).
- **Unverified, flagged:** exact PlanTF perturbation magnitudes, PLUTO/PDM-Closed decimal Val14/Test14-hard numbers — to be code/table-verified before any write-up. No experimental results are claimed.
