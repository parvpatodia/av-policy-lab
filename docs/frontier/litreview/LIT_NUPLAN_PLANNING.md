# Literature Review — Learned Planning on nuPlan & the Open-Loop vs Closed-Loop Problem

**Author:** av-policy-lab (frontier-upgrade)
**Date:** 2026-06-01
**Scope:** Learned planning on nuPlan; the open-loop (OL) vs closed-loop (CL) misalignment; the rule-based PDM-Closed ceiling; data-augmentation / contrastive recipes for covariate shift; gaps and a defensible solo-student question.

> **Verification note.** Numbers below are sourced from arXiv HTML, official GitHub configs/source, and conference PDFs. Where a number could not be read verbatim from the primary source (e.g. binary PDF), it is tagged **[UNVERIFIED]** or **[CROSS-SOURCE — varies by reproduction]**. nuPlan reproductions of *the same baseline* differ by 1–3 CL points across papers; treat all cross-paper comparisons as approximate. The single most reliable anchor is each paper's *own* PDM-Closed row, reproduced in its own harness.

---

## 0. nuPlan metrics primer (so the tables are unambiguous)

nuPlan reports three scores, each in [0,100]:
- **OLS** — open-loop score: displacement-based (ADE/FDE-style) agreement with the logged human trajectory over an 8 s horizon. The ego does **not** control the sim; it is pure forecasting.
- **CLS-NR** — closed-loop score, **non-reactive** agents: the planner actually drives; background agents replay logs (do not react to ego).
- **CLS-R** — closed-loop score, **reactive** agents: background agents are simulated with IDM and react to the ego.

Standard benchmark splits (Dauner et al. / `tuplan_garage`):
- **Val14** — 1118 scenarios, 14 official types.
- **Test14-hard** — a hard split mined for low planner scores; the discriminating benchmark in 2024–2026.

Sources: nuPlan benchmark paper (https://arxiv.org/abs/2106.11810); PDM / Dauner et al. (https://arxiv.org/abs/2306.07962).

---

## 1. The central uncomfortable finding: does rule-based PDM-Closed still beat learned planners in 2026?

**Short answer: largely yes, and it is closer than the marketing suggests.** As of mid-2026, the rule-based **PDM-Closed** remains the reference that learned planners are measured against. On **Val14**, *no* purely learned planner has convincingly and reproducibly beaten PDM-Closed in **both** reactive and non-reactive closed-loop. On **Test14-hard non-reactive**, several learned/RL planners now clearly exceed it. On **Test14-hard reactive**, the picture is mixed and reproduction-dependent.

### 1a. PDM and its variants — Val14 (Dauner et al., CoRL 2023)

Verbatim from the paper's Val14 table (https://arxiv.org/html/2306.07962v1):

| Method | OLS | CLS-NR | CLS-R | Representation / type |
|---|---|---|---|---|
| Urban Driver (OL) | 76 | 45 | 44 | learned, polygon |
| GC-PGP | 82 | 57 | 54 | learned, lane-graph |
| PlanCNN | 64 | 73 | 72 | learned, raster |
| **IDM** | 38 | 76 | 77 | **rule-based** |
| **PDM-Open** | **86** | 50 | 54 | learned MLP, centerline-only |
| **PDM-Closed** | 44 | **93** | **92** | **rule-based** |
| **PDM-Hybrid** | 84 | 93 | 92 | rule + learned correction |
| Log-replay (expert/GT) | 100 | 94 | 80 | reference |

The headline: **PDM-Open has the second-best OLS (86) yet near-worst CLS (50/54); PDM-Closed has near-worst OLS (44) yet the best CLS (93/92).** A pure-learning OL champion is a CL disaster, and vice versa. (Note: the expert/log-replay CLS-R is only **80** — even the ground-truth human trajectory is not a perfect reactive closed-loop driver, a subtle ceiling.)

**How PDM-Closed works** (the thing learned planners keep failing to beat): it is *not* deep. It (1) extracts a centerline via lane-graph search, (2) rolls out **IDM at 5 target speeds** × **3 lateral offsets {−1, 0, +1} m = 15 trajectory proposals**, (3) simulates each proposal forward with a simple ego-forecast + collision/comfort/progress scoring, and (4) executes the best-scored proposal. It is a sampling-plus-scoring rule-based planner with a kinematic prior. (Source: https://arxiv.org/html/2306.07962v1.)

### 1b. The 2024–2026 closed-loop scoreboard vs PDM-Closed

Each block uses **that paper's own** reproductions (the only fair within-row comparison). PDM-Closed is repeated per source because its reproduced value drifts.

**PLUTO (Cheng et al., arXiv 2404.14327)** — first learned planner to claim it surpasses PDM-Closed.
- Val14: PLUTO CLS-R **92.88** vs PDM-Closed **92.84** (a +0.04 "win" — within noise). **[CROSS-SOURCE]**
- Test14-hard: PLUTO NR **89.84**, R **80.08**; PDM-Closed NR **~65–92** (varies wildly by split definition), R **65.08**. **[UNVERIFIED — original PLUTO table not readable from binary PDF; numbers from secondary citations]**
- Critical caveat: PLUTO's headline numbers include a **post-processing / trajectory-refinement** stage. The *imitation-only* PLUTO is materially lower; independent reproductions (Plan-R1 table, below) report PLUTO Val14 NR **88.89** / R **78.11** and Test14-hard NR **70.03** / R **59.74** — i.e. *below* PDM-Closed when the rule-based refinement is stripped. (https://github.com/jchengai/pluto; https://arxiv.org/html/2505.17659v2)

**PlanTF (Cheng et al., ICRA 2024, arXiv 2309.10443)** — "a well-designed pure-IL planner is competitive."
- Val14: OLS **89.18**, CLS-NR **84.83**, CLS-R **76.78**.
- Test14-hard: OLS **83.32**, CLS-NR **72.68**, CLS-R **61.7**.
- PlanTF does **not** beat PDM-Closed in closed-loop on Val14 (84.83/76.78 vs 93/92). Its contribution is that a *pure IL* planner closes much of the gap and generalizes better in long-tail OL. (Source: https://arxiv.org/html/2309.10443.)

**Diffusion Planner (Zheng et al., ICLR 2025 Oral, arXiv 2501.15564)** — Table 1 (verbatim, https://arxiv.org/html/2501.15564v2):

| Split | DP NR | DP R | PDM-Closed NR | PDM-Closed R |
|---|---|---|---|---|
| Val14 | 89.87 | 82.80 | 92.84 | 92.12 |
| Test14 | 89.19 | 82.93 | 90.05 | 91.63 |
| Test14-hard | **75.99** | 69.22 | 65.08 | 75.19 |

→ Diffusion Planner beats PDM-Closed **only on Test14-hard NR** (+10.91). PDM-Closed wins everywhere else, dominating all reactive splits.

**CarPlanner (Zhang et al., CVPR 2025, arXiv 2502.19908)** — first RL planner to beat IL **and** rule-based on nuPlan.
- Test14-random NR: CarPlanner **94.07** vs PDM-Closed **90.05** (+4.02) vs PLUTO **91.92** (+2.15).
- Reactive: CarPlanner **91.1** vs PDM-Closed **91.64** — **loses narrowly**, because it was trained non-reactive only. (Source: https://arxiv.org/html/2502.19908v1.) **[Val14 / Test14-hard rows not reported in the same table — they use Test14-random + reduced-Val14.]**

**CaRL (Jaeger, Dauner et al., CoRL 2025, arXiv 2504.17838)** — scalable RL with a single route-completion reward.
- Val14: CaRL NR **91.3** / R **90.6** vs PDM-Closed NR **92.8** / R **92.1**.
- → CaRL **does NOT beat PDM-Closed on nuPlan**; it is the best *learning-based* planner and runs 7–17× faster, but the rule-based ceiling holds. (Also: CaRL on CARLA Longest6 = 64 DS, beating RL baseline Roach's 22.) (Source: https://arxiv.org/html/2504.17838v2.)

**Plan-R1 (Tang et al., ICLR 2026, arXiv 2505.17659)** — predict-then-RL-finetune with VD-GRPO; Table 1 (verbatim, https://arxiv.org/html/2505.17659v2):

| Split | Plan-R1 NR | Plan-R1 R | PDM-Closed NR | PDM-Closed R |
|---|---|---|---|---|
| Val14 | 87.99 | 84.97 | 92.84 | 92.12 |
| Test14-hard | **75.31** | 73.18 | 65.08 | 75.19 |

→ Same pattern: beats PDM-Closed on Test14-hard NR (+10.23), loses on Val14 (both modes) and Test14-hard R (−1.99).

**StateTransformer / STR2 (Sun et al.; arXiv 2310.19620 → MoE successor 2410.15774)** — scaling-law decoder-only planner. Strong OL accuracy that scales with data/model size, but the authors concede it "struggles to generalize in closed-loop, interaction-heavy settings unless explicitly trained for reactivity." Not a CL leader. **[exact CL numbers UNVERIFIED]** (Sources: https://github.com/Tsinghua-MARS-Lab/StateTransformer; https://arxiv.org/abs/2410.15774.)

**Gen-Drive (Huang et al., ICRA 2025, arXiv 2410.05582)** — generation-then-evaluation diffusion + VLM-preference reward + RL finetune. Reports ~+16 overall nuPlan points and ~50% collision reduction over its IL baseline, but inference is 282–484 ms (slow). Does not claim to beat PDM-Closed across the board. (Source: https://arxiv.org/abs/2410.05582.)

**GameFormer-Planner (Huang et al., ICCV 2023, arXiv 2303.05760)** — hierarchical level-k game-theoretic prediction+planning; a 2023 nuPlan-challenge contender, now used mainly as a learned baseline that PLUTO/PlanTF/CarPlanner beat. **[exact Val14 CL numbers UNVERIFIED from primary]** (Sources: https://arxiv.org/abs/2303.05760; https://github.com/MCZhi/GameFormer-Planner.)

### 1c. Verdict on Q1

- **Val14, both modes:** PDM-Closed (~92–93 NR / ~92 R) is still **unbeaten** by any reproducible *pure*-learning planner. PLUTO's +0.04 R "win" is noise and depends on rule-based post-processing.
- **Test14-hard NR:** **broken** — Diffusion Planner (75.99), Plan-R1 (75.31), CarPlanner, and PLUTO-with-refinement clearly exceed PDM-Closed (65.08), by **~+10**. This is the one regime where learning has genuinely pulled ahead.
- **Test14-hard R / reactive everywhere:** PDM-Closed (75.19 on T14-hard R) remains very hard to beat; RL planners trained non-reactively (CarPlanner, partly Plan-R1) regress under reactive agents.
- **The deepest 2026 twist (When Planners Meet Reality, arXiv 2510.14677):** when the *IDM* reactive agents are replaced with the learned **SMART** agent model, **"IDM-based simulation overestimates planning performance: nearly all scores deteriorate."** IL planners drop most (Urban Driver −8, GC-PGP −6, PlanTF −5 on Val14); the *closed-loop-trained* RL planner **CaRL is the most stable** (93→90) and the **only one to roughly hold up**, edging PDM-Closed (90 vs 89 CLS-SR). The conclusion: **the PDM-Closed ceiling is partly an artifact of the IDM-agent simulator, and closed-loop-trained policies generalize better to realistic agents.** (Source: https://arxiv.org/html/2510.14677v1.) Corroborated by **nuPlan-R** (arXiv 2511.10403), a reactive-multi-agent re-benchmark.

---

## 2. Open-loop vs closed-loop misalignment — the literature and the mechanism

**This is the project's core thesis and it is well-established.**

**Primary establishing paper — Dauner et al. (PDM), CoRL 2023.** The abstract states the two tasks "are fundamentally misaligned and should be addressed independently." The mechanism, quoted from the paper (https://arxiv.org/html/2306.07962v1):

> "learned PDM-Open generates predictions along the lane chosen by the human driver, thereby obtaining a high OLS. Nonetheless, as errors accumulate in its short-term predictions during the simulation, the model's trajectory veers off the drivable area, culminating in a subpar CLS."

And:

> "we observe that the best [open-loop] results are achieved when using only this centerline as scene context (i.e., ignoring all information regarding the map and other agents)."

**The explanation, distilled:**
1. **OLS rewards mimicking the human's *recorded* path.** A model that just extrapolates along the chosen lane (no map, no agents) maximizes OLS — it is a good *forecaster*.
2. **CLS rewards *recovering from your own mistakes*.** In closed loop, the ego executes its prediction, lands in a slightly off-distribution state, and must correct. A forecaster never trained on its own induced states accumulates error (**covariate shift / compounding error**, the classic DAgger problem) and drifts off-road or collides.
3. Therefore **low ADE/FDE (high OLS) does not imply good driving**, and the *best* CL planner (PDM-Closed) is a *poor* forecaster (OLS 44). The metrics are not just uncorrelated — across PDM variants they are **anti-correlated**.

**Corroborating evidence across the literature:**
- **PlanTF (2309.10443)** frames the same phenomenon as the **"imitation gap" / compounding error**, motivating its augmentation study (§3).
- **When Planners Meet Reality (2510.14677)** extends the misalignment to a *third* axis: **closed-loop-with-IDM vs closed-loop-with-realistic-agents** also disagree. So there are really three measurements (OL, CL-IDM, CL-learned-agents) and they each rank planners differently.
- **CaRL (2504.17838)** motivates RL precisely because "RL ... does not suffer from compounding errors like imitation learning" — an explicit acknowledgment that the OL→CL gap is an IL artifact.

**Caveat to log in the project:** the misalignment is sharpest for *naive* IL. PlanTF/PLUTO partly *re-align* OL and CL (a well-augmented IL model can be decent at both). So the honest statement is **"OL and CL are misaligned, and closing the gap requires explicitly training against self-induced distribution shift,"** not "OL is useless."

---

## 3. The exact anti-covariate-shift recipes (for F4)

**These are ground-truth values pulled from the official source/config, not paraphrased from prose.**

### 3a. PlanTF — `state_perturbation` (https://github.com/jchengai/planTF)

Two sets of numbers exist; **the config YAML is the recipe actually used for the reported run**; the source defaults are looser.

**Actually-used config** (`config/data_augmentation/state_perturbation.yaml`):
- `augment_prob: 0.5` (perturbation applied to 50% of training samples)
- `normalize: True` (re-normalization is mandatory — see finding below)
- `dt: 0.1`, `hist_len: 21`
- Uniform perturbation bounds, per-dimension `[x, y, yaw, vel, accel, steer, steer_rate]`:
  - `low  = [-1.0, -0.75, -0.35, -1, -0.5, -0.2, -0.1]`
  - `high = [ 1.0,  0.75,  0.35,  1,  0.5,  0.2,  0.1]`
  - → **longitudinal x: ±1.0 m, lateral y: ±0.75 m, yaw: ±0.35 rad (≈ ±20°)**, velocity ±1 m/s, accel ±0.5, steering ±0.2 rad, steering-rate ±0.1.

**Source-code defaults** (`src/data_augmentation/state_perturbation.py`) are wider — `x∈[0,2.0], y∈[−1.5,1.5], yaw∈[−0.55,0.55]` — and include a **velocity-floor safety clamp** `new_state[3] = max(0.0, v)` and a collision check. **No bicycle-model re-projection of the *future* trajectory** is applied — only the *current* state is perturbed; the GT future is reused. **[FLAG: config vs source defaults disagree — use the YAML values for F4 reproduction.]**

**PlanTF's key empirical finding (quote, https://arxiv.org/html/2309.10443):** perturbation only helps *with proper feature normalization* — "it is important to keep the data distribution close between training and testing." Naively perturbing without re-normalizing the scene frame **hurts**. This is the single most actionable detail for F4: the augmentation and the input-normalization must be co-designed.

PlanTF's three studied augmentations: **Perturbation (P)** → **Re-Normalization (RN)** → **Future Correction (FC)** (correct the GT future via constrained nonlinear optimization so it remains kinematically consistent with the perturbed start). PlanTF also uses a **state attention-dropout encoder, dropout rate 0.75**, to prevent over-reliance on the ego's own history (causal confusion / "shortcut" on ego velocity).

### 3b. PLUTO — contrastive + 6 augmentations (https://github.com/jchengai/pluto, `src/data_augmentation/contrastive_scenario_generator.py`)

PLUTO frames augmentation as **contrastive**: each sample gets a **positive** aug (GT still valid) and a **negative** aug (GT now invalid → used only for the contrastive loss, never as a regression target).

**Triplet contrastive loss:**
L_c = −log [ exp(sim(z,z⁺)/σ) / (exp(sim(z,z⁺)/σ) + exp(sim(z,z⁻)/σ)) ]

**Total loss** = w₁·L_imitation + w₂·L_prediction + w₃·L_auxiliary + w₄·L_contrastive.

**Positive augmentations (𝒯⁺, GT preserved):**
1. **State perturbation** (uniform), per-dim `[x,y,yaw,vel,accel,steer,steer_rate]`:
   - `low  = [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2]`
   - `high = [2.0,  1.5,  0.35,  1,  0.5,  0.2,  0.2]`
   - → longitudinal x ∈ [0, 2.0] m, lateral y ±1.5 m, yaw ±0.35 rad. Includes an **iterative safety rescale**: scale noise by 0.5, up to 5 retries, until collision-free.
2. **Non-interactive agent dropout** — drop agents whose bbox never intersects ego's path; drop probability **0.5**, drop fraction sampled in **[0.1, 1.0]**.

**Negative augmentations (𝒯⁻, GT made invalid → contrastive only):**
3. **Leading-agent dropout** — remove the lead vehicle.
4. **Leading-agent insertion** — insert a synthetic vehicle on `free_path_points` where a collision would occur; velocity = ratio of ego/similar-agent speed scaled by a coeff sampled in **[0.0, 0.8]**; shape perturbed by **×[0.9, 1.1]**.
5. **Interactive-agent dropout** — remove agents with `interaction_label > 0` within `max_interaction_horizon = 40`.
6. **Traffic-light inversion** — flip red→green/unknown in intersection waiting scenes.

Also a **cost-map transform** (500×500 px, 0.2 m/px) shifted/rotated to match the perturbation. **WHY this is the F4 design target:** the negative augmentations explicitly teach *causal* structure — "if I delete the agent I was reacting to, my plan should change" — which is exactly the causal-confusion failure that plagues naive IL (the model latches onto ego-history shortcuts instead of the true cause).

---

## 4. Mistakes, criticisms, gaps — where the field admits it's stuck

**From the papers' own limitations and the community critiques:**

1. **The benchmark itself is partly an illusion (biggest 2025–26 gap).** *When Planners Meet Reality* (2510.14677) and *nuPlan-R* (2511.10403) show the **IDM reactive agents are passive and cannot react to adjacent lanes**, so CLS-R systematically **overestimates** planner quality and **biases rankings**. Swapping in learned agents (SMART) drops nearly all scores and re-orders the leaderboard. → **Many published "we beat PDM-Closed" claims may not survive realistic agents.** This is the field's most honest current admission.

2. **PDM-Closed's dominance is built on the centerline + lane-graph prior.** It implicitly assumes a good route/centerline exists and is roughly followable. Reviewers/critics note it leans on hand-tuned IDM hyperparameters and the same kinematic prior the simulator rewards — it can be **fragile in scenarios without a clean centerline** (unstructured lots, severe occlusion, novel topologies) and does not *learn* the long tail. (Dauner et al. itself frames PDM as evidence that learning has *not yet* paid off in CL, not that rules are the final answer.)

3. **OLS is a near-useless leaderboard axis but still reported.** Because OL and CL are anti-correlated for the best planners, OLS mainly measures "forecasting," not "driving." The field keeps it for legacy/challenge reasons; it actively misleads model selection.

4. **RL planners overfit the training agent-model and reward.** CarPlanner regresses in reactive mode (trained non-reactive only); Plan-R1 loses on Test14-hard R; CaRL explicitly scopes out highway/high-speed and notes failure modes (missed off-ramps, rear-end *by other cars*). RL inherits the simulator's biases — including the IDM-agent bias from gap #1.

5. **Post-processing laundering.** PLUTO and several "SOTA learned" results only beat PDM-Closed *with a rule-based refinement/post-processing layer* bolted on. Strip it and the pure network underperforms PDM-Closed. Independent reproductions (e.g. Plan-R1's table reporting PLUTO Val14 NR 88.89) make this visible. → Reported "learning beats rules" is sometimes "learning + rules beats rules."

6. **Reproducibility drift.** PDM-Closed's reproduced Val14/Test14-hard numbers vary by 1–3+ points across papers (and Test14-hard NR for PDM-Closed is reported anywhere from ~65 to ~92 depending on harness/split version). Cross-paper rankings within 2–3 points are **not** trustworthy.

7. **Causal confusion persists.** Both PlanTF (dropout 0.75) and PLUTO (contrastive negatives) are *mitigations*, not solutions — they reduce, not eliminate, the ego-history shortcut. No planner robustly demonstrates learned, generalizable *interactive* causal reasoning.

---

## 5. Can a SOLO student produce a non-trivial result here? (brutal assessment)

**What a solo student CANNOT competitively do:**
- Train a new SOTA planner from scratch — needs the full nuPlan cache (~tens of TB), multi-GPU weeks, and is a crowded race (PLUTO/CarPlanner/Diffusion-Planner/CaRL/Plan-R1 are well-funded lab efforts). You will at best "reproduce known findings."
- Beat PDM-Closed on the standard benchmark — even CoRL-2025 RL (CaRL) doesn't, on Val14.
- Out-engineer the augmentation recipe — PLUTO/PlanTF already swept it.

**Where the genuine, under-explored questions are (all solo-feasible because they are *analysis/diagnosis*, not new-SOTA-training):**
- **(A) Benchmark-validity / agent-realism.** The *When Planners Meet Reality* result is one paper, one agent model (SMART), aggregate deltas. **Nobody has done a careful, per-scenario-type, per-planner diagnosis of *which specific behaviors* flip when you change the reactive-agent model — and whether the "PDM-Closed beats learning" conclusion is robust to the choice of agent model.** This is pure evaluation work: download released checkpoints (PLUTO, PlanTF, Diffusion-Planner, CaRL, PDM-Closed are all open), run them under ≥2 agent models, and measure *ranking stability*. No training required.
- **(B) The OL↔CL predictivity question, made quantitative.** Everyone *asserts* OL doesn't predict CL. **Nobody has published a rigorous regression/correlation study** over many planners/checkpoints quantifying *exactly how predictive (or anti-predictive) each open-loop metric is of each closed-loop metric, per scenario type* — and whether any *cheap* open-loop-computable proxy (e.g. on-self-induced-states ADE, or a 1-step rollout divergence) **does** predict CL. A positive result here would be genuinely useful: a fast CL-proxy metric.
- **(C) Augmentation-magnitude sensitivity.** The PlanTF config (±1.0 m / ±0.75 m / ±0.35 rad) and PLUTO defaults *disagree*, and the YAML-vs-source mismatch is undocumented. **A controlled sweep of perturbation magnitude vs CL recovery, holding architecture fixed (use PlanTF as the testbed), with the explicit hypothesis that there is an optimal perturbation scale that maximizes CL while OL degrades monotonically** — directly tests the misalignment and is cheap (one small model, repeated).

---

## 6. The single most defensible novel question for a solo student in 2026

> **"Is the rule-based PDM-Closed advantage on nuPlan an artifact of the IDM reactive-agent simulator, and does any *cheap open-loop-computable* metric predict closed-loop ranking once the reactive-agent model is made realistic?"**

**Why this is the right question (research backing):**
- It sits exactly on the project's thesis (OL↔CL misalignment) **and** the freshest, least-saturated crack in the field — agent-realism (When Planners Meet Reality, 2510.14677; nuPlan-R, 2511.10403, both ≤8 months old, both leaving the per-planner / metric-predictivity analysis open).
- It is **solo-feasible**: it uses *released checkpoints* (PDM-Closed, PlanTF, PLUTO, Diffusion-Planner, CaRL are all public on GitHub) and *released agent models* (SMART). **No SOTA training** — the compute is *inference + simulation*, which one A100 can do.
- It produces a **falsifiable, citable claim** regardless of outcome: either "PDM-Closed's lead is robust to agent realism" (defends the rule-based ceiling) or "it collapses under realistic agents" (a publishable benchmark-validity result), **plus** the first quantitative OL→CL predictivity map and, ideally, a *cheap CL proxy* the community would actually use.
- It cannot be dismissed as "reproduces known findings": the known finding is the aggregate −5/−3 deltas; the novel contribution is **(i)** ranking-stability across multiple agent models, **(ii)** per-scenario-type attribution, and **(iii)** the search for an open-loop-side predictor of closed-loop rank under realistic agents.

**Concrete first experiment (week 1):** run PDM-Closed + PlanTF + PLUTO on Val14 and Test14-hard under (a) stock IDM agents and (b) SMART agents; record CLS-NR/CLS-R and the full metric breakdown; compute Spearman rank-correlation of planner ordering between the two agent models. If the ordering is unstable, the project has its thesis. This is the F4 / frontier-upgrade hook.

---

## Source index (URLs)

- nuPlan benchmark — https://arxiv.org/abs/2106.11810
- PDM / Parting with Misconceptions (Dauner et al., CoRL 2023) — https://arxiv.org/abs/2306.07962 · html https://arxiv.org/html/2306.07962v1 · code https://github.com/autonomousvision/tuplan_garage · OpenReview https://openreview.net/forum?id=o82EXEK5hu6
- PlanTF (ICRA 2024) — https://arxiv.org/abs/2309.10443 · html https://arxiv.org/html/2309.10443 · code https://github.com/jchengai/planTF
- PLUTO — https://arxiv.org/abs/2404.14327 · code https://github.com/jchengai/pluto
- GameFormer (ICCV 2023) — https://arxiv.org/abs/2303.05760 · planner https://github.com/MCZhi/GameFormer-Planner
- Diffusion Planner (ICLR 2025) — https://arxiv.org/abs/2501.15564 · html https://arxiv.org/html/2501.15564v2 · code https://github.com/ZhengYinan-AIR/Diffusion-Planner
- CarPlanner (CVPR 2025) — https://arxiv.org/abs/2502.19908 · html https://arxiv.org/html/2502.19908v1
- CaRL (CoRL 2025) — https://arxiv.org/abs/2504.17838 · html https://arxiv.org/html/2504.17838v2 · code https://github.com/autonomousvision/CaRL · OpenReview https://openreview.net/forum?id=1otaE496Vm
- Gen-Drive (ICRA 2025) — https://arxiv.org/abs/2410.05582
- StateTransformer / STR2 — https://github.com/Tsinghua-MARS-Lab/StateTransformer · STR2 (MoE) https://arxiv.org/abs/2410.15774
- Plan-R1 (ICLR 2026) — https://arxiv.org/abs/2505.17659 · html https://arxiv.org/html/2505.17659v2 · code https://github.com/XiaolongTang23/Plan-R1
- When Planners Meet Reality (2025) — https://arxiv.org/abs/2510.14677 · html https://arxiv.org/html/2510.14677v1
- nuPlan-R reactive benchmark (2025) — https://arxiv.org/abs/2511.10403
