# STAGE 5 — EVALUATION & RESULT ANALYSIS

> Status: DESIGN ONLY. Defines protocol, success criteria, falsifiers, table formats. No results are reported — all cells below are placeholders to be filled by real runs.

## 1. Benchmarks & modes

- **Val14** (1,118 scenarios) — the comparable headline split.
- **Test14-hard** (≈272 scenarios; 20% lowest-scoring per type under PDM-Closed) — the long-tail stress test.
- **E8 caveat:** the nuPlan **official challenge test set is withheld**; Val14 and Test14-hard are **community-standard offline splits** (tuPlan-garage / Dauner et al.), not the closed leaderboard. Results here are comparable to the cited papers (which use the same offline splits), **not** to the official leaderboard. Stated to avoid implying leaderboard-grade comparability.
- **Both reactive (R-CLS) and non-reactive (NR-CLS) closed-loop**, plus **open-loop (OLS)**. WHY: Diffusion Planner / PLUTO / PlanTF all report exactly these; reactive vs non-reactive distinguishes "drives well in a frozen world" from "handles interaction." Recent work (arXiv:2510.14677) shows learned-reactive agents shift rankings — so report both honestly.

## 2. Metrics

**Closed-loop (primary): PDM-score decomposition** — report the aggregate AND every sub-term:
- Collision (no at-fault collisions), Drivable-area compliance, Driving-direction compliance, Time-to-collision, Progress, Speed-limit compliance, Comfort. WHY: the repo's Phase-3d already showed the learned policy *trades* collision margin for comfort — only the decomposition reveals this; the aggregate hides it.

**Open-loop (secondary):** ADE, FDE at 3/5/8 s, **miss-rate** (FDE > threshold). WHY: cheap proxy for the HP sweep; but explicitly subordinate — the repo already proved open-loop (0.058 m ADE) is nearly uncorrelated with closed-loop (27 m drift), so OLS is diagnostic only, never the headline.

## 3. Statistical protocol (the repo's existing strength — extend it)

- **Paired tests across scenarios:** Wilcoxon signed-rank (already in `nuplan/statistical_analysis.py`) for each method-vs-baseline pair, on per-scenario PDM-score.
- **Bootstrap 95% CIs** on the aggregate PDM-score (per method).
- **Per-scenario-type stratification:** report PDM-score per nuPlan scenario type (e.g., `starting_left_turn`, `high_magnitude_speed`, `following_lane_with_lead`); the Phase-3d analysis found 63% of error mass in 4 intersection scenarios — stratification is mandatory to avoid that masking.
- **Multiple-comparison awareness:** if reporting many ablations, note Holm-Bonferroni on the key claims.
- WHY: 30 scenarios gave ≈zero power (root cause #4); 1118 + paired tests + CIs is what makes a result *believable* to the reviewer panel.

## 4. Success criteria (pre-registered)

- **PRIMARY:** scene-aware diffusion model **statistically significantly > IDM and > the scene-blind MLP/diffusion baseline** on Val14 NR-CLS (paired Wilcoxon p<0.05, non-overlapping bootstrap CIs).
- **MULTIMODALITY (the Phase-3d redemption):** with the **route-region goal**, diffusion **significantly beats the deterministic regressor at junction scenario types**, while with the **precise near/far goal** the two are **statistically tied** (reproducing the old null). This *controlled pair* is the scientific contribution.
- **COVARIATE SHIFT:** perturbation-augmented model significantly > non-augmented on Val14 closed-loop (expected magnitude in the PlanTF range, ≈+10 pts NR-CLS).
- **COMPETITIVENESS (stretch):** within ~10 PDM-points of PLUTO/Diffusion-Planner on Val14 NR-CLS. (Matching them is a multi-month/team effort — see final plan.)

## 5. Falsifiers (what would prove the thesis WRONG)

- If diffusion **still** ties the MLP even with the multimodal route-goal AND a working scene encoder → the multimodality hypothesis (root cause #1) is wrong; diffusion adds nothing here.
- If perturbation augmentation does **not** improve closed-loop after correct re-normalization → either normalization is broken or the drift mechanism differs from PlanTF's.
- If scene encoder ON vs OFF shows **no** collision/off-road improvement → the encoder isn't actually using the map (check attention maps).
- WHY pre-register falsifiers: prevents post-hoc rationalization; this is what separates a methodologically sound study from a demo.

## 6. Results table format (matches Diffusion Planner / PLUTO reporting)

```
Method                         | Val14 NR-CLS | Val14 R-CLS | Test14-hard NR | Test14-hard R | OLS
-------------------------------|--------------|-------------|----------------|---------------|-----
IDM (rule)                     |     TBD      |    TBD      |      TBD       |     TBD       | TBD
PDM-Closed (rule, ref)         |   92.84*     |    ~93*     |     ~89*       |    ~76*       |  -
Scene-blind MLP (ours, base)   |     TBD      |    TBD      |      TBD       |     TBD       | TBD
Scene-blind Diffusion (ours)   |     TBD      |    TBD      |      TBD       |     TBD       | TBD
+ Scene encoder (ours)         |     TBD      |    TBD      |      TBD       |     TBD       | TBD
+ Route-goal + cross-attn      |     TBD      |    TBD      |      TBD       |     TBD       | TBD
+ Perturbation aug (full)      |     TBD      |    TBD      |      TBD       |     TBD       | TBD
PlanTF (lit ref)               |    84.83     |   76.78     |    72.68       |    61.70      |  -
PLUTO (lit ref)                |    ~93       |   ~87       |    ~84         |    ~76        |  -
Diffusion Planner (lit ref)    |    89.87     |    TBD      |     TBD        |    69.22      |  -
```
*Reference numbers from §Stage-1; re-confirm decimals at write-up. Every "TBD/ours" cell MUST come from a real run — never filled by estimate.

Plus a **PDM-decomposition table** (one row per method, one column per sub-metric) and a **per-scenario-type table**.

## 7. Workshop vs top-venue bar (honest)

- **Workshop-grade:** the controlled multimodality demonstration (route-goal vs precise-goal × diffusion vs deterministic) on Val14, with clean stats and ablations, even if absolute numbers trail PLUTO. This is achievable solo and is genuinely interesting (it explains *when* diffusion planning helps).
- **Top-venue-grade (ICRA/CoRL/NeurIPS):** the above PLUS competitive absolute Val14/Test14-hard numbers, the full ablation matrix, Test14-hard long-tail results, and ideally the joint ego+agent denoising or closed-loop RL finetune. This realistically needs a team or 6+ months — stated plainly in the final plan.

## HANDOFF TO NEXT STAGE (Stage 6 — Scorecard)

- The four pre-registered success criteria + five falsifiers are the rubric the reviewer panel scores against.
- The controlled "route-goal vs precise-goal" experiment is the load-bearing scientific claim — Stage 6 must verify the *design* actually isolates it.
- Eval cost (Val14 closed-loop is CPU-heavy, sharded SLURM array) feeds the feasibility section of the final plan.

## Honesty flags
- All result cells are placeholders. The plan produces NO numbers; producing them is the (compute-heavy) execution phase.
- PDM-Closed / PLUTO reference decimals are from Stage-1 sources and must be re-verified against the original tables before any publication.
