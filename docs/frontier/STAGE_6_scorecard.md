# STAGE 6 — ORCHESTRATOR EVALUATION & ITERATION (Reviewer Panel Scorecard)

> Panel: a Diffusion Planner author (D), a Waymo planning research lead (W), a Physical Intelligence engineer (P). Question per stage: **"Would this design and plan be accepted as methodologically sound by this panel?"** Score 1–10. Below 8 → revise and re-score. Up to 3 passes. Error ledger checked before each stage.

## Pass 1 — initial scores

| Stage | D | W | P | Mean | Verdict | Flaws → ledger |
|---|---|---|---|---|---|---|
| 1 SOTA | 8 | 8 | 7 | 7.7 | revise | E9 (compute gap unquantified) |
| 2 Data | 7 | 7 | 6 | 6.7 | revise | E1 (goal construction), E2 (correction seam), E3 (perturbed-frame inputs) |
| 3 Arch | 7 | 8 | 7 | 7.3 | revise | E4 (horizon undecided), E5 (mode collapse) |
| 4 Train | 7 | 7 | 6 | 6.7 | revise | E6 (non-causal collision loss), E7 (no online expert for DAgger) |
| 5 Eval | 8 | 8 | 8 | 8.0 | minor | E8 (withheld test caveat) — fixed anyway |

Panel commentary (Pass 1):
- **P (Physical Intelligence):** "The collision loss against logged agents is the classic non-causal trap — you'd be training the policy to fit the data's counterfactual. And nuPlan has no online expert; calling the relabel 'DAgger' without saying it's a surrogate oracle would get flagged in review." → E6, E7.
- **D (Diffusion Planner):** "The route-region goal is the right idea and the core contribution, but as written it's a slogan — show the construction from `route_roadblock_ids` and how it stays multimodal at junctions, or a reviewer can't reproduce it." → E1. "K candidates don't guarantee diversity; anchor them." → E5.
- **W (Waymo):** "Perturbation without re-expressing the scene in the perturbed frame and without re-fitting the history is internally inconsistent; and you must say Val14/Test14-hard are offline community splits." → E2, E3, E8.

## Pass 2 — re-scores after applying E1–E10

| Stage | D | W | P | Mean | Verdict |
|---|---|---|---|---|---|
| 1 SOTA | 9 | 8 | 8 | 8.3 | accept |
| 2 Data | 8 | 8 | 8 | 8.0 | accept |
| 3 Arch | 8 | 9 | 8 | 8.3 | accept |
| 4 Train | 8 | 8 | 8 | 8.0 | accept |
| 5 Eval | 9 | 8 | 8 | 8.3 | accept |

Panel commentary (Pass 2):
- All four High/Med flaws resolved. Collision loss is now honestly scoped to a causal proxy + closed-loop; DAgger relabel correctly labeled approximate with a PDM-Closed oracle; goal construction is reproducible; perturbation is internally consistent.
- **Remaining concerns are feasibility, not soundness:** (a) Val14 closed-loop sim is CPU-bound and slow; (b) the 25–50× compute gap to Diffusion Planner means absolute numbers will trail SOTA; (c) the contrastive CIL and joint ego+agent denoising are correctly marked stretch. These are recorded in the final plan's honesty section, not as methodological errors.

**All stages ≥ 8.0 after Pass 2 → iteration stops (2 of 3 passes used).**

## What the panel would still push back on (honest residual gaps)
1. **No absolute-SOTA claim possible** at this compute budget — the contribution is the *controlled when-does-diffusion-help result*, not beating PLUTO. Acceptable for a workshop / strong-analysis paper; a top-venue full paper needs more compute and likely the joint-denoising + RL-finetune extensions.
2. **Closed-loop eval throughput** is the binding constraint on iteration speed, more than GPU training.
3. **Perturbation magnitudes and PLUTO/PlanTF decimals** remain code-to-verify, not yet verified.

## Scorecard summary
- Mean across stages, Pass 2: **8.18 / 10.** All stages individually ≥ 8. Methodologically sound per the panel rubric, with feasibility caveats explicit.
