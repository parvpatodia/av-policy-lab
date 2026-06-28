# Frontier results — does trajectory multimodality help closed-loop driving on nuPlan?

Entry point for the frontier-upgrade study. Full writeup: **[PAPER.md](PAPER.md)**. Derivations and
every decision: **[../../DECISIONS.md](../../DECISIONS.md)** (ADR-013…047). Honest status: complete.

## TL;DR (the honest, artifact-controlled result)

We set out to test H1: *does a diffusion policy's closed-loop advantage over a capacity-matched
deterministic policy grow with scene interaction-criticality (F4)?* The answer, after removing every
confound we could find, is **no — and the more interesting contribution is why the question is so
hard to even pose on standard nuPlan.**

1. **The original experiment could not test H1.** The diffusion policy trained by standard single-
   future imitation **collapsed to a near-deterministic point estimator** (ADR-029); the architecture
   is fine (a synthetic control recovers modes perfectly, ADR-030); and every nuPlan closed-loop
   metric is **saturated** (ADR-033). Under the *pre-registered* inference the first analysis omitted
   (fixed effects + wild-cluster bootstrap + TOST), the moderation is null and **survives ceiling
   removal** (ADR-034) — so it was treatment-absence + saturation, not a clean test.
2. **We recovered a genuinely present treatment.** Supervised fixes (winner-take-all, diversity
   regularization) only fan or *manufacture* uniform modes (ADR-032/037); a **reward-guided RL recipe**
   (validated open-loop proxy reward → GRPO/AWR) produces a **scene-adaptive** multimodal policy that
   **drives in real closed-loop** with no reward-hacking (ADR-038–041), bounded by a proxy-CLS ceiling
   (ADR-042).
3. **With the treatment present, H1 still does not hold.** Executed-CLS moderation flipped from
   wrong-signed-null to +0.035 (suggestive, p=0.063, ADR-043/044) but **did not replicate** in a
   distribution-aware safety oracle; and a **fair best-of-K control** showed the learned multimodality
   provides **~0 safety value beyond matched-random perturbations** of the deterministic policy (a
   coin flip), with no F4-moderation in any metric (ADR-046/047). The "latent value" was a selection
   artifact, retracted.

**Conclusion.** On standard nuPlan closed-loop, the much-assumed multimodal benefit of diffusion /
multi-hypothesis planners is unrealized under standard training, hard to even measure (executed-CLS is
blind to the predictive distribution), and — once a present, scene-adaptive treatment is supplied and
artifacts are controlled — not actually present, and certainly not interaction-criticality-specific.
This is consistent with Dauner et al. *Parting with Misconceptions* (CoRL 2023) and RAPiD
(diffusion→deterministic distillation); the collapse leg matches DIVER (2025).

## Finding map

| # | Finding | Where |
|---|---|---|
| F1 | Moderation null under correct inference (FE + wild-cluster + TOST) | [MODERATION_RESULTS.md](MODERATION_RESULTS.md), ADR-033 |
| F2 | Diffusion policy collapses to a point estimator (3-seed, EMA-ruled-out) | [MULTIMODALITY_FINDINGS.md](MULTIMODALITY_FINDINGS.md), ADR-029 |
| F3 | Architecture CAN be multimodal (synthetic positive control) | ADR-030 |
| F4 | kNN "available multimodality" is proxy-dependent (retracted) | ADR-031/036 |
| F5 | Supervised fixes fan / manufacture, not scene-adaptive | ADR-032/037 |
| F6 | Metrics saturated; null ceiling-robust + wrong-signed at high F4 | ADR-033/034 |
| RL | Reward-guided RL = scene-adaptive multimodality, drives closed-loop | ADR-038–042 |
| H1 | Re-test: CLS flip suggestive but non-replicating; fair control ~0 | ADR-043–047 |

Figures: [diagnostic_figure.png](figures/diagnostic_figure.png) (collapse / control / WTA / saturation),
[slope_flip_figure.png](figures/slope_flip_figure.png) (moderation slope across conditions).

## Reproduce (all under `nuplan/analysis/`, CPU unless noted; results in `results/`)

- Collapse + control: `multimodality_probe.py`, `compare_det_diff.py`, `synth_bimodal_test.py`
- Available-multimodality (both proxies): `available_multimodality.py`, `available_multimodality_v2.py`
- Supervised fixes: `wta_derisk_train.py` (+ `--diversity-weight`), `wta_probe.py`
- Corrected moderation: `merge_eval_full.py` → `analyze_moderation_v2.py` (FE + wild-cluster + TOST;
  `--selftest` validates the bootstrap), `moderation_slices.py` (unsaturated subsets)
- S_inter v1.2 fix: `nuplan/features/f4_score.py` (deadband/clamp) → `recombine_f4_v12.py`
- RL capstone (GPU): `reward_proxy.py` (validate), `rl_train.py` (GRPO/AWR), serve via
  `serving/policy_planner.py --head wta`, eval via `eval/run_cells.py`, `h1_retest.py`,
  `safety_oracle.py` (incl. the fair best-of-K control)

## Limitations (honest)

nuPlan-mini (Vegas-skewed); one encoder/head family; the RL uses an OPEN-LOOP proxy reward (bounded by
the open-loop≠closed-loop gap our own Part I documents — see ADR-045), so it is undertrained vs the
baseline. The only orthogonal lever left is closed-loop-reward RL (lifts CLS, not the moderation). None
of these change the H1 conclusion, which is robust across executed-CLS, a distribution-aware metric, and
an artifact control.
