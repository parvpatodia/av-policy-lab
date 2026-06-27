# Diffusion planners silently collapse to unimodal under standard imitation

**Frontier-upgrade results synthesis (2026-06-27).** Branch `frontier-upgrade`. All numbers below
are from frozen `f5_2x2_v4` checkpoints and the `f0_v3` feature set; scripts in `nuplan/analysis/`.

## TL;DR
We set out to test whether interaction-criticality moderates the closed-loop benefit of a diffusion
planner over a deterministic one. The moderation is a well-powered null. Investigating why, we found
the diffusion policy had silently collapsed to a near-deterministic point estimator, so the
"multimodality" treatment was absent. The collapse is **not** an architecture or objective defect
(a controlled test recovers multimodality perfectly), and the data **does** contain multimodal
supervision the policy discards, concentrated exactly at decision points (traffic lights,
intersections, construction). The widely-assumed multimodal benefit of diffusion planners is
therefore untested by standard single-future imitation.

## Finding 1 — the moderation is a well-powered null (experiment #18)
2x2 (head: deterministic/diffusion x goal: route/precise) x 3 seeds, matched 150-epoch
CLS-selected checkpoints, N=800 tokens/cell, both reactive modes. Headline cross-condition contrast
beta1(route)-beta1(precise) for Delta_CLS ~ F4 interaction-criticality: r0 -0.0064 (p .85), r1
-0.0013 (p .58). All four per-condition slopes <= 0. Realized MDE(80%) 0.0155/0.0170 < the smallest
hypothesized effect (0.02); 95% CIs exclude the hypothesized positive range. => informative null:
F4 interaction-criticality does NOT moderate the diffusion-vs-deterministic CLS gap. (See
MODERATION_RESULTS.md.)

## Finding 2 — the diffusion policy collapsed to a point estimator (ADR-029)
Open-loop multimodality probe (`multimodality_probe.py`): K=32 samples/scene from the diffusion head
as deployed (EMA, 20-step DDIM), endpoints clustered at eps=3.5 m (lane width).
- diff_route seed0 e130, N=2000: endpoint dispersion median 0.13 m on a 35 m path (0.37%);
  0.05% of scenes have >=2 modes. The head is a point estimator.
- 3-seed generality: seed0/1/2 all collapse (0.22-0.78% dispersion, ~0% multimodal). Not a fluke.
- epoch trajectory: ~8x contraction e010->e150; the collapse develops with training.
- EMA control: raw 0.39% ~= EMA 0.37% (not an averaging artifact).
- capstone (`compare_det_diff.py`): the deployed diffusion policy (medoid-of-8) vs the deterministic
  policy differ by a median 0.47 m ADE on a 35 m path (1.33%), sub-lane-width -> functionally a
  near-copy. Experiment #18 compared a policy to a near-copy of itself; the null was near-inevitable.

## Finding 3 — the architecture can be multimodal; the collapse is a data property (ADR-030)
Controlled synthetic-bimodal test (`synth_bimodal_test.py`): 32 contexts each mapping 50/50 to two
arcs 24 m apart, trained with the SAME x0-MSE + CosineSchedule as the real policy. Result: modes
mean 2.0, frac>=2 modes 1.0, samples split 50/50 across both arcs, ZERO mass at the collapse
midpoint, both modes covered in 100% of contexts. => the head + objective + DDIM sampler recover
multimodality perfectly when the supervision is multimodal. The real-data collapse is a property of
single-future-per-scene imitation (nuPlan logs one future per scenario -> the learned conditional is
correctly ~unimodal), not a model bug.

## Finding 4 — "available multimodality" is proxy-dependent and unreliable (ADR-031, revised by ADR-036)
We tried to estimate the per-scene future ambiguity the data carries via k-NN over scene context,
measuring the neighbors' logged-future endpoint dispersion. This estimate is NOT reliable: it swings
3-12 m with the (arbitrary) choice of context similarity, because none of the proxies is a true
matched context.
- random untrained-encoder kNN (available_multimodality.py): median 3.14 m, 46% >=2 modes -- but the
  embedding is degenerate (all scenes ~0.97 cosine; audit C1), so its neighbors are near-random.
- interpretable same-type match on v0/n_par/g_stop/b_r (available_multimodality_v2.py): median 5.64 m,
  68% >=2 modes, tightest-pair future distance median 12.4 m -- but scalar features ignore geometry,
  so same-type "matches" are not geometrically matched.
- for reference: marginal (random pairs) ~38 m; captured (policy) 0.13 m.
Because the estimate is proxy-dependent, the kNN approach CANNOT establish per-scene multimodality
from single-future data. The reliable per-scene estimate is the FULL-scene-conditioned one: train on
the entire scene and read the predictive spread -- exactly Finding 5 (WTA), where conditioning on all
tensors yields a ~2-5 m fan that does NOT split into modes. So ADR-031's "the data holds multimodality
the policy discards" is RETRACTED as a proxy artifact; the trustworthy evidence (Finding 5) indicates
the per-scene future is largely scene-determined.

## Finding 5 — the standard fix (WTA) yields a wider fan, not distinct modes; the future is largely determined (ADR-032)
De-risk before any multi-day retrain: a winner-take-all multi-hypothesis head (`WTAHead`, M=6),
trained on 16k real scenes with the same encoder + x0-loss, two assignment-sharpness settings
(`wta_derisk_train.py` + `wta_probe.py`, N=6000):
- eps=0.05: endpoint dispersion 1.77 m (14x the collapsed diffusion's 0.13 m), best-of-M minADE
  0.35 m, but frac>=2 modes = 0 in every scenario type at lane width.
- eps=0.01 (sharper): the fan WIDENED to 4.65 m but frac>=2 modes is still only 2.3% (max 3), even
  at decision points (traffic-light intersection 5.0 m spread, 3.5% >=2 modes). Sharpening inflated
  the fan width, it did not split it.
Robust verdict: WTA is a well-fit UNIMODAL predictor with adjustable spread, not a maneuver-level
multimodal policy on this data. Only ~2-3% of scenes are genuinely multimodal; the bottleneck is the
data, not the method.

## Finding 6 — corrected inference + ceiling-robustness (ADR-033/034)
The original "well-powered informative null that rules out the moderation" claim is RETRACTED. Under
the pre-registered inference the first analysis omitted (scenario_type fixed effects + wild-cluster
bootstrap + TOST; bootstrap self-test-validated), the headline contrast is null in both modes (r0
p=0.72, TOST-equivalent to 0; r1 p=0.83), and the corrected method overturns a spurious HC3 effect
(drivable_area r0: HC3 t=-1.8 -> wild-cluster p=0.19; cluster SEs ~50% larger throughout). Every
standard nuPlan CLS outcome is saturated (41-100% of tokens at ceiling), so the metric cannot express
a moderated gap; but the null SURVIVES removing the ceiling (null on unsaturated hard subsets, n=200-
649, frac@ceiling 0.0) and the direct H1 prediction is WRONG-SIGNED at high F4 (mean Delta_route -
mean Delta_precise = -0.009/-0.010). So the null is real and ceiling-robust -- it reflects treatment
collapse, not measurement.

## Consequence (scoped to what the evidence earns)
The findings cohere into one contrarian, diagnostic result. On the STANDARD nuPlan closed-loop setup
(this scenario set, this architecture, single-future imitation + medoid selection), the assumed
multimodal benefit of diffusion / multi-hypothesis planners is both UNREALIZED (the policy collapses
to a unimodal point estimator, F2/F3) and UNTESTABLE AS USUALLY SET UP (treatment collapse + a
saturated metric, F6). The "available multimodality" across similar contexts (F4) is largely residual
context variation, and the standard supervised fix (F5, WTA) fans rather than splits -- so the
per-scene future is mostly scene-determined. This is a measurement/method contribution in the Dauner
"Parting with Misconceptions" (CoRL 2023) lineage; the collapse leg is independently confirmed by
DIVER (arXiv:2507.04049). We do NOT claim multimodality is useless for driving in general -- only
that here it is neither present nor measurable, and that demonstrating a benefit would require BOTH a
genuinely multimodal policy (e.g. the RL route DIVER takes, not supervised WTA) AND an unsaturated
evaluation (harder scenarios / a continuous closed-loop metric). The multi-day Tier-3 retrain +
re-moderation is NO-GO on this data: it would test a ~2-3%-present treatment on a ceilinged metric and
reproduce the null.
