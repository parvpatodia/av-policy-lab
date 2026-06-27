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

## Finding 4 — the data holds multimodal supervision the policy discards (ADR-031)
`available_multimodality.py`: context embedding from a RANDOMLY-INITIALIZED (untrained) SceneEncoder
(input geometry, no future leakage); k nearest-neighbor context-mates' logged-future endpoint
dispersion + mode count = the conditional spread the data actually carries.
- captured (policy) 0.13 m  <<  available (kNN, k=16) median 3.14 m  <<  marginal (random pair) 37 m.
  The kNN is valid (neighbors ~12x tighter than random); their futures are ~24x more dispersed than
  the policy emits. 46% of scenes have >=2 future modes.
- k-sensitivity (honesty): available is monotone in k; conservative floor (k=2) 0.79 m / 12.8% >=2
  modes, still ~6x the collapsed policy. Genuine conditional multimodality exists at the tightest
  matching; the k=16 figure is upper-leaning.
- rises with interaction-criticality: high s_inter 3.55 m / 53% >=2 modes vs low 2.94 m / 43%.
- concentrated at decision points: stationary-at-traffic-light 4.4 modes (100% >=2), light
  intersections 3.9, construction 3.2, long-vehicle 3.7; vs low-speed 1.8.

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

## Consequence
The five findings cohere into a single, contrarian result. For richly-conditioned nuPlan planning,
the per-scene future is largely DETERMINED by the scene (agents, map, route, traffic lights): the
diffusion policy collapses (F2) not by defect (F3) but because there is little per-scene multimodality
to learn; the "available multimodality" across similar contexts (F4) is mostly residual context
variation, not per-scene ambiguity; and the standard multimodality fix (F5, WTA) fans rather than
splits. So the much-assumed multimodal benefit of diffusion / multi-hypothesis planners is largely
absent here, which is why the interaction-criticality moderation (F1) is a well-powered null. The
multi-day Tier-3 retrain + re-moderation is NO-GO: it would test a treatment present in ~2-3% of
scenes and reproduce a near-null. The contribution is this diagnostic arc, end to end.
