# Moderation Experiment #18 — Results (interaction-criticality)

Status: COMPLETE, with a corrected analysis (ADR-033/034) that supersedes the original HC3 /
"informative null" framing. Pre-registered (RESEARCH_PROTOCOL, POWER_ANALYSIS, ADR-018/027/028).
Frozen N=800 manifest, both reactive modes, matched CLS-selected v4 checkpoints.

## TL;DR (corrected)

H1 is not supported, and — more importantly — **the study could not have cleanly tested it**, for
two independent reasons that each block detection:

1. **Treatment collapse (ADR-029).** The diffusion policy collapsed to a near-deterministic point
   estimator: its closed-loop trajectories sit a median 0.47 m from the deterministic policy's on a
   35 m horizon, and K head samples disperse only ~0.13 m. The "multimodality" whose benefit H1
   tests was effectively absent, so a near-zero moderation slope is partly forced by construction.
2. **Metric saturation (ADR-033).** Every standard nuPlan closed-loop outcome is at ceiling (CLS
   41-53%, progress 69-72%, comfort ~100%, drivable-area 83-85%, TTC 54-67%, collisions 58-72% of
   tokens at >=0.99), so the metric has little room to express a moderated gap.

Under the **pre-registered inference that the original analysis omitted** (scenario_type fixed
effects + wild-cluster bootstrap + TOST), the headline contrast is null in both modes (r0 p=0.72,
TOST-equivalent to 0; r1 p=0.83), it **survives removal of the ceiling** (null on unsaturated hard
subsets, ADR-034), and the direct H1 prediction is **wrong-signed** at its leverage point. So the
null is real and robust — but it should be read as "the collapsed diffusion policy provides no
interaction-criticality-dependent closed-loop benefit," NOT as "multimodality does not help driving."

This places the result in the lineage of Dauner et al., *Parting with Misconceptions about
Learning-based Vehicle Motion Planning* (CoRL 2023, arXiv:2306.07962): a measured demonstration that
a widely-assumed benefit is, on standard nuPlan, neither realized nor testable as usually set up.
The collapse itself is independently confirmed by DIVER (arXiv:2507.04049, 2025), which states that
diffusion trajectories "collapse around the ground truth under imitation supervision."

## Design

- Cells: 2x2 = head {deterministic, diffusion} x goal {route, precise}, each x 3 seeds = 12 cells.
- Checkpoint per cell: argmax closed-loop CLS over candidate epochs on a disjoint 48-scenario probe
  (ADR-018), from fixed-budget matched 150-epoch training (no early stop).
- Eval set: frozen N=800 manifest, 4 F4 bands x 200, sha256 f3b3b234 (ADR-028), disjoint from the
  CLS-selection probe. Reactive: r0 = CLS-NR (logged-box replay); r1 = CLS-R (reactive IDM agents).
- Metric: official nuPlan closed-loop score (two_stage_controller), per scenario, seed-averaged.
- Moderator: F4 v1.1 = G_stop * (1 - (1 - S_branch)(1 - S_inter)), interaction-criticality (ADR-027).
- Test: per token Delta = CLS(diffusion) - CLS(deterministic); moderation Delta ~ F4 per goal;
  headline = contrast beta1(route) - beta1(precise), predicted > 0.

## Per-cell mean CLS (seed-averaged, n=800 each)

| head | goal | r0 (non-reactive) | r1 (reactive) |
|---|---|---|---|
| deterministic | route   | 0.8654 | 0.8488 |
| diffusion     | route   | 0.8642 | 0.8449 |
| deterministic | precise | 0.9603 | 0.9119 |
| diffusion     | precise | 0.9620 | 0.9156 |

Precise-goal conditioning adds ~0.10 CLS. Diffusion ~= deterministic within each cell (consistent
with the collapse: the two heads are near-copies, ADR-029).

## Moderation — corrected inference (ADR-033, the headline)

Pre-registered method: OLS Delta ~ F4 + scenario_type fixed effects, wild-cluster bootstrap SE/p by
scenario_type (Cameron-Gelbach-Miller 2008; plain HC3 is anticonservative with correlated
within-type residuals), precise-slope equivalence by TOST (Lakens 2017). The bootstrap self-test
passes (Type-I 0.025 under a clustered null; full power under a true effect).

| mode | contrast beta1 | cluster SE | wild-cluster p (2-sided) | naive HC3 (shipped) | TOST equiv @0.02 |
|---|---|---|---|---|---|
| r0 | -0.0028 | 0.0082 | 0.72 | -0.0064 (se 0.0062) | YES |
| r1 | +0.0025 | 0.0112 | 0.83 | -0.0013 (se 0.0068) | (n/a) |

All PDM sub-components null too. The corrected method also **overturns a spurious effect**: on
drivable_area_compliance (r0) the shipped HC3 gives beta1=-0.024, t=-1.8 ("marginal"); the
wild-cluster bootstrap gives p=0.19 — null. Cluster SEs run ~50% larger than HC3 throughout.

The original HC3 table (below) is retained for the record; it is superseded by the corrected
inference above.

<details><summary>Original (uncorrected) HC3 tables — superseded</summary>

r0: route beta1 -0.0116 (HC3 0.0059); precise -0.0052 (0.0029); contrast -0.0064 (0.0062), p 0.85.
r1: route beta1 -0.0050 (0.0061); precise -0.0037 (0.0029); contrast -0.0013 (0.0068), p 0.58.
</details>

## Ceiling-robustness (ADR-034): the null is not a metric artifact

Re-running the contrast on hard / unsaturated subsets (sliced by grand mean CLS, which is orthogonal
to the difference-of-differences contrast, so not outcome-conditioning; ceiling fully removed):

| slice | n | frac at ceiling | contrast beta1 | wild-cluster p |
|---|---|---|---|---|
| r0 unsaturated (M<0.99) | 588 | 0.00 | +0.0011 | 0.91 |
| r0 hardest 25% | 200 | 0.00 | +0.0119 | 0.58 |
| r1 unsaturated (M<0.99) | 649 | 0.00 | +0.0056 | 0.69 |
| r1 hardest 25% | 200 | 0.00 | +0.0199 | 0.30 |

Direct H1 prediction at the top-F4 quartile (where multimodality should pay off): mean(Delta_route) -
mean(Delta_precise) = **-0.0089 r0** (CI [-0.018, +0.0003]), **-0.0099 r1** (CI [-0.020, -0.000],
excludes 0). H1 predicts > 0; the data trend the other way. Where the metric *can* move, there is
still no moderation, and the sign is reversed -> the null reflects treatment collapse, not the
ceiling.

## Retraction: the "informative null / rules out" claim

The earlier version argued the null "RULES OUT" the moderation via a power/MDE analysis. **That claim
is withdrawn.** The power calculation assumed a genuine treatment whose effect size the design could
detect; under treatment collapse (ADR-029) the contrast is ~0 partly by construction, so one cannot
"rule out the moderation of multimodality" from data in which multimodality was absent. The
defensible statements are the ones above: the null holds under the correct inference and survives
ceiling removal. (POWER_ANALYSIS.md remains valid as the a-priori design calculation.)

## Secondary observations (not the pre-registered hypothesis)

- Under reactive agents, diffusion's *comfort* advantage grows with interaction-criticality
  (contrast beta1 +0.0146, wild-cluster p=0.011) — but this is 1 of ~14 outcome tests and comfort is
  near-binary, so it does not survive multiple-comparison correction. Suggestive only.
- Diffusion shows a small mean CLS edge in reactive-precise (+0.0037, CI [+0.0017, +0.0057]).

## Reproducibility

- Corrected results: docs/frontier/results/remod_r{0,1}.json (sub-components, wild-cluster + TOST),
  slices_r{0,1}.json (unsaturated subsets). Original: r{0,1}_result.json.
- Pipeline: eval_array_v4.sbatch -> merge_eval_full.py (seed-average, keep sub-components +
  scenario_type) -> analyze_moderation_v2.py (FE + wild-cluster bootstrap + TOST) and
  moderation_slices.py. F4: features/f4/f4_scores_v11.json. See ADR-029/033/034.
