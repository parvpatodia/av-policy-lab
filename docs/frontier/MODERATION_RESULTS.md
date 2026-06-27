# Moderation Experiment #18 — Results (interaction-criticality)

Status: COMPLETE. Pre-registered (RESEARCH_PROTOCOL, POWER_ANALYSIS, ADR-018/027/028).
Frozen N=800 manifest, both reactive modes, matched CLS-selected v4 checkpoints.

## TL;DR

H1 is NOT supported. Scenario interaction-criticality (F4) does not moderate the closed-loop
advantage of the diffusion policy over the deterministic policy, in either reactive mode. The
pre-registered headline contrast beta1(route) - beta1(precise), predicted > 0, is null and
slightly negative in both modes.

## Design

- Cells: 2x2 = head {deterministic, diffusion} x goal {route, precise}, each x 3 seeds = 12 cells.
- Checkpoint per cell: argmax closed-loop CLS over candidate epochs on a disjoint 48-scenario
  probe (ADR-018), from fixed-budget matched 150-epoch training (no early stop).
- Eval set: frozen N=800 manifest, 4 F4 bands x 200, sha256 f3b3b234 (ADR-028), disjoint from the
  CLS-selection probe.
- Reactive modes: r0 = CLS-NR (non-reactive, logged-box replay); r1 = CLS-R (reactive IDM agents).
- Metric: official nuPlan closed-loop score (two_stage_controller), per scenario, seed-averaged
  over the 3 seeds.
- Moderator: F4 v1.1 = G_stop * (1 - (1 - S_branch)(1 - S_inter)), the pre-registered
  interaction-criticality score (ADR-027). Reconstructs from stored components with 0 error over
  all 800 manifest tokens; f4_version 1.1.
- Test: per token Delta = CLS(diffusion) - CLS(deterministic); OLS Delta ~ F4 with HC3 robust SE,
  per goal. Headline = contrast beta1(route) - beta1(precise), predicted > 0 (diffusion should
  help more under interaction-criticality when the goal does not already resolve it).

## Per-cell mean CLS (seed-averaged, n=800 each)

| head | goal | r0 (non-reactive) | r1 (reactive) |
|---|---|---|---|
| deterministic | route   | 0.8654 | 0.8488 |
| diffusion     | route   | 0.8642 | 0.8449 |
| deterministic | precise | 0.9603 | 0.9119 |
| diffusion     | precise | 0.9620 | 0.9156 |

Precise-goal conditioning adds ~0.10 CLS. Diffusion ~= deterministic within each cell.

## Moderation Delta ~ F4

r0 (non-reactive):

| condition | beta1 | HC3 SE | t | Spearman | mean Delta [95% CI] |
|---|---|---|---|---|---|
| route   | -0.0116 | 0.0059 | -1.97 | -0.102 | -0.0011 [-0.0053, +0.0030] |
| precise | -0.0052 | 0.0029 | -1.78 | -0.031 | +0.0017 [-0.0002, +0.0037] |
| contrast (route - precise) | -0.0064 | 0.0062 | -1.02 | | 1-sided p = 0.85 |

r1 (reactive):

| condition | beta1 | HC3 SE | t | Spearman | mean Delta [95% CI] |
|---|---|---|---|---|---|
| route   | -0.0050 | 0.0061 | -0.82 | -0.017 | -0.0040 [-0.0083, +0.0004] |
| precise | -0.0037 | 0.0029 | -1.26 | -0.083 | +0.0036 [+0.0017, +0.0057] |
| contrast (route - precise) | -0.0013 | 0.0068 | -0.20 | | 1-sided p = 0.58 |

## Conclusion

The pre-registered hypothesis fails in both reactive modes. Both headline contrasts are null and
of the wrong sign (predicted > 0; observed r0 -0.0064 p 0.85, r1 -0.0013 p 0.58). All four
per-condition F4 slopes are <= 0. Interaction-criticality does not moderate the
diffusion-vs-deterministic closed-loop gap. Robust across OLS and rank (Spearman) methods.

Consistent with the F4 validation result: F4 captures real geometric interaction-conflict but does
not track human-perceived decision ambiguity (failed human validation twice; ADR-027), and it also
does not moderate the policy gap.

## Power: the null is informative, not underpowered

Realized from the actual contrast HC3 SEs (r0 0.00624, r1 0.00683) against the pre-registered
effect benchmarks (POWER_ANALYSIS.md: subtle ~0.02, target = slope at F4=1 ~0.035, strong ~0.05):

| mode | contrast beta1 | 95% CI | realized MDE (80%) | power @0.02 | @0.035 | @0.05 |
|---|---|---|---|---|---|---|
| r0 (non-reactive) | -0.0064 | [-0.0186, +0.0059] | 0.0155 | 0.94 | ~1.00 | ~1.00 |
| r1 (reactive)     | -0.0013 | [-0.0147, +0.0120] | 0.0170 | 0.90 | ~1.00 | ~1.00 |

Both modes are well-powered: the realized MDE (~0.016 to 0.017) is below even the "subtle" 0.02
benchmark, with >99.9% power for the pre-registered target (0.035) and strong (0.05) effects, and
0.90 to 0.94 power even for a subtle 0.02 effect. The 95% CI upper bound on the contrast excludes
the entire range of hypothesized positive effects (0.02, 0.035, 0.05) in both modes. So the data
RULE OUT the hypothesized interaction-criticality moderation; they do not merely fail to detect it.
This is an informative null. (Realized via statistics.NormalDist on the committed result JSONs.)

## Secondary observations (not the pre-registered hypothesis)

- Diffusion shows a small but significant mean CLS edge in reactive-precise: +0.0037,
  95% CI [+0.0017, +0.0057] (excludes 0).
- Goal main effect on diffusion benefit: in the reactive mode diffusion does relatively worse under
  route than precise (contrast of mean Delta = -0.0076, 95% CI [-0.0124, -0.0028]). So goal
  precision interacts with diffusion's benefit at the mean level, but interaction-criticality (the
  F4 moderation slope) does not.

## Caveats

- Effects are small against a high CLS ceiling (0.85 to 0.96); both policies are strong, leaving
  limited headroom for a moderated gap to appear.
- F4 is the validated interaction-criticality axis, not human-perceived ambiguity.
- Per-token Delta is zero-inflated (identical CLS on easy scenarios) so Theil-Sen slopes are 0; the
  OLS mean trend and the Spearman rank both agree the moderation is null.

## Reproducibility

- Results: docs/frontier/results/r0_result.json, docs/frontier/results/r1_result.json.
- Pipeline: nuplan/slurm/eval_array_v4.sbatch (12 cells x 16 shards x reactive),
  nuplan/analysis/merge_eval.py (shard-union per seed + seed-average), and
  nuplan/analysis/analyze_moderation.py (OLS HC3 + contrast). F4: features/f4/f4_scores_v11.json.
- Eval completed via resume-safe gap-fill at 2-day walltime for the slow reactive-diffusion shards.
