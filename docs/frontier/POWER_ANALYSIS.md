# Power analysis: sizing the closed-loop moderation eval

Status: pre-registration input, written 2026-06-14, before the manifest freeze.
Reproduce: `python nuplan/analysis/power_analysis.py --f4-scores f4_scores_v11.json`

## Why this exists

The eval costs ~7.5 min/scenario. At 4 cells x 3 seeds plus 3 baselines that is
~15 GPU-runs per scenario, so N is the largest compute lever in the project:
N=500 is ~940 GPU-hours, N=800 is ~1500. Before freezing the manifest we have
to know the minimum detectable effect (MDE) at a given N. If the MDE is larger
than any plausible true effect, the eval is dead on arrival and we would learn
that only after burning the cluster.

## The estimand

Per goal condition, seed-averaged, per scenario i:

    Delta_i = CLS_diff,i - CLS_det,i = beta0 + beta1 * F4_i + eps_i

`beta1` is the moderation slope: does the diffusion head's closed-loop
advantage grow with interaction-multimodality F4. OLS slope variance is
`sigma_eps^2 / (n * Var_n(F4))`, so

    MDE(beta1) = (z_alpha + z_power) * sigma_Delta / sqrt(n * Var_n(F4))   [1-sided]

The **headline** is not either slope but the cross-condition contrast
`beta1(route) - beta1(precise)` (diffusion should help under ambiguity only
when the goal does not already resolve it). Because both conditions are scored
on the same tokens, the contrast is the slope of the per-token difference
`C_i = Delta_route_i - Delta_precise_i` on F4, with residual SD

    sigma_C = sigma_Delta * sqrt(2 * (1 - rho_cond))

where `rho_cond` is the correlation between the two conditions' residuals (a
scenario hard for one is usually hard for the other). `rho_cond > 0` shrinks
`sigma_C` and helps; `rho_cond = 0` (independence) is the worst case.

## Real F4 distribution (5,604 scored scenarios)

| band | count | share |
|---|---|---|
| zero (F4=0) | 3352 | 59.8% |
| low (0, 1/3] | 997 | 17.8% |
| med (1/3, 2/3] | 402 | 7.2% |
| high (>2/3] | 853 | 15.2% |

`Var(F4)`: balanced-X design 0.126, natural 0.114, **design-effect 1.10x**.

> Honest correction to ADR-018. The balanced-X manifest was justified as
> "maximizes slope precision." Quantitatively that gain is ~4% here, because
> the natural F4 distribution is already a high-variance bimodal mass (60%
> zeros + 15% high). The real value of balancing is different and still valid:
> it guarantees >=125 high-band and >=125 med-band scenarios at N=500, whereas
> a random N=500 draw yields only ~76 high and ~36 med, which would starve the
> per-band CIs and the Spearman / Theil-Sen robustness checks. The ADR text
> should be restated to claim band-coverage, not slope precision.

## MDE table (power 0.80, alpha 0.05, one-sided)

Per-condition MDE(beta1):

| sigma_Delta | N=200 | N=300 | N=500 | N=800 |
|---|---|---|---|---|
| 0.05 | 0.025 | 0.020 | 0.016 | 0.012 |
| 0.10 | 0.050 | 0.041 | 0.031 | 0.025 |
| 0.15 | 0.074 | 0.061 | 0.047 | 0.037 |

Headline contrast MDE at sigma_Delta=0.10, by residual correlation:

| rho_cond | N=200 | N=300 | N=500 | N=800 |
|---|---|---|---|---|
| 0.0 | 0.070 | 0.057 | 0.044 | 0.035 |
| 0.3 | 0.059 | 0.048 | 0.037 | 0.029 |
| 0.6 | 0.044 | 0.036 | 0.028 | 0.022 |

Reference: a "moderate but real" effect is a 3.5-point CLS swing from F4=0 to
F4=1, i.e. beta1 ~ 0.035; a "strong" effect ~ 0.05; a "subtle" effect ~ 0.02.

## Two corrections that change the numbers

1. **Bounded CLS.** CLS lives in [0,1] with mass near 1; clipping attenuates
   slopes for high-skill scenes, which the Gaussian formula ignores. Simulation
   on the real balanced pool: bounded MDE is 4-7% larger than the Gaussian MDE
   (ratio 1.04 at sigma=0.05, 1.07 at sigma=0.15). Budget a ~7% margin.

2. **The contrast had no inferential SE.** Both `analyze_moderation.run` and
   `results_table.run` reported `beta1(route)-beta1(precise)` as a bare point
   estimate with no SE or p-value, so the experiment's main hypothesis had no
   significance test. Fixed: `moderation_contrast()` regresses the per-token
   difference C_i on F4, giving one HC3 SE that correctly absorbs the
   within-token correlation. This is now the headline test (the back-compat
   `route_minus_precise_beta1` key still equals this slope).

## Recommendation

The binding constraint is the contrast, whose MDE at N=500 ranges from 0.044
(worst case rho_cond=0) to 0.028 (favorable), and the dominant unknown is
sigma_Delta, which cannot be known until the eval runs. Two design choices
follow.

1. **Freeze the manifest at N=800** (200 per band; feasible, med band has 402
   available). This is the safe superset: at N=800 the contrast MDE covers the
   moderate-effect regime (0.035 worst case) and we never have to re-freeze
   (which would break pre-registration) if variance turns out high.

2. **Do not run all 800 blind. Pre-register an internal-pilot variance
   re-estimation (Wittes & Brittain 1990).** Evaluate the first stratified
   200-token pilot (50 per band) across all 4 cells. Run
   `reestimate_from_pilot()` on the pilot deltas to measure the realized
   sigma_Delta and rho_cond, then size the remaining run for a target contrast
   of 0.035. Decision rule, frozen before unblinding:
   - if the pilot's realized variance already powers the 0.035 contrast at
     n<=200, stop;
   - else complete tokens up to the re-estimated N, capped at the frozen 800.
   The decision uses only the nuisance variance/correlation, never the effect
   estimate, so type-I error is preserved (the slight inflation from variance
   re-estimation is negligible at a 200-token pilot).

Worked example (synthetic pilot, similar heads sigma_head~0.05, rho_cond~0.52):
re-estimation recovers sigma_route=0.051, rho_cond=0.52, and reports
required_n_contrast=95 -> the 200-pilot alone powers the contrast, stop early,
saving ~1100 GPU-hours versus the full 800. If the heads instead diverge
(sigma_Delta~0.15), the rule completes to 800. Either way we never over- or
under-spend on a guessed variance.

## Verification of the existing pre-registered claim

RESEARCH_PROTOCOL.md section 4 states: "n=1000 gives >=0.83 power for
beta1>=0.05 at sigma<=0.20." Checked with this module:

- per-condition slope, n=1000, beta1=0.05, sigma=0.20: power **0.877** -> the
  existing claim is correct *for a single condition's slope*.
- the **contrast** (the actual headline), same configuration, worst case
  rho_cond=0: power **0.633** -> underpowered. It only reaches 0.88 if
  rho_cond >= ~0.5.

So the pre-registered power statement is right for the wrong estimand: it sizes
the per-condition slope, not the contrast the experiment tests. The protocol's
power line must be restated against the contrast. The saving grace is that the
protocol's assumed sigma<=0.20 is conservative; the two heads share data,
optimizer, schedule and architecture except the output head, so sigma_Delta is
more plausibly <=0.10, where even N=500 powers the contrast at 0.88 (rho=0).
Which regime we are in is exactly what the internal pilot measures.

## Honest caveats

- Every single-number recommendation is conditional on sigma_Delta, which is
  unknown pre-eval. The internal-pilot exists precisely so we stop guessing it.
- Power is for the pre-registered one-sided direction (beta1>0). A
  non-significant result is NOT evidence of no effect; pair it with the TOST
  equivalence test already in RESEARCH_PROTOCOL.md.
- These numbers assume a successful seed-averaged CLS; a failed seed raises
  sigma_Delta and the pilot will catch it.
