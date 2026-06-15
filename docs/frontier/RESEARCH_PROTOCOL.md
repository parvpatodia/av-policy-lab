# Research Protocol (Pre-Registration)

Frozen before unblinding any closed-loop result. Amendments are dated and
appended, never silently edited. This is the credibility firewall: the
analysis is fixed before the numbers exist.

## 1. Question and hypotheses

Does the advantage of a generative (diffusion) trajectory policy over a
capacity-matched deterministic one on nuPlan closed-loop depend on (a) how
much the goal conditioning already resolves the scene, and (b) the realism of
the background agents?

- H1 (goal x head): under route-region conditioning, the diffusion-minus-
  deterministic CLS gap increases with per-scenario ambiguity F4 (slope
  beta1_route > 0). Under precise-point conditioning the slope is ~0
  (the goal resolves the ambiguity, so generativity buys nothing).
- H2 (agent realism): the gap and its F4-slope are larger under learned
  reactive agents (SMART) than under IDM. Tested when the Bosch SMART
  integration is available (ADR-020); axes 1-2 do not depend on it.

Both directions are publishable. A null (beta1_route ~ 0) is the finding that
the diffusion advantage on nuPlan is not explained by measured ambiguity.

## 2. Design (2x2, frozen)

Four capacity-matched models, identical except two axes:
  goal in {route-region (corridor sweep), precise (near+far expert points)}
  head in {deterministic regressor, diffusion (x0, DDIM)}
Shared encoder, shared trunk, <10% parameter gap (tests enforce). 3 seeds
{0,1,2} per cell. Trained on f0_v3 (corridor route, recovery perturbation
prob 0.5, parity histories; ADR-017). Identical optimizer/schedule/epochs;
EMA on both heads; checkpoint selected by closed-loop probe, not open-loop
(ADR-018, see 5).

## 3. Moderator F4 (frozen, ADR-013/016, v1.1)

Model-free, computed only on UNPERTURBED extractions, frozen before eval:
  F4 = G_stop * (1 - (1 - S_branch) * (1 - S_inter))
  S_branch: corridor lane-graph branching (map API).
  S_inter : PrET band-pass over path-crossing agents + stationary-pedestrian
            override (no headway term; dropped in v1.1 after it rewarded
            car-following).
  G_stop  : red-light standstill suppressor.
F4 never reads either head's output (no circularity) nor ego_future (no label
leak; ego nominal path is the route corridor). Secondary model-based check:
DDIM sample dispersion (APD/FSD) should correlate with F4 under route-region
and collapse under precise; manipulation check only, never a moderator.

## 4. Evaluation (frozen)

- Set: manifest frozen by freeze_manifest.py (SHA256 + git commit recorded),
  4-band F4-stratified (zero/low/med/high, equal allocation). Balanced-X is
  retained for BAND COVERAGE, not slope precision: on the real F4 distribution
  the design-effect over random sampling is only 1.10x (~4%, the natural
  distribution is already high-variance bimodal), but balancing guarantees
  >=125 high- and med-band scenarios that random N=500 would undersample
  (~76 high, ~36 med), stabilizing the per-band CIs and the robustness checks
  (ADR-019 restated by ADR-023; full analysis in POWER_ANALYSIS.md).
- Power and N (ADR-023, supersedes the n=1000 line below). The HEADLINE is the
  cross-condition contrast beta1(route)-beta1(precise), not a single slope. The
  earlier claim "n=1000 gives >=0.83 power for beta1>=0.05 at sigma<=0.20" is
  correct for a single condition's slope (verified: 0.877) but gives only 0.63
  power for the contrast at worst-case residual correlation rho_cond=0. Because
  the heads share everything but the output head, sigma_Delta is plausibly
  <=0.10, where even N=500 powers the contrast at 0.88; the binding unknown is
  sigma_Delta. DECISION: freeze the manifest at N=800 (the safe superset, never
  re-freeze), then run a pre-registered internal-pilot variance re-estimation
  (Wittes-Brittain): evaluate a stratified 200-token pilot across all 4 cells,
  measure realized sigma_Delta and rho_cond with power_analysis.reestimate_from_pilot,
  and complete tokens up to the re-estimated N (capped at 800), stopping early
  if the 0.035-contrast is already powered. The decision uses only the nuisance
  variance, never the effect estimate, so alpha is preserved.
- Simulator: nuPlan two_stage_controller (LQR + kinematic bicycle), the
  official CLS setup. CLS-NR (box replay) AND CLS-R (IDM) both run; SMART
  added later. Same manifest across all cells and agent modes (paired).
- Trajectory post-processing: IDENTICAL for both heads (path-tangent heading,
  speed from segment length). No head-specific feasibility shaping.
- Diffusion selector (PRE-REGISTERED): medoid of K=8 DDIM samples (the sample
  minimizing mean pairwise xy L2 to the set), eta=0, fixed val seed. Rationale:
  the medoid is the distribution's central mode, the honest "single best guess"
  of a multimodal predictor, and does not peek at the route-progress oracle.
  A route-progress-best-of-K selector is a SEPARATE pre-registered secondary
  variant; both reported, neither cherry-picked. The comparison is explicitly
  "deterministic regressor vs generative sampler + fixed selector," which is
  the deployable question.

## 5. Checkpoint selection (frozen, ADR-018)

Final checkpoint per cell chosen by a 50-scenario closed-loop probe (CLS),
NOT open-loop minADE, because open-loop and closed-loop anti-correlate on
nuPlan (Dauner et al. 2023). The probe set is disjoint from the eval manifest.

## 6. Statistics (frozen)

Per scenario i, seed-averaged CLS per cell, Delta_i = CLS(diff) - CLS(det)
within a goal condition.
- Primary: OLS Delta ~ 1 + F4 with scenario_type fixed effects, wild-cluster-
  bootstrap SE by scenario_type (few clusters). One-sided H1: beta1_route > 0
  (two-sided p also reported).
- Robustness: Spearman(Delta, F4), Theil-Sen slope.
- Cross-condition: beta1_route - beta1_precise (the sharpest H1 statement);
  precise-slope ~0 established by TOST equivalence, not a non-significant test.
- Secondary family (Holm-Bonferroni): S_branch-only and S_inter-only
  moderation; SMART replication.
- Tertile CLS plots are descriptive only (no difference-of-significances
  inference).
- Marginal mean Delta is NOT population-representative under balanced-X; a
  band-reweighted marginal is reported separately if needed.

## 7. Baselines (context, ADR / finding 6)

IDM and log-future-replay (ceiling) reported alongside all cells. PDM-Closed
added before submission (tuplan_garage). These contextualize absolute CLS;
they are not part of the moderation test.

## 8. Validation gates that must pass before unblinding

1. Feature parity offline-vs-serving (60/60, passing).
2. F4 face validity: high scenario_types > low in >=80% of pairwise type-median
   comparisons (v1.1 passed 100% on the re-registered lists; re-run on the
   enriched junction-type inventory).
3. Manifest frozen with hash + commit.
4. Non-degenerate policy: closed-loop progress ratio sanity (the moving smoke
   showed 0.93-0.95, not degenerate).

## 9. Threats to validity (stated, not hidden)

- nuPlan mini scale, Vegas-skewed geography.
- Precise-goal conditioning uses the log future, which diverges from sim state
  after the first deviation (inherent; reported).
- CLS rewards route-following, so branch-level ambiguity (S_branch) is partly
  invisible to the outcome; H1 weight is on S_inter (interaction timing), which
  CLS can express and which is also where reactive agents matter.
- Selector asymmetry (head + selector vs head); addressed by reporting both
  selectors and framing as the deployable comparison.
