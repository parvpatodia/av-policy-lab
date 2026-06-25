# F4: Per-Scenario Interaction-Multimodality Score

Status: spec verified 2026-06-10 (maker/checker research loop, all citations fetched).
Role in the study: pre-treatment MODERATOR of the head effect. Claim under test:
the diffusion-vs-deterministic gap (route-region condition) grows with F4.
This is moderation (effect modification), not mediation.

> AMENDMENT 2026-06-21 (ADR-027): F4 is reframed as an INTERACTION-CRITICALITY moderator, not "ambiguity" - human external validation failed twice; geometric validity (Signal A) holds. Formula unchanged. See F4_VALIDATION_RESULTS.md (Mac repo).

## 1. Definition

F4 in [0,1], model-free, frozen before any closed-loop comparison, identical
across all four experimental cells.

```
F4 = G_stop * (1 - (1 - S_branch) * (1 - 0.5 * S_lane) * (1 - S_inter))
```

Noisy-OR over three ambiguity sources. S_lane carries fixed weight 0.5 (lane
choice is a weaker decision than junction arms). All constants fixed and listed;
no learned weights. Subscores are always reported alongside F4.

### Units shim (mandatory)
Shard tensors are normalized: positions /120 (pos_scale_m), velocities /15
(vel_scale_mps); ego_future is RAW meters. The F4 extractor denormalizes to
meters first and asserts the scale factors against the shard `config` dict.
All thresholds below are in meters / m/s / seconds.

### S_branch + S_lane (lateral decision ambiguity) — map-API primary
Computed at closed-loop/scoring time from the nuPlan map API (shards lack lane
connectivity; a shard-geometry version exists only as a sanity check, see gates).
- v0 = ego speed at t=0; lookahead arc s* = clip(4.0 * max(v0, 3.0), 20, 60) m.
- BFS over lane successor edges from ego's current lane, accumulating arc
  length until >= s*, RESTRICTED to lanes whose roadblock is in the scenario's
  route roadblock sequence (the route corridor: what route-region conditioning
  actually pins down; the single-successor route_polyline is NOT used here
  because it resolves forks and would erase exactly the ambiguity we measure).
- B_R = number of distinct corridor exits reached (terminal lane groups merged
  when laterally adjacent within 4.5 m). S_branch = min(max(B_R - 1, 0), 3) / 3.
- N_par = number of on-route lanes in ego's current roadblock.
  S_lane = min(max(N_par - 1, 0), 2) / 2.

### S_inter (yield-or-go ambiguity from agents)
Inputs: agents (32,20,9) [x,y,sin,cos,vx,vy + one-hot(vehicle,ped,bicycle)],
agent_mask, crosswalks, route corridor path, v0.
- Ego nominal path: roll ego at constant v0 along the route corridor centerline
  for 5 s. NEVER ego_future: the expert future encodes how the expert RESOLVED
  the ambiguity (yielded vs went), which leaks the imitation label into the
  moderator and biases the moderation slope. The route is an input both heads
  condition on; it is symmetric across cells.
- Agent rollout: constant turn-rate (heading rate from sin/cos history),
  capped at 5 s. Constant-velocity over 8 s is indefensible (a 10 deg/s turner
  is tens of meters off the line by 8 s).
- If agent path crosses ego nominal path: predicted encroachment gap
  g_j = |t_ego(cross) - t_agent(cross)|. This is PrET / Time Advantage
  (Westhofen et al. 2023 sec. 5.2.13), NOT PET (a-posteriori). Band-pass:
  I_j = exp(-((g_j - 2.5)/1.5)^2).
- Else if agent's projection onto ego path is within 30 m of ego along-path:
  along-path gap time tau_j = arclength_gap / max(v0, 1.0);
  I_j = exp(-((tau_j - 1.5)/1.0)^2). (Along-path, not euclidean min-dist;
  cutoff at center + 3 sigma so the kernel decays before truncation.)
- Pedestrian override: agent with ped one-hot, speed < 0.5 m/s, within 3 m of a
  crosswalk polyline crossing the ego nominal path: I_j = max(I_j, 0.5).
- Combination: top-3 noisy-OR, S_inter = 1 - prod over 3 largest I_j of (1-I_j).
  (Full noisy-OR saturates under platoons/groups; correlated I_j violate the
  independence read.)
- WHY band-pass, not monotone criticality kernel: ambiguity peaks at
  intermediate gaps where yield and go are both sane; extreme gaps force one
  action. Deliberate, flagged departure from criticality-metric usage.

### G_stop (forced-stop suppressor)
G_stop = 0.25 if (RED connector polyline within 20 m ahead on the corridor,
traffic_lights encoding: GREEN=0 YELLOW=1 RED=2 UNKNOWN=3, -1 none) AND
v0 < 1.0 m/s; else 1.0. Known limitation (accepted): a red-light approach at
speed is not suppressed; covered by the sensitivity grid.

### Exclusions
Scenarios with route_mask all False (route resolution failure) are EXCLUDED
from F4 analysis and counted in the report, never silently scored 0.

## 2. Circularity guard
Primary F4 never touches either policy head or the expert future. The diffusion
head's DDIM dispersion D_K (K=16, route-region cond.; APD = mean pairwise L2
(Yuan & Kitani 2020, DLow) and FSD at the endpoint (Yuan & Kitani ICLR 2020))
is a SECONDARY manipulation check only: expect Spearman(F4, APD) > 0 and
APD_precise < APD_route (paired Wilcoxon) — the direct test of the Phase-3d
mechanism. Disagreement is diagnostic, never grounds to swap moderators.

## 3. Validation gates (all pass before unblinding any paired difference)
1. Per-scenario_type medians: pre-registered high list (starting_left_turn,
   starting_right_turn, starting_unprotected_cross_turn,
   waiting_for_pedestrian_to_cross, traversing_pickup_dropoff) beats low list
   (stationary, following_lane_with_lead, stopping_with_lead) in >= 80% of
   pairwise comparisons.
2. S_branch ~ 0 on lane-follow types; junction fraction consistent with tags.
3. Shard-geometry S_branch vs map-API S_branch on 200 scenarios: report
   Spearman rho as a sanity number (API version is primary regardless).
4. Manipulation check (sec. 2).
5. Non-degeneracy: F4 IQR > 0.1; if upper-tertile mass < 15%, stratified
   closed-loop sampling over F4 deciles (pre-registered).
6. Sensitivity grid: every constant +/-50%, report Spearman rank stability of
   F4 across the grid.

## 4. Statistical analysis plan
- Unit: scenario i, paired across heads. Outcome Delta_i = CLS_i(diffusion) -
  CLS_i(deterministic), route-region condition.
- Primary: OLS Delta on F4 WITH scenario_type fixed effects (F4 varies by type
  by design; FE isolates within-type ambiguity from type difficulty),
  cluster-robust SEs by scenario_type using wild cluster bootstrap (few
  clusters; plain HC3 and naive cluster SEs are both wrong here).
  One-sided H1: beta1 > 0; two-sided p also reported.
- Robustness: Spearman(Delta, F4), Theil-Sen slope.
- Mixed form (same estimand): CLS ~ head x F4 + scenario_type + (1|scenario).
- Secondary family (Holm-Bonferroni, 5): (i) precise-condition slope via TOST
  equivalence (a non-significant test cannot establish ~0); (ii) three-way
  head x goal x F4 (sharpest statement, likely underpowered on mini, stated);
  (iii) S_branch-only; (iv) S_inter-only; (v) reactive-agent replication.
- Tertile plots: descriptive only (difference-of-significances fallacy).
- Power caveat: interaction effects need ~4x the N of main effects; mini-scale
  results are reported with CIs, not just p-values.

## 5. Data prerequisites (DONE 2026-06-10)
Shards carry scenario_token, scenario_type, log_name, iteration per sample
(commit 7a9c0cd); without them gates 1-3 and type clustering are impossible.
Agent type one-hot already present (dims 6:9). traffic_lights encoding and
10 Hz / 2 s agent history confirmed from source.

## 7. v1.1 revision (2026-06-11, ADR-016) — gate-driven, pre-unblinding

The full gate run over 5,604 mini scenarios FAILED gate 1 (17% vs the 80%
requirement) and exposed three defects. No head comparison existed at
revision time; the moderator firewall holds.

What the data showed:
1. Four of five pre-registered high types (starting_left_turn,
   starting_right_turn, starting_unprotected_cross_turn,
   waiting_for_pedestrian_to_cross) DO NOT EXIST in nuPlan mini. Gate 1 was
   unevaluable as registered.
2. S_lane saturated: 69% of scenarios at s_lane = 1.0 (Vegas roadblocks carry
   4-7 parallel lanes), flooring median F4 at 0.50 and destroying contrast.
3. The headway (non-crossing) branch of S_inter rewarded plain car-following:
   following_lane_with_lead (registered LOW) scored median 0.668, above every
   evaluable HIGH type. Mid-headway following is longitudinal regulation, not
   a discrete decision; scoring it was a design error. PrET crossings behaved
   correctly.

v1.1 definition:
    F4 = G_stop * (1 - (1 - S_branch) * (1 - S_inter))
- S_inter: PrET band-pass over path-crossing agents + stationary-pedestrian
  override ONLY. The headway branch is removed.
- S_lane: computed and reported as a descriptive covariate, EXCLUDED from the
  scalar. Revisit only with adjacency-based reachability if a lane-decision
  hypothesis becomes load-bearing.
- All other constants unchanged.

Gate-1 lists re-registered against the types that exist in mini, on the same
a-priori semantics (lateral maneuvers and unstructured zones = high; static /
regulated-stop / lead-following = low), chosen before any v1.1 score was seen:
- HIGH: traversing_pickup_dropoff, high_lateral_acceleration,
  changing_lane_to_left (n=2, reported but excluded from the gate),
  changing_lane_to_right (n=2, idem), traversing_traffic_light_intersection
- LOW: stationary, stopping_with_lead, following_lane_without_lead,
  stationary_at_traffic_light_with_lead, stopping_at_traffic_light_without_lead
Known tension recorded honestly: the original inclusion of
waiting_for_pedestrian_to_cross as HIGH contradicted the band-pass logic
(an active forced yield is unimodal); it is moot on mini but the lists for
full nuPlan must resolve it.

## 6. Bibliography (all fetched and verified)
- Yuan & Kitani, DLow, ECCV 2020, arXiv:2003.08386 — APD definition.
- Yuan & Kitani, DPP forecasting, ICLR 2020, arXiv:1907.04967 — ASD/FSD.
- Chai et al., MultiPath, CoRL 2019, arXiv:1910.05449 — anchor modes.
- Ettinger et al., WOMD, ICCV 2021, arXiv:2104.10133 — interactive-pair
  predicates (crossed paths with time gap).
- Salzmann et al., Trajectron++, ECCV 2020, arXiv:2001.03093 — KDE-NLL.
- Tolstaya et al., arXiv:2104.09959 — interactivity as mutual information
  (model-based; precedent, deliberately not adopted as primary).
- Montali et al., WOSAC, NeurIPS D&B 2023, arXiv:2305.12032 — sim-agent
  realism metrics (distinct construct from F4).
- Florence et al., Implicit BC, CoRL 2021, arXiv:2109.00137 — deterministic
  regressors fail on multi-valued demonstrations.
- Shafiullah et al., BeT, NeurIPS 2022, arXiv:2206.11251 — mode capture.
- Chi et al., Diffusion Policy, RSS 2023, arXiv:2303.04137 — multimodal
  action distributions motivation.
- Deo et al., PGP, CoRL 2021, arXiv:2106.15004 — lateral/longitudinal
  uncertainty decomposition (grounding for S_branch; no published standalone
  per-scenario branching score found — S_branch is presented as new, with the
  search protocol documented).
- Westhofen et al., arXiv:2108.02403 — PET vs PrET/TA definitions (5.2.12/13).
- Treiber et al., Phys. Rev. E 2000, arXiv:cond-mat/0002177 — IDM.
- Dauner et al., PDM, CoRL 2023, arXiv:2306.07962 — rule-based beats learned
  closed-loop on nuPlan; Val14.
- Hagedorn et al., arXiv:2510.14677 — IDM agents inflate scores, rankings
  shift under learned reactive agents.
- Hallgarten et al., interPlan, arXiv:2404.07569 — interactive long-tail
  separates planners.
