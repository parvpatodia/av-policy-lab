# Deployable Goal-Conditioned Imitation Learning for Closed-Loop Driving

**Research design document — av-policy-lab**
Author: Parv Patodia (MS AI, Northeastern SVL) · Advisor: Prof. Nadim
Status: living document · Last updated: 2026-05-28

---

## 1. Research question

> Can a learned driving policy match a hand-engineered rule-based planner (IDM) in
> closed-loop simulation **without access to privileged expert data at inference**,
> using only the information a real autonomous vehicle has — its kinematic state and
> an HD-map route?

This is a deployability question, not a benchmark-chasing one. The contribution is
not "lowest L2 on nuPlan." It is an **ablation that isolates *what kind of goal
information* a closed-loop imitation policy actually needs**, and a demonstration
that the sufficient signal is available from the map alone.

---

## 2. Motivation

Open-loop imitation learning (behavior cloning) is trivially accurate: predicting the
next 1.6 s from a ground-truth state gives 0.058 m ADE. In **closed-loop**, the same
policy reaches 49 m L2 — an 850× collapse. This is covariate shift: the policy is
never trained on the off-distribution states its own errors produce (Ross et al. 2011).

The conventional fixes — more data (DAgger), richer perception (BEV raster), world
models (MILE) — are expensive and, in our Phase 2 results, **ineffective**: all three
plateau at ~49.5 m (Table 2). This motivates a different question: maybe the
bottleneck is not capacity or data, but the *absence of a spatial reference*. A 6-dim
kinematic state looks identical whether the ego is on-road or 50 m off-track. The
policy literally cannot perceive that it is in trouble.

---

## 3. Method — the goal-conditioning ablation ladder

Each planner changes exactly one variable from the previous, isolating the effect of
the **goal source**. Architecture (8→256→256→256→48 MLP) and training data are held
fixed across the goal-conditioned variants; only the goal at inference changes.

| Planner | Goal source | Available at deploy? | Tests the hypothesis |
|---|---|---|---|
| BC | none (6-dim state) | yes | does kinematics alone suffice? |
| IDM | rule-based reference path | yes | rule-based upper baseline |
| **GoalBC** | expert T+8 waypoint (DB) | **no (oracle)** | does *any* global goal break the plateau? |
| MapBC | nearest lane centerline (live query) | yes | is a *local* map query sufficient? |
| RouteMapBC | pre-computed 200 m route, fixed 8 m look-ahead | yes | is a *global* route sufficient? |
| TrainedRouteBC | route goal at train + inference (8 m) | yes | does retraining fix the goal-source gap? |
| **SpeedAdaptiveRouteMapBC** | route, look-ahead = speed × 0.8 s | **yes** | does matching the *temporal* goal horizon fix scale? |
| RoadblockRouteMapBC (planned) | route guided by `route_roadblock_ids` | yes | does intended-route info fix intersection turns? |

**Why this is a clean ablation:** GoalBC and the deployable variants share the *same
weights* (`goal_bc.pt`). The only thing that changes is where the 2-D goal vector
comes from. Any L2 difference is therefore attributable purely to goal-source
fidelity, not to model capacity, training set, or optimization.

---

## 4. Metrics

| Metric | What it measures | Status |
|---|---|---|
| Open-loop ADE/FDE | trajectory accuracy from GT states | done |
| Closed-loop L2 (ego-vs-expert) | drift / covariate-shift severity | done |
| **PDM-Score** (Dauner et al. 2023) | driving *quality*: collision, drivable-area, progress, comfort, TTC | **pipeline ready (`pdm_score.py`), pending re-run** |

L2 is a proxy. A policy can have low L2 and still drive uncomfortably or clip a curb.
PDM-Score is nuPlan's official closed-loop metric and is required for any
publication-grade claim. `eval_production.py` now enables the
`simulation_closed_loop_nonreactive_agents` metric set; `pdm_score.py` parses and
aggregates the seven components into the weighted-product composite.

---

## 5. Experimental protocol

- **Dataset:** nuPlan mini, 64 logs, 100 Hz ego_pose (verified — see §7 below).
- **Simulator:** nuPlan closed-loop, `perfect_tracking_controller`, `box_observation`.
- **Scenario sampling:** `all_scenarios` filter, `shuffle=true`, seeded, N=30 across
  all 64 logs (not 3 from one log — the prior weakness).
- **Statistics:** every planner-vs-planner claim runs through `statistical_analysis.py`:
  exact binomial on win rate, paired Wilcoxon on per-scenario L2, bootstrap 95 % CIs
  (10 000 resamples, seed 42), trimmed means, and tail-mass attribution.
- **Unit tests:** `tests/test_planner_geometry.py` (43 cases) pins the coordinate
  transforms, route construction, arc-length goal walk, and speed-adaptive look-ahead.
- **Pipeline invariants:** `verify_pipeline.py` (6 checks) guards DB rate, goal-timing
  scale, checkpoint presence, and DT consistency before any experiment runs.

---

## 6. Results to date (honest)

**Phase 3a — GoalBC (oracle):** expert T+8 goal → **1.820 m** (96.3 % below BC). A
2-D goal completely breaks the 49.5 m plateau. *The MLP was never the bottleneck.*
**Binding insight: architecture is not the bottleneck — goal representation is.**

**Phase 3b — MapBC:** local map query → 56.3 m (worse than BC). Point queries fail
once the ego drifts off-road. **A local reference is insufficient.**

**Phase 3c — RouteMapBC:** global pre-computed route, fixed 8 m → 32.1 m. Fixes drift
but the 8 m look-ahead is the wrong *scale*: at 100 Hz, the T+8 training goal is ~0.35 m
(speed × 0.08 s), and GoalBC's inference goal is speed × 0.8 s ≈ 3.5 m — never 8 m.

**Phase 3c'' — SpeedAdaptiveRouteMapBC:** look-ahead = speed × 0.8 s matches GoalBC's
temporal horizon at every speed. 3-scenario: **13.7 m** (57 % below RouteMapBC).

**30-scenario diverse eval — the honest headline:**

| Planner | Mean | Median | Trim-4 mean | vs IDM |
|---|---|---|---|---|
| SpeedAdaptiveRouteMapBC | 18.19 m | **7.50 m** | **7.81 m** | tied (Wilcoxon p=0.76) |
| IDM | 13.97 m | 8.50 m | 9.08 m | — |

SpeedAdaptive is **statistically tied with IDM** (binomial p=0.585, Wilcoxon p=0.761,
median-diff 95 % CI [−10.3, +6.4] includes 0). **We do not claim it beats IDM.** Its
entire deficit is concentrated in **4 of 30 scenarios that carry 63 % of total L2
mass** — all intersection-turn scenarios where the centerline route goes straight while
the expert turns. Remove those 4 and the trimmed mean (7.81 m) edges below IDM (9.08 m).

**PDM-Score (driving quality, `pdm_score.py`, 30 scenarios):**

| Planner | PDM-Score | no-collision | drivable | TTC | comfort |
|---|---|---|---|---|---|
| IDM | **0.656** | 0.850 | 0.833 | 0.767 | 0.633 |
| SpeedAdaptiveRouteMapBC | 0.526 | 0.667 | 0.767 | 0.600 | **0.833** |

On the official metric IDM wins overall, **but the learned policy is measurably more
comfortable** (0.833 vs 0.633) while trading away safety (collisions, TTC, drivable-area).
The safety deficit co-locates with the L2 tail — the 4 intersection failures appear as
collisions and off-road excursions, not discomfort. This is the genuine finding: a
deploy-time-only learned policy drives more smoothly than a tuned rule-based planner,
with a localized intersection-safety gap.

**The defensible contribution:** a policy with *no expert data at inference* matches a
tuned rule-based planner on L2, **exceeds it on comfort**, and its single remaining
failure mode (intersection safety) is identified, localized, and has a concrete fix
(Phase 3c''' — `route_roadblock_ids` confirmed populated on all 30/30 scenarios).

---

## 7. Threats to validity

1. **Small n.** 30 scenarios; CIs are wide. *Mitigation:* scale to 100+ scenarios for
   the final result; report CIs always.
2. **L2 ≠ driving quality.** *Mitigation:* PDM-Score (§4), pipeline ready.
3. **GoalBC oracle uses one log's expert DB.** Cross-log evaluation corrupts it
   (documented in `eval_production.py`); the 1.820 m oracle is single-log only and is
   labeled as such, never pooled with the 30-scenario deployable results.
4. **100 Hz vs 10 Hz confusion** caused two earlier wrong diagnoses. *Mitigation:*
   `verify_pipeline.py` Check 1–4 now pins the DB rate and goal-timing scale; the
   `DT=0.1` waypoint-stamp quirk is documented and shown benign (tracker uses spatial
   position, evidenced by GoalBC's 1.82 m).
5. **Route built from expert trajectory proxy at train time.** The deployable variants
   use the HD-map route at inference, but the training goal still derives from the
   expert path. *Mitigation:* TrainedRouteBC tested map-route goals at train time
   (negative result, explained); future work uses map centerline at both.

---

## 8. Contribution statement (paper-level)

> We show, via a controlled goal-source ablation on nuPlan closed-loop simulation,
> that the closed-loop covariate-shift collapse of behavior cloning is caused by the
> absence of a *global spatial goal*, not by model capacity or training-data volume.
> A 2-D goal vector reduces closed-loop L2 by 96 %. We further show the goal need not
> be privileged expert data: an HD-map route, queried at a *speed-matched temporal
> horizon*, brings a learned policy to statistical parity with IDM using only
> deploy-time information, with the residual failure mode localized to intersection
> topology and addressable via the map's intended-route annotation.

---

## 8b. Phase 3c''' result — the pivot (added 2026-05-28)

RoadblockRouteMapBC used `route_roadblock_ids` to pick the correct junction branch.
It **changed the route on 28/30 scenarios** (mechanism confirmed firing) and produced
clean fixes (scen_0000 55.7→4.1m) — but is **statistically tied with its parent**
(Wilcoxon p=0.808): 15 scenarios improved, 13 regressed. Correcting the route direction
fixes gentle turns and breaks sharp ones, because the deterministic MLP — trained on
near-straight expert goals — cannot execute a turn-goal that is out of its training
distribution. **The bottleneck has moved from goal representation (solved across
3a–3c''') to policy execution of multi-modal junction trajectories.** This is now the
empirically-grounded motivation for Phase 3d, and it strengthens the paper: we can show
*with an ablation* that goal correctness and goal executability are distinct, separable
failure axes — most prior covariate-shift work conflates them.

## 9. Roadmap to publication

- [x] **3c''' RoadblockRouteMapBC** — DONE. Route direction now correct (28/30 changed),
      but tied with parent: isolated that policy *execution* of turns is the residual
      bottleneck, not goal representation. (Negative-leaning, high-information result.)
- [ ] **Phase 3d Diffusion Policy** — now the top priority: multi-modal action head to
      represent the turn/straight bimodality the MLP averages away.
- [ ] **3c' redone correctly** — retrain with speed-adaptive *route* goals at training
      time so the policy actually sees turn-goals (the original 3c' used a wrong fixed 8m).
- [ ] **PDM-Score** — re-run `eval_production.py` with metric set enabled; report the
      7 components + composite for all planners.
- [ ] **Scale to 100+ scenarios** — tighten CIs; stratify by scenario type.
- [ ] **Phase 3d Diffusion Policy** — multi-modal goal-conditioned head for
      intersection decisions (lane-change vs straight). Compare to deterministic MLP.
- [ ] **Failure taxonomy** — `failure_analysis.py` output → categorize all >20 m
      scenarios by mechanism (intersection, merge, stop-and-go).
- [ ] **Write-up** — methods + ablation table + PDM-Score + statistical appendix.

---

## References

- Ross et al. 2011. DAgger. AISTATS.
- Treiber et al. 2000. Intelligent Driver Model. Phys. Rev. E 62(2).
- Caesar et al. 2021. nuPlan. arXiv:2106.11810.
- Dauner et al. 2023. PDM / closed-loop planning. arXiv:2306.07962.
- Chi et al. 2023. Diffusion Policy. RSS.
