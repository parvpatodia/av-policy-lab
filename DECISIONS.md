# Architecture Decision Record (ADR) Log — av-policy-lab

> Each entry: Context (why a decision was needed) -> Decision -> Rationale ->
> Consequences -> Status. Append-only; supersede rather than delete so the
> reasoning history is preserved.

---

## ADR-001 — Simulator: nuPlan over CARLA
**Status:** Accepted
**Context:** Need a closed-loop driving benchmark with real data and a trusted metric.
**Decision:** Use nuPlan.
**Rationale:** Real Motional driving logs, a published closed-loop metric (PDM-Score,
Dauner et al. 2023), and built-in reactive agents. CARLA would require building the
entire evaluation stack from scratch.
**Consequences:** Inherit nuPlan's IDM-agent assumption — see ADR-009.

## ADR-002 — Minimal kinematic state for the ablation
**Status:** Accepted (deliberate, with known limits)
**Context:** To attribute closed-loop failure to a single cause, the input must be controlled.
**Decision:** Start from a 6-dim kinematic state + low-dim goal; vary ONLY the goal source.
**Rationale:** A clean ablation needs everything but one variable held fixed.
**Consequences:** The policy is scene-blind (no agents/map geometry) — correct for the
ablation, but the ceiling for performance. The frontier reframe (ADR-008) lifts this.

## ADR-003 — The goal-conditioning ablation ladder
**Status:** Accepted (complete)
**Context:** Is closed-loop covariate-shift collapse caused by architecture, data, or representation?
**Decision:** Hold the MLP fixed; sweep the goal source (none -> oracle -> local map -> global route -> temporally-scaled route -> route-branch -> near+far).
**Rationale:** Each rung isolates one factor, so every delta is attributable.
**Consequences:** Produced the core finding — *goal representation, not architecture, is the bottleneck* — and the Phase-3d null (ADR-007).

## ADR-004 — OOP planner hierarchy (SOLID)
**Status:** Accepted
**Context:** Eight+ planner variants share most logic but differ in one mechanism each.
**Decision:** A single `AbstractPlanner`-rooted hierarchy; `RouteMapBCPlanner` defines
reusable hooks (`_build_route`, `_get_route_goal`, `_select_successor`); variants subclass
and override ONE hook. Fundamentally different policies (DiffusionPolicyPlanner) inherit
`AbstractPlanner` directly rather than feigning reuse.
**Rationale:** Open/Closed (extend by override, not edit), Liskov (subclasses degrade to
parent on edge cases), Single-Responsibility (one goal-source per class), Dependency-
Inversion (harness depends on the interface).
**Consequences:** New variants are ~3-line subclasses; 55 unit tests confirm parent
behavior is preserved (e.g. `RoadblockRouteMapBCPlanner` == parent when no route id matches).

## ADR-005 — Statistical rigor is mandatory
**Status:** Accepted
**Context:** Early comparisons on 3 scenarios produced misleading point estimates.
**Decision:** No planner-vs-planner claim without paired Wilcoxon + exact binomial +
bootstrap 95% CI (seeded) + trimmed mean, via `statistical_analysis.py`, on >=30 scenarios.
**Rationale:** Mean L2 hides bimodal distributions; small-n ties are uninformative.
**Consequences:** Several "wins" were correctly re-labeled statistical ties (e.g. p=0.76).

## ADR-006 — PDM-Score as the quality metric (not just L2)
**Status:** Accepted
**Context:** L2-to-expert measures deviation, not driving quality.
**Decision:** Report the full 7-component PDM-Score (collision, drivable, progress, TTC,
comfort, direction, speed-limit) via `pdm_score.py`, alongside L2.
**Rationale:** A policy can have low L2 yet collide or drive uncomfortably. PDM is nuPlan's
official closed-loop metric.
**Consequences:** Surfaced the comfort/safety dissociation (learned = smoother, less safe).

## ADR-007 — Diffusion policy as the multimodality fix (Phase 3d)
**Status:** Accepted; result was a NULL (informative)
**Context:** DualHorizon proved a deterministic MLP averages multi-modal junction trajectories.
**Decision:** Add a DDPM denoiser head on the same 10-dim conditioning; keep capacity matched.
**Rationale:** Generative policies can represent multiple modes a regressor averages.
**Consequences:** Diffusion **tied** the MLP (p=0.79). Diagnosis: the conditioning admits no
multimodality, so diffusion correctly collapses to the mean. Motivated ADR-008.

## ADR-008 — Reframe: from "build a SOTA planner" to "simulator-validity contribution"
**Status:** Accepted (supersedes the implicit SOTA goal)
**Context:** Four-thread literature review found (a) rule-based PDM-Closed still beats every
reproducible learned planner on Val14 — unbeatable solo; (b) the field asserts but never
measures multimodality; (c) nuPlan's IDM agents are unrealistic and reorder the leaderboard
under realistic agents (arXiv 2510.14677, verified).
**Decision:** Pivot the contribution to: *measure multimodality and test whether the
diffusion-vs-MLP conclusion is a simulator artifact* (the 2x2x2 of ADR-009).
**Rationale:** Addresses the thought leaders' #1 stated gap (we cannot trust closed-loop
eval); novel, solo-feasible, inference-cheap, falsifiable either way.
**Consequences:** F0-F6 build is retained but its GOAL changes; success is a mechanism
claim, not a leaderboard number.

## ADR-009 — Evaluate under IDM agents AND realistic SMART agents
**Status:** Accepted
**Context:** ADR-001 inherited nuPlan's IDM reactive agents, now shown to inflate scores.
**Decision:** Run every closed-loop eval under both the standard IDM agents and the
open-sourced SMART reactive agents.
**Rationale:** If the diffusion-vs-MLP margin changes with agent realism, architectural
conclusions in the literature are partly simulator artifacts — the central claim.
**Consequences:** Adds a SMART-agent integration dependency; de-risked first via a
released-checkpoint pilot (`docs/frontier/STAGE_0_PILOT_RUNBOOK.md`).

## ADR-010 — HPC sized to the question, not to scale
**Status:** Accepted
**Context:** Northeastern Explorer offers H200/A100/V100; instinct is to "max compute."
**Decision:** Design for a single H200 (140 GB), 8 h job ceiling with checkpoint/resume,
`/scratch` for data; do NOT pursue multi-GPU unless a model exceeds one GPU.
**Rationale:** Our planners are ~1-2.5M params — trivially GPU-bound; the real bottleneck
is CPU-bound feature extraction + closed-loop sim. Compute scale is not the contribution.
**Consequences:** Explorer's value is CPU + storage for parallel sim, not FLOPs.

## ADR-011 — Encoder before closed-loop training (build order)
**Status:** Accepted
**Context:** Phase-2 DAgger added on-policy data but did not improve a scene-blind policy.
**Decision:** Implement the vectorized scene encoder (F0/F1) BEFORE any perturbation /
closed-loop fine-tuning (F4).
**Rationale:** The DAgger null proved closed-loop data is wasted on a road-blind
representation — fix perception first, then teach recovery.
**Consequences:** Locks the F-stage order; prevents repeating the DAgger mistake.

## ADR-012 — Scenario identifiers stored in every shard sample (2026-06-10)
F0 v2 shards carry scenario_token, scenario_type, log_name, iteration per
sample. Without them, per-type validation gates, scenario_type-clustered
statistics, and joining offline F4 scores to closed-loop runs are impossible.
Cost: cancelled extraction 7558604 (~1.6 h compute) rather than lose 7 h to a
later full re-run. Anonymous tensors cannot be audited.

## ADR-013 — F4 is model-free and never reads ego_future (2026-06-10)
The interaction-multimodality score is computed from scene structure only:
corridor lane-graph branching (map API) + PrET band-pass interaction geometry.
Two traps closed at review: (a) using the diffusion head's sample spread as
the moderator is near-tautological; (b) using ego_future leaks the expert's
resolution of each yield-or-go into the moderator and biases the moderation
slope. Ego's nominal path rolls along the route corridor instead. Model-based
dispersion (APD/FSD) is a secondary manipulation check only.
Full spec with verified citations: docs/frontier/F4_SPEC.md.

## ADR-014 — SMART axis via CAT-K warm start; deliverable is a recipe, not weights (2026-06-11)
SMART-tiny checkpoints (pre_bc_E31, clsft_E9) obtained from the CAT-K authors
for academic use. They are WOMD-derived: per the Waymo dataset terms, neither
they nor weights fine-tuned from them are redistributable. The repo therefore
publishes the nuPlan port + fine-tune recipe; weights stay private. Plan:
nuPlan data adapter -> vocab compatibility check -> BC fine-tune (frozen
embeddings, per the authors' own finetune.py recipe) -> AbstractObservation
wrapper at the model's native 0.5 s token rate.

## ADR-015 — Separate envs for extraction and training, both hard-guarded (2026-06-11)
The nuplan conda env ships CPU-only torch; training in it silently ran on 4
CPU cores (job 7600266, caught after one epoch failed to finish in an hour).
A second failure mode followed: ~/.local pip packages shadowed the conda env
(broken user-site torch, job 7605681). Extraction runs in the nuplan env
(devkit), training in pytorch_env (CUDA) with PYTHONNOUSERSITE=1, and every
sbatch asserts its env can import its deps AND see its hardware before
starting. Silent fallback is treated as a bug class, not a config nit.

## ADR-016 — F4 v1.1: drop headway branch and S_lane after gate failure (2026-06-11)
The validation gates ran over all 5,604 scenarios BEFORE any model comparison
and failed (17% on gate 1). Three causes, all data-diagnosed: pre-registered
high types absent from mini; S_lane saturated by Vegas multi-lane roadblocks
(69% at ceiling, F4 floored at 0.5); headway branch scoring car-following as
ambiguity. v1.1 removes the headway branch, demotes S_lane to a covariate,
and re-registers the gate lists against mini's actual type inventory.
Revision happened pre-unblinding; the moderator firewall holds. Spec sec. 7.

## ADR-017 — v3 dataset: parity histories, corridor route, recovery perturbation, type enrichment (2026-06-12)
Four changes, each closing an audit finding, all in one re-extraction (f0_v3):
1. Per-iteration history queries replace batched past-trajectory sampling.
   The parity gate (nuplan/serving/parity_check.py) caught real skew (max
   feature diff 1.9) between offline extraction and what the simulator hands
   the planner; after the change the gate passes 60/60 tensors. Offline must
   match serving, not the other way around.
2. Route channel switches from lane centerline to roadblock-corridor sweep
   (route_mode="corridor"). The centerline picked one successor per fork and
   leaked the expert's branch choice into every cell, partially collapsing
   the ambiguity the route-region condition exists to preserve (open-loop
   evidence: det_route reached 0.35 m minADE, near the precise cells).
   Corridor mode = turn-by-turn navigation input, lane choice left open.
   Lane mode kept for the leak ablation.
3. Recovery perturbation (perturb_prob 0.5): blend the history tail toward a
   perturbed current pose (lat sigma 0.3 m clip 1.0, heading sigma 0.05 rad
   clip 0.2), label stays the true expert future in the perturbed frame.
   Without it, closed-loop measures off-distribution brittleness (our own
   DAgger null), not policy quality. Deterministic per (token, iteration).
4. CORRECTION to ADR-016: the canonical junction types DO exist in mini
   (46 types unrestricted vs 32 captured). The chunked per-type filter
   dropped them, suspected remove_invalid_goals=True interaction, under
   investigation. v3 adds an enrichment task (task_0016) explicitly
   requesting 11 junction/turn/pedestrian types across all logs. F4 gate
   lists will be re-run against the enriched inventory; the original
   pre-registered high types become evaluable after all.

