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
