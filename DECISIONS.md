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

## ADR-018 — Closed-loop harness validated end-to-end; eval budget corrected (2026-06-12)
The full verdict pipeline runs: PolicyPlanner -> nuPlan two_stage_controller
sim -> prod_eval_metrics -> aggregator CLS, parsed by
nuplan/analysis/analyze_moderation.py (paired Delta, OLS+HC3, Spearman,
Theil-Sen, bootstrap CI; 11 unit tests against known constructions).
Env required pip installs (pytorch_lightning, ray, pyarrow, etc.); the
bokeh-dependent metric_summary_callback is dropped via main_callback override
(bokeh 2.4 vs this env's numpy is broken) — CLS comes from the metric_file +
aggregator parquets, which are unaffected.

Validation, brutally checked: a stationary-only smoke is uninformative (reward
for not moving). The non-stationary smoke (high_magnitude_speed, expert
progress 113-175 m) showed the policy drives at 0.93-0.95 of expert progress
(NOT degenerate) and CLS discriminated clean runs (~0.99) from drivable-area
failures (~0.62). Harness is scientifically valid.

BUDGET CORRECTION: two_stage_controller costs ~7.5 min/scenario, not the
1-3 min estimated in the audit. Consequence: a 500-scenario eval per cell is
~62 CPU-h; 4 cells x 2 agent modes ~500 CPU-h. MUST run as a scenario-sharded
SLURM array (split tokens across array tasks), never serial. Eval manifest is
frozen and committed before unblinding (token list -> scenario_tokens filter).

## ADR-019 — Eval manifest: 4-band F4 stratification, frozen and hashed (2026-06-12)
The eval scenario set is frozen before unblinding (nuplan/eval/freeze_manifest.py
-> manifest.json with token list, per-band counts, SHA256, git commit).
Sharded eval runs it via nuplan/slurm/eval_array.sbatch (array = cell x shard,
tokens[shard::N]).

Stratification is over 4 F4 BANDS {zero, low(0,1/3], med(1/3,2/3], high(2/3,1]},
NOT deciles. A dry run exposed the trap: 60% of scenarios score F4 == 0 exactly,
so equal-count deciles collapse (bottom 6 deciles all [0,0]) and the eval set
would have contained zero low-ambiguity anchor scenarios, silently biasing the
moderation. Equal allocation per band (125 each at n=500) is a balanced-X
design: selection on the regressor F4 does not bias the OLS slope and maximizes
its precision via leverage at the extremes. Caveat recorded: marginal mean Delta
and intercept are then NOT population-representative; the analysis is
slope-focused (the hypothesis is about how the gap scales with ambiguity), and
a separate population-weighted mean Delta can be computed post hoc by reweighting
bands to natural frequency if a marginal estimate is needed.

## ADR-020 — SMART axis: use the Bosch nuPlan checkpoint, wait for the codebase (2026-06-13)
Steffen Hagedorn (Bosch) shared a nuPlan-trained SMART checkpoint
(epoch=07_1180.ckpt, 7.16M params, config embedded: nuplan, 11/80 steps, 2048
tokens, hidden 128, original-SMART architecture). This SUPERSEDES the catk
WOMD checkpoint for our use: no fine-tuning needed (already nuPlan-trained),
collapsing the SMART axis from "port preprocessing + fine-tune + wrap" to just
the observation wrapper. catk kept only as a fallback.
Decision: do NOT reverse-engineer the SMART model/rollout/observation now.
Bosch open-sources the full codebase (incl. their nuPlan IDM-replacement
integration) by end of June 2026; that supplies the matching model class and
the wrapper. Reverse-engineering would be redundant in ~2 weeks. The core 2x2
under IDM is branch-independent and runs first; SMART slots in on release with
the checkpoint already validated. Details: docs/frontier/SMART_INTEGRATION.md.
Checkpoint and any derived weights are never committed to this repo (nuPlan
non-commercial license; cite Hagedorn et al. 2025). The shgd95 collaboration
repo is read-only to us; nothing is pushed there without Parv's review.

## ADR-021 — Per-iteration timeout guard in extraction (2026-06-13)
v3 job 7626090 task 14 wrote 15 shards then HUNG ~20h on a single scenario
(stuck inside the devkit map queries; a hang is not a Python exception, so
the existing per-iteration try/except could not catch it). It silently blocked
the v3 completion gate while holding a CPU allocation doing nothing.
Fix: a SIGALRM wall-clock budget (FeatureConfig.iteration_timeout_s=90, main
thread / Sequential worker) wraps each extract_sample; a hang becomes a
catchable _IterationTimeout and that one frame is skipped, the job continues.
0 disables (tests). Recovery: scancel the element, resubmit just --array=14;
shard-level resume skips the 15 written shards, the guard skips the bad frame.
Tested: guard fires in 1.0s on a 10s hang, no lingering alarm, 173 tests green.
This is the standard production pattern (no single pathological input may stall
a batch job); it should have been in v1.

## ADR-022 — v3 training data frozen at 722 shards; 12 cells in 2 waves (2026-06-14)
Task 14's last shard (1 of ~16/task) never regenerated cleanly before a 4-day
cluster maintenance window (mapo-2026, 2026-06-15..19, 765 nodes). Rather than
block 12-cell GPU training on 0.14% of data (~600/440k samples) AND risk a
fairness violation (if the shard landed mid-training, staggered array cells
would glob different shard sets and different train/val splits), the dataset is
FROZEN at the verified 722 shards. task14 resubmit cancelled; its 14 valid
shards are kept. Fairness (identical data across all cells) over completeness.
The frozen set includes task_0016 (perturbed junction enrichment, 59 shards).
GPU QOS caps 8 submitted / 4 running, so the 12 cells (4 x 3 seeds) run in
2 waves: wave 1 = array 0-7 (seeds 0,1), wave 2 = 8-11 (seed 2) when slots
free. Data freeze makes wave staggering fairness-safe. All HPC compute (train
7715872, f4enrich2 7712443) is queued for after maintenance ends 2026-06-19.

## ADR-023 — Power analysis: contrast is the estimand; N=800 + internal pilot (2026-06-14)
Re-derived the eval sizing properly before the manifest freeze (full writeup in
docs/frontier/POWER_ANALYSIS.md; machinery in nuplan/analysis/power_analysis.py
with 11 tests). Three findings, all corrections to the prior plan.
(1) The HEADLINE is the cross-condition contrast beta1(route)-beta1(precise),
not a single slope. The prior power_analysis.py (2026-06-12) only simulated a
single condition's slope, so the pre-registered claim "n=1000 -> 0.83 power for
beta1>=0.05 at sigma<=0.20" is for the WRONG estimand: verified correct for one
slope (0.877) but only 0.63 power for the contrast at worst-case residual
correlation rho_cond=0. Fix: moderation_contrast() regresses the per-token
difference C_i = Delta_route_i - Delta_precise_i on F4, giving one HC3 SE that
absorbs the within-token correlation (slope(a-b)=slope(a)-slope(b) with shared
X). The contrast had NO SE/p in analyze_moderation.run or results_table.run
before this; the experiment's main hypothesis was untestable. Both now emit it.
(2) Balanced-X design-effect on the real F4 distribution is only 1.10x (~4%),
not a meaningful precision gain (natural F4 is already high-variance bimodal:
60% zeros + 15% high). ADR-019's "slope precision" justification is restated:
balanced-X earns its place by guaranteeing >=125 high/med-band scenarios that
random N=500 undersamples (~76 high, ~36 med), not by shrinking the slope SE.
(3) sigma_Delta is unknown pre-eval and dominates the answer; the Gaussian MDE
is also 4-7% optimistic under bounded CLS (clipping at 1). DECISION: freeze the
manifest at N=800 (safe superset, never re-freeze), then run a pre-registered
internal-pilot variance re-estimation (Wittes-Brittain 1990): a stratified
200-token pilot across all 4 cells measures realized sigma_Delta and rho_cond
via reestimate_from_pilot(), then complete tokens up to the re-estimated N
(cap 800), stopping early if the 0.035-contrast is already powered. Decision
uses only the nuisance variance, never the effect, so alpha is preserved.
Verification gate: analytic Gaussian MDE matches Monte-Carlo through the real
estimator within MC error; null holds alpha. 31 analysis tests green (HPC
nuplan env + local). Supersedes the n=1000 power line in RESEARCH_PROTOCOL.md.

## ADR-024 — S_inter materially over-fires; freeze blocked pending rating (2026-06-14)
Pre-freeze construct-validity check (docs/frontier/S_INTER_DIAGNOSTIC.md;
tool nuplan/analysis/s_inter_diagnostic.py, reuses the production s_inter on
250 high-F4 scenes from local f0_v2). The tool is validated: a self-test
classifies a straight orthogonal agent as cross and a heading-rate-bent
same-direction agent as curvature-only, and the stored agent heading agrees
with velocity direction to median 0.1deg, so s_inter rolls along the right
direction. Findings: only 6% of high-F4 scenes are led by genuine cross/oncoming
conflicts; 84% are led by SAME-DIRECTION near-parallel agents (lead relative
heading median 5deg, crossing angle median 4deg); and 58.8% of high-F4 score
mass is a CURVATURE ARTIFACT, a crossing that vanishes when the agent is
re-rolled straight (median hr only 1.3deg/s, but near-parallel geometry makes
5 s CTRV-rollout intersection hypersensitive to noisy 2-point hr). PrET gap
median 2.3 s sits at the band-pass peak, so grazes score ~1.0. The curvature
artifact is an interpretation-free bug; the same-direction dominance is a
moderator-scope question. DECISION: do NOT change F4 or freeze the manifest yet.
The blind rating is the external arbiter of this exact question and its answer
key is valid only against the current F4; this diagnostic predicts weak
F4-human correlation in the high band. Sequence: rating first -> decide scope
and fix (cap agent rollout horizon + hr deadband at minimum; optional
crossing-angle gate) -> re-score v1.2 + re-run face-validity gate + regenerate
rating sheet -> then freeze. f4_score.py untouched. This supersedes the
"freeze N=800" timing in ADR-023: the N decision stands, but the freeze waits
on F4 v1.2.



## ADR-027 - F4 reframed: "ambiguity" -> "interaction-criticality" moderator (2026-06-21)
Decision (Parv): relabel F4 from a "scene ambiguity" moderator to an INTERACTION-CRITICALITY
(interaction-conflict + route-branch presence) moderator. The score formula
G_stop*(1-(1-S_branch)*(1-S_inter)) and the pre-registered moderation analysis (Delta ~ 1 + F4,
cross-condition contrast) are UNCHANGED; only the construct LABEL is corrected to what validation
supports.
Why: F4 failed human external convergent validity TWICE - v1.1 ~0.15 vs 52 ratings (see ADR-024
over-fire), and a fresh blind protocol gave Spearman 0.020 (p=0.90), ROBUST to a temporal render
that drew the real agent futures (so not a render artifact). A 4-persona AI panel reached only 0.31
and dropped on high-s_inter scenes once futures were shown. F4's GEOMETRIC validity DOES hold:
s_inter detects real space-time crossings (Signal A logged-future replication, Cliff delta 0.345,
p<1e-4, survives partialling out agent-count + ego-speed); s_branch weakly tracks real agent
route-divergence (Signal D, rank-biserial 0.16). Reading: a path-crossing is real but usually a
clear yield/go, not a decision; F4 measures interaction-conflict + branching - which is what we now
call it. H1 now reads "the route-vs-precise CLS gap grows with interaction-criticality F4."
Validation (pre-registered before results): Mac repo docs/frontier/F4_VALIDATION_PROTOCOL.md +
F4_VALIDATION_RESULTS.md. Supersedes the "ambiguity" label in RESEARCH_PROTOCOL.md H1 and F4_SPEC.md.

## ADR-028 — Eval manifest + CLS-selection probe FROZEN (pre-registration) (2026-06-23)
Froze BOTH closed-loop scenario sets BEFORE any CLS is computed (no unblinding, no
cherry-picking). Files in /scratch/.../av-policy-lab/eval_tokens/:
- eval manifest: manifest.json, N=800, seed 0, 4 F4 bands x 200 (available zero/low/med/high
  = 3352/997/402/853, all >= 200), sha256 f3b3b234..., source f4_scores_v11.json.
- CLS-selection probe (ADR-018): cls_probe50.json, N=48 (12/band), seed 1, sha256 f859eba2...,
  VERIFIED DISJOINT from the manifest (overlap 0) so checkpoint selection never sees the test set.
freeze_manifest.py extended with a backward-compatible --exclude option to draw the disjoint
probe. Both carry sha256 + git commit for audit. Re-freezable before the eval runs if N changes;
immutable once the eval starts. NOTE moderator label is interaction-criticality (ADR-027), not
"ambiguity"; the stratification F4 = g_stop*(1-(1-s_branch)(1-s_inter)) is unchanged.

## ADR-029 — Diffusion policy mode-collapses to a point estimator; the #18 null is treatment-absent (2026-06-27)

Finding (rigor-upgrade #1). The diffusion policy in this lab does NOT produce multimodal
trajectories in closed loop. It collapses to a near-deterministic point estimator. This is a
mechanistic, internal-validity finding that REFRAMES the ADR-027 / experiment-#18 moderation null:
the experiment compared two policies that emit near-identical trajectories, so the multimodality
"treatment" was effectively absent and the null was near-inevitable. It is NOT evidence that
interaction-criticality fails to moderate a genuinely-multimodal-vs-unimodal gap; that question
stays open and needs a non-collapsed policy.

Evidence (all from frozen f5_2x2_v4 checkpoints, scene_shard f0, route head):

1a SELECTOR (code-locked). serving/policy_planner.py compute_planner_trajectory: the diff path
   draws K=8 DDIM samples (EMA weights, 20 steps) then select_medoid() returns the most-central
   sample (argmin mean pairwise xy-L2); the det path is a single head forward (conditional-mean
   trajectory). Both share identical heading-from-path post-processing. So at deployment the only
   det-vs-diff difference is mean-point (det) vs medoid-of-8 (diff). The medoid is central by
   construction and would collapse any multimodality at selection time. The planner header even
   pre-flagged a mode-committing selector as a separate pre-registered variant.

1b HEAD COLLAPSE (open-loop probe, nuplan/analysis/multimodality_probe.py). diff_route_seed0
   e130 (CLS-selected), n=2000 scenes, K=32 samples/scene from the head as deployed (EMA, 20-step
   DDIM), endpoints clustered by union-find at eps = 3.5 m (lane width):
   - endpoint dispersion: median 0.131 m, mean 0.192 m, p90 0.315 m, max 2.42 m
   - dispersion / mean displacement = 0.37% (mean path length 35.4 m)
   - modes: mean 1.0005, frac>=2 modes = 0.0005 (1 of 2000), max 2
   - medoid offset from centroid 0.012 m (the cloud is a point, so medoid == mean here)
   K=32 noise draws land ~13 cm apart on a 35 m path: the head is a point estimator, not a
   multimodal sampler. So 1a is moot: there is essentially no multimodality for the medoid to
   collapse.

EPOCH TRAJECTORY (seed0, n=2000, K=32). Collapse develops over training, it is not present at
   init-then-fixed:
     e010  disp/disp 1.76%  median 0.623 m  frac>=2 modes 1.55% (max 3 modes)
     e050  disp/disp 0.68%  median 0.243 m  frac>=2 modes 0.00%
     e090  disp/disp 1.03%  median 0.362 m  frac>=2 modes 0.45%
     e130  disp/disp 0.37%  median 0.131 m  frac>=2 modes 0.05%
     e150  disp/disp 0.22%  median 0.078 m  frac>=2 modes 0.05%
   ~8x contraction e010 -> e150 (non-monotone wobble at e090). Early training retains modest
   diversity (some 3-mode scenes at e010); training drives the head toward a delta function.

EMA CONTROL. e130 RAW (no EMA) disp/disp 0.39% vs EMA 0.37%: near-identical. The collapse is in
   the trained weights, not an EMA-averaging artifact, so there is no cheap deployment-time fix.

CAPSTONE det-vs-diff (nuplan/analysis/compare_det_diff.py). det_route_seed0 e120 vs
   diff_route_seed0 e130 (deployed: medoid-of-8) on the SAME 2000 scenes:
   - det-vs-diff trajectory ADE: median 0.470 m, mean 0.580 m, p90 1.197 m, max 6.25 m
   - ADE / mean displacement = 1.33% (35.3 m path)
   - endpoint offset: median 1.031 m, p90 3.241 m
   The two separately-trained networks are NOT the identical function (the det-diff gap 0.47 m is
   ~3.6x the within-diff sample spread 0.13 m, as expected for two distinct heads), but they are
   functionally near-equivalent: median sub-half-meter ADE, sub-lane-width endpoint, on a 35 m
   horizon. Experiment #18 compared a policy to a near-copy of itself.

GENERALITY (seeds 1, 2 at CLS-best epochs, n=2000, K=32).
   - seed1 e150: disp/disp 0.78% (median 0.273 m, frac>=2 modes 0.00%, max 1 mode)
   - seed2 e140: disp/disp 0.25% (median 0.087 m, frac>=2 modes 0.00%, max 1 mode)
   All three independently-trained seeds collapse (disp/disp 0.22-0.78%, ~0% multimodal scenes).
   Not a single-seed fluke; the collapse is a property of the training recipe.

ROOT CAUSE. Mean-regression. The head is trained with x0-prediction MSE against a SINGLE
   ground-truth future per scene. The Bayes-optimal x0 predictor under MSE is E[x0 | context], the
   conditional mean, which is independent of the noise input; the network therefore learns to
   ignore its sampling stochasticity and emit the mean trajectory, so K noise draws collapse to one
   point. This is the standard regression-to-the-mean failure of single-target imitation. Genuine
   trajectory multimodality requires either an explicit mixture / multi-hypothesis (winner-take-all)
   loss, or training the diffusion model against a DISTRIBUTION of plausible futures rather than one
   logged future.

Decision / consequence:
- Report this as the headline internal-validity result for the rigor-upgrade phase: the #18 null
  is mechanistically explained (treatment absent), not a mysterious flat effect.
- #1c (mode-committing selector) is moot until the head is multimodal; deferred.
- The real next contribution is a training fix that yields a genuinely multimodal policy (multi-
  future targets / winner-take-all / variance-preserving objective), then re-run the moderation
  (#2 validated Signal A moderator, #3 lowered CLS ceiling) against a policy whose treatment is
  actually present. Until then, no moderation claim can separate "F4 does not moderate" from
  "there was nothing to moderate."

Scope / honesty notes:
- The probe measures GLOBAL endpoint dispersion over 2000 scenes, NOT stratified by
  interaction-criticality as #1b was originally specified. The stratified "does multimodality rise
  with s_inter" test is moot: frac>=2 modes = 0.0005 globally (~1 scene in 2000), so there is no
  multimodality distribution to condition on. The collapse is unconditional.
- Probe covers the ROUTE head only (the precise head needs the GT goal, absent from the offline
  shard). Route is the LESS-constrained case (no goal pinning the endpoint); it still collapses, so
  the precise head, which is additionally conditioned on the goal, collapses a fortiori. Closed-loop
  #18 covered both conditions and showed diff~=det in both, consistent with this.

Artifacts: nuplan/analysis/{multimodality_probe.py, compare_det_diff.py}; per-config JSON
mm_diff_route_*.json + cmp_detdiff_route_seed0.json in /scratch/.../av-policy-lab/.

## ADR-030 — Diffusion architecture CAN represent multimodality; the collapse is a data property (Exp 0) (2026-06-27)
Finding: a controlled synthetic-bimodal test proves the DiffusionHead + x0-MSE + DDIM (the exact
objective and sampler used in this lab) recovers a multimodal conditional WHEN the supervision is
multimodal. So the ADR-029 collapse is NOT a model/objective/sampler defect; it is a property of
single-future-per-scene imitation (nuPlan logs one future per scenario -> the learned conditional is
correctly ~unimodal).
Setup (nuplan/analysis/synth_bimodal_test.py): 32 synthetic contexts (random fixed `memory`, encoder
bypassed), each mapping 50/50 to two mirrored arc futures 24 m apart (>> 3.5 m lane). Train the
DiffusionHead with the SAME x0-MSE + CosineSchedule(T=100) as train_policy.compute_loss (4000 steps,
Adam 1e-3). DDIM-sample K=64/context (20 steps); measure mode recovery (union-find eps=3.5 m).
Result: modes mean 2.0, frac>=2 modes 1.0, samples split 50/50 across the two arcs (near-A 0.50,
near-B 0.50), ZERO mass at the collapse midpoint (0.00), both modes covered in 100% of contexts,
endpoint dispersion 12.96 m (~ the arc separation), train loss 0.024. Smoke (C=4/200 steps) already
decisive; the full run confirms with a perfect split.
Consequence: the architecture is sound -> a genuinely multimodal AV policy requires multimodality in
the SUPERVISION. Whether it exists (similar real contexts -> materially different logged futures,
esp. at interaction-critical scenes) is the gating question = Experiment 1 (SPEC drafted; f0_v3
shards carry ego_future + scenario_token + scenario_type, so it is feasible CPU-only). #2/#3 on the
collapsed checkpoints remain not worth compute. The Tier-3 retrain (multi-day) decision is deferred
to Parv pending Experiment 1 evidence on whether there is multimodal supervision to capture.
Artifact: synth_bimodal_test.py; result /scratch/.../av-policy-lab/synth_bimodal_result.json.

## ADR-031 — The data contains multimodal supervision the policy discards, concentrated at decision points (Exp 1) (2026-06-27)
Finding: the real training data (f0_v3) contains substantial multimodal future supervision that the
collapsed policy (ADR-029) discards, and it concentrates at semantically decision-heavy scenes. This
makes a multimodality-capturing retrain viable and explains the #18 null as treatment-absent at the
DATA level, not just the policy level.
Method (nuplan/analysis/available_multimodality.py, CPU, N=4000, k=16): context embedding = a
RANDOMLY-INITIALIZED (untrained) SceneEncoder, mean-pooled (input geometry only, NO future leakage --
a trained encoder would make neighbors share futures tautologically). Per scene: k=16 cosine nearest
neighbors; dispersion + mode-count (union-find eps=3.5m) of the neighbors' logged-future endpoints =
"available" conditional spread. s_inter joined by scenario_token (f4_scores_v11.json).
Results:
- captured (policy per-scene sample dispersion, ADR-029) 0.131m  <<  available (kNN) median 3.14m /
  mean 5.29m  <<  marginal (random-pair) 37.4m. The kNN is VALID: neighbor futures are ~12x tighter
  than random pairs (finds genuinely similar contexts), yet ~24x MORE dispersed than the policy
  emits. 46.3% of scenes have >=2 future modes at lane width.
- Interaction-criticality: high s_inter (>=0.5, n=1228) available 3.55m / frac>=2 0.532 / 2.07 modes
  vs low (n=2772) 2.94m / 0.433 / 1.88. Available multimodality RISES with interaction-criticality
  (data-level support for the original H1 premise).
- Scenario type (most multimodal): stationary_at_traffic_light_without_lead 10.0m / frac>=2 1.00 /
  4.38 modes; traversing_traffic_light_intersection 8.85m / 0.94 / 3.92; near_construction_zone_sign
  6.48m / 0.84; near_long_vehicle 5.28m / 0.94. Least: low_magnitude_speed 3.29m / 0.45. The
  multimodality concentrates exactly at decision points (go/stop at lights, intersections, construction).
Honesty caveats: kNN-neighbor dispersion is an UPPER bound on the true conditional spread (residual
context differences inflate it); random-encoder similarity is a proxy. Strong suggestive evidence,
not a point estimate of conditional entropy. Refinements: k-sensitivity, within-scenario_type matching.
Consequence: there IS multimodality to capture -> Tier 3 (WTA/multi-hypothesis retrain + mode-
committing selector + re-run moderation with the validated Signal A moderator and a lowered CLS
ceiling) is viable. Decision deferred to Parv (multi-day compute, LEARN-mode "decide later" item);
staged de-risk plan: train ONE WTA route cell + probe before committing the full retrain.
Artifact: available_multimodality.py; result /scratch/.../av-policy-lab/avail_mm_result.json.

### ADR-031 addendum — k-sensitivity (intellectual honesty) (2026-06-27)
Available-multimodality grows monotonically with k (larger neighborhoods pull in less-similar
contexts -- the upper-bound caveat made quantitative). Available endpoint dispersion / frac>=2 modes
by k (N=4000): k=2 0.79m/0.128; k=4 1.44m/0.246; k=8 2.20m/0.354; k=16 3.14m/0.463; k=32 5.78m/0.637.
The CONSERVATIVE FLOOR (k=2, single nearest neighbor) = 0.79m / 12.8% >=2 modes -- still ~6x the
collapsed policy's 0.13m / 0.05%. So genuine conditional multimodality exists even at the tightest
context matching; the k=16 headline (3.14m / 46%) is an upper-leaning estimate inflated by
less-similar neighbors. The conclusion (multimodality present, policy discards it, Tier 3 viable)
holds at the conservative floor.

## ADR-032 — WTA de-risk: the standard multimodality fix yields a wider FAN, not distinct modes; Tier-3 NO-GO (2026-06-27)
Tier-3 Step-1 de-risk (Parv chose: stage before any multi-day retrain). Question: does a winner-take-
all multi-hypothesis head recover the maneuver-level multimodality the diffusion policy discards
(ADR-029/031)?
Setup: standalone bounded trainer (nuplan/analysis/wta_derisk_train.py) -- SceneEncoder + WTAHead
(M=6), 16000 real f0_v3 scenes in-memory, 4000 steps, the SAME relaxed-WTA x0-loss + encoder as the
real cells (~3 min GPU). Probe (nuplan/analysis/wta_probe.py): M-hypothesis endpoint dispersion +
mode count (union-find eps=3.5m lane width) + best-of-M minADE + scenario_type strata. N=6000.

Result (M=6):
- eps=0.05: dispersion median 1.77m (vs collapsed diffusion 0.13m = ~14x more diverse), best-of-M
  minADE 0.35m (accurate), but frac>=2 modes = 0.0 in EVERY scenario type at lane width, including
  decision-point types where ADR-031 found 4+ available modes (traversing_crosswalk 2.10m,
  on_intersection 2.06m -- all single-cluster).
- eps=0.01 (sharper assignment, rules out the soft-WTA confound): dispersion WIDENED to median 4.65m
  (max 8.7m), best-of-M minADE 0.46m, but frac>=2 modes only 0.023 (2.3%, max 3) -- even at
  decision-point types 0-3.5% (traversing_traffic_light_intersection 5.04m disp but 3.5% >=2 modes).
  Sharpening inflated the fan WIDTH, it did not SPLIT it into distinct modes.

Verdict (robust across eps): WTA does not collapse like diffusion, but it is a well-fit UNIMODAL
predictor with adjustable spread, NOT a maneuver-level multimodal policy on this data. Only ~2-3% of
scenes are genuinely multimodal; the trend (eps 0.05->0.01 widened the fan but barely moved frac>=2
from 0% to 2.3%) shows the bottleneck is the DATA, not the method.

Implication: supports a stronger reading of ADR-031 -- most of the kNN-neighbor "available
multimodality" is RESIDUAL CONTEXT VARIATION across similar-but-different scenes, not per-scene
decision ambiguity. Per scene, the rich conditioning (agents, map, route, traffic lights) largely
DETERMINES the future, so a well-fit policy fans (predictive uncertainty) rather than splits
(committed alternatives). This is why the diffusion policy collapsed (ADR-029) and why the #18
moderation was near-null: for richly-conditioned nuPlan planning, the much-assumed "multimodal
benefit" of diffusion/multi-hypothesis planners is largely absent because conditioning resolves the
future to near-unimodal. (Contrarian, honest, and consistent across 5 lines of evidence.)

Decision: NO-GO on the multi-day Tier-3 full retrain + re-moderation -- it would test a treatment
present in ~2-3% of scenes and predictably reproduce a near-null; not worth the compute. The de-risk
correctly gated this BEFORE the multi-day commit. The contribution is the diagnostic arc
(ADR-029/030/031/032 + #18). De-risk limitations (honest): 16k-scene subset, 4000 steps (loss
converged by ~step 500), M=6; a full-budget retrain could shift the 2-3% marginally but the
data-bottleneck trend makes a maneuver-level multimodal policy unlikely from this recipe.
Artifacts: wta_derisk_train.py, wta_probe.py; results /scratch/.../wta_derisk_probe6k.json,
wta_derisk_eps01_probe6k.json.

## ADR-033 — Definitive re-analysis: pre-registered inference + sub-component sensitivity; the null is real but metric-SATURATED (2026-06-27)
Closes audit items B (pre-registered stats never shipped), E (CLS-ceiling alternative), and tightens
A (overclaim). Built analyze_moderation_v2.py (scenario_type FIXED EFFECTS + restricted WILD-CLUSTER
bootstrap SE/p by scenario_type, Cameron-Gelbach-Miller; + TOST equivalence, Lakens) and
merge_eval_full.py (re-merge keeping all PDM sub-components + real scenario_type; the original
merge_eval kept only the composite and clobbered scenario_type). Wild-cluster bootstrap SELF-TEST
passed (Type-I 0.025 under a clustered null = properly conservative, vs plain HC3 which is
anticonservative here; full power under a true effect). N=800/cell, 3 seeds each, both reactive modes.

HEADLINE (route-minus-precise F4-slope contrast, the H1 test):
- composite CLS: r0 beta1=-0.0028 clSE=0.0082 wild-cluster p=0.72 (TOST-EQUIVALENT at 0.02);
  r1 beta1=+0.0025 p=0.83. NULL, now under the registered inference.
- ALL sub-components null too (progress, TTC, collisions, drivable-area, comfort, making-progress),
  EXCEPT one honest secondary: ego_is_comfortable r1 beta1=+0.0146 wild-cluster p=0.011 (diffusion's
  comfort edge grows with interaction-criticality under reactive agents). Does NOT survive multiple-
  comparison correction (~14 tests; Bonferroni 0.0036) and comfort is near-binary -> SUGGESTIVE only.

RIGOR WIN (why the registered method matters): on drivable_area_compliance r0 the naive HC3 that
SHIPPED gives beta1=-0.0243 se=0.0134 (t=-1.8, ~"marginally significant"); the correct wild-cluster
bootstrap gives p=0.188 (NULL). The shipped HC3 was anticonservative and would have reported a
spurious effect. Cluster SEs are systematically larger (e.g. comfort r1 0.0064 vs HC3 0.0042).

THE DEEPER FINDING (sensitivity analysis result, empirically confirms audit-E): the sub-components
are NOT more sensitive than the composite -- they are MORE ceilinged. Frac at ceiling (>=0.99), pooled
4 cells: CLS 0.41-0.53; ego_progress 0.69-0.72; ego_is_comfortable 0.99-1.00; drivable_area 0.83-0.85;
TTC 0.54-0.67; collisions 0.58-0.72. The only continuous open-loop metric (ego_expert_L2_error) is
ALL-NaN in closed-loop sim. => the standard nuPlan closed-loop metric suite is SATURATED for these
policies; it cannot express a fine-grained moderation effect even if one existed.

REVISED CONCLUSION (supersedes the "informative null that rules out" framing of MODERATION_RESULTS):
H1 cannot be answered on standard nuPlan CLS for two INDEPENDENT reasons, each sufficient to block
detection: (1) treatment collapse -- the diffusion policy is a near-copy of the deterministic one
(ADR-029); (2) metric saturation -- every outcome is at ceiling (this ADR). The honest contribution
is therefore a measurement/method result in the Dauner "Parting with Misconceptions" (CoRL 2023)
lineage and consistent with the nuPlan-too-easy / reactive / long-tail literature (nuPlan-R, interPlan):
testing fine-grained planner-quality hypotheses on standard nuPlan CLS is confounded by treatment-
collapse AND ceiling, and needs BOTH a genuinely multimodal policy AND an unsaturated eval (harder
scenario slice / a continuous closed-loop metric). Artifacts: analyze_moderation_v2.py,
merge_eval_full.py; results docs/frontier/results/remod_r0.json, remod_r1.json.

## ADR-034 — Unsaturated-subset robustness: the null is NOT a ceiling artifact (refines ADR-033) (2026-06-27)
Closes the ADR-033 open question (is the null due to metric saturation?). moderation_slices.py re-runs
the headline route-minus-precise F4-slope contrast on HARD / unsaturated subsets (sliced by grand
mean CLS M=(4 cells)/4, which is ~orthogonal to the difference-of-differences contrast C, so not
outcome-conditioning; headroom reported per slice) with the pre-registered wild-cluster bootstrap, plus
the direct H1 paired prediction on the top-F4 quartile.

Results (B=4999, both reactive modes):
- Removing the ceiling does NOT reveal an effect. On unsaturated subsets (frac M at ceiling = 0.0,
  real contrast spread mean|C| ~0.05) the contrast stays NULL: r0 unsat(M<0.99) beta1=+0.0011 p=0.91,
  bottom25%-M beta1=+0.0119 p=0.58; r1 unsat beta1=+0.0056 p=0.69, bottom25% beta1=+0.0199 p=0.30.
- The DIRECT H1 prediction is WRONG-SIGNED at its leverage point (high F4 = interaction-critical):
  mean(Delta_route) - mean(Delta_precise) = -0.0089 r0 (CI95 [-0.0181, +0.0003]); -0.0099 r1 (CI95
  [-0.0198, -0.0000], excludes 0). H1 predicts > 0; the data trend the OTHER way.
- Faint non-significant positive in the hardest reactive scenes (r1 bottom25% beta1=+0.020 p=0.30) =
  noise, not a finding.

Conclusion: the null is ROBUST to the ceiling. Where there is genuine detection headroom there is
still no moderation, and the direction is if anything reversed. So the null reflects TREATMENT
COLLAPSE (ADR-029, diff ~= det -> mean|C| ~0.05 unstructured by F4), not metric saturation. ADR-033's
saturation is real (it inflates SEs / removes the "informative" claim on the full sample) but is NOT
the cause of the null. Combined verdict across ADR-029/033/034: H1 is genuinely unsupported AND the
study could not have shown it anyway under treatment-collapse; the contribution is the diagnostic +
the corrected, ceiling-robust inference. Artifacts: moderation_slices.py; results
docs/frontier/results/slices_r0.json, slices_r1.json.


## ADR-035 — S_inter v1.2 (heading-rate deadband/clamp): artifact removed, null robust to the moderator fix (2026-06-27)
Closes the last CRITICAL audit item (the unfixed S_inter curvature artifact). Root cause in code:
f4_score._agent_state estimated the agent heading-rate hr from the last 2 valid points over a 0.1 s
baseline, so position jitter (~2 deg) became ~20 deg/s, and the 5 s CTRV rollout bent it into fake
crossings (~59% of high-F4 mass, S_INTER_DIAGNOSTIC/ADR-024). FIX: HR_DEADBAND_RAD_S=0.05 (sub-noise
-floor -> straight rollout) + HR_MAX_RAD_S=0.35 clamp; F4_VERSION 1.1->1.2. Cheap recompute (no map
pass): only s_inter changed, so recompute it (score_f4 pass_shards, patched) and recombine with the
stored, unchanged s_branch/s_lane/g_stop. WHY s_lane=0 in the recombine: v1.1 (ADR-016) DROPPED S_lane
from the scalar; combine() still accepts it (audit m1 footgun) -- passing the stored raw s_lane
resurrects the deprecated v1.0 formula (caught: it inflated F4 mean 0.22->0.47). Guard: reproduce v11
f4 from components with s_lane=0 first (max_err 0.0), then change only s_inter.

Impact: of 979 high(v11) s_inter scenes, median s_inter 0.907->0.774 and 34.9% drop below 0.5
(artifact confirmed + removed); s_inter changed in 1234 tokens; F4 mean 0.222->0.209.

Clean-moderator (v1.2) re-analysis (wild-cluster bootstrap, both modes) -- the verdict is ROBUST to
the fix: contrast NULL on CLS (r0 beta1=-0.0025 p=0.77 TOST-EQUIVALENT; r1 beta1=+0.011 p=0.16) and on
all PDM sub-components. NOTE the comfort-in-reactive signal flagged earlier (v1.1 p=0.011) does NOT
replicate under v1.2 (p=0.20): it was 1 of ~14 tests and specification-dependent -> noise, not a
result. So no secondary signal survives the moderator correction; the null is clean. Artifacts:
f4_score.py (v1.2), recombine_f4_v12.py, f4_scores_v12.json, remod_v12_r{0,1}.json.


## ADR-036 — Finding-4 "available multimodality" is proxy-dependent; retracted as evidence (2026-06-27)
WS-D: the ADR-031 available-multimodality estimate rested on a random-untrained-encoder kNN whose
embedding is degenerate (audit C1: all scenes ~0.97 cosine). Built a defensible alternative
(available_multimodality_v2.py): match contexts by INTERPRETABLE future-independent descriptors
(ego speed v0, agent count n_par, g_stop, b_r) restricted to the SAME scenario_type -- no learned
embedding. Result (N=6000, k=8): available disp median 5.64 m (68% >=2 modes), tightest same-type
scalar-match pair future distance median 12.4 m -- LARGER than the random encoder's 3.14 m, not
smaller. Reason: scalar features ignore geometry (two same-speed same-type scenes can be entirely
different intersections), and the random encoder, though degenerate, at least ingests map/route. So
the kNN "available multimodality" swings 3-12 m by proxy choice and none is a true matched context.
DECISION: the kNN approach CANNOT establish per-scene future ambiguity from single-future data;
ADR-031's "the data holds multimodality the policy discards" is RETRACTED as a proxy artifact. The
reliable per-scene estimate is the full-scene-conditioned WTA fan (Finding 5 / ADR-032): conditioning
on the entire scene gives a ~2-5 m fan that does not split into modes -> the per-scene future is
largely scene-determined. This strengthens (does not weaken) the central diagnostic; it just rests it
on the trustworthy estimator. Artifacts: available_multimodality_v2.py, available_v2.json.

## ADR-037 — Positive-experiment GATE-1: supervised diversity manufactures uniform modes; supervised fixes exhausted (2026-06-27)
Goal (Parv: "the real positive experiment"): a genuinely multimodal policy (treatment PRESENT) for an
H1 re-test. Cheapest path tested first = diversity-regularized WTA (relaxed-WTA best-of-M + inter-mode
REPULSION to lane width + winner CE; wta_derisk_train.py wta_div_loss, --diversity-weight 2.0
--div-margin 0.35, one route cell, 16k scenes, 4k steps).
Result (wta_div_probe6k.json, N=6000): repulsion DOES create distinct modes -- frac>=2 = 1.0,
frac>=3 = 1.0, mean 5.8 / max 6 distinct modes/scene, endpoint dispersion median 5.57 m (vs plain WTA
1.77 m, collapsed diffusion 0.13 m). BUT it FAILS the quality bar:
- accuracy degraded: best-of-M minADE 0.35 -> 0.66 m (1.9x; gate <=1.5x), minFDE 1.38 m.
- diversity is SCENE-INAPPROPRIATE (the decisive tell): dispersion is HIGHER at trivial scenes
  (stationary 6.47 m, stopping_with_lead 6.43 m, stationary_in_traffic 6.29 m) than at genuine
  decision points (on_intersection 5.55 m, traversing_traffic_light_intersection 5.64 m) -- BACKWARDS
  from real ambiguity. A fixed repulsion forces ~6 spread modes everywhere, even for a stopped car.
Verdict: GATE-1 FAILS. Supervised diversity MANUFACTURES diversity decoupled from scene ambiguity; it
cannot recover genuine scene-conditioned multimodality from single-future data, because there is no
per-scene ambiguity signal to make diversity adaptive. Together with WTA fanning (ADR-032), the
SUPERVISED fixes are EXHAUSTED: no-repulsion -> unimodal fan; strong-repulsion -> uniform manufactured
modes. Neither yields useful scene-appropriate multimodality.
Consequence: a genuinely-treated H1 re-test requires either (a) RL with a designed reward (DIVER-style
GRPO: diversity + SAFETY/realism, so diversity becomes scene-adaptive and plausible) -- multi-day, the
real capstone; or (b) data with genuine multiple-futures-per-scene (not available in nuPlan single-log
imitation). This is itself a clean constructive result: it explains WHY DIVER needs RL, and bounds what
supervised training can do on this data. Artifacts: wta_derisk_train.py (wta_div_loss), wta_div_probe6k.json.

## ADR-038 — RL capstone GATE-RL-1 PASSED: validated open-loop proxy reward (2026-06-27)
The positive-experiment RL path (Parv chose "the real positive experiment") needs a per-sample
reward; closed-loop CLS is too expensive, so built an OPEN-LOOP proxy reward (reward_proxy.py) from
scene tensors, reusing f4_score machinery: R = w_prog*progress_along_route - w_coll*collision_risk
(vs agent CTRV rollouts, time-aligned) - w_off*off_route(hinge) - w_comf*discomfort.
Validation (N=500 scenes): the EXPERT (ego_future) beats perturbations -- offroute 0.90, collision
0.87, jerky 0.95, reversed 0.94 (all > 0.8 gate); expert is the only positive-median category
(median R +0.24). GATE-RL-1 PASS.
Three reward bugs diagnosed + fixed en route (the gate doing its job): (1) offroute used point-to-
VERTEX not point-to-SEGMENT distance (overestimated for sparsely-sampled route polylines);
(2) the route polyline is SYSTEMATICALLY ~3 m laterally offset from the ego path -- diagnosed
directly (route Y ~+3 vs ego Y ~0 in every scene), a frame/centerline offset NOT off-road driving ->
added a tolerance HINGE (penalize only deviation beyond 4 m) so the expert pays ~0 and gross
departures penalize; (3) collision sigma 2.0 -> 1.0 so safe passing (2-3 m) does not fire, only
genuine near-overlap (<1.5 m).
NEXT: GATE-RL-2 -- a reward-weighted / GRPO update on the multi-hypothesis head; confirm modes shift
toward higher reward AND diversity becomes SCENE-ADAPTIVE (more spread at decision types than at
stationary -- the ADR-037 failure inverted). Artifacts: reward_proxy.py, reward_gate1.json,
SPEC-rl-capstone.md, SPEC-positive-experiment.md.

## ADR-039 — RL capstone GATE-RL-2: reward-guided RL achieves SCENE-ADAPTIVE diversity (partial pass) (2026-06-27)
Built rl_train.py: AWR/GRPO on the multi-hypothesis head -- per scene/mode, perturb E times, reward
each via the validated open-loop proxy (reward_proxy), per-scene group-relative advantage (GRPO
baseline), AWR-regress each mode toward its advantage-weighted perturbations + GT realism anchor +
score CE. Bounded run: sbatch 7911097 (GPU, n=4000, 1500 steps, B16/E4, sigma 0.15, anchor-w 0.5,
warm-start plain-WTA) -> rl_s1500.pt. Training reward climbed (maxR -0.57 -> +0.4).
Result (probe N=6000): endpoint dispersion median 11.84 m (plain WTA 1.77, diversity-reg 5.57,
collapsed 0.13); frac>=2 modes 1.0, mean 4.74.
KEY POSITIVE -- diversity is now SCENE-ADAPTIVE, the GATE-1/ADR-037 uniform-diversity failure
INVERTED. Dispersion by scenario type: on_intersection 18.0 m, on_pickup_dropoff 15.8 m,
traversing_crosswalk 15.3 m, following_lane_with_slow_lead 14.8 m (decision-heavy = TOP) vs
stationary_in_traffic 11.6 m, low_magnitude_speed 11.7 m (trivial = BOTTOM). The reward makes the
model spread where multiple options are good -- the mechanism supervised repulsion lacked. Confirms a
PRESENT, scene-appropriate multimodality treatment is achievable via RL on this data.
BUT accuracy degraded: best-of-M minADE 0.35 (plain WTA) -> 1.37 m (gate <= ~0.53). Modes over-spread
(11.6 m even at stationary is too much); the open-loop reward is permissive so modes drift from
realistic paths. PARTIAL PASS: scene-adaptivity (the hard part) achieved; accuracy needs tuning.
NEXT: tune for accuracy -- stronger GT anchor (0.5 -> ~2), smaller exploration sigma (0.15 -> ~0.08),
optionally an expert-proximity reward term, to pull minADE back toward ~0.5 m while keeping scene-
adaptive spread; then GATE-RL-3 closed-loop smoke. Artifacts: rl_train.py, rl_probe6k.json.

## ADR-040 — GATE-RL-2 tuning: a present scene-adaptive multimodal policy achieved (accuracy/spread tradeoff) (2026-06-27)
Tuned the reward-guided RL (anchor-w 0.5->1.5, sigma 0.15->0.08) to rein in the GATE-RL-2 over-spread.
Result (rl_tuned_s1500, probe N=6000): best-of-M minADE 1.37 -> 0.768 m (recovered, sub-meter),
dispersion median 11.84 -> 10.50 m, frac>=2 modes 1.0, mean 4.86 modes. Training reward improved
(meanR -1.1 -> -0.4, maxR +0.5).
Scene-adaptivity PERSISTS but COMPRESSED: top types = on_traffic_light_intersection 12.4 m,
stationary_at_traffic_light_without_lead 12.4 m, traversing_traffic_light_intersection 11.7 m (all
genuine decision points -- go/stop, turn options) vs near_pedestrian_on_crosswalk 10.8 m (range
10.8-12.4 vs untuned 11.6-18.0). So an anchor-strength TRADEOFF: anchor 0.5 = strong scene-adaptivity
+ poor accuracy (1.37 m); anchor 1.5 = weaker scene-adaptivity + good accuracy (0.77 m).
VERDICT: GATE-RL-2 QUALIFIED PASS -- a PRESENT, scene-adaptive multimodal policy now exists (sub-meter
best-of-M, ~5 distinct modes, diversity concentrated at genuine decision points). best-of-M 0.77 vs
unimodal 0.35 m is expected for a 6-mode policy (modes cover alternatives, not all hug the expert).
This is the treatment the H1 re-test needs. Two usable operating points (rl_s1500 high-diversity,
rl_tuned_s1500 high-accuracy).
CRITICAL CAVEAT / next test: optimizing against the OPEN-LOOP proxy reward risks REWARD-HACKING
(spread trajectories that score on the proxy but drive badly closed-loop). GATE-RL-3 = closed-loop
smoke (wrap the multi-hypothesis head as a planner with a mode-committing selector = top-scored mode;
run the real nuPlan sim on a few scenarios; confirm it drives + CLS is sane) is the decisive
validation BEFORE any full retrain/eval. Artifacts: rl_train.py, rl_tuned_probe6k.json.

## ADR-041 — RL capstone GATE-RL-3 PASSED: the RL multimodal policy drives closed-loop (no reward-hacking) (2026-06-27)
Ran the tuned RL policy (rl_tuned_s1500, head_type=wta, TOP-SCORED-mode selector) through the REAL
nuPlan closed-loop sim (run_cells --head wta, 6 scenarios, non-reactive). Result: 6/6 simulations
SUCCESSFUL (0 failed); per-scenario CLS 0.38-0.99, final_score 0.654 (high_magnitude_speed 0.867,
stationary 0.603, traversing_pickup_dropoff 0.38).
VERDICT: GATE-RL-3 PASS. The policy DRIVES -- sane drivable closed-loop trajectories, no crashes.
Reward-hacking RULED OUT at smoke level: a policy gaming the open-loop proxy would crash (CLS ~0);
instead CLS 0.65. The open-loop proxy reward is trustworthy enough that optimizing it yields a
deployable policy. CLS 0.65 < det/diff baseline ~0.85 = UNDERTRAINING (bounded de-risk RL: 1500 steps,
4000 scenes, M=6, warm-started quick WTA, score head trained on proxy not CLS), NOT hacking.
MILESTONE: ALL de-risk gates passed -- RL-1 (reward valid), RL-2 (scene-adaptive multimodality),
RL-3 (drives closed-loop). The positive-experiment pipeline is validated END-TO-END: a scene-adaptive
multimodal policy, trained by reward-guided RL on an open-loop proxy, drives in real nuPlan closed-loop.
The treatment H1 needs is genuinely PRESENT and DEPLOYABLE.
NEXT: close the CLS gap -- longer RL (more steps/data) toward baseline maturity. If CLS -> ~0.85,
proceed to the full matched retrain + unsaturated eval + H1 re-test (negative+positive capstone); if it
plateaus below baseline, that bounds open-loop-proxy RL (a finding). Artifacts: sim_results/rl_smoke,
run_cells (--head wta), serving/policy_planner.py (wta path).

## ADR-042 — Longer RL: open-loop-proxy RL CLS ceilings ~0.71 (below baseline); H1 re-test is feasible via the moderation slope (2026-06-27)
Longer RL (rl_long_s6000: 6000 steps, 8000 scenes, warm-start rl_tuned) closed-loop eval (20
scenarios): mean CLS 0.707, median 0.750, final 0.712, 20/20 successful. vs 1500-step 0.65; det/diff
baseline ~0.85. More training improved CLS (0.65->0.71) but PLATEAUED below baseline; the training
proxy reward also plateaued (meanR ~-0.5 across 1500->6000 steps).
FINDING: open-loop-proxy RL yields a deployable, scene-adaptive multimodal policy that drives
(CLS 0.71) but is NOT baseline-competitive -- the open-loop proxy is an imperfect closed-loop surrogate
(reward-CLS gap), so CLS ceilings ~0.71 without closed-loop-in-the-loop reward (the expensive
DIVER-full path). This BOUNDS what open-loop-proxy RL achieves -- a clean, citable boundary.
H1 RE-TEST feasibility: a matched-maturity multimodal-vs-unimodal level comparison isn't achievable
via proxy-RL (0.71 vs 0.85). BUT the MODERATION is robust to a constant level offset: Delta =
CLS(RL) - CLS(det) ~ F4 tests whether the gap's SLOPE varies with interaction-criticality; the
intercept absorbs the maturity gap. So an honest H1 re-test against a PRESENT, scene-adaptive treatment
IS feasible.
NEXT: closed-loop eval rl_long on a manifest subset spanning F4 bands (matched to the #18 det_route
baseline), regress Delta(RL-det) ~ F4 (analyze_moderation_v2 inference). Either result publishable:
positive slope = multimodality helps relatively more at interaction-critical scenes (H1 supported with
a present treatment); null = even present scene-adaptive multimodality does not F4-moderate the
closed-loop gap (the strongest form of the negative result). Artifacts: rl_long_eval sim_results.

## ADR-043 — H1 RE-TEST CAPSTONE: with a PRESENT scene-adaptive treatment, the moderation slope FLIPS POSITIVE (suggestive H1 support) (2026-06-27)
Evaluated the RL scene-adaptive multimodal policy (rl_long_s6000, head=wta, top-scored mode) CLOSED-LOOP
on a 200-token F4-stratified manifest subset (matched to the #18 det_route baseline). 200/200 sims
successful. Delta = CLS(RL) - CLS(det_route) ~ F4, scenario_type FE + wild-cluster bootstrap (31 clusters):
- mean CLS: RL 0.758 vs det 0.864 (Delta -0.106 = maturity-gap INTERCEPT, expected; RL undertrained, ADR-042).
- beta1 (the moderation SLOPE) = +0.054 (cluster SE 0.049, t 1.11, one-sided p 0.132 for H1: beta1>0).
THE KEY RESULT: with the COLLAPSED policy (#18) the slope was NULL + WRONG-SIGNED (-0.006/-0.013);
with a PRESENT, scene-adaptive multimodal treatment it FLIPS to POSITIVE (+0.054, the H1-predicted
direction, effect size IN the pre-registered range 0.02-0.05). NOT significant at N=200 (p=0.13;
cluster SE large for 31 clusters / single-seed RL eval) -> SUGGESTIVE, not confirmed.
INTERPRETATION: making the treatment present flipped the moderation from wrong-signed-null to
positive-in-direction -- direct evidence that (a) treatment-ABSENCE (not a true negative) drove the
original null, and (b) when multimodality is genuinely present + scene-adaptive, the closed-loop
advantage trends with interaction-criticality as H1 predicts. The full 800-token eval (4x N ->
~halved SE) would confirm/refute significance if the effect holds.
This is the CAPSTONE: negative diagnostic (collapse + saturation + supervised fixes fail) + a
constructive RL recipe (scene-adaptive multimodality, deployable) + a positive directional FLIP of
the moderation once the treatment is present. Artifacts: rl_h1_eval sim_results, h1_retest_result.json,
h1_retest.py.

## ADR-044 — H1 RE-TEST FINAL (N=800): moderation slope flips to the H1-predicted POSITIVE direction; suggestive, not significant (2026-06-28)
Full 800-token closed-loop eval of the RL scene-adaptive multimodal policy (rl_long_s6000, head=wta,
top-scored mode) vs the #18 det_route baseline (800/800 matched). Delta = CLS(RL) - CLS(det) ~ F4,
scenario_type FE + wild-cluster bootstrap (39 clusters):
- mean CLS RL 0.751 / det 0.865 (Delta -0.114 = maturity-gap INTERCEPT; RL undertrained, ADR-042).
- beta1 (moderation SLOPE) = +0.035 (cluster SE 0.023, t 1.56, one-sided p 0.063, CI90 [-0.002, +0.073]).
Progression across N (slope / one-sided p): N=200 +0.054 / 0.13 ; N=600 +0.045 / 0.055 ; N=800 +0.035 / 0.063.
VERDICT: with a PRESENT scene-adaptive multimodal treatment the moderation slope is POSITIVE (the
H1-predicted direction, effect size in the pre-registered 0.02-0.05 range) -- a marked qualitative
change from the collapsed-policy experiment, where it was NULL and WRONG-SIGNED (#18: -0.006/-0.013).
But it does NOT reach conventional significance at N=800 (one-sided p 0.063; CI90 just includes 0).
SUGGESTIVE, not confirmatory.
INTERPRETATION (honest capstone): the directional FLIP (wrong-signed-null -> positive) is direct
evidence the original null was TREATMENT-ABSENCE, not a true negative; making multimodality present +
scene-adaptive moves the moderation toward H1 as predicted. Full confirmation falls short, most
plausibly because the proxy-RL policy is undertrained (CLS 0.75 vs 0.86, ADR-042) -> the treatment is
at reduced strength, and a subtle-range effect needs more power / a more mature closed-loop-RL policy.
COMPLETE ARC: negative diagnostic (collapse + saturation + supervised fixes fail) -> constructive RL
recipe (scene-adaptive multimodality, deployable, drives closed-loop) -> positive directional flip of
the moderation with a present treatment (suggestive). The positive experiment is CLOSED at proxy-RL
scale; a fully-confirmatory result would need closed-loop-in-the-loop RL (out of current scope).
Artifacts: rl_h1_eval + s0/s1/s2a/s2b sim_results, h1_retest_result.json, h1_retest.py.

## ADR-045 — Why the positive result is bounded: conceptual/metric barriers + an open-loop-proxy tension (brutally-honest diagnosis) (2026-06-28)
Question (Parv): are the results "not good", and is it a conceptual / implementation / literature-matching
/ illogical issue? Triage:
- The NEGATIVE diagnostic (Part I) is correct, clean, and LITERATURE-CONSISTENT (collapse = DIVER
  arXiv:2507.04049; saturation = the nuPlan-too-easy critique, Dauner CoRL'23). Not "not good."
- The POSITIVE result (Part II) is suggestive-not-significant (beta1 +0.035, p 0.063) + sub-baseline
  policy (CLS 0.75 vs 0.86). The "not good" is here. Causes, in order:

(1) CONCEPTUAL / METRIC (dominant, field-confirmed). Multimodality is a property of the predictive
DISTRIBUTION; closed-loop CLS scores the single EXECUTED trajectory. The car drives ONE path, so
policy-multimodality moves CLS only when the SELECTOR's chosen mode beats the deterministic policy's
executed trajectory. nuPlan's own paper: open-loop distance "is not a suitable indicator in a
multi-modal scenario" -- but executed-CLS is ALSO near-blind to "having modes" unless they change the
executed choice. Plus: genuine per-scene bimodality is rare (~2-3%, ADR-037); the deterministic mean
is usually fine on a saturated benchmark; RAPiD (arXiv:2602.07339, 2026) distills diffusion -> a
DETERMINISTIC policy with competitive nuPlan CLS. So a SMALL H1 effect is conceptually EXPECTED, not a
failure -- it matches the literature's skepticism.

(2) LOGICAL TENSION (sharpest self-critique). We trained the RL policy on an OPEN-LOOP proxy reward,
but Part I's thesis (Dauner) is open-loop != closed-loop. So the proxy-RL is bounded by the exact
misconception this paper studies; the CLS ceiling (proxy reward plateaued while CLS did not follow,
ADR-042) is the direct symptom. Not broken, but a known trap we partially fell into. The field's fix
is closed-loop-reward RL (CaRL arXiv:2504.17838; CarPlanner 2502.19908; DiffusionDriveV2 2512.07745) =
the multi-week step flagged in SPEC-rl-capstone.

(3) IMPLEMENTATION (real, secondary, fixable). Bounded de-risk RL (6k steps, single seed, M=6, proxy
reward, proxy-trained selector). Fixable with compute/closed-loop RL -- but per (1) that lifts CLS
toward baseline WITHOUT necessarily enlarging the H1 effect (the conceptual ceiling remains).

ILLOGICAL? No fundamental illogic. The one honest caveat: the H1 operationalization (selected-mode
policy vs deterministic, executed-CLS) conflates "having modes" with "the selector's choice" and uses
a metric structurally near-blind to the predictive distribution -- a known limitation, not a bug.

VERDICT: the results are GOOD as science (correct, honest, literature-consistent). They look "not
good" only against an expectation -- a big clean significant positive -- that the problem structurally
does NOT offer on nuPlan executed-CLS. The small/suggestive effect IS the finding.

HIGHER-LEVERAGE NEXT MOVE (cheaper + sharper than closed-loop RL): test multimodality's value with a
metric that CAN see the distribution -- a BEST-OF-MODES SAFETY ORACLE (does ANY of the K modes avoid a
collision / off-road that the deterministic policy commits, at interaction-critical scenes?). If
multimodality's benefit shows up there but not in executed-CLS, that PINPOINTS the metric-blindness as
the barrier (a clean, novel result) -- runnable on existing data (reuse reward_proxy collision/offroute
+ the RL modes), no closed-loop RL. Closed-loop-reward RL remains the orthogonal path to lift CLS.

## ADR-046 — Safety-oracle: multimodality has latent (uniform, partly-artifactual) value, but H1's F4-moderation is NULL in the distribution-aware metric too (2026-06-28)
Best-of-modes safety oracle (safety_oracle.py, N=2000): per scene, min unsafety (collision + off-route
via the validated reward_proxy) over the RL policy's 6 modes vs the DETERMINISTIC policy's single
executed trajectory.
- mean det unsafety 0.431 -> best-mode 0.109 (advantage 0.322); a mode is SAFER than det in 74% of
  scenes. So multimodality carries latent value that executed-CLS (one trajectory) cannot credit --
  the metric-blindness (ADR-045) is REAL.
- BUT the advantage does NOT grow with interaction-criticality: high s_inter 0.329 vs low 0.318;
  moderation beta1 = -0.031 (cluster SE 0.045, one-sided p 0.76) = NULL / wrong direction.
HONEST CAVEAT: the 74% / 0.322 magnitude is partly a BEST-OF-K SELECTION ARTIFACT (min over 6 spread
modes vs 1 trajectory on a noisy metric usually finds a lower one); a fair control would be
best-of-K-modes vs best-of-K-perturbations-of-det. So the latent-value SIZE is inflated; the robust
readout is the F4-MODERATION, which is NULL.
CROSS-METRIC CONCLUSION (the real answer to "are the results good / why"): H1 -- the SPECIFIC claim
that multimodality's closed-loop benefit GROWS with interaction-criticality -- is NOT robustly
supported on nuPlan in EITHER metric: executed-CLS N=800 beta1 +0.035 (suggestive, p 0.063) AND the
distribution-aware safety oracle beta1 -0.031 (null). The executed-CLS directional flip is NOT robust
across metrics -> treat it as noise, not support. What DOES hold: (a) the original null was
treatment-ABSENT (collapse, ADR-029); (b) with a present scene-adaptive treatment, H1's
interaction-criticality moderation STILL does not appear; (c) multimodality has latent safety value
but it is UNIFORM (not decision-point-specific) and partly a best-of-K artifact.
HONEST FINAL: not "multimodality is useless" and not "H1 confirmed" -- "H1's interaction-criticality
moderation is genuinely unsupported even after recovering a present treatment and using a
distribution-aware metric; multimodality's value, where present, is broad not F4-specific." Consistent
with Dauner (deterministic competitive) + RAPiD (diffusion->deterministic distillation). This REVISES
the ADR-043/044 "suggestive positive" down to "not robust across metrics." Artifacts: safety_oracle.py,
safety_oracle.json.

## ADR-047 — Fair best-of-K control: the RL multimodality provides ~0 safety value beyond matched-random perturbations of the deterministic policy (2026-06-28)
Resolves the ADR-046 caveat (the 74%-of-scenes "a mode is safer" was "partly a best-of-K artifact").
Fair control (safety_oracle.py): RL best-of-K modes vs the deterministic trajectory + K MATCHED-
DISPERSION random perturbations (per-scene noise calibrated to the RL modes' endpoint std, ramped to
the endpoint), both reduced by best-of-K. N=2000.
- det single-traj unsafety 0.431; RL best-of-K 0.109; det best-of-K (matched random) 0.116.
- UNFAIR advantage (RL bestK vs det 1 traj) = 0.322, 74% of scenes -> ALMOST ENTIRELY a SELECTION
  ARTIFACT.
- FAIR advantage (RL bestK vs det matched-random bestK) = +0.0075 (~0); RL learned modes beat
  matched-random in only 48.7% of scenes (~coin flip). F4-moderation of the FAIR advantage:
  beta1 -0.061 (negative; one-sided p 0.76, NOT H1).
VERDICT: the RL policy's LEARNED multimodality provides essentially NO safety value beyond K random
perturbations of the deterministic trajectory of the same dispersion. The 74%/0.32 "latent value"
(ADR-046) was the best-of-K selection effect; netting it out, the learned modes are not placed in
safer regions than random spread (48.7% = coin flip).
COMPLETE, FINAL, HONEST CONCLUSION (across ALL metrics): even with a present, scene-adaptive multimodal
policy, there is NO evidence its multimodality yields closed-loop value -- executed-CLS H1 slope +0.035
(p 0.063) did NOT replicate; the artifact-controlled safety advantage is ~0 / coin-flip -- and NO
interaction-criticality moderation in any metric. The original null was TREATMENT-ABSENCE; the
corrected experiment, with a present treatment AND an artifact-controlled metric, shows the multimodal
benefit is genuinely not there on nuPlan. This SUPERSEDES the ADR-046 "latent value (74%)" framing
(retracted as a selection artifact). Consistent with Dauner (deterministic competitive) + RAPiD
(diffusion->deterministic distillation). INTEGRITY NOTE: this fair control was built specifically to
falsify the positive-leaning interpretation, and it did -- the conclusion is the stronger for it.
Artifacts: safety_oracle.py (fair control), safety_oracle_fair.json.

## ADR-048 — GATE-CL-1: the open-loop proxy reward is ~orthogonal to closed-loop CLS (r=0.11); open-loop reward engineering is futile (2026-06-28)
Option-1 (closed-loop-reward RL) de-risk. Correlated the validated open-loop proxy reward
(reward_proxy, on the RL policy's executed TOP-SCORED mode) vs ACTUAL closed-loop CLS over n=301
rl_h1 tokens: Pearson 0.108, Spearman 0.049 -- NEAR ZERO.
VERDICT: the open-loop proxy reward does NOT predict the closed-loop outcome. Therefore (a) improving
the open-loop proxy is FUTILE -- a "better" open-loop reward faces the same ~0 transfer; (b) the RL CLS
ceiling (0.75, ADR-042) was a fundamental REWARD-SIGNAL gap, not a training-budget issue; (c) a
closed-loop-good policy REQUIRES sim-in-the-loop reward (real CLS in the loop) -- the heavy path.
SHARP, CITABLE FINDING: open-loop reward FIDELITY (ranking single trajectories: GATE-RL-1 expert beats
perturbations 87-95%) does NOT imply closed-loop reward fidelity (r=0.11 with CLS). The Dauner
open-loop != closed-loop thesis quantified for REWARD signals -- explains why proxy-RL ceilinged and
why the H1 re-test was inconclusive.
NEXT (GATE-CL-2, the worth-it test): is the SELECTOR the bottleneck or the MODES? Closed-loop
best-of-modes oracle -- execute EACH of the K modes in the sim on a subset, take best-of-K CLS. If
>> top-scored 0.75 and approaches/beats baseline 0.86 -> the modes are good, a closed-loop-trained
SELECTOR is the tractable fix, AND multimodality has real closed-loop value (a good mode exists, just
mis-selected). If ~0.75 -> the modes themselves are the ceiling -> only sim-in-the-loop policy RL
(prohibitive, low H1-payoff per ADR-045/047). Artifacts: cl_corr.py, cl_corr.json.

## ADR-049 — GATE-CL-2: the SELECTOR is the bottleneck, not the modes; best-of-modes oracle BEATS the deterministic baseline (2026-06-28)
Closed-loop best-of-modes oracle: executed each of the 6 WTA modes in the sim on 120 F4-stratified
tokens (720 sims, 120/120 successful each), took best-of-K CLS per token. Per-mode mean CLS 0.71-0.80
(each mode is a reasonable policy); BEST-OF-MODES mean CLS = 0.883 vs det baseline 0.868 vs deployed
top-scored 0.75.
DECISIVE: best-of-modes (0.883) >> deployed top-scored (0.75) AND > det baseline (0.868). The modes
CONTAIN baseline-beating capability; the 0.75->0.883 gap (0.13 CLS) is ENTIRELY the SELECTOR's fault
(trained on the ~0-corr open-loop proxy, ADR-048). The MODES are good; the SELECTOR is the fixable
bottleneck.
CAVEATS (honest): (a) best-of-modes beats det by only +0.016 on average (frac a-mode-beats-det 44%);
(b) its advantage does NOT grow with F4 (beta1 -0.023, null) -> consistent w/ ADR-046/047:
multimodality's value is broad+small, NOT interaction-criticality-specific; (c) best-of-modes is an
ORACLE (post-hoc, uses true CLS) = the CEILING of a perfect selector; a real closed-loop-trained
selector captures only PART of the 0.75->0.883 gap.
TRACTABLE OPTION-1 PATH (GATE-CL-3): train a CLOSED-LOOP selector on per-mode CLS labels (modes frozen;
far cheaper than sim-in-the-loop policy RL), deploy + eval -> measure how much of the 0.75->0.883
headroom a learnable selector realizes = the genuine tractable route to a baseline-mature multimodal
policy. Artifacts: clo_oracle.py, clo_oracle.json, sim_results/clo_mode{0..5}.


## ADR-050: Pivot to standard-split, realism-axis evaluation (SMART reactive agents)
Date: 2026-06-29. Status: ACTIVE.

Context. Hagedorn et al., "When Planners Meet Reality: How Learned, Reactive Traffic
Agents Shift nuPlan Benchmarks" (arXiv:2510.14677; Steffen Hagedorn, Bosch / U. Lubeck)
show nuPlan IDM agents OVERESTIMATE planning performance and shift the deterioration
pattern: imitation-learned planners degrade on simple scenes, rule-based planners degrade
on harder interaction scenes, once IDM is replaced by the learned SMART reactive-agent
model. Steffen's email confirms their SMART checkpoint was selected purely on next-token
classification accuracy (an open-loop proxy), with no closed-loop metric in selection,
and invites closed-loop results with newer planners. This matches our own GATE-CL-1
finding (open-loop proxy ~ orthogonal to closed-loop CLS, r=0.11, ADR-048).
Measured fact (this fire): all prior eval (H1, selector, CLS) ran under IDM on a custom
800-token set (eval_tokens/manifest.json). Overlap with canonical Val14 (1118 tokens) and
Test14-hard (272 tokens) is EXACTLY ZERO. Prior numbers are therefore not comparable to
the literature, and the f0_v3 cache (722 shards, 17 task dirs, no token index) does not
contain the canonical tokens.

Decision. Re-scope the capstone to a two-phase, realism-axis study on standard splits.
- Phase 1 (no dependency on Steffen): re-baseline the ego zoo (det, multimodal/WTA,
  closed-loop selector, PDM-Closed) under IDM on canonical Val14 + Test14-hard + interPlan,
  with the pre-registered stratified inference (scenario fixed effects, wild-cluster
  bootstrap, TOST) and interaction-criticality (F4) stratification. Requires feature
  regeneration for canonical tokens (cache is disjoint). Standalone, literature-comparable.
- Phase 2 (gated on Steffen's nuPlan tokenization config + motion-token codebook): swap
  IDM -> SMART reactive agents and re-run the same zoo. Headline questions: does sim
  realism change the multimodality / scene-adaptivity conclusion, and does CLS-under-IDM
  predict CLS-under-SMART (planner-ranking shift)?

Consequences. Prior IDM / custom-subset numbers (det 0.868, deployed selector 0.75,
oracle best-of-modes 0.883) are retained only as INTERNAL references, not headline
results. The H1 null is reframed as CONDITIONAL on sim realism (IDM under-reacts and
masks interaction), to be retested under SMART. The nuPlan<->SMART adapter is built in
tokenizer-independent parts now (harness hooks, agent read/write, validation gate);
tokenization is gated on Steffen. First concrete step: f0_v3 token index (job f0v3idx)
-> true canonical coverage -> feature-regeneration scope for missing tokens.


## ADR-051: GATE-CL-3 deployed-selector result + cluster env map (IDM regime)
Date: 2026-06-29. Status: internal reference (superseded as headline by ADR-050 pivot).

Closed-loop selector (score head retrained on per-mode closed-loop CLS, encoder/queries/
mode_emb/trunk frozen), deployed argmax on the held-out 120 IDM scenes:
  deployed_selector_CLS = 0.771 (n=120)
  refs: prev_deployed (open-loop / next-token-style selection) 0.75 | det baseline 0.868 |
        oracle best-of-modes 0.883.
Reading: the learnable closed-loop selector lifts deployed CLS by only +0.021 (0.75 ->
0.771), i.e. it captures ~16% of the 0.133 oracle headroom and stays ~0.10 below the
deterministic baseline. So "the selector is the bottleneck" (ADR-049, an oracle/privileged
result) is only partly actionable: a realistic non-privileged learned selector recovers
little of the oracle advantage. The modes-good (oracle 0.883) vs deployable (0.771) gap is
a selection-LEARNABILITY gap, and it is large. Consistent with ADR-045: under IDM the
multimodal policy stays sub-baseline even with closed-loop-aware selection. Headline work
is now the standard-split + SMART realism pivot (ADR-050); this number is an IDM-regime
internal reference only.

Cluster env map (to stop env-guessing; cost three failed job attempts this session):
  - TRAINING / torch.load: /home/patodia.pa/.conda/envs/pytorch_env/bin/python
    (torch 2.4.1+cu121; NO pyarrow).
  - ANALYSIS / nuPlan sim / aggregator parquet: /home/patodia.pa/.conda/envs/nuplan/bin/python
    (pyarrow + nuplan-devkit).
  - ALWAYS export PYTHONNOUSERSITE=1 first; a broken ~/.local torch shadows imports otherwise.
  - There is NO conda env named "fell". The vitcd venv (/scratch/patodia.pa/venvs/vitcd)
    belongs to vit-from-scratch, not av-policy-lab. Call interpreters by absolute path; do
    not rely on `module load` + `conda activate` in non-interactive ssh.

### ADR-050 coverage result (2026-06-29)
f0_v3 cache = 5568 unique tokens. Canonical-split coverage: Val14 5/1118, Test14-hard 2/272. => feature regeneration required for ~1383 canonical tokens before any Phase-1 standard-split eval.


## ADR-052: Data reality (mini-only) + token-targeted extraction tooling
Date: 2026-06-29. Status: ACTIVE (decision made with Parv: acquire val data).

Finding (verified). Only the nuPlan `mini` split (64 log DBs, 14GB) is staged at
/scratch/patodia.pa/nuplan/data/cache/mini; no val/trainval/train/test anywhere on
scratch or home. The ENTIRE project to date (H1, the closed-loop selector, all CLS
numbers) ran on `mini`, a dev subset sampled by scenario type (num-scenarios-per-type),
NOT a benchmark split. Canonical Val14 (1118 tokens) and Test14-hard (272 tokens) come
from the nuPlan `val` log split, which is absent. That is the true cause of the ~0 cache
coverage (ADR-050). So standard-split, Steffen-comparable evaluation REQUIRES downloading
the nuPlan val logs (credentialed; Parv's action). Decision with Parv: acquire the val data.

Tooling added (this commit). scene_features.py now accepts --tokens-file (a JSON
{"tokens": [...]}) and threads it into ScenarioFilter(scenario_tokens=...), so extraction
targets ONLY the canonical tokens instead of walking all of val (far less compute).
Verified on mini: --tokens-file with one token yields exactly "Extracting 1 scenarios"
(token selection correct). The shard-write path is unchanged from the one that built
f0_v3 (5568 tokens). Canonical token-files staged: eval_tokens/{val14, test14hard,
canonical_val14_test14hard}.json (1118 / 272 / 1390).

Next. Parv downloads the nuPlan val logs -> token-targeted array extraction over val
(--tokens-file canonical_val14_test14hard.json) -> Phase-1 standard-split IDM re-baseline
of the ego zoo with the pre-registered stratified inference; then the SMART realism axis
(Phase 2) once Steffen provides the nuPlan tokenizer/codebook.


## ADR-053: Feature regen NOT needed for eval; val data acquired; run_cells --db-dir
Date: 2026-06-29. Status: ACTIVE.

Correction to ADR-050/052. The closed-loop EVAL does not need a feature cache. Verified in
nuplan/serving/policy_planner.py: the planner instantiates SceneFeatureExtractor() and calls
build_input_features on the live PlannerInput each sim step (the SAME extractor as offline),
so features are built on-the-fly during simulation. The offline f0_v3 cache was only for
TRAINING the policy and the offline selector analysis (cl_corr / clo_selector). Therefore the
Phase-1 standard-split re-baseline needs only: the val DB files + the existing checkpoints +
run_cells.py targeting the val tokens. No val feature extraction is required. (Feature
extraction on val would only be needed to RETRAIN on val, e.g. a val-CLS-labelled selector,
which is optional and later.) The earlier "feature regeneration required" was an overstatement.

Val data acquired (autonomously, no manual portal download). nuPlan is on the public AWS Open
Data bucket s3://motional-nuplan (ap-northeast-1, --no-sign-request); boto3 is in the nuplan
env and HPC reaches S3 via the cluster proxy. Detached job 7986967 downloaded
public/nuplan-v1.1/nuplan-v1.1_val.zip (90.3 GB) and unzipped the val log DBs to
/scratch/patodia.pa/nuplan/val_stage/data/cache/val/ (sensor blobs deliberately skipped).

Tooling. run_cells.py now takes --db-dir (default mini) so eval can target the val DB dir.

Honest scope note. The zoo checkpoints were TRAINED on the mini split (64 logs). Evaluating
them on standard Val14/Test14-hard is a valid benchmark eval, but the absolute CLS reflects a
mini-trained policy; the H1 / realism questions are about RELATIVE comparisons (multimodal vs
det; IDM vs SMART) which remain valid. Training on the full train split is a separate, larger
future lift (train_* zips are ~40-175 GB each).

Next. Once unzip completes: run_cells --db-dir <val> --tokens-file val14.json / test14hard.json
across the zoo (det, multimodal/WTA, selector, PDM-Closed) under IDM -> CLS per token -> the
pre-registered stratified inference + F4 stratification = the Phase-1 standard-split result.


## ADR-054: Val14 = Phase-1 eval set; Test14-hard is in the test split; Val14 zoo eval launched
Date: 2026-06-29. Status: ACTIVE.

Split membership (verified via the nuPlan builder + ScenarioFilter scenario_tokens on the val DBs):
  Val14 1118/1118 present in the val split; Test14-hard 0/272 present. Test14-hard tokens live in
  the held-out TEST split, not val (planTF test14-hard.yaml uses log_names: null). So Val14 is the
  Phase-1 primary benchmark (also what Hagedorn reports); Test14-hard is deferred pending the 96GB
  test.zip, once that split membership is confirmed. interPlan (30) also later.

Val data: 1381 val log DBs at /scratch/patodia.pa/nuplan/val_stage/data/cache/val (from the 90GB
  nuplan-v1.1_val.zip pulled via the public S3 bucket, job 7986967). No feature cache needed: the
  eval builds features live (ADR-053).

Pipeline fixes (verified): run_cells --db-dir targets the val DBs; PDM-Closed requires
  PYTHONPATH += /home/patodia.pa/tuplan_garage for the PDMClosedPlanner import. Val scenario
  extraction verified (a proof built 20 Val14 scenarios); det + pdm proofs run clean.

Launched: nuplan/slurm/val14_zoo_array.sbatch (job 7987964, array 0-63%16) =
  {PDM-Closed, det/route, WTA/route, closed-loop-selector/route} x 16 shards of the 1118 Val14
  tokens, under IDM (reactive=1), resume-safe. -> per-config Val14 CLS, then the pre-registered
  stratified inference (scenario fixed effects, wild-cluster bootstrap, TOST) + F4 stratification.

Honest scope: the checkpoints were trained on the mini split; eval-on-Val14 is a valid benchmark
  eval, but absolute CLS reflects a mini-trained policy. The multimodal-vs-deterministic and (Phase 2)
  IDM-vs-SMART RELATIVE comparisons are the research signal, not the absolute numbers.


## ADR-055: Phase-1 Val14 standard-split CLS (under IDM) -- multimodality underperforms deterministic
Date: 2026-06-30. Status: result (CLS headline complete; F4/s_inter moderation pending job 8004148).

Setup. Ego zoo evaluated closed-loop on the canonical Val14 split (1118 scenarios) under IDM
reactive agents (run_cells --db-dir <val> --reactive 1), with the CLS-selected det checkpoint
and the wta_derisk rl_long / rl_clsel checkpoints (all mini-trained; ADR-054 scope caveat).
Features built live during sim (no cache; ADR-053). 64-task array, all 16/16 shards per config,
0 unreadable shards.

Per-config Val14 CLS (n=1118 each):
  PDM-Closed (rule-based)            0.9688
  det (deterministic IL)            0.8049
  sel (closed-loop selector on WTA) 0.7410
  wta (multimodal / WTA)            0.7121

Reading (brutally honest):
- Among the LEARNED policies, deterministic is best. MULTIMODALITY UNDERPERFORMS: WTA 0.712 is
  -0.093 below det 0.805 on the standard split under IDM. This confirms the multimodality-collapse
  / mean-regression finding (mini: ADR-029/045) on the full Val14 benchmark, not just the dev split.
- The closed-loop selector recovers part of the WTA deficit (0.712 -> 0.741, +0.029) but does NOT
  reach deterministic. The selection-learnability gap (ADR-051; mini 0.75->0.771) reproduces on the
  standard split (val 0.712->0.741).
- PDM-Closed (rule-based) dominates all learned policies (0.969), as expected -- the learned policies
  are mini-trained, so ABSOLUTE CLS is not the headline; the RELATIVE result is.
- The relative conclusion (multimodality net-negative vs deterministic; selector partial recovery)
  is robust to the move from the dev split to the standard Val14 benchmark.

Pending. Per-token interaction-criticality (s_inter, job 8004148) -> the H1 retest via
analyze_val14.py: does the WTA-det gap vary with interaction-criticality, with scenario_type fixed
effects + wild-cluster bootstrap + TOST? Phase 2 (SMART reactive agents) then tests whether this
conclusion changes under realistic agents (the realism axis), and whether CLS-under-IDM predicts
CLS-under-SMART.


## ADR-056: Phase-1 H1 retest on Val14 under IDM -- the NULL replicates (TOST equivalence)
Date: 2026-06-30. Status: result. Phase-1 IDM arm COMPLETE.

Pre-registered inference: CLS difference regressed on per-token interaction-criticality s_inter,
with scenario_type FIXED EFFECTS, wild-cluster bootstrap (B=9999, clustered by scenario_type, 33
clusters), and TOST equivalence (margin 0.05), on Val14 (n=1118). Moderator s_inter from
f4_score (peak over scene samples); distribution skewed high (mean 0.67, p50 0.97, p90 1.0).

Contrasts:
  MULTIMODALITY (WTA - det): mean_delta -0.0928; slope beta1 +0.0168 (se .0135, t 1.25, p2 .244);
    TOST equivalent=True, CI90 [-0.005, +0.039].
  SELECTOR (sel - det):      mean_delta -0.0639; slope beta1 +0.0014 (p2 .941); TOST equivalent.
  SELECTOR vs WTA (sel - wta): mean_delta +0.0289; slope beta1 -0.0154 (t -1.71, p2 .109); TOST equiv.

Reading (brutally honest):
- The H1 NULL REPLICATES on the standard Val14 split under IDM. The multimodality deficit (WTA is
  -0.093 below det) does NOT vary with interaction-criticality: slope p=.244, and TOST establishes
  EQUIVALENCE within +/-0.05. Multimodality underperforms BROADLY, not specifically in
  interaction-critical scenes. This is a strong null (equivalence, n=1118), consistent with the
  mini-split H1 (ADR-029/045); it is NOT revived by CLS engineering.
- The closed-loop selector improves on WTA (+0.029) but does not reach deterministic (-0.064), and
  its advantage shows no s_inter moderation either.

Caveat: s_inter is skewed high (the over-fire tendency, tasks #4/#8), limiting slope power in the
high-interaction region; the low tail provides identifying variation and TOST equivalence holds, but
a less-saturated moderator would strengthen the test. A single-representative-sample s_inter (faster,
less saturated) is the planned refinement if the moderation is revisited.

Phase-1 status: IDM arm COMPLETE -- CLS headline (ADR-055) + this H1-retest moderation. The remaining
open question is the REALISM AXIS (Phase 2): does the null hold under SMART learned reactive agents
(the principled reason it might change, per Hagedorn arXiv:2510.14677), and does CLS-under-IDM predict
CLS-under-SMART? Phase 2 is gated on Steffen Hagedorn providing the nuPlan SMART tokenizer/codebook.


## ADR-057: Phase-1 Test14-hard under IDM -- CLS pattern replicates; moderation inconclusive (s_inter saturated)
Date: 2026-06-30. Status: result. Phase-1 IDM standard-benchmark coverage COMPLETE (Val14 + Test14-hard).

Setup. Zoo evaluated closed-loop on the canonical Test14-hard split (272 scenarios, from the nuPlan
TEST split, pulled via public S3) under IDM reactive agents. Mini-trained checkpoints (ADR-054 caveat),
features built live (ADR-053).

Per-config Test14-hard CLS:
  PDM-Closed  0.907 (n=272, 8/8)
  det         0.746 (n=272, 8/8)
  selector    0.693 (n=204, 6/8)
  WTA         0.654 (n=238, 7/8)
  (wta/sel had a few sim shards still finishing at write time; CLS was stable across shard accrual
   and the ordering/conclusion is locked.)

Reading (brutally honest):
- The CLS pattern REPLICATES Val14: PDM > det > selector > WTA. Multimodality underperforms
  deterministic on the hard split too (WTA - det approx -0.09 to -0.10); the selector recovers part of
  the WTA deficit (+0.04) but does not reach det. Everything drops relative to Val14 (Test14-hard is
  harder), PDM still dominates. The relative conclusion is robust across BOTH standard splits.
- Moderation on Test14-hard is INCONCLUSIVE: WTA-det slope on s_inter beta1 ~ -0.065 (p ~ .24), TOST
  NOT equivalent (CI90 wide, ~[-0.15, +0.03]). Honest cause: s_inter is SATURATED on the hard split
  (mean 0.77, p50 0.996) -- Test14-hard scenes are near-uniformly high-interaction, so there is too
  little moderator variance (and smaller n) to test moderation. Test14-hard is the right split for the
  CLS comparison but NOT for the moderation test. The Val14 moderation (equivalence, ADR-056) remains
  the primary H1 result.

Phase-1 IDM arm COMPLETE across both standard splits (Val14 ADR-055/056 + Test14-hard here). Remaining
open axis: Phase 2 (SMART learned reactive agents) -- does the multimodality/interaction conclusion
change under realism, and does CLS-under-IDM predict CLS-under-SMART? Gated on Steffen's nuPlan
tokenizer/codebook.

### ADR-057 FINAL (8/8, n=272 each): PDM 0.9074, det 0.7463, selector 0.6951, WTA 0.6622. WTA-det=-0.084. Conclusion unchanged (multimodality underperforms; PDM>det>sel>WTA).


## ADR-058: Selection-bottleneck confirmed on the standard split (Val14) -- oracle-vs-selector gap is large
Date: 2026-06-30. Status: result. Selection-bottleneck study, step 1.

Per-mode closed-loop eval on Val14 (n=300 subset), WTA modes 0..5 forced via WTA_MODE_INDEX, IDM.
Per-mode CLS: [0.711, 0.674, 0.747, 0.796, 0.792, 0.657].
  oracle (best-of-6)   0.8679
  det                  0.8095
  learned selector     0.7471
  WTA (default score)  0.7050
Gaps:
  latent value  (oracle - det)              +0.058   modes contain trajectories that beat deterministic
  realized      (selector - det)            -0.062   the learned selector is BELOW det
  UNREALIZED selection gap (oracle - sel)   +0.121   selector realizes ~none of the latent value
  frac scenes some mode beats det:           0.48

Reading. The multimodality deficit is a SELECTION failure, not a modes failure. On the standard split
the modes carry real latent value (oracle > det by +0.058; ~half of scenes have a mode that beats det),
but the learned selector captures none of it (0.121 below oracle, 0.062 below det). This replicates
ADR-049 (old subset) on Val14 and confirms the selection-bottleneck framing that Steffen Hagedorn
independently flagged.

Caveats: n=300 Val14 subset; mini-trained checkpoints (provisional numbers). Full-Val14 + retrain pending.

GATE (half 1 of 2): the oracle-vs-selector gap HOLDS strongly on the standard split. Next: the mechanism
(cl_corr -- do pre-decision scene features predict per-mode closed-loop CLS on Val14? prior r~0.11). If
the mechanism holds too, commit to the full-train retrain for credible numbers.


## ADR-059: Selection mechanism -- feedforward mode selection is worse than random; GATE PASSED
Date: 2026-06-30. Status: result. Selection-bottleneck study, step 2 (mechanism) + gate.

On Val14 (n=300), per-mode CLS means: [0.711, 0.674, 0.747, 0.796, 0.792, 0.657]
  -> random-mode baseline = mean of per-mode means = 0.730.
Comparison on the same tokens:
  oracle (best-of-6)             0.868
  det                            0.810
  learned CLS-selector           0.747   (barely above random)
  RANDOM mode                    0.730
  default WTA (imitation score)  0.705   (BELOW random)

Mechanism. The imitation score head ranks modes WORSE than random for closed-loop outcome
(0.705 < 0.730), i.e. its confidence is anti-informative for closed-loop selection. A selector
trained directly on closed-loop CLS labels barely beats random (0.747) and remains 0.121 below the
oracle. So closed-loop mode-value is not recoverable from pre-decision signals -- feedforward
selection is the bottleneck. This is consistent with GATE-CL-1 (open-loop proxy vs closed-loop CLS,
r=0.11) and it is the novel, mechanistic part of the contribution.

GATE (both halves): PASSED.
  half 1 -- oracle-vs-selector gap +0.121 on the standard split (ADR-058).
  half 2 -- mechanism: feedforward mode selection <= random; purpose-trained selector still 0.121
            below oracle.
-> Proceed to the full-train retrain (credibility step), then the selection study becomes the paper's
core; SMART realism is the later robustness axis.

Caveats: n=300 Val14 subset; mini-trained checkpoints (provisional). The retrain is the credibility step
and the reviewer-critical requirement.


## ADR-060: Retrain data pipeline de-risked -- exact train-split keys/sizes confirmed on public S3
Date: 2026-07-10. Status: readiness. Selection-bottleneck study, pre-step-3 (retrain gate).

Before committing the full-train retrain, verified the download path end-to-end WITHOUT downloading:
S3 (bucket motional-nuplan, region ap-northeast-1, UNSIGNED via cluster proxy 10.99.0.130:3128)
lists cleanly; dl_split.py head_object path is correct. Exact train-split objects:
  public/nuplan-v1.1/nuplan-v1.1_train_boston.zip       35.5 GB
  public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip   28.5 GB
  public/nuplan-v1.1/nuplan-v1.1_train_singapore.zip    32.6 GB
  (vegas_1..6 skipped -- ~850 GB, off-distribution-heavy, low marginal value)
Subset bos+pitt+sing = 96.6 GB compressed (~150 GB extracted).

Implication for the retrain decision: a SCOPED de-risk run (Boston only, 35.5 GB) validates the full
pipeline (download -> extract -> token-targeted f0 -> train det+wta+selector -> selection study) on
real train data in days, not the ~1-1.5 wk the full subset costs on shared A100s. If Boston-only
reproduces the standard-split gate (oracle-vs-selector gap + feedforward-unpredictability mechanism),
scale to the full subset with confidence; if it does not, the result is fragile and 1+ wk of cluster
time is saved. This is the capital-efficient path and does not commit the large spend prematurely.

Gate status unchanged (ADR-058/059 PASSED). Retrain GO/NO-GO is Parv's call (shared-cluster capital +
paper-strategy bet); surfaced with recommendation = scoped Boston-first de-risk. SMART realism axis
(Hagedorn) remains the later differentiator, blocked on code release (no timeline).


## ADR-061: Scoped Boston-first de-risk retrain -- APPROVED; verified pipeline recorded
Date: 2026-07-10. Status: in-progress. Selection-bottleneck study, step 3 (retrain), scoped.

Parv approved the scoped Boston-first de-risk (over full bos+pitt+sing up front): download Boston only,
run the full pipeline on real train data, and scale to the full subset ONLY if the standard-split gate
(oracle-vs-selector gap + feedforward-unpredictability mechanism) reproduces. Capital-efficient; caps
downside at ~1 city if the result is fragile.

VERIFIED pipeline (reconnaissance this session, corrects an earlier wrong assumption):
- Training does NOT read raw DBs live. The production trainer nuplan/training/train_policy.py reads
  PRE-EXTRACTED f0 feature shards via F0ShardDataset(--data-root). (train_diffusion_policy.py with its
  hardcoded Mac DB_DIR is an older/unused path -- ignore it.)
- So f0 extraction over Boston is a REQUIRED step and the real bottleneck, not the download.
- Mini training set reference: features/f0_v3 = 43 GB, 17 shard-task dirs.
- Trainer args: train_policy.py --head {det,diff,wta} --n-modes 6 --goal {route,precise}
  --data-root <f0dir> --ckpt-dir <out> --epochs 150 (GPU partition, pytorch_env, ABSOLUTE python).
- Selector (closed-loop mode selection, rl_clsel.pt / rl_long_s6000.pt) is trained on top of the frozen
  WTA head by nuplan/analysis/wta_derisk_train.py + clo_selector.py.
- Eval split held CONSTANT at Val14 (val_stage already downloaded); only training data changes
  mini -> Boston, isolating the data-quality effect. s_inter for Val14 already extracted (f0_val).

Ordered steps:
  1. [LAUNCHED] Download+extract Boston (job 8271334, dl_boston.sbatch -> /scratch/.../nuplan/train_stage).
  2. f0 extraction over Boston scenarios -> features/f0_boston (scope: match/modestly exceed mini scale
     for a clean same-scale mini-vs-real comparison; balanced sampling via --num-scenarios-per-type).
  3. Train det + wta(6) on f0_boston (train_policy_array-style, GPU); then train the selector (rl_clsel)
     on the frozen Boston WTA head.
  4. CLS-select best epoch (cls_select on frozen disjoint probe).
  5. Re-run the selection study (per-mode oracle + zoo) on Val14 with Boston checkpoints; compare gaps to
     ADR-058/059. Reproduce -> scale to full subset; not -> report fragility (still a real finding).

Files added: nuplan/slurm/dl_boston.sbatch. Gate ADR-058/059 unchanged. SMART axis still later/blocked.


## ADR-062: Boston de-risk retrain step 2 -- matched-scale f0 extraction launched
Date: 2026-07-10. Status: in-progress. Selection-bottleneck study, step 3 (retrain), Boston de-risk.

Step 1 (download+extract) DONE: job 8271334, 1647 Boston log DBs at
/scratch/patodia.pa/nuplan/train_stage/data/cache/train_boston (35.5 GB compressed).

Scope decision (matched-scale, isolates data SOURCE from scale): built a 64-log Boston subset
(boston_sub64/, symlinks to the first 64 sorted DBs) to MIRROR mini's 64 DBs exactly. f0 extraction
uses the IDENTICAL f0_v3 recipe -- 16 array tasks (4 DBs/task), --num-scenarios-per-type 20 --stride 5
--perturb-prob 0.5 -- changing ONLY the data root mini -> real Boston train-split. So a reproduced gate
at matched scale is a clean "real data reproduces" signal; the full bos+pitt+sing run later adds scale.
WHY subset not all 1647: the balanced per-type cap is per-task, so all-Boston (~103 DBs/task) would
blow up both wall time (linear in DBs scanned) and cache size; matched-scale is the disciplined first cut.

Smoke note: an interactive login-node smoke was SIGKILLed at 9 s (rc=137) when the Boston map layer
loaded -- a login-node cgroup artifact, NOT a pipeline fault (the builder enumerated Boston scenarios
fine before the kill). Extraction runs on compute nodes (16 G/task). Task 0 of the array is the canary.

LAUNCHED: job 8271900 (f0_boston_array.sbatch, array 0-15) -> features/f0_boston. Next: verify task 0
writes shards, then train det + wta(6) + selector on f0_boston (train_policy.py, GPU), CLS-select, and
re-run the selection study on Val14. Files added: nuplan/slurm/f0_boston_array.sbatch.


## ADR-063: Boston de-risk training scope -- core (det+wta) first, selector deferred to scale-up
Date: 2026-07-10. Status: prepped. Selection-bottleneck study, step 3 (retrain), Boston de-risk.

The full selection-study training chain has three parts: det (train_policy.py), the 6-mode WTA head
(wta_derisk_train.py: SceneEncoder+WTAHead, 16000 scenes, 4000 steps), and the closed-loop SELECTOR
(clo_selector.py retrains ONLY the score head on per-mode closed-loop CLS labels from sim -- which
first requires running the WTA head per-mode through closed-loop on TRAIN tokens to produce those
labels; the base scorer comes from rl_train.py).

De-risk sequencing (capital-efficient): the GATE's core reproduces from det + wta ALONE, evaluated on
Val14 --
  - latent value: per-mode oracle (WTA_MODE_INDEX 0..5) > det?
  - mechanism half: default WTA imitation-score selection <= random mode?
These need no selector. The purpose-trained selector (rl_clsel: rl_train + per-mode train-CLS labels +
clo_selector, the expensive sim-heavy chain) is the REFINEMENT (ADR-058's "even a CLS-trained selector
leaves +0.121") -- add it in the scale-up ONLY if the core reproduces on real Boston data.

Prepped sbatches (both gated on f0_boston completion; GPU/pytorch_env, absolute python):
  train_boston_det.sbatch -- train_policy.py --head det --goal route --data-root f0_boston, 150 ep,
      patience 9999, ckpt-every 25 -> runs/boston_derisk (mirrors f5 det_route, only data changes).
  train_boston_wta.sbatch -- wta_derisk_train.py --n-modes 6 --steps 4000 --n-scenes 16000 (mini scale)
      --shard-glob f0_boston -> runs/boston_derisk/wta_boston_s4000.pt.
Shard layout confirmed identical to f0_v3 (task_NNNN/scene_shard_*.pt), so both trainers find Boston data.

Next after training: eval on Val14 -- per-mode array (WTA_MODE_INDEX 0..5, Boston wta ckpt) + det zoo ->
oracle vs det vs default-WTA vs random; compare gaps to ADR-058/059. Reproduce -> add selector + scale to
full subset; not -> report fragility (still a real finding). Files added: train_boston_{det,wta}.sbatch.


## ADR-064: Boston f0 extraction healthy -- empty-task diagnosis + scale confirmed
Date: 2026-07-10. Status: monitoring. Selection-bottleneck study, step 3, Boston de-risk.

Mid-run check of job 8271900 (f0_boston): 10/16 tasks productive; 6 (3,4,5,7,10,15) exited
"No scenarios found -- exiting cleanly." Diagnosis: the 64-log subset was the FIRST 64 sorted logs,
which are temporally clustered (same recording sessions veh-28/veh-40, Aug 18-20); some consecutive
segments carry no tagged scenarios (nuPlan scenario-type tags are sparse), so num_scenarios_per_type
found nothing there. NOT a pipeline fault -- the 10 productive tasks extract normally.

Scale is NOT a concern: each shard holds 616 samples (measured), so ~42 shards so far = ~25k samples
and growing; the trainers cap at 16000 (wta) / batch over all (det), so abundant. The empty tasks only
mildly reduce log DIVERSITY (~40 distinct logs of real Boston data), acceptable for the de-risk. If the
core reproduces, the full bos+pitt+sing scale-up restores full diversity (and a RANDOM log sample, not
first-N-sorted, will avoid the clustering).

Fixed a cosmetic sbatch bug: the trailing "ls TASK_OUT/*.pt | wc -l" tripped pipefail on empty dirs,
marking cleanly-skipped tasks FAILED 2:0. Switched to find (exits 0 on no match). Real extraction was
always fine. Next: let f0 finish, then launch train_boston_det + train_boston_wta.


## ADR-065: Boston f0 finalized; det+wta training launched
Date: 2026-07-10. Status: in-progress. Selection-bottleneck study, step 3, Boston de-risk.

f0_boston finalized: 14/16 tasks completed; the 2 slow tasks (12,14, at 3:49 on rich logs) were
cancelled for a clean stable snapshot -- their marginal data is unneeded. Final set = 180 shards / 11 GB
= ~110k samples (616/shard), 7x the 16k the trainers cap at. Data path de-risked by construction:
f0_boston layout is identical to f0_v3 (proven with train_policy.py in the f5 run).

LAUNCHED (GPU, detached):
  8277524 bos-det -- train_policy.py --head det --goal route --data-root f0_boston, 150 ep -> runs/boston_derisk
  8277525 bos-wta -- wta_derisk_train.py --n-modes 6 --steps 4000 --n-scenes 16000 -> runs/boston_derisk/wta_boston_s4000.pt
Both PD (Priority) awaiting a GPU. Next fire: confirm they start + train; on completion, eval the
selection study on Val14 (per-mode WTA_MODE_INDEX 0..5 with the Boston wta ckpt + det zoo) and compare
oracle/det/default-WTA/random gaps to ADR-058/059. Reproduce -> add the selector + scale to full subset.


## ADR-066: Boston WTA trained; per-mode Val14 eval launched (ckpt compat verified)
Date: 2026-07-10. Status: in-progress. Selection-bottleneck study, step 4, Boston de-risk.

Boston WTA training DONE: wta_boston_s4000.pt (loss 5.26 -> 0.177 over 4000 steps on 16k Boston scenes).
Boston DET training running cleanly (job 8277524, ~171s/epoch, minADE 3.97 -> 0.32 by epoch 8; ~7h to
epoch 150, fits the 7:55 wall).

Ckpt-compat verified before launch (avoided a load-failure job): the eval planner
(nuplan/serving/policy_planner.py load_ema_into) uses ONLY ckpt["encoder"] + ckpt["head"]; it ignores
the "rl" key that rl_long has and wta_boston lacks. WTA path returns per-mode (trajs, scores) from the
head itself: WTA_MODE_INDEX forces a mode (oracle); default = scores.argmax() = the "imitation-score"
default-WTA of ADR-058. So wta_boston fully supports oracle + default-WTA + random on Val14.

LAUNCHED (parallel with det training): job 8278183 (permode_boston_array, array 0-23%16) -- per-mode
closed-loop CLS on the SAME 300-token Val14 subset as ADR-058, reactive IDM, goal route, with the
Boston WTA head -> eval/permode_boston_r1. Next: when det finishes, run the det zoo eval on the same
tokens, then analyze_permode -> compare Boston oracle/det/default-WTA/random gaps to ADR-058/059
(mini). Reproduce -> add selector + scale to full subset; not -> report fragility.
Files added: nuplan/slurm/permode_boston_array.sbatch.

## ADR-067: Boston de-risk zoo eval -- default-WTA launched, det pending
Date: 2026-07-10. Status: in-progress. Step 4.
boston_zoo_array.sbatch (2 cfgs x 4 shards on val14_sub300, IDM): cfg0 det_route (Boston det best.pt),
cfg1 wta_route (Boston WTA DEFAULT argmax = imitation-score default-WTA). Launched cfg1 now (job 8279921,
array 4-7). cfg0 (det, array 0-3) launches when det training finishes (best.pt). Together with
permode_boston_r1 (oracle+random) -> full core: oracle/det/default-WTA/random vs ADR-058/059. sel deferred.
det 8277524 at epoch ~42 minADE 0.126; per-mode 8278183 running (closed-loop, few h).

## ADR-068: Boston closed-loop eval validated (sane per-mode CLS)
Date: 2026-07-10. Status: monitoring. Step 4.
Partial per-mode Boston CLS on Val14 sub300: mode means 0.625-0.638, all in [0,1], mode0 complete n=300.
Confirms the Boston-trained WTA closed-loop eval produces sane numbers (not garbage). Early/non-conclusive:
Boston per-mode means cluster ~0.63, tighter+lower than mini per-mode [0.657,0.796] (ADR-058); the oracle
(best-of-6) + det + default-WTA decide the gate and are still running. det 8277524 epoch 63 minADE 0.094;
modes 4,5 + default-WTA (8279921) still in closed-loop. det zoo (array 0-3) launches on final best.pt.

## ADR-069: det zoo chained via SLURM dependency (autonomous advance)
Date: 2026-07-10. Status: monitoring. Step 4.
per-mode modes 0-3 done (4/4); modes 4,5 + default-WTA (8279921) still in closed-loop (~1-1.5h). det
8277524 epoch 71 minADE 0.085 (~4h to epoch 150, ~40min wall margin). Queued det zoo 8281301
(boston_zoo det_route array 0-3) with --dependency=afterany:8277524 -> auto-launches on det end using
final best.pt (afterany so it runs even if det times out; boston_zoo guards on best.pt existing). When
per-mode(0-5)+det+default-WTA all land: run analyze_permode_boston.py permode_boston_r1 boston_zoo_r1.


## ADR-070: Boston de-risk PARTIAL -- selection signal looks WEAK; recipe confound flagged
Date: 2026-07-10. Status: preliminary (det pending). Selection-bottleneck study, step 4, Boston de-risk.

Partial core on real Boston data, Val14 sub300, n=300 (all 6 modes + default-WTA done; det pending):
  per-mode means  {0:0.629, 1:0.639, 2:0.625, 3:0.636, 4:0.600, 5:0.640}   (TIGHT, ~0.60-0.64)
  ORACLE (best-of-6)   0.678
  default-WTA          0.647
  RANDOM               0.628
  default-WTA vs random  +0.019   [mini ADR-058: -0.025, i.e. BELOW random]
  oracle vs default-WTA  +0.031   [mini: +0.163]
  oracle vs random       +0.050

HONEST READ (preliminary): the dramatic mini selection-bottleneck signal does NOT cleanly reproduce at
matched scale on real Boston data. (1) The headline mini MECHANISM -- imitation-score selection WORSE
than random -- FLIPS: on Boston default-WTA is slightly ABOVE random (+0.019). (2) The oracle headroom
COLLAPSES: modes cluster tightly (0.60-0.64), so best-of-6 barely exceeds random (+0.050) or default-WTA
(+0.031), vs mini's +0.163 oracle-over-default-WTA. Fewer diverse "good modes" on real data.

CONFOUND (leading caveat, must not be buried): mini's per-mode + default-WTA used the rl_long_s6000
(RL-trained scorer) lineage; the Boston de-risk used plain wta_derisk_train (no RL scorer). So this
changes DATA and RECIPE together -- not a clean isolation. The score-head calibration differs by KIND,
which alone could move default-WTA-vs-random. A clean test needs the SAME recipe (rl_long) on Boston.

STILL PENDING before any gate judgment: (a) det (oracle-vs-det latent value -- the PRIMARY claim; det
8277524 at epoch ~103/150, ~2h; det zoo 8281301 chained). (b) cl_corr mechanism (recipe-independent;
job 8282240 launched -> pearson(open-loop proxy, closed-loop CLS) vs mini r=0.11).

Do NOT conclude reproduction or non-reproduction yet. If, after det + cl_corr, the signal stays weak,
surface to Parv the gate decision: clean rl_long-on-Boston reproduction vs proceed-to-full vs reconsider.
n=300, Boston-only, matched-scale, single seed, plain-WTA recipe -> provisional.


## ADR-071: Boston MECHANISM reproduces (cl_corr r=0.071) -- novel claim survives on real data
Date: 2026-07-11. Status: result (det still pending). Selection-bottleneck study, step 4, Boston de-risk.

cl_corr on real Boston models (Boston WTA head, Val14 f0 features, Boston default-WTA deployed CLS,
n=300): pearson(open-loop proxy reward, closed-loop CLS) = 0.071, spearman = -0.00 (proxy_mean -0.385,
cls_mean 0.647). Mini (GATE-CL-1/ADR-048) was r=0.11. => the open-loop proxy is ~orthogonal to closed-
loop CLS on REAL data too; the feedforward-unpredictability MECHANISM REPRODUCES (even slightly lower r).

This is the recipe-INDEPENDENT test (uses the reward proxy on the deployed top-scored mode, not the
score-head calibration), so unlike the default-WTA-vs-random magnitude (ADR-070, confounded by
wta_derisk vs rl_long) it is a clean reproduction. It is exactly the "novel part" per the honesty note:
the closed-loop-unpredictability mechanism. It HOLDS on real Boston training data.

Coherent honest picture so far:
- MECHANISM (open-loop ⊥ closed-loop, feedforward can't predict per-mode closed-loop value): reproduces
  robustly (Boston r=0.071 ~ mini 0.11; near-zero spearman).
- GAP MAGNITUDE (oracle headroom; default-WTA-vs-random): weaker/smaller on Boston plain-WTA
  (oracle-default +0.031 vs mini +0.163; default-WTA +0.019 ABOVE random vs mini -0.025). Depends on
  mode diversity + scorer miscalibration, which differ by recipe (confounded).

PENDING: det (oracle-vs-det latent value, PRIMARY claim; det 8277524 epoch 113/150 ~1.75h; det zoo
8281301 chained). After det: full verdict + gate decision to surface to Parv. n=300, Boston-only, single
seed, plain-WTA -> provisional.


## ADR-072: Two confounds framing the imminent oracle-vs-det read (interpret with care)
Date: 2026-07-11. Status: methodology. Selection-bottleneck study, step 4, Boston de-risk.

Before det zoo lands, pin down what the oracle-vs-det number can and cannot say, so it is not misread.

Confound A (recipe): mini per-mode/default-WTA used rl_long_s6000 (RL-trained scorer); Boston used plain
wta_derisk_train (no RL scorer). Affects the default-WTA-vs-random magnitude (ADR-070).

Confound B (training budget, quantified): det (train_policy) = ~1521 steps/epoch x 150 ep = ~228k
optimizer steps (~14.6M sample-views); the Boston WTA head = 4000 steps (256k sample-views). det is
optimized ~57x longer. det converged to minADE 0.070 (very strong); the WTA modes are bounded-budget.
=> if oracle(best-of-6 WTA modes) <= det, that is PARTLY because det is far better optimized, NOT proof
that "the modes carry no latent value / selection is fine." A clean latent-value test needs det and wta
trained to MATCHED budgets.

What is NOT confounded and DID reproduce: cl_corr (ADR-071, r=0.071 ~ mini 0.11) -- it uses the reward
proxy on the deployed mode, independent of recipe and of absolute policy strength. The
feedforward-unpredictability MECHANISM (the novel claim) holds on real data.

Plan after det zoo: report oracle/det/default-WTA/random WITH both confounds explicit; do NOT claim the
gate reproduces or fails from the magnitudes alone. Then surface to Parv the gate decision: whether the
full retrain should use MATCHED recipe+budget for det+wta+selector (the clean design), given (i) the
mechanism reproduces cleanly but (ii) the raw Boston magnitudes are weak+confounded. n=300, single seed,
Boston-only -> provisional. Consolidates ADR-070/071.


## ADR-073: Partial oracle-vs-det = -0.055 (det BEATS oracle) -- confound B decisive; matched-budget WTA launched
Date: 2026-07-11. Status: in-progress. Selection-bottleneck study, step 4, Boston de-risk.

Partial (det zoo shard 2 done, n=75 of 300; all 6 modes + det + default-WTA on those tokens):
  ORACLE (best-of-6) 0.695   det 0.750   default-WTA 0.665   RANDOM 0.641
  oracle - det (LATENT VALUE)  -0.055   [mini ADR-058: +0.058]  -> det BEATS best-of-6 oracle
  frac scenes some mode beats det  0.347   [mini 0.48]
  det - random  +0.109  (det far stronger than the modes)

READ (honest): on real Boston data the plain-WTA modes carry NO latent value over det -- the opposite
sign to mini. BUT this is confound B (ADR-072) biting exactly as pre-registered: det got ~57x the WTA
head's training budget (train_policy 150 ep, minADE 0.070) vs wta_derisk_train 4000 steps. det simply
being a much better-optimized policy explains det >> the under-trained WTA modes. So -0.055 does NOT
cleanly refute latent value; the plain-WTA de-risk is UNINTERPRETABLE on the primary claim.

ACTION (evidence-justified, no longer speculative): launched a MATCHED-BUDGET WTA -- train_policy.py
--head wta --n-modes 6 --goal route on f0_boston, 150 epochs, IDENTICAL trainer/optimizer/schedule as
det (job 8285695, GPU). This removes confound B AND the recipe confound A (same trainer as det) in one
shot. Then re-run per-mode eval with the matched WTA -> a FAIR oracle-vs-det. Both outcomes are clean
findings: matched-oracle > det => latent value reproduces (earlier -0.055 was a budget artifact);
matched-oracle <= det => latent value genuinely does not reproduce on real data (honest negative).

Note: this extends the "scoped de-risk" by ~7h train + ~3h eval, but it is the only interpretable test
of the primary claim and stays far cheaper than the full bos+pitt+singapore retrain. The plain-WTA n=300
verdict (chained, 8283561) still completes as the confounded baseline. Mechanism (cl_corr r=0.071) is
unaffected and still reproduces. n=75 partial, single seed, Boston -> provisional.


## ADR-074: Steffen's code released -- SMART reactive agents integration understood; Phase 2 plan
Date: 2026-07-24. Status: plan. Selection-bottleneck study, Phase 2 (SMART realism axis) UNBLOCKED.

Repo: github.com/boschresearch/interactive-closed-loop (AGPL-3.0). Paper: Hagedorn, Distelzweig,
Condurache, "When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks",
ICRA 2026, arXiv:2510.14677. It is a FORK of nuplan-devkit's `planning` module (setup: rm -rf
nuplan/planning && git clone this repo AS planning) + the SMART repo (rainmaker22/SMART) + a nuPlan-
trained SMART checkpoint (planning/checkpoint/epoch=07_1180.ckpt, token_size 2048, 11 hist / 80 fut @0.1s).

MECHANICS (verified from code):
- SMARTAgents(AbstractMLAgents) is a standard nuPlan OBSERVATION (drop-in for IDMAgents). Every 5 frames
  (0.5s) it runs SMART inference (match_token_map -> sample_pt_pred -> inference -> pred_traj/pred_head),
  then moves each non-ego agent to the predicted pose (velocity from finite diff). Ego is excluded
  (planner-controlled). SMARTWrapper wraps `smart.model.smart.SMART`; SMARTFeatureBuilder tokenizes the
  nuPlan scene (token_size 2048).
- closed_loop_smart_reactive_agents.yaml overrides ONLY /observation: smart_agents; it KEEPS
  two_stage_controller, simulation_closed_loop_reactive_agents metrics, and the reactive weighted-average
  aggregator -- IDENTICAL to the standard IDM closed_loop_reactive_agents. => CLS under SMART is directly
  comparable to our entire IDM-based study (ADR-055..073). The ego planner is orthogonal/overridable.

IMPACT ON OUR WORK (this is the differentiator that lifts us off the workshop ceiling):
- Our result to date is ALL under IDM: modes carry latent value (Boston matched oracle 0.828 > det 0.737,
  +0.091; ADR-073), feedforward selection fails to realize it (imitation-score < random; cl_corr r~0.07).
- Steffen's thesis: IDM misranks planners; SMART shifts rankings. So the novel, non-overlapping question
  our work can now answer: DOES THE MODE-SELECTION BOTTLENECK PERSIST OR CHANGE UNDER REALISTIC (SMART)
  REACTIVE AGENTS? Both outcomes publish: (a) persists -> selection failure is robust, not an IDM artifact
  (strengthens us + extends him); (b) shifts -> agent realism modulates the selection landscape (a new
  finding built directly on his contribution). Early-mover: SMART-in-nuPlan just released.

PLAN (Phase 2):
  1. Stand up on cluster: nuplan-devkit + SMART repo + this fork AS nuplan/planning + `nuplan_smart` conda
     env. Smoke-test closed_loop_smart_reactive_agents on 1 scenario (checkpoint loads, agents roll out).
  2. Drop OUR policy_planner (det/wta/selector) into his fork as a planner config (it's a standard
     AbstractPlanner); load our matched-budget Boston checkpoints.
  3. Re-run the selection study (per-mode oracle 0..5, det, default-WTA, selector, cl_corr) under
     observation=smart_agents on the SAME Val14 sub300, matched-budget ckpts -> directly comparable
     IDM-vs-SMART table.
  4. Compare: oracle-det, oracle-vs-selection gap, cl_corr under SMART vs IDM. Interpret invariance/shift.
RISKS: SMART rollout slower than IDM (autoregressive transformer / 0.5s) -> costlier eval; env/devkit
version pinning; AGPL-3.0 copyleft attaches to any derived code we release (our Phase-2 code should be
AGPL if distributed). Non-blocking for research.
DECISION PENDING (Parv): commit to the ~multi-day env build + planner integration now (recommended), vs
first reproduce his 1-scenario smoke only. Boston IDM de-risk stands independently (matched latent value
reproduced). Push remains MANUAL (public repo).


## ADR-075: IDM Boston de-risk COMPLETE (matched-consistent) -- gate reproduces on real data
Date: 2026-07-24. Status: result. Selection-bottleneck study, step 3 (Boston de-risk), FINAL matched.

Final matched-consistent verdict on real Boston data, Val14 sub300, n=300 (matched-budget WTA head for
oracle/random/default-WTA; det = train_policy 150ep; all matched trainer/budget):
  per-mode means {0:0.756, 1:0.750, 2:0.770, 3:0.789, 4:0.796, 5:0.777}
  oracle 0.8276   det 0.7365   default-WTA 0.7771   random 0.7731
  latent value  (oracle - det)              +0.0911   [mini +0.058]
  unrealized    (oracle - default-WTA)      +0.0505
  selection     (default-WTA - random)      +0.0041   ~= random (NOT below)
  frac scenes some mode beats det            0.590     [mini 0.48]
  mechanism cl_corr pearson(open-loop, CLS)  0.071     [mini 0.11]

HONEST READING (supersedes the plain-WTA confounded ADR-070/073 for the final claim):
- LATENT VALUE reproduces strongly on real data and is LARGER than mini (+0.091 vs +0.058); 59% of
  scenes contain a mode beating det; even random-mode (0.773) beats det (0.737). The multimodality
  deficit is NOT a modes problem on real data.
- SELECTION is the bottleneck: the imitation-score head selection (0.777) is statistically ~random
  (+0.004) and leaves +0.051 of oracle headroom unrealized. The mini "worse than random" (-0.025) was an
  undertraining artifact; the well-trained matched head is simply UNINFORMATIVE for closed-loop selection
  -- a cleaner, more defensible claim, consistent with cl_corr r=0.071 (open-loop ~orthogonal to CLS).
- The earlier "det beats oracle" (-0.059) was purely the ~57x budget confound (ADR-072/073), now removed.

CONCLUSION: the mode-selection bottleneck REPRODUCES on real nuPlan training data under IDM. This is the
solid IDM foundation for Phase 2 (SMART realism axis, ADR-074). Provisional caveats: n=300 subset, single
seed, Boston only; the full bos+pitt+singapore + multi-seed is the paper-grade version. Gate: PASSED on
real data.


## ADR-076: Phase 2 integration recipe -- plug our PolicyPlanner into Steffen's SMART sim
Date: 2026-07-24. Status: recipe (env building, job 8693723). Selection-bottleneck Phase 2.

Verified both sides:
- OUR PolicyPlanner (av-policy-lab/nuplan/serving/policy_planner.py) is a std AbstractPlanner. Deps:
  stock nuplan.common/.planning.simulation APIs (his fork provides) + OUR top-level pkgs
  features.scene_features, models.{policy_heads,scene_encoder,samplers,f0_dataset}. Builds features LIVE
  at sim time (no cache). Loads our matched-budget Boston ckpts (det/wta) via encoder+head state_dict.
- HIS fork registers planners via a yaml with `_target_: <class>` (+ params), e.g. ml_planner.yaml,
  simple_planner.yaml, smart_planner.yaml, under
  smart-stack/nuplan-devkit/nuplan/planning/script/config/simulation/planner/.

INTEGRATION STEPS (once nuplan_smart env is up):
  1. SMOKE his stack alone (no our-code): run_simulation.py +simulation=closed_loop_smart_reactive_agents
     planner=simple_planner scenario_filter=<1 Val14 token> scenario_builder=nuplan, NUPLAN_DATA_ROOT=our
     val_stage. Confirms checkpoint loads + SMART agents roll out on OUR data.
  2. Register our planner: PYTHONPATH += /home/patodia.pa/av-policy-lab/nuplan (gives features/, models/,
     serving/) + SMART_ROOT. Add policy_planner.yaml (_target_: serving.policy_planner.PolicyPlanner,
     params head/goal/ckpt). Run his sim with planner=policy_planner observation=smart_agents on 1 token.
  3. Scale: replicate our per-mode (WTA_MODE_INDEX 0..5) + det + default-WTA under observation=smart_agents
     on Val14 sub300, SAME matched Boston ckpts -> IDM-vs-SMART comparable table. + cl_corr under SMART.

RISKS: (a) devkit VERSION skew -- his cloned nuplan-devkit may differ from the one our PolicyPlanner /
SceneFeatureExtractor were written against; AbstractPlanner/DetectionsTracks/scenario API drift could
break our planner (find at step 2). (b) torch 2.1.0 (his env) vs 2.4.1 (our training) loading our
state_dicts -- low risk. (c) WTA_MODE_INDEX override path must exist in the planner code we deploy here.
(d) SMART rollout slow -> keep to sub300. Reference IDM result: ADR-075 (oracle 0.828 > det 0.737 +0.091,
selection ~random, cl_corr 0.071).

## ADR-077: nuplan_smart env needed base devkit deps (his requirements.txt gap)
Date: 2026-07-25. Status: setup. Phase 2.
Env build (8693723) COMPLETED (torch 2.1.0+cu121, pyg 2.6.1, pl 1.3.8). But import check failed:
ModuleNotFoundError shapely -- his planning/requirements.txt omits the BASE nuplan-devkit deps
(shapely, geopandas, Fiona, rasterio, hydra-core==1.1.0rc1, rtree, SQLAlchemy...). Fix: install
nuplan-devkit/requirements.txt into nuplan_smart (job 8694572, detached compute), re-pin numpy 1.23.4 +
urllib3 1.26.20 after. hydra 1.1.0rc1 from that req is also REQUIRED for nuplan configs. Next: re-verify
imports (shapely+geopandas+hydra+torch+pyg+smart), then SMOKE closed_loop_smart_reactive_agents on 1
Val14 token. Stack /home/patodia.pa/smart-stack; PYTHONPATH=nuplan-devkit:SMART; NUPLAN_DATA_ROOT=val_stage.

## ADR-078: nuplan_smart env IMPORT-CLEAN; SMART smoke launched
Date: 2026-07-25. Status: in-progress. Phase 2.
Env now imports cleanly: core (shapely/geopandas/hydra 1.1.0rc1/torch 2.1.0+cu121/pyg/pl) + his SMARTAgents
observation + smart.model.smart.SMART. Fixes applied beyond his README: (a) installed base nuplan-devkit
deps filtered to drop jupyter/testbook/dev lines that pulled nbclient>=py3.10 (ADR-077); (b) downgraded
tensorboard 2.21->2.11.2 to fix protobuf 3.20.3 runtime_version ImportError (tensorboard only via PL
logger, not the sim); (c) STUB for waymo_open_dataset.protos.sim_agents_submission_pb2 at
smart-stack/waymo_stub -- SMART imports it ONLY in joint_scene_from_states (Waymo submission), NOT on the
nuPlan inference path; avoids pulling TF 2.12. PYTHONPATH = nuplan-devkit:SMART:waymo_stub.
SMOKE launched (job 8696046): run_simulation.py +simulation=closed_loop_smart_reactive_agents
planner=simple_planner scenario_builder=nuplan db_files=val_stage/data/cache/val scenario_filter=val14_split
limit 1, GPU. Verifies checkpoint epoch=07_1180.ckpt loads + SMART agents roll out on our val DBs.

## ADR-079: scratch PURGED extracted stage DBs; re-extracting val (smoke blocker)
Date: 2026-07-25. Status: fix. Phase 2.
SMART smoke got past ALL imports (env fixes worked) and failed only at scenario_builder: the val DB dir
was EMPTY. Cause: HPC scratch auto-purged the extracted stage DBs -- val_stage/test_stage/train_stage
are all 0 bytes (extracted Jul 10-13; purge ~2wk). SURVIVING: the 216G downloads/ zips (incl
nuplan-v1.1_val.zip 91G), mini data/ (14G), AND critically our matched Boston ckpts
(runs/boston_derisk/{det_route_seed0,wta_route_seed0}/best.pt + wta_boston_s4000.pt) and features/f0_boston
(11G) -- Phase 2 NOT set back. Fix: re-extract val zip (job 8696826) -> val_stage/data/cache/val; no
re-download. train_stage NOT needed (f0_boston already extracted); test_stage not needed for Phase 2.
OPERATIONAL RISK: extracted val DBs will re-purge in ~2wk; keep the SMART eval runs close together, or
touch/re-extract as needed. Next: after 8696826, re-run smoke 8696397-style on the restored val DBs.
