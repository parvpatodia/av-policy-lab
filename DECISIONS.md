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
