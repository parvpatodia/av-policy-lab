# F4 Validation Protocol (pre-registration amendment)

> Amends RESEARCH_PROTOCOL.md (HPC) / F4_SPEC.md. Written 2026-06-20, BEFORE any
> validation result is computed. Design is fixed here so the analysis cannot be
> rationalized post-hoc.

## 0. What F4 is and what "valid" means here

F4 is a per-scenario scalar in [0,1], a pre-treatment moderator of the head effect.
The **frozen** definition (RESEARCH_PROTOCOL.md, ADR-013/016, v1.1; confirmed
against `score_f4.pass_combine` and the deployed `f4_scores_v11.json`) is:

    F4 = G_stop · (1 − (1 − S_branch) · (1 − S_inter))      # noisy-OR; S_lane EXCLUDED

i.e. F4 fires on **two** ambiguity sources, not one:
- **S_inter** (yield-or-go interaction): ego route polyline rolled at constant
  speed, each agent rolled at constant turn-rate, PrET Gaussian around 2.5 s over
  path crossings, plus a stationary-pedestrian override. Geometric; no planner, no
  ego-future (the ego future is excluded as a label leak).
- **S_branch** (lateral/route decision ambiguity): corridor lane-graph branching
  from the map API, S_branch = min(max(B_R−1,0),3)/3.
- S_lane is computed but EXCLUDED from F4 (demoted to a covariate in v1.1).
- G_stop gates out red-light/forced-stop frames.

> Correction (2026-06-20): an earlier working note had F4 = g·s_inter with S_branch
> dropped. That was wrong — only **S_lane** was dropped; **S_branch is a primary
> term**. 949/5604 scenes (17%) have F4>0 driven entirely by S_branch with
> S_inter≈0. Validation must therefore be **component-wise**, not against a single
> "F4 = interaction" assumption.

Per RESEARCH_PROTOCOL lines 132–133 the experiment's H1 weight is on **S_inter**
(CLS, the closed-loop outcome, is partly blind to branch-level ambiguity), and the
protocol already pre-registers **S_branch-only / S_inter-only / reactive-agent
replication** as validation hooks. This protocol fills those hooks.

"Validating F4" = showing each component, and the combined moderator, track scene
ambiguity as measured by methods **independent of F4's own formula** (convergent
validity).

## 1. Why this amendment exists

The pre-registered plan was human pairwise ratings (friends) → a scaled ambiguity
ground truth → correlate with F4. That data collection fell through (no usable
returns). This amendment replaces it with a method that does not depend on
recruiting raters, and corrects a flaw discovered while scoping the originally
proposed replacement.

### 1.1 The discovered flaw (why behavioral-multimodality cannot be the sole primary)

The proposed replacement was "behavioral multimodality": run an ensemble of
independent planners on each scene and treat their **trajectory dispersion** as the
ambiguity ground truth (ambiguous scene ⇒ competent planners disagree). Inspecting
the available independent planners (`nuplan/planners.py`) shows the ensemble is too
ego-centric to measure *interaction* ambiguity:

| Planner | Checkpoint | Input actually consumed | Sees the conflicting agent? |
|---|---|---|---|
| BC | `bc_best.pt` | ego 6-dim kinematics only | **No** |
| MILE | `mile_policy.pt` | ego 6-dim kinematics only | **No** |
| BEV | `bev_cnn.pt` | raster of **ego-only** history (`_rasterize_ego_bev`) | **No** |
| IDM | rule-based | ego + a **direct lead** (`|y_e|>3 m` rejected) | only longitudinal |

Three of four planners are blind to other agents; the fourth handles only
car-following, not the lateral path-crossing conflict that `s_inter` scores. Their
dispersion therefore reflects each model's kinematic-extrapolation bias, not scene
interaction ambiguity. Using it as the primary criterion would manufacture a
correlation (or a null) that says nothing about F4's construct. It is **demoted to
an exploratory lower-bound** (Signal C), reported with this caveat, never as proof.

## 2. Design: component-wise convergent validity

F4 has two primary components, so validation is per-component plus holistic, mapped
onto the protocol's pre-registered hooks. Signal A validates S_inter (the
H1-weighted component); Signal D validates S_branch; the Signal B panel validates
the combined moderator holistically. Each method is independent of F4's formula.

- **Signal A — Ground-truth-conflict convergent (primary, internal).**
  Re-derive interaction conflict using each agent's **logged future trajectory**
  (nuPlan `get_future_tracked_objects`, 5 s horizon) instead of F4's constant-turn-rate
  rollout, crossed against the same ego route rollout. Produces `a_gt` ∈ [0,1] from
  the **same geometric construct but a different, more accurate motion model**.
  Independence: replaces F4's central approximation (constant turn-rate) with reality;
  a correlation shows F4's ambiguity is not an artifact of that approximation. Uses
  agent futures only (never ego future), so no imitation-label leak. Limitation: same
  construct family as F4 (tests robustness-of-operationalization, not an external
  criterion) — which is why B exists.

- **Signal B — Vision-LLM rater PANEL on rendered scenes (primary, external; validates combined F4).**
  Render a top-down image of each scene at iteration 19 (ego, all agents, lanes,
  crosswalks, traffic-light state) and have a **panel of 3–4 distinct judge personas**
  rate OVERALL driving ambiguity 0–1 from a frozen rubric — capturing both "which way
  to go" (S_branch) and "yield or go" (S_inter), so it matches the combined moderator.
  Personas are genuinely different perspectives (e.g. cautious defensive driver,
  assertive driver, traffic-safety analyst, AV-planning expert), each on the same
  rubric, mirroring a multi-rater human study. Report inter-persona reliability
  (Krippendorff α / ICC) and use the panel median as the score. Independence: the
  panel sees **pixels**, not F4's features. Limitations (disclosed): AI personas
  share a base model so their errors are correlated (not truly independent like
  separate humans), hence (a) reliability is reported, not assumed, and (b) the panel
  is **calibrated against Parv's human ratings** on the same stratified sample when
  available — the human set is the ultimate external anchor; the panel is the
  scalable stand-in until then.

- **Signal D — Independent corridor-branch count (validates S_branch).**
  Recompute the number of distinct downstream route branches from the map lane-graph
  with a different traversal than `f4_map_branch.corridor_branching`, and check it
  recovers B_R / S_branch (the protocol's shard-geometry-vs-map-API S_branch check).
  Plus: the panel's "which way" sub-judgment should rise with S_branch.

- **Signal C — Behavioral toy-planner dispersion (exploratory).**
  As in §1.1. Reported with the ego-centric caveat. A positive result is a weak
  bonus; a null is uninformative about F4.

## 3. Scene set, alignment, and the zero-inflation problem

- Set: the 5,604 F4-scored scenarios (`features/f4/f4_scores_v11.json`), keyed by
  nuPlan token, scored at a fixed `iteration = 19`. All 54 source logs are present
  in the local nuPlan-mini split (verified), so every scene is runnable on the Mac.
- All three signals are computed at **iteration 19** to match F4's scoring frame.
- Distribution is heavily zero-inflated: ~65% of scenes have F4 < 0.1 (no crossing
  conflict ⇒ s_inter = 0), ~20% have F4 > 0.5; median 0. A single Pearson/Spearman
  over the whole set would be dominated by the zero mass. Analysis is therefore
  **two-part** and fixed here:
  1. **Group contrast.** Scenes with F4 = 0 ("no conflict") vs F4 > 0 ("some
     conflict"): the validation signal (a_gt / VLM score / dispersion) must be
     higher in the F4 > 0 group. Report effect size (Cliff's δ, rank-biserial) +
     bootstrap 95% CI + Mann–Whitney permutation p.
  2. **Monotone within-conflict.** Among F4 > 0 scenes only, Spearman ρ between F4
     and the signal, with bootstrap 95% CI (resample scenes) + a permutation null
     (shuffle the F4↔signal pairing). One-sided H1: ρ > 0, α = 0.05.

## 4. Pre-registered hypotheses and decision rule

- **H-A:** a_gt is higher for F4 > 0 than F4 = 0 (Cliff's δ > 0, CI excludes 0), and
  Spearman ρ(F4, a_gt) > 0 within F4 > 0.
- **H-B:** VLM ambiguity is higher for F4 > 0 than F4 = 0, and ρ(F4, VLM) > 0 within
  F4 > 0.
- **H-C (exploratory):** dispersion ρ — reported, not used for the verdict.
- **Verdict:** F4 is considered a validated ambiguity moderator iff **both A and B**
  pass the group contrast (CI excludes 0) **and** at least one shows a positive
  within-conflict ρ (CI excludes 0). A mixed result (A passes, B fails or vice versa)
  is reported as partial validity with the specific failure named — not rounded up.

## 5. Confounds controlled (stated before looking)

- **Competence confound (Signal C):** dispersion can rise because a planner is bad,
  not because the scene is ambiguous. Mitigated by demoting C to exploratory and, if
  reported, by excluding degenerate rollouts (off-route / NaN) and noting per-planner
  open-loop error.
- **Difficulty confound (all signals):** ambiguity could co-vary with raw scene
  busyness (agent count, ego speed). For the primary signals, also report the
  group contrast / ρ **partialling out** `n_par` (agent count) and `v0` (ego speed),
  both already in the F4 records, so the claim is "F4 tracks ambiguity beyond mere
  busyness."
- **Multiple comparisons:** two primary signals × two tests = 4. Holm correction
  across the 4 primary p-values.

## 6. Build order

1. `scene_loader.py` — load a scenario by token at iteration 19; expose ego state,
   tracked objects, **logged agent futures**, map, route, traffic lights, and a
   minimal `PlannerInput` for Signal C. De-risk on one scene first.
2. `signal_a_gt_conflict.py` — Signal A over a pilot (~200 scenes spanning the F4
   range), then the full set. First real number.
3. `signal_b_vlm_judge.py` — render + rubric + judge, pilot then full.
4. `signal_c_dispersion.py` — exploratory.
5. `analyze_validation.py` — the §3 two-part stats + §5 confound controls + §4 verdict.

Pilot before full on every signal: confirm the pipeline and the sign of the effect
on a stratified ~200-scene sample before spending the full 5,604.
