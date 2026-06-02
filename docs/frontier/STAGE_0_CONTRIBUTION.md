# Stage 0 — The Contribution (research-backed, reframed)

> Synthesis of four literature reviews (LIT_DIFFUSION, LIT_NUPLAN_PLANNING,
> LIT_ENCODER_AND_COVARIATE, THOUGHT_LEADERS_GAP) + HPC_NORTHEASTERN.
> This file defines WHAT we are actually claiming and WHY it is novel. Designs only.

## The genuine gap (what the frontier labs are missing)

The field justifies generative/diffusion planners with one sentence: *"driving is
multimodal, so a deterministic regressor mode-averages and a generative policy is
better."* Three things are wrong with how the field treats this:

1. **It never MEASURES the multimodality** — it is asserted, never instrumented.
2. **It never decomposes the SOURCE** of multimodality (route/intention ambiguity
   vs. agent-interaction ambiguity vs. human-demonstration inconsistency).
3. **It draws these architectural conclusions inside a simulator (IDM reactive
   agents) that was just shown to be unrealistic** and to *re-order the leaderboard*
   when realistic learned agents (SMART, arXiv:2510.14677, Oct 2025) are substituted.

Put together: **the field's architectural conclusions (“diffusion helps”) may be
artifacts of (a) unmeasured/assumed multimodality and (b) an unrealistic simulator.**
Nobody has tested this, because it needs a controlled capacity-fixed rig + the new
SMART agents + a multimodality instrument — none of which leaderboard culture rewards.
This maps onto the thought leaders' #1 ranked anxiety: *we cannot currently trust
closed-loop evaluation* (Geiger/Chitta “Parting with Misconceptions”; “Is Ego Status
All You Need”; “When Planners Meet Reality”).

## The contribution (one sentence)

> Whether a generative (diffusion) policy *appears* superior to a deterministic one
> is contingent on the simulator's agent realism, and the margin is **predictable
> from a directly-measured interaction-multimodality score** — demonstrating that a
> published architectural conclusion can be manufactured or erased by changing only
> the simulator's background agents.

This reframes the project from "which planner wins" (niche; the leaders say the
deterministic-vs-generative *form* axis is not the bottleneck) to a **meta-validity
result about evaluation** (the field's deepest stated gap).

## The experiment: 2×2×2 + an instrumented mediator

| Axis | Levels | Removes / adds |
|---|---|---|
| Goal conditioning | precise near+far point  vs  route-conditioned | route axis removes route/intention ambiguity |
| Policy head (capacity-matched) | deterministic MLP  vs  diffusion | the form axis under test |
| Simulator agents | IDM (standard)  vs  SMART (realistic, 2510.14677) | SMART axis adds agent-interaction ambiguity |

**Mediator (the novel instrument):** a per-scenario *interaction-multimodality score*
— e.g. (i) diversity of the diffusion policy's K samples (pairwise distance / mode
count via clustering), and (ii) an independent interaction-density measure (number of
other agents whose near-horizon predicted paths conflict with the ego route). Logged
for every scenario under both agent models.

**Pre-registered hypothesis:** the diffusion-minus-MLP margin is ~0 when measured
multimodality is low (IDM + precise goal) and grows monotonically with the measured
multimodality (SMART + route goal). I.e. the margin tracks the mediator, not the
architecture per se.

## Why this survives the fatal threat (be honest about it)

**The threat (from THOUGHT_LEADERS_GAP, and Parv's own F2 instinct):** once the ego
route is given, the *ego's* optimal plan may be near-unimodal — the multimodality
lives in *other agents* (prediction), not the ego plan (planning). So SMART realism
might NOT re-introduce ego-plan multimodality, and diffusion might tie the MLP even
under realistic agents.

**Why the reframe is robust to it:** because we **measure** multimodality instead of
assuming it, every outcome is a result:
- If the margin tracks measured multimodality → architectural superiority is a
  simulator artifact (the strong, field-relevant claim).
- If ego-plan multimodality stays ~0 even under SMART → *the field's core
  justification for generative planners is overstated for the ego-planning task* —
  itself a publishable, contrarian, well-grounded finding.

A null result is no longer a failure; it is a claim about the field's assumptions.

## Build order (locked by the research)

```
PREREQ  Explorer account + PI-sponsored /projects allocation (hard blocker)
F0  vectorized scene WITH agents (~32, 2s/20-step), lanes+connectors+crosswalks+route+lights
      — agents are essential: interaction-multimodality cannot exist without them
F1  encoder, ~1–2.5M params (literature scale, NOT max — bigger overfits closed-loop)
F2  the two goal conditionings: precise near+far point  AND  route-conditioned
F3  capacity-matched twin heads: deterministic MLP  AND  temporal cross-attn diffusion
F4  perturbation, then light closed-loop SFT — only AFTER F0/F1 (DAgger-null proved
      closed-loop training is wasted on a road-blind encoder); enough to make the pair
      a FAIR test, not to chase SOTA
SMART  integrate drop-in SMART reactive agents into the nuPlan closed-loop sim
METRIC instrument the interaction-multimodality score
F6  the 2×2×2 eval on Val14 (+ Test14-hard), full PDM decomposition, paired stats,
      margin-vs-mediator regression
```

## Feasibility (sized to reality, not scale)

- **Not compute-bound.** ~2.5M-param planners on a single H200 (140 GB) — trivial.
  The bottleneck is CPU-bound nuPlan feature extraction + closed-loop sim, and the
  8 h job ceiling (checkpoint/resume). Explorer's value is CPU + storage, not FLOPs.
- Inference-only comparison; no RL, no world-model training, no SOTA chase.
- Released checkpoints (PDM-Closed, Diffusion Planner, PLUTO) enable a cheap pilot of
  the SMART-vs-IDM reordering BEFORE building our controlled pair.

## Must-verify before any claim
- Efficient Virtuoso (2509.03658) exact scope (closest prior on the goal axis).
- PlanTF/PLUTO perturbation magnitudes from source (config-vs-default discrepancy).
- Reproduce one released-checkpoint PDM number under IDM before trusting the harness.
- Confirm SMART agents drop-in API + nuPlan-devkit version compatibility.

## Housekeeping flagged by the research
The repo's `context.md` / `DECISIONS.md` still describe the old robotics project and
old baselines, not this AV thesis. Update them so a reviewer reading the repo cold
sees the actual contribution.
