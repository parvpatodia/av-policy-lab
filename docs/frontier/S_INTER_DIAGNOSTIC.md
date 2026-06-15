# S_inter construct-validity diagnostic (pre-freeze)

Status: finding, 2026-06-14. Tool: nuplan/analysis/s_inter_diagnostic.py.
Does NOT change F4. The decision it informs is below.

## Question

Before freezing the eval manifest, does F4's interaction branch S_inter measure
what it claims, genuine interaction-timing ambiguity (a yield-or-go fork against
crossing traffic), or does it over-fire on same-direction traffic and on
rollout artifacts? This was the open question flagged when scene S28
(F4=0.93) looked like ordinary traffic.

## Method

Rerun the exact production s_inter agent loop (imported from f4_score, not
reimplemented) on 250 high-F4 scenes (F4 > 2/3) drawn from local f0_v2 shards.
For each agent that contributes I_j >= 0.05 to the score, record: the band-pass
value, the PrET gap, the agent heading relative to the ego (same <45deg / cross
/ oncoming >135deg), the crossing angle of the two paths, the heading-rate
magnitude, and a counterfactual: **does the crossing survive re-rolling the
agent straight (hr=0)?** If not, the crossing exists only because the
5 s constant-turn-rate rollout bent the agent across the ego corridor. Each
scene is attributed to its top (score-leading) contributor.

## Validity of the tool (checked first)

- Self-test: a constructed orthogonal straight agent is classified `cross`,
  not curvature-only; a same-direction agent given a heading rate that bends it
  across is classified `same`, curvature-only. Both correct.
- Heading-channel gate: for the lead agents, the stored heading agrees with the
  velocity direction to **median 0.1deg (p90 0.9deg)**. So s_inter rolls agents
  along the right direction and the diagnostic is not an artifact of a wrong
  channel assumption.

## Result (250 high-F4 scenes)

| lead contributor | share of scenes | share of score mass |
|---|---|---|
| genuine cross | 5.2% | 5.1% |
| oncoming | 0.8% | 0.7% |
| **same-direction** | **84.0%** | **84.4%** |
| pedestrian override | 10.0% | 9.8% |

- **Curvature-only lead (crossing vanishes when rolled straight): 58.8%** of
  scenes, 58.2% of score mass.
- Lead-agent medians: relative heading **5.1deg**, crossing angle **4.2deg**,
  |heading rate| **1.3deg/s**, PrET gap **2.3 s** (right at the band-pass peak
  of 2.5 s, so near-parallel grazes score ~1.0).

## Reading (split by confidence)

1. **Curvature artifact — high confidence, indefensible.** ~59% of high-F4
   score mass comes from crossings that do not exist if the agent travels
   straight. The median heading rate is only 1.3deg/s; near-parallel geometry
   (crossing angle ~4deg) makes whether two 5 s rollouts intersect
   hypersensitive to tiny curvature, so a noisy 2-point heading-rate estimate
   on quantized 10 Hz history manufactures the conflict. The spec already
   rejected constant-velocity over 8 s as indefensible; constant-turn-rate over
   5 s with a noisy hr is the same failure in the other direction. This is a bug
   independent of any interpretation debate.

2. **Same-direction dominance — needs a scope decision.** 84% of high-F4 leads
   are same-direction near-parallel paths, not cross traffic. Some of these are
   arguably legitimate interaction (a merge or a `starting_right_turn` is a real
   gap-accept-or-yield fork). But a 4deg median crossing angle says most are
   near-parallel grazes (driving behind/beside same-direction traffic), which a
   human would not call a yield-or-go fork. Whether this is "over-fire" depends
   on what S_inter is meant to capture. That is a moderator-definition decision,
   not an engineering one.

## Candidate fixes (for after the decision, not applied)

- Remove the artifact: require the straight-roll (hr=0) crossing to also exist,
  OR shorten the agent rollout to where hr is reliable (2-3 s, not 5 s), OR
  estimate hr robustly (multi-point with a deadband). Cheapest defensible fix:
  cap the agent rollout horizon and add an hr deadband.
- Resolve the scope: add a minimum crossing-angle / relative-heading gate so
  S_inter scores genuinely conflicting paths, and let same-direction route
  choice live in S_branch instead. Only if the decision is "S_inter = cross
  traffic."

## Decision sequence (recommended, ordered)

1. **Do the blind rating first; do not change F4 yet.** The rating is the
   EXTERNAL arbiter of exactly this question (does F4 match human ambiguity).
   The answer key is valid only against the current F4; rescoring now would
   throw the rating away. This diagnostic predicts the rating will show weak
   F4-human correlation concentrated in the high band.
2. With internal (this) + external (rating) evidence in hand, decide the
   S_inter scope and fix.
3. Then: re-score all scenarios (v1.2), re-run the F4 face-validity gate,
   regenerate the rating sheet if the high band changes materially, and only
   then freeze the manifest.

## Honest caveats

- nuPlan mini scale; Vegas-skewed geography; same-direction traffic is common
  here, which is part of why the artifact bites.
- "Same-direction" includes some legitimate merges; the 84% is not all false
  positive. The 59% curvature-only number is the clean, interpretation-free
  indictment.
- Not yet run: a control on the low/med bands to confirm S_inter scores
  path-graze prevalence rather than interaction type. The rating largely
  subsumes this; run the control if the rating is ambiguous.
