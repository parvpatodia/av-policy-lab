# SPEC — Positive experiment: a genuinely multimodal policy + unsaturated eval + H1 re-test

## Goal
Convert the negative diagnostic (collapse + saturation) into a negative+positive result: produce a
policy whose treatment (multimodality) is genuinely PRESENT, evaluate on an UNSATURATED metric/slice,
and test whether the multimodal-vs-unimodal benefit grows with interaction-criticality (H1). Either
outcome is publishable: benefit appears (positive H1) OR forced diversity manufactures unsafe/unhelpful
modes (a strong, constructive extension of the negative result).

## Component 1 — a genuinely multimodal policy (supervised WTA only fans: F5 = 2.3% >=2 modes)
Levers, cheapest decisive first:
- (1a) DIVERSITY-REGULARIZED multi-hypothesis head: relaxed-WTA best-of-M regression (modes still fit
  GT) + explicit inter-mode REPULSION (push endpoints apart up to ~lane width) + winner CE. No RL.
- (1b) If (1a) modes become unrealistic (minADE blows up / off-road), add a realism/feasibility term
  (on-road + kinematic), or escalate to RL (DIVER-style GRPO: diversity+safety reward). Heavier.
Acceptance gate (probe, no eval): frac>=2 modes materially up (>~20%) at decision-point types AND
best-of-M minADE within ~1.5x of the unimodal head AND modes stay plausible (endpoints not absurd).

## Component 2 — unsaturated eval
Define a HARD slice from the frozen data: bottom-CLS-quartile scenarios (real headroom, ADR-034
showed frac@ceiling 0 there) + decision-heavy types (intersections, unprotected turns). A NEW policy
needs the closed-loop sim harness to get CLS (multi-day); FIRST gate on open-loop multimodality +
a small closed-loop smoke before committing the full eval.

## Component 3 — H1 re-test
With a present treatment + unsaturated slice, re-run the moderation (analyze_moderation_v2, the
corrected FE + wild-cluster + TOST inference). Report honestly either way.

## De-risk order (no multi-day commit until gates pass)
1. THIS STEP: diversity-reg WTA de-risk — extend the de-risk trainer with a repulsion term, train one
   route cell (16k scenes, bounded), probe modes + best-of-M minADE + plausibility. GATE 1.
2. If GATE 1 passes (distinct + accurate + plausible) -> realism check -> full retrain (route+precise
   x seeds, the fairness-matched cells) -> closed-loop eval on the unsaturated slice -> H1 re-test.
3. If GATE 1 fails (diversity wrecks accuracy/plausibility) -> either add realism constraint / RL, or
   conclude per-scene determinism is fundamental on nuPlan (negative extension, also a result).

## Discipline
Spec-first (this), cheap-verify, fairness-matched training for any cell that enters the comparison,
correct inference, NO multi-day eval until the open-loop gate + a closed-loop smoke pass. Commit no
AI attribution; push from Mac fast-forward.
