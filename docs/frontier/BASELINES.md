# Baselines: integration validation (smoke, n=4)

Reference planners for the closed-loop results table. All run through the same
run_cells.py / two_stage_controller harness as the trained cells.

- log_future (LogFuturePlanner): replays the expert log. Upper bound / ceiling.
- idm (IDMPlanner): devkit IDM as ego planner. Weak reference.
- pdm_closed (PDMClosedPlanner, tuplan_garage, Apache-2.0): rule-based, the
  planner that beats learned planners on nuPlan closed-loop (Dauner et al.,
  CoRL 2023). Centerline + lateral-offset proposals x batch-IDM longitudinal
  policies, each simulated forward and scored against the nuPlan metric; best
  proposal executed. No learning, no checkpoint. Deps: scikit-learn,
  positional-encodings. Runtime PYTHONPATH must include tuplan_garage.

## Smoke head-to-head (2026-06-14, 4 high-speed scenarios)

| planner                    | mean CLS | notes |
|----------------------------|----------|-------|
| pdm_closed                 | 1.0000   | 4/4 perfect: no collision, on-road, full progress |
| det_route (v2 LEAKY pilot) | 0.8043   | 2/4 drifted off-road + collided (CLS 0.62) |

Reproduces the field's central closed-loop finding at small scale: open-loop
imitation accuracy (our det head matched expert progress 0.93-0.95) does NOT
imply closed-loop quality; a planner that scores candidates against the
simulator's own dynamics avoids the off-distribution drift that sinks a
BC-style network. CAVEATS: n=4 (not significant); the policy here is the v2
leaky-route pilot, NOT v3 (whose recovery perturbation targets this exact
off-road-drift failure mode), so the real gap is expected to be smaller.
The full stratified eval (incl. junction/interaction scenarios where PDM is
less dominant) is the real test.
