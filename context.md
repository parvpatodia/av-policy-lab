# Project Context: av-policy-lab
> Persistent project memory. Read at session start. Last updated: 2026-06-02.

## 1. What this project is
A controlled study of **closed-loop imitation learning for autonomous-driving motion
planning** on the [nuPlan](https://www.nuplan.org/) benchmark. It is a *research /
evaluation-validity* project, not a product and not a SOTA-leaderboard chase.

## 2. The contribution (current thesis)
**Measure multimodality; do not assume it.** The field justifies generative
(diffusion) planners with "driving is multimodal," but never (a) measures the
multimodality, (b) decomposes its source, or (c) checks whether that conclusion
survives a realistic simulator. We run a controlled **2x2x2** experiment
— (precise-point goal vs route-conditioned goal) x (deterministic MLP head vs
diffusion head) x (IDM agents vs realistic SMART agents) — with a per-scenario
**interaction-multimodality readout** as the mediator, to test whether a
published architectural conclusion ("diffusion > MLP") is an artifact of the
simulator's background agents. Falsifiable either way; both outcomes are findings.
Full spec: `docs/frontier/STAGE_0_CONTRIBUTION.md`.

## 3. How we got here (the ablation ladder, all complete)
Each planner changes ONE variable from the previous; architecture held fixed
across the goal-conditioned variants — a clean isolation of the goal-source effect.

| Planner | Closed-loop signal | Key finding |
|---|---|---|
| BC | kinematic state only | 49.4 m — covariate-shift collapse |
| GoalBC (oracle) | expert T+8 goal | 1.82 m — **goal representation, not architecture, is the bottleneck** |
| MapBC | local lane query | 56 m — a *local* reference fails off-road |
| RouteMapBC | global route, fixed 8 m | 32 m — global route helps; 8 m is the wrong scale |
| SpeedAdaptiveRouteMapBC | route, speed x 0.8 s | tied with IDM (Wilcoxon p=0.76) |
| RoadblockRouteMapBC | route_roadblock_ids branch | correct goal != executable goal (15 fix / 13 break) |
| DualHorizon | near + far (20 m) goal | mode swap: fixes turns, breaks straights |
| DiffusionPolicy (3d) | DDPM head, same 10-dim cond | **tied the MLP (p=0.79)** — the null that motivated the reframe |

**Why the null happened (verified by literature):** the conditioning nearly fully
specifies one trajectory, so `p(traj|cond)` is ~unimodal and diffusion provably
collapses to the conditional mean. The reframe (Section 2) fixes the premise.

## 4. Software architecture (SOLID / OOP)
All planners live in `nuplan/planners.py` and implement nuPlan's `AbstractPlanner`
interface (`name`, `observation_type`, `initialize`, `compute_planner_trajectory`).
The inheritance graph is deliberate, not incidental:

```
AbstractPlanner (nuPlan)
├─ BCPlanner, IDMPlanner                      # independent baselines
├─ GoalBCPlanner                              # expert-goal oracle
├─ MapBCPlanner                               # local map query
└─ RouteMapBCPlanner                          # global route; defines reusable hooks:
   │   _build_route(), _get_route_goal(look_ahead_m), _select_successor(), _straight_route()
   ├─ TrainedRouteBCPlanner                   # Liskov: only the checkpoint differs
   ├─ SpeedAdaptiveRouteMapBCPlanner          # adds speed_adaptive flag (no logic dup)
   │   └─ RoadblockRouteMapBCPlanner           # OPEN/CLOSED: overrides _select_successor only;
   │                                           #   Liskov-safe (identical to parent when no route id)
   └─ DualHorizonRouteMapBCPlanner            # in_dim=10 (near+far goal)
DiffusionPolicyPlanner(AbstractPlanner)        # direct inheritance — the head is
                                               #   fundamentally different; route helpers
                                               #   carried with _dp suffix, no misleading reuse
```
SOLID notes: **S** — one planner, one goal-source responsibility. **O** — new
junction logic added by overriding `_select_successor`, not editing the base loop.
**L** — every subclass degrades to its parent's behavior on the empty/edge case.
**D** — the eval harness depends on the `AbstractPlanner` interface, never a concrete class.

## 5. Tooling around the model (engineering hygiene)
| File | Purpose |
|---|---|
| `nuplan/eval_production.py` | multi-scenario closed-loop harness; cleans stale output; composes L2 + PDM metrics |
| `nuplan/pdm_score.py` | parses + aggregates the 7 PDM-Score components (Dauner et al. 2023) |
| `nuplan/statistical_analysis.py` | paired Wilcoxon, exact binomial, bootstrap CIs, trimmed mean, tail attribution |
| `nuplan/verify_pipeline.py` | 6 pre-run invariant checks (DB rate, goal-timing scale, checkpoints) |
| `tests/test_planner_geometry.py` | 55 unit tests pinning transforms, route walk, denoiser shapes |
| `nuplan/train_*.py` | training scripts (dual-horizon, diffusion) with mandatory `--sanity` gates |

Rules: never commit without `pytest tests/ -q` green; every non-obvious line gets a
`# WHY:` comment; no statistical claim without a paired test + CI; simulation
artifacts (`sim_results/`, large `.pt`) are git-ignored.

## 6. Tech stack
- Python 3.9 (conda env `nuplan`) | PyTorch (MPS on M1 local, CUDA on Explorer H200)
- nuPlan devkit + mini dataset (100 Hz ego_pose) | Hydra (sim config) | pytest
- HPC: **Northeastern Explorer** (MGHPCC) — H200/A100/V100; `gpu` partition 8 h /
  single-GPU default; `/scratch` for data (auto, purged), `/projects` (PI) for long-haul.

## 7. Current status
- [x] Ablation ladder BC -> DiffusionPolicy complete; all findings reproduced + documented
- [x] 55 unit tests, statistical + PDM + verification tooling in place
- [x] Four-thread literature review + thought-leader gap analysis (`docs/frontier/litreview/`)
- [x] Reframe to the simulator-validity contribution (`docs/frontier/STAGE_0_CONTRIBUTION.md`)
- [ ] **NEXT: Stage-0 pilot** — released checkpoints under IDM vs SMART agents (de-risk)
- [ ] F0 vectorized scene -> F1 encoder -> F2 dual goals -> F3 twin heads -> F4 perturbation -> F6 2x2x2 eval

## 8. Reframed roadmap (F0-F6)
See `docs/frontier/FRONTIER_UPGRADE_PLAN.md` and `STAGE_0_CONTRIBUTION.md`. Build
order is locked by the research: **encoder before closed-loop training** (the
DAgger-null proved closed-loop data is wasted on a road-blind representation).

## 9. Run commands
```bash
conda activate nuplan
python nuplan/verify_pipeline.py                 # invariant gate
pytest tests/ -q                                 # 55 tests
python nuplan/eval_production.py --n_scenarios 30 --planners idm,speedadaptive,dualhorizon,diffusion
python nuplan/statistical_analysis.py --a DiffusionPolicyPlanner --b DualHorizonRouteMapBCPlanner
```

## 10. Session log
Maintained in user memory (`session_log.md`); per-result details in `PROGRESS.md`
and `notes/phase3_roadmap.md`.
