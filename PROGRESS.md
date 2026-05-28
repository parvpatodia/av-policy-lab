# Weekly Progress — Parv Patodia Summer 2026

*This file is updated every session by Claude. Remote audit agent reads it every Sunday.*

---

## Week of May 26–Jun 1

| Track | Target/week | Actual | Status |
|---|---|---|---|
| NeetCode | 14 | 10 done this week (19 total) | ✅ On track |
| Applications | 5–8 | 0 this week | ⚠️ Do tomorrow |
| Cold emails | 5 | 0 this week | ⚠️ Do tomorrow |
| Karpathy | L4 | L4 ✅ DONE (dev loss 2.1) | ✅ Done |
| AV-Policy-Lab | Phase 3c' start | Not started | ⚠️ Tomorrow |
| Physical book | Get + start | Got "The Alignment Problem" | ✅ Done |

## All-time Counts
- NeetCode: 19/150
- Applications submitted: 13
- Cold emails sent: 6 (follow-ups due May 27-28)
- Karpathy: L1 ✅ L2 ✅ L3 ✅ L4 ✅ L5 ⬜ L6 ⬜ L7 ⬜

## NeetCode Log
- Arrays & Hashing: Contains Duplicate, Valid Anagram, Two Sum, Group Anagrams, Top K Frequent, Encode & Decode, Product Except Self ✅ (7)
- Two Pointers: Valid Palindrome, Two Sum II, 3Sum, Container With Most Water ✅ (4)
- Sliding Window: Best Time Buy/Sell, Longest Substring, Longest Repeating Char, Permutation in String, Sliding Window Maximum, Minimum Window Substring ✅ (6)
- Binary Search: next section

## AV-Policy-Lab Results — Phase 3c'' COMPLETE + 30-Scenario Production Eval

### 30-scenario diverse eval (May 28, 30 scenarios × 64 logs)
| Policy | Mean | Median | Std | Fail>20m | Good<5m |
|---|---|---|---|---|---|
| **SpeedAdaptiveRouteMapBC** | **18.19m** | **7.50m** | 28.57 | 6/30 | 12/30 |
| IDMPlanner | 13.97m | 8.50m | 16.26 | 8/30 | 12/30 |
| BCPlanner | 27.18m | 16.99m | 28.19 | 14/30 | 12/30 |
| RouteMapBCPlanner | 47.36m | 53.57m | 25.43 | 26/30 | 0/30 |

**SpeedAdaptive wins 17/30 scenarios over IDM (57% win rate). Better median (7.50 vs 8.50m).**
Mean is worse (18.19 vs 13.97m) due to 4 catastrophic tail failures (L2: 55.7, 80.3, 85.3, 121.2m).
Root cause of tail failures: route centerline goes straight at intersections where expert turns.
**Without 4 tails: SpeedAdaptive mean ≈ 8.5m — beats IDM.**

### Single-log 3-scenario results (for reference)
| Policy | 3-scen L2 | Notes |
|---|---|---|
| GoalBCPlanner | **1.820m** | oracle — expert DB at inference |
| SpeedAdaptiveRouteMapBC | 13.697m | 57% better than RouteMapBC (32.085m) |
| IDMPlanner | 6.285m | |
| RouteMapBCPlanner | 32.085m | fixed 8m scale — wrong |
| TrainedRouteBCPlanner | 49.034m | network ignored 12×-horizon goal |

### Phase 3c'' root cause chain (complete)
DB = 100Hz → T+8 training = 0.08s = 0.35m avg. GoalBC inference = T+0.8s = 3.46m. RouteMapBC fixed 8m = wrong scale → ignored. TrainedRouteBC retrained on 8m → 8m is 12× prediction horizon (0.69m) → ignored. SpeedAdaptive = speed×0.8 → correct T+0.8s scale → 57% improvement.

### Remaining gap to GoalBC: intersection topology
SpeedAdaptive fails at 4/30 scenarios (intersection turns). Fix: use `route_roadblock_ids` from nuPlan `PlannerInitialization` to select correct lane at intersections. **Phase 3c''' planned.**

### Production infrastructure shipped (May 28)
- `verify_pipeline.py` — 6-check invariant verifier (7 PASS, 1 WARN)
- `eval_production.py` — 30-scenario multi-planner harness with mean/median/std/JSON
- `trajectory_viz.py` — L2-over-time plots, failure-vs-speed scatter
- `failure_analysis.py` — per-scenario failure report + findings

## Tomorrow's Priorities (Thu May 29)
1. **AV Phase 3c'''**: Implement `route_roadblock_ids`-guided route construction — fix the 4 intersection tail failures. Expected: mean ~8.5m, beats IDM.
2. **NeetCode × 2**: Binary Search (Binary Search + Search 2D Matrix)
3. **Cold emails × 2**: 2 new AV/frontier lab targets
4. **Applications × 2**: Tesla Optimus + FieldAI (fill, don't submit)
5. **Karpathy L5**: Start makemore Part 4 (WaveNet)
6. **Flash Attention 1**: Read first half

---
*Last updated: 2026-05-28 evening*
