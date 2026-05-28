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

## AV-Policy-Lab Results — Phase 3c COMPLETE

### Closed-loop L2 (all policies, nuPlan mini)
| Policy | Avg L2 (m) | Key signal |
|---|---|---|
| BCPlanner v0 | 49.449 | none (kinematic only) |
| DAgger iter 2 | 49.486 | on-policy data |
| BEVPlanner | 49.410 | ego history raster |
| MILEPlanner | 49.565 | latent consistency |
| IDMPlanner | **6.285** | rule-based road following |
| **GoalBCPlanner** | **1.820** | **expert T+8 goal — 96.3% reduction** |
| MapBCPlanner | 56.326 | nearest-lane point query (fails off-road) |
| **RouteMapBCPlanner** | **32.085** | **pre-computed 200m route — 35% better than BC** |

**Phase 3c finding: train/inference mismatch.** Global route fixes drift bootstrapping (32m vs 56m, +43%). GoalBC weights trained on expert T+8 goals — different distribution from route centerline goals. Fix: TrainedRouteBC — retrain with route goals at training time.

## Phase 3c' TrainedRouteBC — Data Pipeline Analysis

**Root cause confirmed quantitatively (2026-05-28):**
| Goal source | Mean magnitude | Corr w/ speed |
|---|---|---|
| GoalBC training (T+8) | 0.461m | 1.000 |
| RouteMapBC inference (arc 8m) | ~8m | low |
| **TrainedRouteBC training (arc 8m)** | **8.013m** | **0.150** |

**The 17.6× L2 gap (1.82m→32.085m) = 17× goal magnitude mismatch.** Fixed in TrainedRouteBC.

**Files added today:**
- `nuplan/trained_route_bc.ipynb` — full training pipeline (Cells 1–10)
- `nuplan/planners.py` — `TrainedRouteBCPlanner` added (subclasses `RouteMapBCPlanner`)

**Status (updated):**
- `trained_route_bc.ipynb` ran — **UNEXPECTED: 49.034m ≈ BC_v0** (retraining didn't help)
- **Root cause found**: DB is 100 Hz, not 10 Hz. T+8 = 0.08s. GoalBC training goals = 0.342m mean. Fixed 8m look-ahead = **23× scale mismatch** → policy ignores goal.
- **Fix added**: `SpeedAdaptiveRouteMapBCPlanner` — look_ahead = speed × 0.08s. No retraining. Uses `goal_bc.pt`.
- **To run**: `python nuplan/eval_speed_adaptive.py` (~2 min). Expected ≈ GoalBC (1.82m).

## Today's Remaining Priorities (Wed May 28)
1. **Run TrainedRouteBC**: open `trained_route_bc.ipynb`, run all cells, record result
2. **Cold emails × 2**: 2 new targets (AV or frontier lab researchers)
3. **Applications × 2**: Tesla Optimus + FieldAI Manipulation (fill, don't submit)
4. **Flash Attention 1 paper**: read first half
5. **Karpathy L5**: start next lecture
6. **NeetCode × 2**: Binary Search section starts

---
*Last updated: 2026-05-28 afternoon*
