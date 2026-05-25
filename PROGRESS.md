# Weekly Progress — Parv Patodia Summer 2026

*This file is updated every session by Claude. Remote audit agent reads it every Sunday.*

---

## Week of May 19–25

| Track | Target/week | Actual | Status |
|---|---|---|---|
| NeetCode | 14–21 | 9 (Valid Sudoku + Longest Consecutive done May 24) | ⚠️ Need 5 more Mon–Wed |
| Applications | 8 (2/day Mon-Thu) | 13 submitted (all-time) | ✅ Ahead |
| Cold emails | 10 | 6 sent (May 20-21) | ⚠️ Need 4 more Mon/Tue |
| Karpathy | L4 due | L3 done, L4 not done | ⚠️ Must do today (Mon) |
| AV-Policy-Lab | Phase 3c eval | RouteMapBC 32.085m — Phase 3c COMPLETE | ✅ Phase 3c done |

## All-time Counts
- NeetCode: 9/150
- Applications submitted: 13
- Cold emails sent: 6 (4 follow-ups due May 27-28)
- Karpathy: L1 ✅ L2 ✅ L3 ✅ L4 ⬜ L5 ⬜ L6 ⬜ L7 ⬜

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

**Phase 3b finding: drift bootstrapping.** Map point queries fail once ego drifts off-road → query returns 0 lanes → straight-ahead fallback → worse than BC.

**Phase 3c finding: train/inference mismatch.** Global route fixes drift bootstrapping (32m vs 56m, +43%). But GoalBC weights were trained on expert T+8 goals — a systematically different distribution from route centerline goals. Policy learned "goal offset = where expert will be in 800ms" — route goals violate that mapping. Fix: **TrainedRouteBC** — retrain with route goals at training time.

## Next Priorities (Mon May 25 afternoon — today)
1. **NeetCode**: Valid Palindrome + 4 more (need 14/150 by EOW today)
2. **Karpathy L4**: BatchNorm — non-negotiable, do after gym
3. **Applications**: Tesla Optimus + FieldAI Manipulation (fill, don't submit)
4. **Cold emails**: 2 today, 2 Tue (need 4 more this week)
5. **AV Phase 3c'**: TrainedRouteBC notebook — retrain GoalBC with route-based goals

---
*Last updated: 2026-05-25 11:25 AM (Phase 3c eval complete)*
