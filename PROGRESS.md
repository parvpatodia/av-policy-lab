# Weekly Progress — Parv Patodia Summer 2026

*This file is updated every session by Claude. Remote audit agent reads it every Sunday.*

---

## Week of May 19–25

| Track | Target/week | Actual | Status |
|---|---|---|---|
| NeetCode | 14–21 | 9 (Valid Sudoku + Longest Consecutive done May 24) | ⚠️ Need 5 more Mon–Wed |
| Applications | 8 (2/day Mon-Thu) | 13 submitted (all-time) | ✅ Ahead |
| Cold emails | 10 | 6 sent (May 20-21) | ⚠️ Need 4 more Mon/Tue |
| Karpathy | L4 due | L3 done, L4 not done | ⚠️ Must do Monday |
| AV-Policy-Lab | Phase 3b complete | GoalBC 1.82m, MapBC 56.3m (drift bootstrapping) | ✅ Phase 3b done |

## All-time Counts
- NeetCode: 9/150
- Applications submitted: 13
- Cold emails sent: 6 (4 follow-ups due May 27-28)
- Karpathy: L1 ✅ L2 ✅ L3 ✅ L4 ⬜ L5 ⬜ L6 ⬜ L7 ⬜

## AV-Policy-Lab Results — Phase 3b COMPLETE

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

**Phase 3b finding: drift bootstrapping problem.** Map point queries only work ON the road. Once ego drifts 2-3m off-road, query returns 0 lanes → fallback to straight-ahead → worse than BC. GoalBC works because expert T+8 is a GLOBAL reference. Phase 3c: pre-compute route at scenario start → route tracker = always valid.

## Next Session Priorities (Monday May 25)
1. Valid Palindrome + 4 NeetCode to catch up (target: 14/150 by EOW)
2. Karpathy L4: BatchNorm — non-negotiable
3. 2 applications: Tesla Optimus + FieldAI Manipulation (fill, don't submit)
4. 2 cold emails (need 4 more this week total)
5. AV-Policy-Lab Phase 3c: RouteMapBCPlanner (pre-computed route, no expert needed)

---
*Last updated: 2026-05-25 (overnight)*
