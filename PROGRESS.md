# Weekly Progress — Parv Patodia Summer 2026

*This file is updated every session by Claude. Remote audit agent reads it every Sunday.*

---

## Week of May 19–25

| Track | Target/week | Actual | Status |
|---|---|---|---|
| NeetCode | 14–21 | 9 (Valid Sudoku + Longest Consecutive done May 24) | ⚠️ Need 1 more today (Valid Palindrome) |
| Applications | 8 (2/day Mon-Thu) | 13 submitted (all-time) | ✅ Ahead |
| Cold emails | 10 | 6 sent (May 20-21) | ⚠️ Need 4 more (2 today, 2 Mon/Tue) |
| Karpathy | L4 due | L3 done, L4 tonight | ⚠️ Due today |
| AV-Policy-Lab | Phase 3a GoalBC | ✅ COMPLETE — 1.820m L2 | ✅ Done |

## All-time Counts
- NeetCode: 9/150
- Applications submitted: 13
- Cold emails sent: 6 (4 follow-ups due May 27-28)
- Karpathy: L1 ✅ L2 ✅ L3 ✅ L4 ⬜ L5 ⬜ L6 ⬜ L7 ⬜

## AV-Policy-Lab Results — Phase 3a COMPLETE

### Closed-loop L2 (all policies, nuPlan mini, 3 scenarios)
| Policy | Avg L2 (m) | vs BC_v0 |
|---|---|---|
| BCPlanner v0 | 49.449 | baseline |
| BCPlanner v1 (DAgger 1) | 49.470 | +0% |
| BCPlanner v2 (DAgger 2) | 49.486 | +0% |
| IDMPlanner | 6.285 | –87.3% |
| BEVPlanner | 49.410 | –0.08% |
| MILEPlanner | 49.565 | +0.2% |
| **GoalBCPlanner** | **1.820** | **–96.3%** |

**Key insight: GoalBC beats IDM by 3.5×. Perception was the bottleneck, not capacity.**

## AV-Policy-Lab Open-loop Results
| Policy | ADE (m) | FDE (m) |
|---|---|---|
| BC MLP | 0.058 | 0.063 |
| BEV CNN | 0.051 | 0.059 |
| MILE | 0.060 | 0.068 |
| GoalBC | 0.004 | 0.008 |

## Next Session Priorities
1. NeetCode: Valid Palindrome (1 problem to hit 10 today)
2. Applications: Tesla Optimus + FieldAI Manipulation (fill, don't submit)
3. Cold emails: 2 new targets
4. Karpathy L4: BatchNorm — watch + implement + dead-reckon
5. AV-Policy-Lab Phase 3b: MapBC (centerline-conditioned, no expert leak) — background

---
*Last updated: 2026-05-24*
