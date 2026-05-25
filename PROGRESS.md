# Weekly Progress — Parv Patodia Summer 2026

*This file is updated every session by Claude. Remote audit agent reads it every Sunday.*

---

## Week of May 19–25

| Track | Target/week | Actual | Status |
|---|---|---|---|
| NeetCode | 14–21 | 8 (Valid Sudoku done May 24) | ⚠️ Behind — need 6 more by May 25 |
| Applications | 8 (2/day Mon-Thu) | 13 submitted (all-time) | ✅ Ahead |
| Cold emails | 10 | 6 sent (May 20-21) | ⚠️ Need 4 more (2 today, 2 Mon/Tue) |
| Karpathy | L4 due | L3 done, L4 tonight | ⚠️ Due today |
| AV-Policy-Lab | DAgger iter2 + closed-loop eval | ✅ ALL COMPLETE | ✅ Done |

## All-time Counts
- NeetCode: 8/150
- Applications submitted: 13
- Cold emails sent: 6 (4 follow-ups due May 27-28)
- Karpathy: L1 ✅ L2 ✅ L3 ✅ L4 ⬜ L5 ⬜ L6 ⬜ L7 ⬜

## AV-Policy-Lab Results — COMPLETE

### Open-loop (ADE / FDE on nuPlan mini val split)
| Policy | ADE (m) | FDE (m) |
|---|---|---|
| BC MLP | 0.058 | 0.063 |
| BEV CNN | 0.051 | 0.059 |
| MILE | 0.060 | 0.068 |
| IDM | 3.898 | 7.871 |

### Closed-loop (nuPlan L2 vs expert, 3 scenarios)
| Policy | Avg L2 (m) | Max L2 (m) | p90 L2 (m) |
|---|---|---|---|
| BCPlanner v0 | 49.449 | 104.614 | 91.526 |
| BCPlanner v1 (DAgger 1) | 49.470 | 104.656 | 91.564 |
| BCPlanner v2 (DAgger 2) | 49.486 | 104.689 | 91.593 |
| IDMPlanner | **6.285** | **24.308** | **15.733** |
| BEVPlanner | 49.410 | 104.543 | 91.416 |
| MILEPlanner | 49.565 | 104.834 | 91.723 |

**Central finding:** All imitation policies plateau ~49.4–49.6m regardless of architecture. IDM wins by 8x. Feedback beats representation.

## Next Session Priorities
1. NeetCode: Longest Consecutive + Valid Palindrome (2 more today to hit 10)
2. Applications: Tesla Optimus + FieldAI Manipulation (fill, don't submit)
3. Cold emails: 2 new targets
4. Karpathy L4: BatchNorm — watch + implement + dead-reckon

---
*Last updated: 2026-05-24*
