# F4 Validation Results

> Results for the protocol in F4_VALIDATION_PROTOCOL.md. Each section is the
> component it validates. Numbers are reproducible from nuplan/f4_validation/.

## Signal A — S_inter convergent validity (reactive-agent replication hook)

`a_gt` = logged-future conflict: the SAME geometry as `s_inter` (ego route rollout
at v0, PrET Gaussian, top-3 noisy-OR) with the one change of using each agent's
**logged nuPlan future** instead of the constant-turn-rate rollout. Isolates F4's
agent motion model. Computed at iteration 19 on all 5,604 scenes (5604/5604 scored).

Full real distribution (76.7% s_inter=0, 23.3% s_inter>0):

| Test | Statistic | 95% CI | p | Verdict |
|---|---|---|---|---|
| Group contrast (s_inter=0 vs >0) | Cliff's δ = 0.345 | [0.314, 0.375] | perm 5e-4 | **PASS** |
| Within-conflict Spearman(s_inter, a_gt) | rho = 0.085 | [0.030, 0.135] | perm 3.5e-3 | PASS, weak |
| Partial (control n_par, v0) | partial rho = 0.185 | — | — | holds beyond busyness |

Both primary tests pass Holm correction. Means: a_gt = 0.23 (s_inter=0) vs 0.49
(s_inter>0); medians 0.00 vs 0.56.

**Constant-turn-rate blind spot:** among scenes s_inter scores as 0, **32%** have a
real logged-future crossing (a_gt>0) and **24%** a strong one (a_gt>0.5). So s_inter
materially **under-fires** relative to ground-truth agent motion.

**Interpretation (honest).** S_inter is valid at the binary "is there a yield-or-go
conflict" level: an independent, more accurate motion model agrees, and the effect
survives controlling for agent count and ego speed. But (a) its fine-grained
magnitude is only weakly corroborated (within-conflict rho ≈ 0.09), and (b) it
misses a real crossing in ~24% of the scenes it labels zero. Because H1 weights
S_inter, the under-firing plausibly **attenuates** the moderation slope (some truly
ambiguous scenes are coded zero) — a conservative bias, not an inflationary one,
which is the safe direction for the headline claim but worth stating.

Reproduce: `run_signal_a.py` (compute) then `analyze_signal_a.py` (stats).

## Signal B — combined-F4 rater panel (n=80, stratified 20/band)

A 4-persona AI panel (cautious driver, assertive driver, traffic-safety analyst,
AV-planning engineer) rated 80 anonymized top-down renders blind (val_NN.png; private
manifest), one frozen rubric, core instruction *decision ambiguity, not busyness*. Run
via vision-capable subagents (no external auth needed; the `claude` CLI is not logged in
headless and there is no API key). Panel score = mean across personas. A 20-scene pilot
gave rho 0.52 but over-estimated; n=80 is the reported result.

| Metric | Result (n=80) |
|---|---|
| Inter-rater reliability | Cronbach α 0.907; ICC(2,k) 0.902; mean pairwise Pearson 0.714, Spearman 0.715 |
| Panel vs **F4 combined** | Spearman rho = 0.313, 95% CI [0.096, 0.512], p = 0.005 |
| Panel vs S_inter | rho = 0.235, CI [0.016, 0.434], p = 0.036 |
| Panel vs S_branch | rho = 0.142, CI [−0.083, 0.365], p = 0.21 (n.s.) |
| Group contrast f4=0 vs f4>0 | mean 0.34 vs 0.48; MW p = 0.0017, rank-biserial 0.44 |
| Band means | zero 0.34 < mid 0.42 < hi_inter 0.49 ≈ hi_branch 0.51 |

**Interpretation (honest).** The panel has **moderate convergent validity with the
combined F4** (rho 0.31, CI excludes 0, p=0.005) and a significant zero-vs-nonzero group
contrast. Two findings the scale-up surfaced: (1) the n=20 pilot over-estimated (0.52 →
0.31) — small-n optimism, now corrected; (2) the panel tracks **S_inter** (rho 0.24, just
significant) but **not S_branch** (rho 0.14, CI includes 0) — visual judges do not reliably
read map-branch/route ambiguity off a top-down render, so S_branch (17% of F4's firing
mass) cannot be carried by the panel and needs the independent map check (Signal D). The
zero band still averages 0.34 (residual busyness↔ambiguity conflation). Reliability dropped
from the pilot's 0.97 to 0.90 on the harder, more varied set — more realistic.

Caveat: the 4 personas share a base model (correlated error); Parv's human ratings (50-item
tool, in progress) are the external anchor and will (a) calibrate the panel and (b) give the
gold human-vs-F4 number. Raw per-persona ratings in `data/panel_ratings_n80/`.

Reproduce: `render_validation_set.py` → 8 persona subagents (2 batches × 4) → `panel_full_analyze.py`.

## Human anchor + temporal test — the central validity finding

Parv rated 50 items (40 unique + 10 repeats) on the static renders. Self-consistent
(test-retest: exact-agree 0.70, mean abs diff 0.10, Pearson 0.82), conservative
(mean 0.22, 47% zeros, max 0.75).

**Gold human-vs-F4 is NULL:** Spearman(human, F4) = 0.020 (p=0.90); vs S_inter
-0.045; vs S_branch 0.044. Group contrast f4=0 vs f4>0: 0.17 vs 0.23, p=0.33 (ns).
Human band means are flat (zero 0.17, mid 0.23, hi_branch 0.24, hi_inter 0.24).
Panel-vs-human is weak (rho 0.24, ns) with a +0.2 panel over-rating bias.

**Temporal test (rejects the "render too limited" explanation).** Re-rendered the 40
human scenes with each agent's ACTUAL logged 5s future + the ego route drawn (ground
truth, independent of F4 → no leak), re-ran the panel:

|  | vs F4 | vs S_inter | vs human |
|---|---|---|---|
| static panel | +0.424 | +0.371 | +0.235 (ns) |
| temporal panel | +0.422 | +0.352 | +0.282 (ns) |
| human | +0.020 (ns) | −0.045 (ns) | — |

On s_inter>0.6 scenes: static panel 0.53 → temporal 0.41 → human 0.24. Drawing the
real futures made judges rate F4's "high interaction" scenes LOWER, because many of
those crossings are parallel/non-conflicting once trajectories are shown.

**Synthesis (honest).** s_inter detects real geometric crossings (Signal A, δ=0.35),
but a crossing is not the same as a *decision*: a 2.5 s-gap path-crossing is usually a
clear yield or clear go. Neither a human nor the panel-with-futures perceives F4's
high-s_inter scenes as decision-ambiguous. The temporal overlay did not lift the F4
correlation, so the human null is not a render artifact — it is external evidence that
**s_inter over-fires as an ambiguity measure (quantifies ADR-024)** and that F4 is
better described as an **interaction-conflict + branch-presence** index than a
"decision ambiguity" moderator. Implication: H1 ("the route-vs-precise gap grows with
ambiguity") may need reframing toward interaction-criticality, or s_inter needs a
gap-clarity / right-of-way filter (a moderator change with pre-registration cost).
Caveats: single human rater, n=40, coarse 5-point scale, AI personas share a base
model. A larger pairwise human study on temporal renders would make this definitive.

## Signal D — S_branch behavioral check (independent of f4_map_branch)

Independent question: at the ego's location, do other agents' LOGGED futures actually
fan out across turn directions (left/straight/right)? d_branch = normalized turn-direction
entropy of agents within 40 m (no ego-future, no map-graph reuse). Scored 4808/5604 (rest
had <2 turning agents nearby).

| Test | Result |
|---|---|
| Spearman(d_branch, S_branch) | 0.131, p = 8.8e-20 |
| Spearman(d_branch, B_R) | 0.061 |
| Group d_branch: s_branch=0 vs >0 | 0.117 vs 0.199; MW p=5.4e-20, rank-biserial 0.155 |

**Interpretation.** S_branch has a **real but weak** behavioral signal: agents do fan out
across routes slightly more at high-S_branch junctions (highly significant by the huge n,
but rho 0.13 / rbc 0.16). So S_branch is not noise, but its tie to actual route-divergence
is weak — and the visual panel could not read it at all (rho 0.14 ns). S_branch is the
weakest-validated component.

## Signal C — toy dispersion (NOT RUN, by decision)

The only independent planner ensemble available (IDM/BC/BEV/MILE) is ego-centric: 3 of 4
cannot perceive the conflicting agent and IDM handles only a direct lead (see
F4_VALIDATION_PROTOCOL §1.1). Its trajectory dispersion would reflect model bias, not
scene ambiguity, so running it would produce a meaningless number. Skipped deliberately.

## Verdict (LOCKED 2026-06-21)

**F4 is a validated interaction-conflict + route-branch presence index, but NOT a
validated measure of human-perceived decision ambiguity.**

What IS validated:
- s_inter detects genuine space-time crossings (Signal A, Cliff δ=0.35, p<1e-4, holds
  partialling out busyness).
- s_branch weakly but really tracks route-divergence (Signal D, rbc 0.16).

What is NOT:
- Human-vs-F4 convergent validity is **null and robust to the render**:
  static rho=+0.020, temporal rho=+0.020 (both p≈0.90); vs s_inter +0.06/−0.05; group
  contrast f4=0 vs >0 ns in both. Drawing the actual agent futures did not move it
  (human static↔temporal agree 0.51; high-s_inter scenes 0.24→0.28).
- The 4-persona panel was only moderate (0.31) and dropped on high-s_inter once futures
  were shown.
- This replicates the prior round (F4 v1.1 ≈0.15 vs 52 earlier ratings; ADR-024
  s_inter over-fire). F4 has failed human external validation twice.

Interpretation: a path-crossing is real (Signal A) but usually a *clear* yield/go, not a
*decision*; F4 measures interaction-conflict + branching, which is geometrically real but
is not what a human calls "ambiguity." Caveats kept: 1 human, n=40, coarse 5-point scale —
but 1 human + 4 AI personas all fail to link F4 to perceived ambiguity, so it is converging
evidence, not one noisy rater.

Implication for the experiment (H1 "the route-vs-precise CLS gap grows with ambiguity F4"):
the moderator should not be labeled "ambiguity." Open framing decision for Parv:
- (A) **Reframe** F4 as an interaction-criticality / scene-conflict index — most honest,
  matches what Signal A/D validated, preserves the moderation result on its own terms.
- (B) **Fix** s_inter with a gap-clarity / right-of-way filter so it tracks genuine
  close-calls (post-hoc moderator change, pre-registration cost).
- (D) **Document** the null as a limitation and lean on the geometric validity.
(C bigger human study is now low-value: the null is robust, unlikely to flip.)

**DECISION (2026-06-21, Parv): (A) reframe.** F4 is relabeled from "ambiguity" to an
**interaction-criticality** (interaction-conflict + route-branch) moderator. The score
formula and the pre-registered moderation analysis (Delta ~ F4) are UNCHANGED — only the
construct label is corrected to what the validation supports. Recorded as ADR-027; H1 in
RESEARCH_PROTOCOL.md and F4_SPEC.md marked accordingly. The eval writeup uses
"interaction-criticality," not "ambiguity."
