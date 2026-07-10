# The Mode-Selection Bottleneck in Multimodal Closed-Loop Planning

AV-policy-lab. Consolidated result (ADR-049, GATE-CL-1/ADR-048, ADR-055/057/058/059).
Status: standard-split result complete on mini-trained checkpoints; full-train retrain is the
credibility step; SMART realism is a later robustness axis.

## Question
Multimodal / diffusion imitation policies underperform simple deterministic ones in nuPlan
closed-loop (Dauner 2023; DIVER arXiv:2507.04049). Is that because the *modes* are bad, or because
the system *selects* the wrong mode? We separate the two.

## Setup
nuPlan closed-loop, IDM agents, PDM-style CLS. Ego zoo: deterministic IL head (det), a
winner-take-all multi-hypothesis head that emits 6 candidate ego trajectories (WTA), a learned
score head that picks one (default WTA selection), a selector retrained on closed-loop CLS labels
(sel), and PDM-Closed (rule-based reference). Standard splits Val14 / Test14-hard. Per-mode CLS
obtained by forcing each mode through closed-loop (WTA_MODE_INDEX).

## Result (Val14, n=300, under IDM)
  oracle (best-of-6 modes)         0.868
  det                              0.810
  learned CLS-selector (sel)       0.747
  random mode                      0.730
  default WTA (imitation score)    0.705
Gaps: latent value oracle-det = +0.058; realized sel-det = -0.062; unrealized oracle-sel = +0.121.
~48% of scenes contain a mode that beats det.

Full-zoo means (Val14 n=1118 / Test14-hard n=272) confirm the ordering PDM > det > sel > WTA on both
splits (ADR-055/057).

## Findings
1. The modes carry real latent value: an oracle over the 6 modes beats the deterministic policy
   (+0.058) on ~half of scenes. The multimodality deficit is NOT a modes problem.
2. Selection is the bottleneck: the default (imitation-score) selection is WORSE than random mode
   selection (0.705 < 0.730), and a selector trained directly on closed-loop CLS barely beats random
   (0.747) and leaves +0.121 of oracle headroom on the table.
3. Mechanism: closed-loop mode-value is not recoverable from pre-decision signals. Open-loop proxy
   reward correlates with closed-loop CLS at r=0.11 (GATE-CL-1), and imitation confidence is
   anti-informative for closed-loop selection. A feedforward selector is therefore fundamentally
   limited; realizing multimodality's value likely needs closed-loop / receding-horizon selection.

## Relation to prior work
- DIVER / Dauner: multimodal & complex IL underperform in closed-loop. We localize the cause to
  selection, not the modes.
- Hagedorn et al. (arXiv:2510.14677): sim realism (SMART agents) shifts planner rankings. Our
  planned robustness axis: does the selection bottleneck persist under SMART agents?

## Honest caveats
- Numbers are on mini-trained checkpoints and an n=300 Val14 subset -> provisional. The full-train
  retrain is the reviewer-critical credibility step.
- Novelty is workshop/report-ceiling: the oracle-vs-selected gap is well known in *prediction*
  (minK metrics); the contribution is the *planning* framing plus the closed-loop-unpredictability
  mechanism and the "worse-than-random imitation selection" result.

## Next
1. Full-train retrain (subset: boston/pitt/singapore) -> re-run the selection study for credible numbers.
2. SMART robustness axis when the code releases.
