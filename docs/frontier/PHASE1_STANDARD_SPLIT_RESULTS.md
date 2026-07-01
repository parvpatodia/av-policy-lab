# Phase 1 — Multimodality vs Determinism in nuPlan Closed-Loop, on Standard Splits (under IDM)

AV-policy-lab. Status: Phase-1 IDM arm complete (Val14 + Test14-hard). Decisions: DECISIONS.md ADR-050..057.

## Question
Does a multimodal imitation policy (winner-take-all / diffusion) beat a deterministic one in nuPlan
closed-loop, and does any advantage concentrate in interaction-critical scenes (H1)? And (Phase 2,
open) does that conclusion change under realistic learned reactive agents (SMART) vs the default IDM
agents (Hagedorn et al., arXiv:2510.14677)?

## Setup
- Splits: canonical Val14 (1118) and Test14-hard (272), from the planTF / PDM benchmark lineage
  (Val14 from the nuPlan val split; Test14-hard from the test split).
- Simulation: nuPlan closed-loop, two_stage_controller, IDM reactive agents (reactive=1), PDM-style CLS.
- Ego zoo: deterministic IL head (det), multimodal WTA head (wta), a closed-loop-trained selector on
  the WTA modes (sel), and PDM-Closed as the rule-based reference (pdm).
- Checkpoints trained on the nuPlan mini split (scope caveat: absolute CLS reflects limited training;
  the relative comparisons are the signal). Features are built live in-sim (no offline cache).
- Inference (pre-registered): per-token CLS difference regressed on interaction-criticality (s_inter)
  with scenario_type fixed effects, wild-cluster bootstrap (clustered by scenario_type), and TOST
  equivalence (margin 0.05).

## Results — CLS under IDM
| config              | Val14 (n=1118) | Test14-hard (n=272) |
|---------------------|----------------|---------------------|
| PDM-Closed (rule)   | 0.969          | 0.907               |
| det (IL)            | 0.805          | 0.746               |
| selector (on WTA)   | 0.741          | 0.693               |
| WTA (multimodal)    | 0.712          | 0.654               |

## Findings
1. Multimodality UNDERPERFORMS determinism in closed-loop, on BOTH standard splits
   (WTA - det = -0.093 on Val14, ~ -0.09 on Test14-hard). Confirms multimodality-collapse /
   mean-regression under single-future imitation (our mini findings ADR-029/045; cf. DIVER
   arXiv:2507.04049).
2. The closed-loop-trained selector recovers part of the WTA deficit (+0.03 Val14, +0.04 Test14-hard)
   but does NOT reach deterministic: a selection-learnability gap (the modes are decent, but a
   learnable non-oracle selector captures little of the oracle headroom; ADR-051).
3. H1 (interaction-criticality moderation): on Val14 the WTA-det gap does NOT vary with s_inter —
   TOST establishes EQUIVALENCE within +/-0.05 (slope p=.24, n=1118). The null replicates on the
   standard benchmark with equivalence, not merely failure-to-reject. On Test14-hard the moderation
   is INCONCLUSIVE because s_inter saturates (p50 0.996; the hard split is near-uniformly
   high-interaction, so there is too little moderator variance).
4. PDM-Closed (rule-based) dominates all learned policies, as expected for mini-trained IL.

## Honest caveats
- Checkpoints are mini-trained, so absolute CLS is well below SOTA; the RELATIVE conclusion
  (multimodal vs deterministic) is the contribution, not the absolute numbers.
- s_inter over-fires (skews high). Val14 retains enough spread for a valid equivalence test;
  Test14-hard does not.

## Open question — Phase 2 (realism axis)
Everything above is under nuPlan's IDM reactive agents. Hagedorn et al. (arXiv:2510.14677) show IDM
overestimates planning performance and shifts the IL-vs-rule-based deterioration pattern under learned
SMART reactive agents. Phase 2 swaps IDM -> SMART (gated on the SMART nuPlan tokenizer/codebook) to
test whether the multimodality / interaction conclusion changes under realism, and whether
CLS-under-IDM predicts CLS-under-SMART. This is the principled reason the conclusion might change, and
the natural extension of Hagedorn's result with a multimodality lens.

## Reproduce
- Split token-files: eval_tokens/{val14,test14hard}.json (from planTF config/scenario_filter).
- Eval: nuplan/slurm/{val14_zoo_array,test14hard_zoo_array}.sbatch (run_cells --db-dir <split> --reactive 1).
- Collect / analyze: nuplan/analysis/{collect_val14,analyze_val14}.py <split>.
- s_inter: nuplan/analysis/producer_val_sinter.py (from token-targeted f0 extraction).
- Data pulled via public S3 (dl_split.py); membership via check_membership.py.
