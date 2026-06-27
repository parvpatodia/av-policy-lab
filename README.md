> **Frontier result:** see [docs/frontier/PAPER.md](docs/frontier/PAPER.md) -- *When Multimodality Doesn't Help: A Diagnostic of Diffusion Planners on nuPlan Closed-Loop* (Dauner-lineage negative result; collapse + saturation).

# av-policy-lab

A controlled study of when generative (diffusion) planners actually beat
deterministic ones on the [nuPlan](https://nuplan.org) closed-loop benchmark,
and whether that conclusion survives realistic background traffic.

MS research project, Northeastern University. Work in progress; this README
states exactly what is done, what is running, and what is not built yet.

## The question

The field justifies diffusion planners with "driving is multimodal," but
rarely measures the multimodality or checks whether the conclusion depends on
the simulator. In an earlier phase of this repo, a capacity-matched diffusion
head and a deterministic MLP head TIED (avg displacement 27.59 m vs 27.55 m,
Wilcoxon p = 0.79) when both were conditioned on a precise future goal point.
The precise goal removed the ambiguity, and with it any reason for the
generative head to win. That null result became the design for the real
experiment.

## The experiment

Train four identical-capacity models that differ only along two axes, then
evaluate closed-loop:

|  | route-region goal | precise-point goal |
|---|---|---|
| **deterministic head** | cell 1 | cell 2 |
| **diffusion head** | cell 3 | cell 4 |

- Hypothesis: the diffusion head wins only under route-region conditioning,
  and only on scenarios that contain real decision ambiguity.
- Ambiguity is measured, not assumed: a per-scenario interaction-multimodality
  score (F4) computed from scene structure alone (lane branching, predicted
  encroachment gaps, crosswalk geometry). Never from either model's output,
  so the moderation claim is not circular. Spec with verified citations:
  [docs/frontier/F4_SPEC.md](docs/frontier/F4_SPEC.md).
- Third axis (in progress): repeat the comparison with learned reactive
  background agents (SMART, warm-started from NVIDIA CAT-K checkpoints)
  instead of rule-based IDM, testing whether the conclusion itself flips.
  Motivated by Hagedorn et al. (arXiv:2510.14677), who showed IDM agents
  inflate nuPlan scores and reshuffle planner rankings.

## Status

Done and tested:
- F0: vectorized scene extraction from nuPlan mini. 707 shards, 431,508
  samples, every sample carries scenario identifiers and an 8 s future label.
- F1: Wayformer-style scene encoder, 1.03M params, 32 latent queries.
- F2/F3: goal conditioning + capacity-matched twin heads (5.4% param gap,
  shared trajectory trunk, x0-parameterized diffusion, cosine schedule).
- F5: training loop with bit-exact checkpoint resume, DDIM sampler,
  deterministic train/val shard split. 146 tests pass.
- F4 (partial): shard-side interaction score components.

Running now:
- The four training cells on HPC GPUs (SLURM array).

Not built yet:
- F4 map-API branching score and the score-to-scenario join.
- Closed-loop evaluation harness for the trained heads.
- SMART agent integration (checkpoints in hand, port not started).

Earlier phases (BC, DAgger, BEV CNN, MILE world model, goal-representation
ablations) live in the notebooks and [docs/](docs/); their results stand but
they are not the current contribution.

## Layout

```
nuplan/features/    F0 extractor, F4 score components
nuplan/models/      encoder, twin heads, DDIM sampler, shard dataset
nuplan/training/    train_policy.py (one experiment cell per invocation)
nuplan/slurm/       extraction / training / verification jobs
tests/              full suite (run on a compute node: pytest tests/ -q)
docs/frontier/      specs, decision log, literature reviews
```

## Reproducing

Requires the nuPlan mini dataset and devkit (see
[docs/frontier/HPC_NORTHEASTERN.md](docs/frontier/HPC_NORTHEASTERN.md) for the
cluster setup used here).

```
pytest tests/ -q                                  # 146 tests
sbatch nuplan/slurm/extract_features_array.sbatch # F0 extraction (16 tasks)
python nuplan/slurm/merge_shards.py --strict      # shard verification gate
sbatch nuplan/slurm/train_policy_array.sbatch     # the 2x2 (4 cells)
```

## Notes

- SMART checkpoints are WOMD-derived and are NOT redistributed in this repo,
  per the Waymo Open Motion Dataset license. Thanks to Zhejun Zhang for
  sharing the CAT-K checkpoints for academic use.
- Commit convention: type(scope): description. Tests gate every commit.
