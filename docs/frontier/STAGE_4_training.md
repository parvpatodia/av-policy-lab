# STAGE 4 — TRAINING & OPTIMIZATION

> Status: DESIGN ONLY + runnable SLURM scaffolding (templates, not yet executed). Builds on Stage 3 model and Stage 2 data.

## 1. Loss design

Total loss (Stage A, open-loop imitation pretrain):

```
L = L_diff                              # x0 diffusion MSE on the trajectory
  + λ_imit · L_imit                     # smooth-L1 on x̂0 vs expert future  (PLUTO)
  + λ_da   · L_drivable                 # off-drivable-area penalty on x̂0    (PLUTO, ESDF)
  + λ_col  · L_collision                # soft collision penalty vs constant-velocity agent boxes (see E6)
  + λ_cls  · L_mode_ce                  # cross-entropy on K-candidate scorer (optional)
```

- **L_diff:** `‖x̂0 − x0‖²` (x0-param, Stage 3). REF: arXiv:2501.15564.
- **L_imit:** smooth-L1 on predicted vs expert waypoints — anchors metric accuracy. REF: arXiv:2404.14327.
- **L_drivable:** signed-distance penalty pushing waypoints inside drivable area, computed via an **ESDF with N_c covering circles** approximating the ego footprint (PLUTO's batch-efficient trick). WHY: injects map awareness into the *objective*, directly attacking the off-road failures the brief flags as structurally inevitable without scene. REF: arXiv:2404.14327.
- **L_collision (E6 fix — causality):** in open-loop imitation, penalizing the ego against agents' *logged* future positions is **non-causal** (the logged agents reacted to the logged ego, not to ours). So use only a **soft, low-weight** penalty against agents rolled forward under **constant velocity** (a causal, ego-independent proxy), PLUTO-style. The **real interaction-safety signal comes from the closed-loop / DAgger stage** (§5), where agents react to *our* ego. This limitation is stated explicitly, not papered over.
- Start weights: `λ_imit=1.0, λ_da=0.1, λ_col=0.1, λ_cls=0.5` — **sweep** (§6).

Stage B (closed-loop / DAgger fine-tune) adds on-policy recovery data — see §5.

## 2. Optimizer & schedule

- **AdamW**, weight decay 1e-4. WHY: standard for transformers; PlanTF/PLUTO use Adam-family.
- **LR:** peak **1e-3** (PlanTF) for the encoder/decoder, **cosine decay to 0**; **5% linear warmup**. WHY: PlanTF uses 1e-3 cosine→0; warmup stabilizes transformer training.
- **EMA** of weights (decay 0.999–0.9999). WHY: standard and important for diffusion sampling quality (Diffusion Policy / DDPM practice). REF: arXiv:2303.04137.
- **AMP (bf16)** on A100; grad clip 1.0.
- **Batch size 256–512** per the parameter budget; effective batch scaled by DDP world size.
- **Epochs:** ~100 over ~1M frames (PLUTO/PlanTF regime), early-stop on Val14 open-loop proxy.

## 3. Diffusion specifics

- T_train = 100; cosine (ablate VP). Sampler **DPM-Solver++ 10–15 steps**. K=8 candidates. (Stage 3 §5.)

## 4. SLURM / HPC pipeline (A100, DDP)

**4a. Data staging** (`scripts/slurm/stage_data.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=avlab_stage
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage_%j.out
set -euo pipefail
module load anaconda3
source activate avlab
# WHY: extract+cache vectorized tensors once (SQLite -> .pt shards) so GPU jobs are not I/O-bound
python -m avlab.data.cache_features \
  --nuplan_root $NUPLAN_DATA_ROOT \
  --split train --num_frames 1000000 \
  --out $SCRATCH/avlab_cache/train --num_workers 32
```

**4b. Multi-GPU training** (`scripts/slurm/train_ddp.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=avlab_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
set -euo pipefail
module load anaconda3 cuda/12.1
source activate avlab
export OMP_NUM_THREADS=8
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
# WHY: torchrun handles DDP rendezvous; 4xA100 for throughput, model fits on one
srun torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m avlab.train \
  --config configs/diffusion_scene.yaml \
  --data $SCRATCH/avlab_cache/train \
  --batch_size 128 --epochs 100 --amp bf16 --ema 0.9999
```

**4c. Val14 closed-loop eval** (`scripts/slurm/eval_val14.sbatch`): single GPU, runs nuPlan closed-loop sim (reactive + non-reactive) over the 1118-scenario split; writes PDM decomposition (Stage 5). Note: closed-loop sim is **CPU-heavy and slow** — budget many CPU hours, parallelize by scenario shard via `--array`.

**Checkpointing:** save EMA weights every N steps to `$SCRATCH`; resume-safe (preemption-tolerant). WHY: HPC jobs get preempted; a 48h wall clock needs restartable training.

## 5. Closed-loop / DAgger fine-tune (Stage B — the covariate-shift cure in training)

1. Train Stage A (imitation + perturbation augmentation, Stage 2 §5) to convergence.
2. **Approximate-DAgger loop (E7 fix — no online expert):** nuPlan has **no queryable online expert** at off-distribution states, so classic DAgger is impossible. Substitute the relabeling oracle with **PDM-Closed** (the strong rule-based planner) — or nearest-expert lane-following — to label drifted states. Roll out the current policy closed-loop on a subset of train logs; collect drift states; relabel with the PDM-Closed future; aggregate; retrain. This is **approximate DAgger**, stated as such, not classic DAgger. WHY: it still trains on the *policy's own* state distribution (the covariate-shift cure), just with a surrogate expert. The repo already has a DAgger v2 (~12.7K samples) using a heuristic relabel — formalize the oracle and scale it on the scene-aware model.
3. **(Stretch) PLUTO contrastive CIL:** add positive/negative augmented scenes with the triplet contrastive loss. REF: arXiv:2404.14327.
4. **(Far stretch) CaRL-style RL fine-tune** with a route-completion reward — explicitly flagged as out of solo-student scope (500M-sample budget). REF: arXiv:2504.17838.

## 6. Hyperparameter search plan

| Param | Range | Budget |
|---|---|---|
| LR peak | {3e-4, 1e-3, 3e-3} | 3 runs |
| Perturbation σ_lat / σ_θ | {0.5/0.05, 1.0/0.1} | 2 runs |
| Loss weights λ_da, λ_col | {0.05, 0.1, 0.2} | grid-lite, 3 runs |
| Sampler steps | {10, 15, 25} | inference-only sweep (cheap) |
| K candidates | {1, 4, 8, 16} | inference-only |

Use **Val14 open-loop proxy (ADE/FDE on held-out log split)** for the sweep (cheap), confirm finalists with **full Val14 closed-loop** (expensive). WHY: closed-loop sim is too slow to sweep directly; open-loop proxy + a handful of closed-loop confirmations is the standard cost-aware protocol.

## 7. Ablations (attribute every gain — paper-grade)

Each toggles ONE root-cause fix, holding all else fixed:

| Ablation | Isolates | Expected effect |
|---|---|---|
| scene encoder OFF (kinematics only) | root cause #2 | large closed-loop drop, esp. collisions/off-road |
| cross-attn → concat conditioning | root cause #1 mechanism | multimodality collapse at junctions |
| route-region goal → precise near/far goal | root cause #1 | diffusion ≈ MLP again (reproduces Phase-3d null) |
| temporal decoder → flat-48 MLP | root cause #5 | worse long-horizon turns |
| perturbation augmentation OFF | root cause #3 | closed-loop drift up (PlanTF: -14 pts NR-CLS) |
| x0 → ε prediction | diffusion param choice | stability/quality |
| DPM-Solver++ → DDIM | sampler | speed/quality |
| full nuPlan → mini | root cause #4 | variance up, significance lost |

The **route-goal and cross-attn ablations are the headline**: they convert the Phase-3d "diffusion = MLP" null into a *controlled demonstration* of *when* diffusion helps.

## HANDOFF TO NEXT STAGE (Stage 5 — Evaluation)

- Eval must consume EMA checkpoints and run **Val14 + Test14-hard, reactive AND non-reactive**, with full **PDM-score decomposition** + open-loop ADE/FDE/miss-rate.
- The K-candidate scorer needs a selection rule (PDM proxy) — Stage 5 defines it.
- Ablation matrix (§7) defines the rows of the results table; each needs the same eval protocol + paired stats.
- SLURM eval array job is the harness Stage 5 formalizes.

## Honesty flags
- DAgger closed-loop rollouts + Val14 closed-loop sim are **CPU-bound and very slow**; this, not GPU training, is the real time sink. Costed in the final plan.
- LR 1e-3 is PlanTF's for a regression transformer; a diffusion transformer may prefer lower (3e-4) — hence the sweep, not a fixed claim.
