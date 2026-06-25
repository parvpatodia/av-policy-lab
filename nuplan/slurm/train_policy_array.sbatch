#!/bin/bash
# F5: train the 2x2 — array task i selects (head, goal) cell i.
# Matched re-run (v4): early-stop DISABLED (--patience 9999) so every cell trains the
# full fixed budget, + periodic checkpoints (--ckpt-every) for closed-loop CLS selection.
# Resume: each cell picks up its latest.pt. The 8h wall is spanned by an external cron
# resubmitter (this cluster has JobRequeue=0, so in-job self-requeue is unavailable).
#SBATCH --job-name=f5-2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11   # 4 cells x 3 seeds (audit: single-seed is unpublishable)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f5-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f5-%A_%a.err

set -euo pipefail

HEADS=(det det diff diff)
GOALS=(route precise route precise)
SEEDS=(0 1 2)
CELL=$(( SLURM_ARRAY_TASK_ID % 4 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
HEAD="${HEADS[$CELL]}"
GOAL="${GOALS[$CELL]}"

DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_v3"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4"
REPO_ROOT="/home/patodia.pa/av-policy-lab"
mkdir -p "$CKPT_DIR" /scratch/patodia.pa/av-policy-lab/logs

# WHY direct binary: conda activate fell back to base on compute nodes
# (jobs 7510782_15, 7555465_*); absolute path cannot.
module purge
# WHY PYTHONNOUSERSITE: ~/.local/lib pip packages shadow the conda env
# (a broken user-site torch killed job 7605681 with missing libnvJitLink).
export PYTHONNOUSERSITE=1
# WHY pytorch_env: the nuplan env ships CPU-only torch (2.8.0+cpu); training
# in it ran on 4 CPU cores (job 7600266). pytorch_env has torch 2.4.1+cu121.
PYBIN="/home/patodia.pa/.conda/envs/pytorch_env/bin/python"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -c "import torch, models.scene_encoder; assert torch.cuda.is_available(), "no CUDA"" \
  || { echo "FATAL: env import or CUDA check failed" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "cell: head=$HEAD goal=$GOAL  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node=$(hostname)"

cd "$REPO_ROOT"
# Foreground call. The job runs until it finishes 150 epochs or is killed at the 8h wall
# (latest.pt is on disk from the per-epoch save). The cron resubmitter restarts any
# incomplete cell, which resumes from latest.pt. --patience 9999 disables early-stop so
# every cell trains the full matched budget; --ckpt-every 10 keeps trajectory checkpoints
# for closed-loop CLS selection (ADR-018).
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head "$HEAD" --goal "$GOAL" \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --patience 9999 --ckpt-every 10 \
    --batch-size 64 --num-workers 2 --seed "$SEED" --device cuda
