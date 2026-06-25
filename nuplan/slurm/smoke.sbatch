#!/bin/bash
# SMOKE test of the v4 training change: 1 cell, 2 epochs, ckpt-every 1, throwaway dir.
# Verifies --patience/--ckpt-every parse, periodic epoch_*.pt are written, run completes.
#SBATCH --job-name=f5-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/smoke-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/smoke-%A_%a.err

set -euo pipefail

HEADS=(det det diff diff)
GOALS=(route precise route precise)
SEEDS=(0 1 2)
CELL=$(( SLURM_ARRAY_TASK_ID % 4 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
HEAD="${HEADS[$CELL]}"
GOAL="${GOALS[$CELL]}"

DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_v3"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/SMOKE_v4test"
REPO_ROOT="/home/patodia.pa/av-policy-lab"
mkdir -p "$CKPT_DIR" /scratch/patodia.pa/av-policy-lab/logs

module purge
export PYTHONNOUSERSITE=1
PYBIN="/home/patodia.pa/.conda/envs/pytorch_env/bin/python"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -c "import torch, models.scene_encoder; assert torch.cuda.is_available(), "no CUDA"" \
  || { echo "FATAL: env import or CUDA check failed" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "SMOKE cell: head=$HEAD goal=$GOAL node=$(hostname)"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head "$HEAD" --goal "$GOAL" \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 2 --patience 9999 --ckpt-every 1 \
    --batch-size 64 --num-workers 2 --seed "$SEED" --device cuda
