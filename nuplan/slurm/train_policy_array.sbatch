#!/bin/bash
# F5: train the 2x2 — array task i selects (head, goal) cell i.
# Resume: resubmit this same file; each cell picks up its latest.pt.
#SBATCH --job-name=f5-2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f5-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f5-%A_%a.err

set -euo pipefail

HEADS=(det det diff diff)
GOALS=(route precise route precise)
HEAD="${HEADS[$SLURM_ARRAY_TASK_ID]}"
GOAL="${GOALS[$SLURM_ARRAY_TASK_ID]}"

DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_v2"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/f5_2x2"
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
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head "$HEAD" --goal "$GOAL" \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --batch-size 64 --num-workers 2 --seed 0 --device cuda
