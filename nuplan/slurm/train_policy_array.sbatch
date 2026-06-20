#!/bin/bash
# F5: train the 2x2 — array task i selects (head, goal) cell i.
# Resume: each cell picks up its latest.pt. Self-requeues near the wall clock
# (USR1 trap below) so 150 epochs spans windows without a manual resubmit.
#SBATCH --job-name=f5-2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11   # 4 cells x 3 seeds (audit: single-seed is unpublishable)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f5-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f5-%A_%a.err
# WHY --requeue + --signal: gpu QOS caps the user at 8 submitted / 4 running
# jobs and 7h55m wall time, but 150 epochs needs ~4 windows. A dependency chain
# would blow the 8-submit cap; scontrol requeue keeps the SAME job id (no extra
# submit slot) and resumes from latest.pt. USR1 is delivered to the batch shell
# (B:) 300s before the wall so the trap can requeue cleanly.
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

HEADS=(det det diff diff)
GOALS=(route precise route precise)
SEEDS=(0 1 2)
CELL=$(( SLURM_ARRAY_TASK_ID % 4 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
HEAD="${HEADS[$CELL]}"
GOAL="${GOALS[$CELL]}"

DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_v3"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v3"
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
# WHY background + wait + trap: under set -e a foreground python blocks the shell
# from handling USR1 until it returns, so the wall-clock warning would be missed.
# Backgrounded + wait lets the trap fire. On USR1: latest.pt is already on disk
# (saved every epoch), so requeue this same job id, stop the run, exit clean.
# A cell already at --epochs exits in seconds (empty loop) BEFORE USR1 fires, so
# finished cells never requeue. A real crash exits non-zero before USR1 too, so it
# FAILS under set -e instead of requeue-looping. Net: self-terminating, cap-safe.
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head "$HEAD" --goal "$GOAL" \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --batch-size 64 --num-workers 2 --seed "$SEED" --device cuda &
TRAIN_PID=$!
trap 'echo "[requeue] USR1 at $(date -Is), resubmitting ${SLURM_JOB_ID}"; scontrol requeue "${SLURM_JOB_ID}"; kill -TERM "${TRAIN_PID}" 2>/dev/null || true; exit 0' USR1
wait "${TRAIN_PID}"
