#!/bin/bash
# Boston de-risk retrain step 3b: 6-mode WTA head on Boston f0 (ADR-062).
# Mirrors the wta_derisk_train.py recipe (n-modes 6, 4000 steps, 16000 scenes to match
# mini scale) changing ONLY --shard-glob f0_v3 -> f0_boston. GPU / pytorch_env.
# Launch AFTER f0_boston completes: sbatch nuplan/slurm/train_boston_wta.sbatch
#SBATCH --job-name=bos-wta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/boswta-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/boswta-%j.err
set -euo pipefail
REPO_ROOT="/home/patodia.pa/av-policy-lab"
SHARDS="/scratch/patodia.pa/av-policy-lab/features/f0_boston/task_*/scene_shard_*.pt"
OUT="/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/wta_boston_s4000.pt"
mkdir -p /scratch/patodia.pa/av-policy-lab/runs/boston_derisk /scratch/patodia.pa/av-policy-lab/logs
module purge
export PYTHONNOUSERSITE=1
PYBIN="/home/patodia.pa/.conda/envs/pytorch_env/bin/python"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -c "import torch, models.scene_encoder; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "FATAL: env import or CUDA check failed" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$REPO_ROOT"
echo "boston wta 6-mode  node=$(hostname)  shards=$SHARDS"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/analysis/wta_derisk_train.py \
    --shard-glob "$SHARDS" \
    --n-scenes 16000 --n-modes 6 --steps 4000 --seed 0 \
    --out "$OUT"
echo "boston wta DONE $(date) -> $OUT"
