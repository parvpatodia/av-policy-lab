#!/bin/bash
# Boston de-risk retrain step 3a: deterministic head on Boston f0 (ADR-062).
# Mirrors the f5 det_route recipe (train_policy.py --head det --goal route, 150 epochs,
# early-stop disabled, periodic ckpts) changing ONLY --data-root mini(f0_v3) -> f0_boston.
# GPU / pytorch_env. Launch AFTER f0_boston completes: sbatch nuplan/slurm/train_boston_det.sbatch
#SBATCH --job-name=bos-det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosdet-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosdet-%j.err
set -euo pipefail
REPO_ROOT="/home/patodia.pa/av-policy-lab"
DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_boston"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/boston_derisk"
mkdir -p "$CKPT_DIR" /scratch/patodia.pa/av-policy-lab/logs
module purge
export PYTHONNOUSERSITE=1
PYBIN="/home/patodia.pa/.conda/envs/pytorch_env/bin/python"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -c "import torch, models.scene_encoder; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "FATAL: env import or CUDA check failed" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$REPO_ROOT"
echo "boston det_route seed0  node=$(hostname)  DATA=$DATA_ROOT"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head det --goal route \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --patience 9999 --ckpt-every 25 \
    --batch-size 64 --num-workers 2 --seed 0 --device cuda
echo "boston det DONE $(date)"
