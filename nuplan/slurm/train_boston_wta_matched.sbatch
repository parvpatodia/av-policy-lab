#!/bin/bash
# Boston de-risk: MATCHED-BUDGET WTA (removes confound B, ADR-072/073). Trains the 6-mode WTA head with
# the IDENTICAL trainer/budget/schedule as det (train_policy.py --head wta, 150 epochs, same f0_boston,
# same optimizer) so oracle(best-of-6) vs det is a FAIR latent-value test. The plain wta_derisk_train
# (4000 steps) gave det >> oracle (-0.055), confounded by det's ~57x larger budget. GPU/pytorch_env.
#SBATCH --job-name=bos-wtaM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/boswtaM-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/boswtaM-%j.err
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
echo "boston wta MATCHED (train_policy, 150ep)  node=$(hostname)  DATA=$DATA_ROOT"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head wta --n-modes 6 --goal route \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --patience 9999 --ckpt-every 25 \
    --batch-size 64 --num-workers 2 --seed 0 --device cuda
echo "boston wta MATCHED DONE $(date)"
