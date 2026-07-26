#!/bin/bash
# ADR-091 paper-grade Wave 1: MULTI-SEED training. det + WTA at seeds 1,2 (matched 150ep budget, same
# recipe/data as seed0 det_route_seed0/wta_route_seed0) -> {det,wta}_route_seed{1,2}. For across-seed
# error bars on the ADR-089/090 latent-value + selection claims. array 0-3: 0=det s1,1=det s2,2=wta s1,3=wta s2.
#SBATCH -J bos-seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosseed-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosseed-%A_%a.err
set -euo pipefail
REPO_ROOT="/home/patodia.pa/av-policy-lab"
DATA_ROOT="/scratch/patodia.pa/av-policy-lab/features/f0_boston"
CKPT_DIR="/scratch/patodia.pa/av-policy-lab/runs/boston_derisk"
mkdir -p "$CKPT_DIR" /scratch/patodia.pa/av-policy-lab/logs
t=$SLURM_ARRAY_TASK_ID
if   [ "$t" -eq 0 ]; then HEAD=det; SEED=1; EXTRA=""
elif [ "$t" -eq 1 ]; then HEAD=det; SEED=2; EXTRA=""
elif [ "$t" -eq 2 ]; then HEAD=wta; SEED=1; EXTRA="--n-modes 6"
else                      HEAD=wta; SEED=2; EXTRA="--n-modes 6"; fi
module purge
export PYTHONNOUSERSITE=1
PYBIN="/home/patodia.pa/.conda/envs/pytorch_env/bin/python"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -c "import torch, models.scene_encoder; assert torch.cuda.is_available(), 'no CUDA'" \
  || { echo "FATAL: env import or CUDA check failed" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$REPO_ROOT"
echo "boston MULTI-SEED head=$HEAD seed=$SEED node=$(hostname) DATA=$DATA_ROOT -> ${HEAD}_route_seed${SEED}"
PYTHONPATH="$REPO_ROOT/nuplan" "$PYBIN" -u nuplan/training/train_policy.py \
    --head "$HEAD" $EXTRA --goal route \
    --data-root "$DATA_ROOT" \
    --ckpt-dir  "$CKPT_DIR" \
    --epochs 150 --patience 9999 --ckpt-every 25 \
    --batch-size 64 --num-workers 2 --seed "$SEED" --device cuda
echo "boston seed DONE head=$HEAD seed=$SEED $(date)"
