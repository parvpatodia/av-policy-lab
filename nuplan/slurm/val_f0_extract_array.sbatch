#!/bin/bash
# Token-targeted f0 feature extraction for the 1118 Val14 tokens (for F4 scoring /
# H1 stratification only -- the closed-loop eval itself builds features live and needs
# no cache). Uses the --tokens-file path; array chunks the val logs across 16 tasks.
#SBATCH -J val_f0
#SBATCH --partition=short
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/valf0-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/valf0-%A_%a.err
set -euo pipefail
module purge
export PYTHONNOUSERSITE=1
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
REPO=/home/patodia.pa/av-policy-lab
DATA=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
MAP=/scratch/patodia.pa/nuplan/maps
OUT=/scratch/patodia.pa/av-policy-lab/features/f0_val
TOK=/scratch/patodia.pa/av-policy-lab/eval_tokens/val14.json
cd "$REPO"
"$PYBIN" -c "import nuplan, torch" || { echo "FATAL: nuplan/torch import" >&2; exit 1; }
"$PYBIN" -u nuplan/features/scene_features.py \
    --data-root "$DATA" --map-root "$MAP" --out-dir "$OUT" \
    --tokens-file "$TOK" \
    --array-task-id "$SLURM_ARRAY_TASK_ID" --num-array-tasks 16 \
    --stride 5
echo "val_f0 task $SLURM_ARRAY_TASK_ID done"
