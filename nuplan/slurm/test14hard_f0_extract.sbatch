#!/bin/bash
# Token-targeted f0 extraction for the 272 Test14-hard tokens (for s_inter / H1
# stratification on the hard split). Array chunks the test logs across 8 tasks.
#SBATCH -J t14_f0
#SBATCH --partition=short
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/t14f0-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/t14f0-%A_%a.err
set -euo pipefail
module purge
export PYTHONNOUSERSITE=1
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
REPO=/home/patodia.pa/av-policy-lab
DATA=/scratch/patodia.pa/nuplan/test_stage/data/cache/test
MAP=/scratch/patodia.pa/nuplan/maps
OUT=/scratch/patodia.pa/av-policy-lab/features/f0_test14hard
TOK=/scratch/patodia.pa/av-policy-lab/eval_tokens/test14hard.json
cd "$REPO"
"$PYBIN" -c "import nuplan, torch" || { echo "FATAL import" >&2; exit 1; }
"$PYBIN" -u nuplan/features/scene_features.py \
    --data-root "$DATA" --map-root "$MAP" --out-dir "$OUT" \
    --tokens-file "$TOK" \
    --array-task-id "$SLURM_ARRAY_TASK_ID" --num-array-tasks 8 \
    --stride 5
echo "t14_f0 task $SLURM_ARRAY_TASK_ID done"
