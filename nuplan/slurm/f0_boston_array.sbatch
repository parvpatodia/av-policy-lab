#!/bin/bash
#================================================================
# Boston de-risk retrain, step 2: F0 feature extraction (ADR-061).
# Mirrors extract_features_array.sbatch (the f0_v3 mini recipe) EXACTLY, changing only
# the data root: mini -> a matched-scale 64-log Boston subset (boston_sub64). This
# isolates data SOURCE (mini vs real train-split) from scale for the de-risk comparison.
# Same sampling: --num-scenarios-per-type 20 --stride 5 --perturb-prob 0.5, 16 tasks / 4 DBs.
# Launch (only after Boston unzip fully done): sbatch nuplan/slurm/f0_boston_array.sbatch
#================================================================
#SBATCH --job-name=f0-boston
#SBATCH --partition=short
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=47:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f0boston-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f0boston-%A_%a.err
set -euo pipefail

DATA_ROOT="/scratch/patodia.pa/nuplan/train_stage/data/cache/boston_sub64"
MAP_ROOT="/scratch/patodia.pa/nuplan/maps"
OUT_DIR="/scratch/patodia.pa/av-policy-lab/features/f0_boston"
REPO_ROOT="/home/patodia.pa/av-policy-lab"
LOG_DIR="/scratch/patodia.pa/av-policy-lab/logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"

TASK_OUT="$OUT_DIR/task_$(printf '%04d' ${SLURM_ARRAY_TASK_ID})"
if compgen -G "$TASK_OUT/*.pt" >/dev/null; then
    echo "skip (done): task ${SLURM_ARRAY_TASK_ID} already has shards"; exit 0
fi

module purge
export PYTHONNOUSERSITE=1
PYBIN="/home/patodia.pa/.conda/envs/nuplan/bin/python"
"$PYBIN" -c "import nuplan, torch" || { echo "FATAL: $PYBIN cannot import nuplan/torch" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$REPO_ROOT"

echo "F0-boston task ${SLURM_ARRAY_TASK_ID}/15  node=$(hostname)  DATA=$DATA_ROOT"
"$PYBIN" -u nuplan/features/scene_features.py \
    --data-root  "$DATA_ROOT" \
    --map-root   "$MAP_ROOT" \
    --out-dir    "$OUT_DIR" \
    --array-task-id   "${SLURM_ARRAY_TASK_ID}" \
    --num-array-tasks 16 \
    --num-scenarios-per-type 20 \
    --stride 5 \
    --perturb-prob 0.5
NSHARDS=$(ls "$TASK_OUT"/*.pt 2>/dev/null | wc -l)
echo "task ${SLURM_ARRAY_TASK_ID} DONE $(date) — shards: $NSHARDS"
