#!/bin/bash
#================================================================
# SLURM array job: nuPlan F0 vectorized scene-feature extraction
#
# WHY array (not single job):
#   nuPlan map queries (SQLite spatial lookups) make single-process
#   extraction of mini take ~90 h.  16 parallel tasks, each handling
#   4 of the 64 mini DB files, complete in ~1-2 h wall-clock on the
#   `short` partition (48 h limit, 45+ mix nodes available).
#
# Usage:
#   sbatch nuplan/slurm/extract_features_array.sbatch
#
# After all tasks finish:
#   python nuplan/slurm/merge_shards.py --verify-only   # check all shards OK
#
# Outputs:
#   /scratch/$USER/av-policy-lab/features/f0_balanced/task_NNNN/  (16 sub-dirs)
#   /scratch/$USER/av-policy-lab/logs/f0-array-{JOB}_{TASK}.out
#================================================================
#SBATCH --job-name=av-f0-array
#SBATCH --partition=short
#SBATCH --array=0-15          # 16 tasks, each handles 4 of 64 mini DB files
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=47:00:00        # 47 h per task; each processes ~4 logs (~90 min)
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f0-array-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f0-array-%A_%a.err

set -euo pipefail

echo "======================================="
echo "F0 array task ${SLURM_ARRAY_TASK_ID}/15"
echo "Date  : $(date)"
echo "Node  : $(hostname)"
echo "Job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "======================================="

# ── Paths (hardcoded to actual staged layout) ─────────────────
DATA_ROOT="/scratch/patodia.pa/nuplan/data/cache/mini"
MAP_ROOT="/scratch/patodia.pa/nuplan/maps"
OUT_DIR="/scratch/patodia.pa/av-policy-lab/features/f0_v2"
REPO_ROOT="/home/patodia.pa/av-policy-lab"
LOG_DIR="/scratch/patodia.pa/av-policy-lab/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── Conda ─────────────────────────────────────────────────────
# WHY module purge first: avoids stale PATH/LD_LIBRARY_PATH from
# login-shell init that can shadow conda's Python.
module purge
module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nuplan

# WHY OMP_NUM_THREADS = allocated CPUs: numpy (heavy in map queries)
# spawns OMP threads; capping to our allocation prevents oversubscription.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_ROOT"

echo ""
echo "Python : $(which python)"
echo "DATA   : $DATA_ROOT"
echo "MAP    : $MAP_ROOT"
echo "OUT    : $OUT_DIR/task_$(printf '%04d' ${SLURM_ARRAY_TASK_ID})"
echo ""

# ── Extract ───────────────────────────────────────────────────
# --array-task-id and --num-array-tasks tell the script which DB-file
# chunk to process.  It writes to a task-specific sub-dir to prevent
# shard-name collisions under concurrent writes.
python -u nuplan/features/scene_features.py \
    --data-root  "$DATA_ROOT" \
    --map-root   "$MAP_ROOT" \
    --out-dir    "$OUT_DIR" \
    --array-task-id   "${SLURM_ARRAY_TASK_ID}" \
    --num-array-tasks 16 \
    --num-scenarios-per-type 20 \
    --stride 5

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} DONE at $(date)"
    NSHARDS=$(ls "$OUT_DIR/task_$(printf '%04d' ${SLURM_ARRAY_TASK_ID})"/*.pt 2>/dev/null | wc -l)
    echo "Shards written: $NSHARDS"
else
    echo "Task ${SLURM_ARRAY_TASK_ID} FAILED (exit $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

echo "======================================="
