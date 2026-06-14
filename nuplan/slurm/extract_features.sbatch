#!/bin/bash
#SBATCH --job-name=av-f0-extract
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/av-policy-lab/logs/f0-extract-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/f0-extract-%j.err
# WHY NO --gres / NO GPU: scene-feature extraction is CPU-bound (SQLite reads +
#   numpy geometry). A GPU would sit 100% idle — and RC bots email + flag idle GPU
#   jobs (HPC_NORTHEASTERN.md §3). Extraction belongs on the CPU `short` partition.
# WHY partition=short: it is the general-access CPU partition with the longest
#   wall-clock confirmed on this account (48h via `sinfo`). Full-split extraction is
#   a multi-hour-to-multi-day job (see throughput honesty in F0_IMPLEMENTATION.md),
#   so we take the full 48h window and shard incrementally (resumable).
# WHY cpus-per-task=16 + mem=64G: extraction is embarrassingly parallel across
#   scenarios; 16 cores lets the (single-process) walk overlap SQLite I/O, and the
#   nuPlan map cache + a shard buffer of samples fit comfortably in 64G. Tune up if
#   you raise --scenarios-per-shard.

set -euo pipefail

# ── Cluster: Explorer (login.explorer.northeastern.edu); compute nodes have internet ──
# Paths (HPC_NORTHEASTERN.md §4):
#   /scratch  — large, shared, NOT backed up, purged monthly  -> dataset + shards (transient)
#   /home     — ~75 GB, backed up                              -> code (this repo)
#   /projects — persistent per-PI                              -> move KEEPER shards here after
#
# WHY read AND write under /scratch: dataset lives on /scratch and shards go to
# /scratch. After the run completes, copy keeper shards to /projects/<group>/
# (scratch is purged on the first Tuesday of the month).

# --- user-tunable env (override on the command line: VAR=... sbatch extract_features.sbatch) ---
# WHY these defaults: they mirror the actual staged layout and respect the env vars
# already exported in ~/.bashrc (NUPLAN_DATA_ROOT is the PARENT holding split subdirs;
# NUPLAN_MAPS_ROOT is the maps dir). DATA_SPLIT selects which split to extract.
: "${NUPLAN_DATA_ROOT:=/scratch/$USER/nuplan/data/cache}"   # parent dir of split subdirs
: "${DATA_SPLIT:=mini}"                                      # 'mini' (have it) | 'trainval' (full, once downloaded)
: "${NUPLAN_MAPS_ROOT:=/scratch/$USER/nuplan/maps}"
: "${OUT_DIR:=/scratch/$USER/av-policy-lab/features/f0_balanced}"
: "${SCENARIOS_PER_SHARD:=8}"
: "${STRIDE:=1}"
: "${LIMIT:=}"   # leave empty for the whole split; set e.g. 2000 for a costed pilot
: "${CONDA_ENV:=nuplan}"
# WHY a fixed REPO_ROOT default in /home: code lives in the small backed-up space.
: "${REPO_ROOT:=$HOME/av-policy-lab}"

DATA_DIR="$NUPLAN_DATA_ROOT/$DATA_SPLIT"   # e.g. /scratch/$USER/nuplan/data/cache/mini

mkdir -p "/scratch/$USER/av-policy-lab/logs" "$OUT_DIR"

echo "[extract] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[extract] data_dir=$DATA_DIR  (split=$DATA_SPLIT)"
echo "[extract] map_root=$NUPLAN_MAPS_ROOT"
echo "[extract] out_dir=$OUT_DIR  per_shard=$SCENARIOS_PER_SHARD  stride=$STRIDE  limit=${LIMIT:-<all>}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[extract] ERROR: data dir not found: $DATA_DIR" >&2
  echo "          set NUPLAN_DATA_ROOT (parent) and DATA_SPLIT correctly, or stage the data." >&2
  exit 2
fi

# ── conda env (nuPlan devkit lives here) ──
# WHY `source ... conda.sh` then activate: `conda activate` needs the shell hook,
# which is not sourced in a non-interactive SLURM shell by default.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Help BLAS/OMP not oversubscribe the allocated cores (numpy is the hot path).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$REPO_ROOT"

# WHY run the module by path under nuplan/: matches the repo's conftest convention
# (nuplan/ on sys.path). The extractor's devkit imports resolve from the conda env.
LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then LIMIT_ARG="--limit ${LIMIT}"; fi

srun python -u nuplan/features/scene_features.py \
    --data-root "$DATA_DIR" \
    --map-root  "$NUPLAN_MAPS_ROOT" \
    --out-dir   "$OUT_DIR" \
    --scenarios-per-shard "$SCENARIOS_PER_SHARD" \
    --stride "$STRIDE" \
    $LIMIT_ARG

echo "[extract] done. shards under: $OUT_DIR"
echo "[extract] REMINDER: copy keeper shards to /projects/<group>/ before the monthly /scratch purge."

# ── Optional: array-shard a >24h run across two jobs if `short` caps below 48h ──
# Add `#SBATCH --array=0-1` above and split the scenario set with --limit + an
# offset flag (extend the CLI before doing this). Documented, not enabled here, to
# keep the default a single resumable job.
