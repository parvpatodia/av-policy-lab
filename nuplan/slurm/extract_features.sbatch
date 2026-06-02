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
#   wall-clock the brief documents (48h). Full-nuPlan extraction is a multi-hour-to
#   -multi-day job (see throughput honesty in docs/frontier/F0_IMPLEMENTATION.md), so
#   we take the full 48h window and checkpoint at the shard level (resumable).
#   NOTE: confirm the live `short` max wall-clock against rc.northeastern.edu/partitions/
#   — RC docs list it as version-sensitive; if it is 24h on your account, split the
#   run across two array jobs (see --array note below) rather than one 48h job.
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
# WHY read AND write under /scratch: the brief specifies dataset on /scratch and
# shards to /scratch. After the run completes, copy the shards you want to keep to
# /projects/<group>/ (scratch is purged on the first Tuesday of the month).

# --- user-tunable env (override on the command line: VAR=... sbatch extract_features.sbatch) ---
: "${NUPLAN_DATA_ROOT:=/scratch/$USER/nuplan/dataset/nuplan-v1.1/splits/trainval}"   # full split
: "${NUPLAN_MAP_ROOT:=/scratch/$USER/nuplan/dataset/maps}"
: "${OUT_DIR:=/scratch/$USER/av-policy-lab/features/f0}"
: "${SCENARIOS_PER_SHARD:=8}"
: "${STRIDE:=1}"
: "${LIMIT:=}"   # leave empty for the full split; set e.g. 2000 for a costed pilot
: "${CONDA_ENV:=nuplan}"
# WHY a fixed REPO_ROOT default in /home: code lives in the small backed-up space.
: "${REPO_ROOT:=$HOME/av-policy-lab}"

mkdir -p "/scratch/$USER/av-policy-lab/logs" "$OUT_DIR"

echo "[extract] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[extract] data_root=$NUPLAN_DATA_ROOT"
echo "[extract] map_root=$NUPLAN_MAP_ROOT"
echo "[extract] out_dir=$OUT_DIR  per_shard=$SCENARIOS_PER_SHARD  stride=$STRIDE  limit=${LIMIT:-<all>}"

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
    --data-root "$NUPLAN_DATA_ROOT" \
    --map-root  "$NUPLAN_MAP_ROOT" \
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
