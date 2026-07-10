#!/bin/bash
# Scoped Boston-first de-risk retrain (ADR-060, Parv-approved 2026-07-10).
# Download + extract the nuPlan train_boston split (35.5 GB compressed) from public S3.
# Compute nodes have direct outbound (no proxy env, matching the working dl_test.sbatch).
#SBATCH -J nuplan_boston_dl
#SBATCH --partition=short
#SBATCH -t 36:00:00
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -o /home/patodia.pa/nuplan_boston_dl-%j.out
#SBATCH -e /home/patodia.pa/nuplan_boston_dl-%j.err
set -e
export PYTHONNOUSERSITE=1
PY=/home/patodia.pa/.conda/envs/nuplan/bin/python
DLDIR=/scratch/patodia.pa/nuplan/downloads
STAGE=/scratch/patodia.pa/nuplan/train_stage
mkdir -p "$DLDIR" "$STAGE"
echo "[dl] start $(date)"
"$PY" /home/patodia.pa/dl_split.py public/nuplan-v1.1/nuplan-v1.1_train_boston.zip "$DLDIR/nuplan-v1.1_train_boston.zip"
echo "[unzip] start $(date)"
cd "$STAGE"
unzip -n -q "$DLDIR/nuplan-v1.1_train_boston.zip"
echo "[unzip] done $(date)"
echo "[db sample]"; find "$STAGE" -iname "*.db" | head -3
echo "[db count] $(find "$STAGE" -iname "*.db" | wc -l)"
echo "[ALL DONE] $(date)"
