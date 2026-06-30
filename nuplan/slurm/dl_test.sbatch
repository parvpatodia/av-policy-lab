#!/bin/bash
#SBATCH -J nuplan_test_dl
#SBATCH --partition=short
#SBATCH -t 36:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o /home/patodia.pa/nuplan_test_dl-%j.out
#SBATCH -e /home/patodia.pa/nuplan_test_dl-%j.err
set -e
export PYTHONNOUSERSITE=1
PY=/home/patodia.pa/.conda/envs/nuplan/bin/python
DLDIR=/scratch/patodia.pa/nuplan/downloads
STAGE=/scratch/patodia.pa/nuplan/test_stage
mkdir -p "$DLDIR" "$STAGE"
echo "[dl] start $(date)"
"$PY" /home/patodia.pa/dl_split.py public/nuplan-v1.1/nuplan-v1.1_test.zip "$DLDIR/nuplan-v1.1_test.zip"
echo "[unzip] start $(date)"
cd "$STAGE"
unzip -n -q "$DLDIR/nuplan-v1.1_test.zip"
echo "[unzip] done $(date)"
echo "[db sample]"; find "$STAGE" -iname "*.db" | head -3
echo "[db count] $(find "$STAGE" -iname "*.db" | wc -l)"
