#!/bin/bash
# Boston de-risk FINAL matched verdict: fully matched-consistent oracle / det / default-WTA / random.
# Reads matched per-mode (permode_boston_matched_r1) + boston_zoo_matched_r1 (matched default-WTA +
# det symlinked from boston_zoo_r1). Chained afterany on the matched default-WTA array.
#SBATCH -J bosMfinal
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosMfinal-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosMfinal-%j.err
set -euo pipefail
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/boston_derisk_matched_FINAL_verdict.txt
module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
echo "Boston de-risk FINAL matched verdict (all matched-consistent) $(date)" | tee "$OUT"
PYTHONPATH="$REPO/nuplan" "$PYBIN" -u nuplan/analysis/analyze_permode_boston.py permode_boston_matched_r1 boston_zoo_matched_r1 2>&1 | grep -v -iE "warning|iloc|elements\[" | tee -a "$OUT"
echo "verdict written to $OUT"
