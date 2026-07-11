#!/bin/bash
# Boston de-risk VERDICT: runs the core analyzer (oracle/det/default-WTA/random on Val14 sub300) once
# the det zoo lands, writing the result to a file. Chained via --dependency=afterany on det zoo so the
# verdict computes autonomously. Light analysis -> CPU/short.
#SBATCH -J bos_verdict
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosverdict-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosverdict-%j.err
set -euo pipefail
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/boston_derisk_verdict.txt
module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
echo "Boston de-risk verdict $(date)" | tee "$OUT"
PYTHONPATH="$REPO/nuplan" "$PYBIN" -u nuplan/analysis/analyze_permode_boston.py permode_boston_r1 boston_zoo_r1 2>&1 | grep -v -i warning | tee -a "$OUT"
echo "--- cl_corr (mechanism) ---" | tee -a "$OUT"
cat /scratch/patodia.pa/av-policy-lab/sim_results/cl_corr_boston.json 2>/dev/null | tee -a "$OUT"
echo "verdict written to $OUT"
