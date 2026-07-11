#!/bin/bash
# Boston de-risk MATCHED verdict: fair oracle-vs-det once the matched per-mode eval lands. Reads the
# matched oracle/random from permode_boston_matched_r1 and det (0.737) from the existing boston_zoo_r1.
# NOTE: the default-WTA line printed is the PLAIN wta (boston_zoo_r1); the MATCHED claim is oracle-vs-det
# and oracle-vs-random (both come from the matched per-mode). Chained afterany on the matched per-mode.
#SBATCH -J bosMverdict
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosMverdict-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosMverdict-%j.err
set -euo pipefail
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/boston_derisk_matched_verdict.txt
module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
echo "Boston de-risk MATCHED verdict (fair oracle-vs-det) $(date)" | tee "$OUT"
echo "matched oracle/random from permode_boston_matched_r1; det from boston_zoo_r1 (existing)" | tee -a "$OUT"
PYTHONPATH="$REPO/nuplan" "$PYBIN" -u nuplan/analysis/analyze_permode_boston.py permode_boston_matched_r1 boston_zoo_r1 2>&1 | grep -v -i warning | tee -a "$OUT"
echo "verdict written to $OUT"
