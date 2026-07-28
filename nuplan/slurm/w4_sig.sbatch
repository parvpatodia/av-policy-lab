#!/bin/bash
#SBATCH -J w4sig
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH -o /home/patodia.pa/w4sig-%j.out
#SBATCH -e /home/patodia.pa/w4sig-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1
cd /home/patodia.pa/av-policy-lab
/home/patodia.pa/.conda/envs/nuplan/bin/python -u nuplan/analysis/w4_significance.py
echo "[w4sig] DONE rc=$?"
