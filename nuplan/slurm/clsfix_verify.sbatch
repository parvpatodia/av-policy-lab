#!/bin/bash
#SBATCH --job-name=clsfix
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clsfix-%j.out
set -euo pipefail
REPO=/home/patodia.pa/av-policy-lab
PY=/home/patodia.pa/.conda/envs/nuplan/bin/python
CKPT=/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4/det_route_seed0/epoch_150.pt
module purge; export PYTHONNOUSERSITE=1
cd "$REPO"
rm -rf /scratch/patodia.pa/av-policy-lab/sim_results/clssel_fixtest
PYTHONPATH="$REPO/nuplan" "$PY" nuplan/eval/run_cells.py --planner policy --head det --goal route \
  --ckpt "$CKPT" --tokens-file /scratch/patodia.pa/av-policy-lab/eval_tokens/_fixtest2.json \
  --exp-name clssel_fixtest/det_route_seed0
