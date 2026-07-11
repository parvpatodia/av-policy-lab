#!/bin/bash
# Boston de-risk mechanism half (ADR-048/GATE-CL-1 reproduction): does the open-loop proxy reward
# predict the closed-loop CLS on real Boston models? Uses the Boston WTA head + Val14 scene features
# (f0_val) + the Boston default-WTA deployed CLS (boston_zoo wta_route). CPU/short (light forward pass
# on ~300 tokens; avoids GPU contention with det training). Compare pearson to mini r=0.11.
#SBATCH -J clcorr_bos
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clcorrbos-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/clcorrbos-%j.err
set -euo pipefail
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
PYTHONPATH="$REPO/nuplan" "$PYBIN" -u nuplan/analysis/cl_corr.py \
    --rl-ckpt /scratch/patodia.pa/av-policy-lab/runs/boston_derisk/wta_boston_s4000.pt \
    --shard-glob "/scratch/patodia.pa/av-policy-lab/features/f0_val/task_*/scene_shard_*.pt" \
    --cls-glob "eval/boston_zoo_r1/wta_route_shard*" \
    --min-tokens 300 \
    --out /scratch/patodia.pa/av-policy-lab/sim_results/cl_corr_boston.json
echo "cl_corr_boston DONE $(date)"
