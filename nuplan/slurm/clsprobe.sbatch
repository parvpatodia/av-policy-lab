#!/bin/bash
# DE-RISK for #16 CLS-selection: prove run_cells.py produces an official-CLS number
# for a v4 cell checkpoint on a CPU node (policy_planner defaults to device=cpu, so
# no GPU contention with the running training). One done cell, one ckpt, 2 scenarios.
#SBATCH --job-name=cls-probe
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clsprobe-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/clsprobe-%j.err
set -euo pipefail

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
CKPT=/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4/det_route_seed0/epoch_150.pt
[ -f "$CKPT" ] || { echo "FATAL: no ckpt $CKPT" >&2; exit 1; }

module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
echo "probe: det/route epoch_150 on 2 scenarios, two_stage_controller (official CLS), CPU"
PYTHONPATH="$REPO/nuplan" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head det --goal route --ckpt "$CKPT" \
    --n-scenarios 2 --controller two_stage_controller \
    --exp-name clsprobe/det_route_seed0_e150
echo "EXIT $?"
