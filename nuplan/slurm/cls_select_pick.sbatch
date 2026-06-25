#!/bin/bash
# Run cls_select.py --select on a COMPUTE node (login node OOM-kills pandas).
# Picks best-epoch-per-cell from the sweep parquets -> cls_selected.json. Fast (~min).
#SBATCH --job-name=cls-pick
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clspick-%j.out
set -euo pipefail
PY=/home/patodia.pa/.conda/envs/nuplan/bin/python
BASE=/scratch/patodia.pa/av-policy-lab/sim_results/clssel/f5_2x2_v4
OUT=/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4/cls_selected.json
CELLS="det_route_seed0 det_precise_seed0 diff_route_seed0 diff_precise_seed0 det_route_seed1 det_precise_seed1 diff_route_seed1 diff_precise_seed1 det_route_seed2 det_precise_seed2 diff_route_seed2 diff_precise_seed2"
cd /home/patodia.pa/av-policy-lab
"$PY" nuplan/eval/cls_select.py --select "$BASE" --cells $CELLS --epochs 110 120 130 140 150 --out "$OUT"
echo "=== cls_selected.json ==="; cat "$OUT"
