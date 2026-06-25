#!/bin/bash
# Per-scenario CLS timing: det vs diff head, 6 scenarios each, official two_stage_controller,
# CPU. Gives a real per-scenario rate (vs the noisy n=2 probe) to size the moderation eval
# and decide CPU-vs-GPU. array 0=det, 1=diff (route, seed0 done cell, epoch_150).
#SBATCH --job-name=cls-time
#SBATCH --partition=short
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clstiming-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/clstiming-%A_%a.err
set -euo pipefail
HEADS=(det diff)
H="${HEADS[$SLURM_ARRAY_TASK_ID]}"
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
CKPT=/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4/${H}_route_seed0/epoch_150.pt
[ -f "$CKPT" ] || { echo "no ckpt $CKPT" >&2; exit 1; }
module purge; export PYTHONNOUSERSITE=1
cd "$REPO"
echo "TIMING $H route seed0 epoch_150, 6 scenarios, CPU, $(date -Is)"
/usr/bin/time -v env PYTHONPATH="$REPO/nuplan" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head "$H" --goal route --ckpt "$CKPT" \
    --n-scenarios 6 --controller two_stage_controller \
    --exp-name clstiming/${H}_route_seed0 2>&1 | grep -E "Simulation duration|Building simulations from|successful sim|Elapsed \(wall"
echo "DONE $H $(date -Is)"
