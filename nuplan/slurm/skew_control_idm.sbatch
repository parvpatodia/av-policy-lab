#!/bin/bash
# SKEW CONTROL (ADR-081): OUR det PolicyPlanner in OUR OWN devkit under IDM on the EXACT SAME 5 val14
# tokens that scored {0,0,0,1.0,0.823} (final 0.365) in Steffen's fork (job 8717075). Same planner,
# same ckpt, same tokens, same val DBs, same reactive=IDM -> the ONLY difference is his-fork devkit vs
# ours. If our devkit reproduces the off-road/collision zeros -> genuine weak-planner behavior (skew NOT
# the cause) -> safe to scale SMART. If our devkit scores clean -> his fork corrupts trajectories (skew)
# -> must fix before trusting any SMART number.
#SBATCH -J skew_ctrl
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/patodia.pa/skew_ctrl-%j.out
#SBATCH --error=/home/patodia.pa/skew_ctrl-%j.err
set -euo pipefail
echo "[ctrl] START $(date) node=$(hostname)"

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
CKPT=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/det_route_seed0/best.pt
SLICE=$ET/skew_control_5tok.json
EXP="eval/skew_control_idm"

mkdir -p "$ET"
# EXACT 5 tokens from job 8717075 (his fork), in order.
cat > "$SLICE" <<'JSON'
{"tokens": ["33a911fcc99150b1", "00301fbca75b5ce4", "9da92ed8992a561d", "ced53747c9745b55", "6980261fb6c5566c"]}
JSON
echo "[ctrl] tokens:" && cat "$SLICE"
[ -f "$CKPT" ] || { echo "FATAL: ckpt missing $CKPT"; exit 1; }

module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
echo "[ctrl] RUN det/route IDM in OUR devkit on 5 pinned tokens"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head det --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$SLICE" --reactive 1 --exp-name "$EXP"
echo "[ctrl] rc=$? DONE $(date)"
