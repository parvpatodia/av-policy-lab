#!/bin/bash
# Run ONE zoo config in closed-loop on the val DBs. Env-parametrized so the same
# script serves det / wta / selector (policy planners) and pdm_closed (rule-based).
# tuplan_garage is added to PYTHONPATH so PDMClosedPlanner imports.
#SBATCH -J val_eval
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/valeval-%j.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/valeval-%j.err
set -euo pipefail
module purge
export PYTHONNOUSERSITE=1
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
PLANNER="${PLANNER:?set PLANNER}"; TOKENS="${TOKENS:?set TOKENS}"; EXP="${EXP:?set EXP}"
REACTIVE="${REACTIVE:-1}"
ARGS=(--planner "$PLANNER" --db-dir "$VAL_DB" --tokens-file "$TOKENS" --reactive "$REACTIVE" --exp-name "$EXP")
[ -n "${HEAD:-}" ] && ARGS+=(--head "$HEAD")
[ -n "${GOAL:-}" ] && ARGS+=(--goal "$GOAL")
[ -n "${CKPT:-}" ] && ARGS+=(--ckpt "$CKPT")
cd "$REPO"
echo "[val_eval] ${ARGS[*]} $(date)"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py "${ARGS[@]}"
echo "[val_eval] done $(date)"
