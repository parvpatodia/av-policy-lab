#!/bin/bash
# Phase-1 Val14 zoo eval (under IDM). array task = (config x shard). 4 configs
# (PDM-Closed, det/route, WTA/route, closed-loop-selector/route) x 16 shards of the
# 1118 Val14 tokens. ~70 tokens/shard x ~4.3 min ~= 5h/task < 48h. Resume-safe.
# Launch:  sbatch --array=0-63%16 val14_zoo_array.sbatch
#SBATCH -J val14_zoo
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/val14zoo-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/val14zoo-%A_%a.err
set -euo pipefail

N_SHARDS=16
RUNS=/scratch/patodia.pa/av-policy-lab/runs
# name|planner|head|goal|ckpt
CONFIGS=(
  "pdm|pdm_closed|||"
  "det_route|policy|det|route|$RUNS/f5_2x2_v4/det_route_seed0/epoch_120.pt"
  "wta_route|policy|wta|route|$RUNS/wta_derisk/rl_long_s6000.pt"
  "sel_route|policy|wta|route|$RUNS/wta_derisk/rl_clsel.pt"
)
idx=$SLURM_ARRAY_TASK_ID
cfg=$(( idx / N_SHARDS )); shard=$(( idx % N_SHARDS ))
IFS='|' read -r NAME PLANNER HEAD GOAL CKPT <<< "${CONFIGS[$cfg]}"
REACTIVE=${REACTIVE:-1}

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
SLICE=$ET/val14_${NAME}_s${shard}.json
EXP="eval/val14_zoo_r${REACTIVE}/${NAME}_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done: $EXP"; exit 0; fi

"$PYBIN" - "$ET/val14.json" "$shard" "$N_SHARDS" "$SLICE" <<'PY'
import json, sys
toks = json.load(open(sys.argv[1]))["tokens"]
s, k, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump({"tokens": toks[s::k]}, open(out, "w"))
print("shard", s, "/", k, ":", len(toks[s::k]), "tokens")
PY

module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
ARGS=(--planner "$PLANNER" --db-dir "$VAL_DB" --tokens-file "$SLICE" --reactive "$REACTIVE" --exp-name "$EXP")
[ -n "$HEAD" ] && ARGS+=(--head "$HEAD")
[ -n "$GOAL" ] && ARGS+=(--goal "$GOAL")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
echo "RUN $NAME shard $shard reactive $REACTIVE: ${ARGS[*]}"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py "${ARGS[@]}"
echo "done $NAME shard $shard"
