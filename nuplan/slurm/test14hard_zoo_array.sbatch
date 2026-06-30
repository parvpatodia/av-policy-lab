#!/bin/bash
# Phase-1 Test14-hard zoo eval (under IDM). 4 configs x 8 shards of the 272 Test14-hard
# tokens, on the TEST split DBs. Self-confirms membership (logs "Building simulations
# from N scenarios"). Resume-safe.  Launch:  sbatch --array=0-31%16 this.sbatch
#SBATCH -J t14_zoo
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/t14zoo-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/t14zoo-%A_%a.err
set -euo pipefail

N_SHARDS=8
RUNS=/scratch/patodia.pa/av-policy-lab/runs
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
TEST_DB=/scratch/patodia.pa/nuplan/test_stage/data/cache/test
SLICE=$ET/test14hard_${NAME}_s${shard}.json
EXP="eval/test14hard_zoo_r${REACTIVE}/${NAME}_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done: $EXP"; exit 0; fi

"$PYBIN" - "$ET/test14hard.json" "$shard" "$N_SHARDS" "$SLICE" <<'PY'
import json, sys
toks = json.load(open(sys.argv[1]))["tokens"]
s, k, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump({"tokens": toks[s::k]}, open(out, "w"))
print("shard", s, "/", k, ":", len(toks[s::k]), "tokens")
PY

module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
ARGS=(--planner "$PLANNER" --db-dir "$TEST_DB" --tokens-file "$SLICE" --reactive "$REACTIVE" --exp-name "$EXP")
[ -n "$HEAD" ] && ARGS+=(--head "$HEAD")
[ -n "$GOAL" ] && ARGS+=(--goal "$GOAL")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
echo "RUN $NAME shard $shard reactive $REACTIVE: ${ARGS[*]}"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py "${ARGS[@]}"
echo "done $NAME shard $shard"
