#!/bin/bash
# Boston de-risk step 4 zoo eval on the SAME 300-token Val14 subset as ADR-058, under IDM.
# 2 configs x 4 shards: det_route (Boston det best.pt) and wta_route (Boston WTA, DEFAULT head
# argmax = "imitation-score" default-WTA). Together with permode_boston_r1 (oracle + random) this
# gives the full core comparison oracle/det/default-WTA/random to compare against ADR-058/059.
# The learned CLS-selector (sel) is deferred (ADR-063). cfg 0 = det (needs det ckpt), cfg 1 = wta (ready).
# Launch wta-only now: sbatch --array=4-7 nuplan/slurm/boston_zoo_array.sbatch
# Launch det when training done: sbatch --array=0-3 nuplan/slurm/boston_zoo_array.sbatch
#SBATCH -J boston_zoo
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/boszoo-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/boszoo-%A_%a.err
set -euo pipefail

N_SHARDS=4
RUNS=/scratch/patodia.pa/av-policy-lab/runs
# name|planner|head|goal|ckpt
CONFIGS=(
  "det_route|policy|det|route|$RUNS/boston_derisk/det_route_seed1/best.pt"
  "wta_route|policy|wta|route|$RUNS/boston_derisk/wta_route_seed1/best.pt"
)
idx=$SLURM_ARRAY_TASK_ID
cfg=$(( idx / N_SHARDS )); shard=$(( idx % N_SHARDS ))
IFS='|' read -r NAME PLANNER HEAD GOAL CKPT <<< "${CONFIGS[$cfg]}"
REACTIVE=${REACTIVE:-1}

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
SLICE=$ET/boston_zoo_seed1_${NAME}_s${shard}.json
EXP="eval/boston_zoo_matched_seed1_r${REACTIVE}/${NAME}_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done: $EXP"; exit 0; fi
[ -n "$CKPT" ] && [ ! -f "$CKPT" ] && { echo "SKIP: ckpt not ready yet: $CKPT"; exit 0; }

"$PYBIN" - "$ET/val14_sub300.json" "$shard" "$N_SHARDS" "$SLICE" <<'PY'
import json, sys
toks = json.load(open(sys.argv[1]))["tokens"]
s, k, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump({"tokens": toks[s::k]}, open(out, "w"))
print("shard", s, "/", k, ":", len(toks[s::k]), "tokens")
PY

module purge
export PYTHONNOUSERSITE=1
cd "$REPO"
# NOTE: no WTA_MODE_INDEX -> wta uses head argmax (default-WTA / imitation score).
ARGS=(--planner "$PLANNER" --head "$HEAD" --goal "$GOAL" --ckpt "$CKPT" \
      --db-dir "$VAL_DB" --tokens-file "$SLICE" --reactive "$REACTIVE" --exp-name "$EXP")
echo "RUN $NAME shard $shard reactive $REACTIVE"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py "${ARGS[@]}"
echo "done $NAME shard $shard"
