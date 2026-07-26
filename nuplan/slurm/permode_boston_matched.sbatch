#!/bin/bash
# Boston de-risk retrain step 4: per-mode closed-loop eval on the SAME 300-token Val14 subset
# as ADR-058, but with the BOSTON-trained WTA head (wta_route_seed0/best.pt). Forces each mode
# (WTA_MODE_INDEX 0..5) -> per-mode CLS -> best-of-modes oracle; the default (no override) path
# uses the head's own argmax score = "imitation-score" default-WTA. eval planner needs only
# encoder+head (verified: load_ema_into ignores the rl key). Compare gaps to ADR-058/059.
# Launch: sbatch --array=0-23%16 nuplan/slurm/permode_boston_array.sbatch
#SBATCH -J permode_bos
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/permodebos-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/permodebos-%A_%a.err
set -euo pipefail

N_SHARDS=4
idx=$SLURM_ARRAY_TASK_ID
mode=$(( idx / N_SHARDS )); shard=$(( idx % N_SHARDS ))
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
CKPT=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/wta_route_seed0/best.pt
SLICE=$ET/permode_boston_sub300_m${mode}_s${shard}.json
EXP="eval/permode_boston_matched/mode${mode}_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done"; exit 0; fi

"$PYBIN" - "$ET/val14_sub300.json" "$shard" "$N_SHARDS" "$SLICE" <<'PY'
import json, sys
toks = json.load(open(sys.argv[1]))["tokens"]
s, k, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump({"tokens": toks[s::k]}, open(out, "w")); print("shard", s, k, len(toks[s::k]))
PY

module purge
export PYTHONNOUSERSITE=1
export WTA_MODE_INDEX=$mode
cd "$REPO"
echo "BOSTON MODE $mode shard $shard  ckpt=$CKPT"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head wta --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$SLICE" --reactive 1 --exp-name "$EXP"
echo "done boston mode $mode shard $shard"
