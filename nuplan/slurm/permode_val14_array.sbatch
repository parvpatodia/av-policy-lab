#!/bin/bash
# Selection-bottleneck study, step 1: per-mode closed-loop eval on a Val14 subset.
# Forces each WTA mode (WTA_MODE_INDEX 0..5) through closed-loop -> per-mode CLS ->
# best-of-modes oracle, to complete the standard-split oracle-vs-selector picture
# (ADR-049 was on an old subset). 6 modes x 4 shards of a 300-token Val14 subset.
# Launch: sbatch --array=0-23%16 permode_val14_array.sbatch
#SBATCH -J permode_v14
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/permode14-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/permode14-%A_%a.err
set -euo pipefail

N_SHARDS=4
idx=$SLURM_ARRAY_TASK_ID
mode=$(( idx / N_SHARDS )); shard=$(( idx % N_SHARDS ))
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
CKPT=/scratch/patodia.pa/av-policy-lab/runs/wta_derisk/rl_long_s6000.pt
SLICE=$ET/permode_v14_sub300_m${mode}_s${shard}.json
EXP="eval/permode_val14_r1/mode${mode}_shard${shard}"
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
echo "MODE $mode shard $shard"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head wta --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$SLICE" --reactive 1 --exp-name "$EXP"
echo "done mode $mode shard $shard"
