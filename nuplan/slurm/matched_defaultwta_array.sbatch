#!/bin/bash
# Boston de-risk: MATCHED-head default-WTA (imitation-score / head-argmax selection, NO WTA_MODE_INDEX)
# on the same 300-token Val14 subset, so the selection-failure magnitude (oracle - default-WTA,
# default-WTA vs random) is stated with matched-consistent numbers (the earlier verdict mixed the PLAIN
# head's default-WTA with matched random). 4 shards. Launch: sbatch --array=0-3 nuplan/slurm/matched_defaultwta_array.sbatch
#SBATCH -J bosMdwta
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/bosMdwta-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/bosMdwta-%A_%a.err
set -euo pipefail
shard=$SLURM_ARRAY_TASK_ID
REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
CKPT=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/wta_route_seed0/best.pt
SLICE=$ET/bosMdwta_sub300_s${shard}.json
EXP="eval/boston_zoo_matched_r1/wta_route_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}
if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done"; exit 0; fi
[ ! -f "$CKPT" ] && { echo "FATAL: matched ckpt missing" >&2; exit 1; }
"$PYBIN" - "$ET/val14_sub300.json" "$shard" 4 "$SLICE" <<'PY'
import json, sys
toks = json.load(open(sys.argv[1]))["tokens"]
s, k, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump({"tokens": toks[s::k]}, open(out, "w")); print("shard", s, k, len(toks[s::k]))
PY
module purge
export PYTHONNOUSERSITE=1
# NO WTA_MODE_INDEX -> planner uses the head's own argmax score (default-WTA / imitation selection)
cd "$REPO"
echo "BOSTON-MATCHED default-WTA shard $shard  ckpt=$CKPT"
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head wta --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$SLICE" --reactive 1 --exp-name "$EXP"
echo "done matched default-WTA shard $shard"
