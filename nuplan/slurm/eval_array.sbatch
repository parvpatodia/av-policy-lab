#!/bin/bash
# Sharded closed-loop eval: array task = (cell x shard). ADR-018 budget fix.
# 4 cells x N_SHARDS shards. Each task evals its token slice of the frozen
# manifest for one cell. Resume-safe: skip if the cell-shard output exists.
#SBATCH --job-name=f5-eval
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/eval-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/eval-%A_%a.err
set -euo pipefail

N_SHARDS=8
HEADS=(det det diff diff)
GOALS=(route precise route precise)
CELL=$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % N_SHARDS ))
HEAD="${HEADS[$CELL]}"; GOAL="${GOALS[$CELL]}"

REPO=/home/patodia.pa/av-policy-lab
RUNS=/scratch/patodia.pa/av-policy-lab/runs/${RUN_TAG:?set RUN_TAG}
MANIFEST=${MANIFEST:?set MANIFEST}
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
EXP="${HEAD}_${GOAL}_shard${SHARD}"
SLICE=/scratch/patodia.pa/av-policy-lab/eval_tokens/${RUN_TAG}_${EXP}.json
mkdir -p /scratch/patodia.pa/av-policy-lab/eval_tokens

# slice the frozen manifest: tokens[shard::N_SHARDS]
"$PYBIN" - "$MANIFEST" "$SHARD" "$N_SHARDS" "$SLICE" <<PY
import json, sys
m, shard, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
toks = json.load(open(m))["tokens"]
json.dump(toks[shard::k], open(out, "w"))
print(f"shard {shard}/{k}: {len(toks[shard::k])} tokens")
PY

CKPT="$RUNS/${HEAD}_${GOAL}_seed0/best.pt"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint $CKPT" >&2; exit 1; }

echo "cell=$CELL head=$HEAD goal=$GOAL shard=$SHARD reactive=${REACTIVE:-0}"
cd "$REPO"
PYTHONPATH="$REPO/nuplan" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head "$HEAD" --goal "$GOAL" \
    --ckpt "$CKPT" --tokens-file "$SLICE" \
    --reactive "${REACTIVE:-0}" \
    --exp-name "${RUN_TAG}/${EXP}"
