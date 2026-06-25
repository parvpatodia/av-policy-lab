#!/bin/bash
# Moderation eval (#18), v4. array task = (cell x manifest-shard). 12 cells (det/diff x
# route/precise x 3 seeds) each evaluated at its CLS-SELECTED checkpoint on a shard of the
# FROZEN manifest (N=800). CPU short: ~4.3 min/scenario measured, 800/8=100 scen/task ~= 7.2h
# < 12h wall. Run once per reactive mode (REACTIVE=0 CLS-NR box replay; REACTIVE=1 CLS-R IDM).
# Resume-safe: skip if this cell-shard output exists.
#
# Launch (after cls_select produces SELECTED):
#   RUN_TAG=f5_2x2_v4 MANIFEST=.../eval_tokens/manifest.json \
#   SELECTED=.../runs/f5_2x2_v4/cls_selected.json REACTIVE=0 \
#   sbatch --array=0-95%24 nuplan/slurm/eval_array_v4.sbatch
#SBATCH --job-name=f5-eval-v4
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=11:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/evalv4-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/evalv4-%A_%a.err
set -euo pipefail

N_SHARDS=16
HEADS=(det det diff diff); GOALS=(route precise route precise); SEEDS=(0 1 2)
idx=$SLURM_ARRAY_TASK_ID
cell=$(( idx / N_SHARDS )); shard=$(( idx % N_SHARDS ))
head="${HEADS[$(( cell % 4 ))]}"; goal="${GOALS[$(( cell % 4 ))]}"; seed="${SEEDS[$(( cell / 4 ))]}"
CELL="${head}_${goal}_seed${seed}"
REACTIVE=${REACTIVE:-0}

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
RUNS=/scratch/patodia.pa/av-policy-lab/runs/${RUN_TAG:?set RUN_TAG}
MANIFEST=${MANIFEST:?set MANIFEST}
SELECTED=${SELECTED:?set SELECTED (cls_selected.json: cell -> best_epoch)}
SLICE=/scratch/patodia.pa/av-policy-lab/eval_tokens/${RUN_TAG}_r${REACTIVE}_${CELL}_s${shard}.json
EXP="eval/${RUN_TAG}_r${REACTIVE}/${CELL}_shard${shard}"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then echo "skip done: $EXP"; exit 0; fi

# CLS-selected epoch for this cell
EPOCH=$("$PYBIN" -c "import json,sys; print(json.load(open('$SELECTED')).get('$CELL',{}).get('best_epoch',''))")
[ -n "$EPOCH" ] || { echo "no selected epoch for $CELL (cls_select not done?)"; exit 0; }
CKPT=$(printf "%s/%s/epoch_%03d.pt" "$RUNS" "$CELL" "$EPOCH")
[ -f "$CKPT" ] || { echo "FATAL: no ckpt $CKPT" >&2; exit 1; }

mkdir -p /scratch/patodia.pa/av-policy-lab/eval_tokens
"$PYBIN" - "$MANIFEST" "$shard" "$N_SHARDS" "$SLICE" <<'PY'
import json, sys
m, shard, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
toks = json.load(open(m))["tokens"]
json.dump(toks[shard::k], open(out, "w"))
print(f"shard {shard}/{k}: {len(toks[shard::k])} tokens")
PY

module purge; export PYTHONNOUSERSITE=1
cd "$REPO"
echo "EVAL $CELL epoch $EPOCH shard $shard reactive $REACTIVE"
PYTHONPATH="$REPO/nuplan" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head "$head" --goal "$goal" --ckpt "$CKPT" \
    --tokens-file "$SLICE" --reactive "$REACTIVE" --exp-name "$EXP"
