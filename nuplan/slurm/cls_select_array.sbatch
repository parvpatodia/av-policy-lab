#!/bin/bash
# CLS-selection sweep (ADR-018). array task = (cell x candidate-epoch). Each task runs the
# official closed-loop CLS on the FROZEN 50-probe (cls_probe50.json, DISJOINT from the eval
# manifest). After all tasks, cls_select.py picks argmax-CLS epoch per cell.
# CPU short partition: measured ~4.3 min/scenario, so 48 probe scenarios ~= 3.5h/task < wall;
# independent of the training GPUs (no contention). Resume-safe: skip if aggregator exists;
# skip (exit 0) if the cell has not yet reached the candidate epoch.
#
# Launch: RUN_TAG=f5_2x2_v4 PROBE=/scratch/.../eval_tokens/cls_probe50.json \
#         sbatch --array=0-59%20 nuplan/slurm/cls_select_array.sbatch
#SBATCH --job-name=cls-select
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=11:55:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/clssel-%A_%a.out
#SBATCH --error=/scratch/%u/av-policy-lab/logs/clssel-%A_%a.err
set -euo pipefail

CAND_EPOCHS=(110 120 130 140 150)
NE=${#CAND_EPOCHS[@]}
HEADS=(det det diff diff); GOALS=(route precise route precise); SEEDS=(0 1 2)

idx=$SLURM_ARRAY_TASK_ID
cell=$(( idx / NE )); ei=$(( idx % NE ))
head="${HEADS[$(( cell % 4 ))]}"; goal="${GOALS[$(( cell % 4 ))]}"; seed="${SEEDS[$(( cell / 4 ))]}"
EPOCH="${CAND_EPOCHS[$ei]}"
CELL="${head}_${goal}_seed${seed}"

REPO=/home/patodia.pa/av-policy-lab
PYBIN=/home/patodia.pa/.conda/envs/nuplan/bin/python
RUNS=/scratch/patodia.pa/av-policy-lab/runs/${RUN_TAG:?set RUN_TAG}
PROBE=${PROBE:?set PROBE (frozen disjoint probe tokens)}
CKPT=$(printf "%s/%s/epoch_%03d.pt" "$RUNS" "$CELL" "$EPOCH")
EXP="clssel/${RUN_TAG}/${CELL}_e$(printf %03d "$EPOCH")"
OUT=/scratch/patodia.pa/av-policy-lab/sim_results/${EXP}

if compgen -G "$OUT/**/aggregator_metric/*.parquet" >/dev/null; then
  echo "skip (done): $EXP"; exit 0
fi
[ -f "$CKPT" ] || { echo "skip (cell not yet at epoch $EPOCH): $CELL"; exit 0; }

module purge; export PYTHONNOUSERSITE=1
cd "$REPO"
echo "CLS-select $CELL epoch $EPOCH on frozen probe ($PROBE)"
PYTHONPATH="$REPO/nuplan" "$PYBIN" nuplan/eval/run_cells.py \
    --planner policy --head "$head" --goal "$goal" --ckpt "$CKPT" \
    --tokens-file "$PROBE" --reactive "${REACTIVE:-0}" --exp-name "$EXP"
