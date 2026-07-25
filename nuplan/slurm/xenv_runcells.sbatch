#!/bin/bash
# ADR-085 isolation: OUR run_cells.py harness on the 5 pinned tokens, but under the nuplan_smart env
# (torch2.1/numpy1.23). Holds harness+config+devkit(our)+planner constant; ONLY the env changes.
#   ~0.365 => ENV is the cause.  ~0.799 => env exonerated, his HARNESS/config is the cause.
#SBATCH -J xenv_rc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /home/patodia.pa/xenv_rc-%j.out
#SBATCH -e /home/patodia.pa/xenv_rc-%j.err
echo "[xenv] START $(date) node=$(hostname)"; set -x
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
REPO=/home/patodia.pa/av-policy-lab
OURDEVKIT=/home/patodia.pa/nuplan-devkit
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
CKPT=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/det_route_seed0/best.pt
SLICE=$ET/skew_control_5tok.json   # the same 5 pinned tokens (created by skew_control_idm.sbatch)
[ -f "$SLICE" ] || { echo '{"tokens": ["33a911fcc99150b1","00301fbca75b5ce4","9da92ed8992a561d","ced53747c9745b55","6980261fb6c5566c"]}' > "$SLICE"; }
export PYTHONNOUSERSITE=1
cd "$REPO"
# nuplan_smart env; `nuplan` resolves to OUR devkit clone (same commit e924167); serving/models/features
# from our repo; tuplan_garage for the reactive controller helpers.
PYTHONPATH="$OURDEVKIT:$REPO/nuplan:/home/patodia.pa/tuplan_garage" \
$C run -n nuplan_smart python nuplan/eval/run_cells.py \
    --planner policy --head det --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$SLICE" --reactive 1 --exp-name eval/xenv_runcells_smart
echo "[xenv] rc=$? DONE $(date)"
