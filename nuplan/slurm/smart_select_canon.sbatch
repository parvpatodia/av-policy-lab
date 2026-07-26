#!/bin/bash
# ADR-090: SMART SELECTION sweep (canonical CLS). 8 configs under his closed_loop_smart_reactive_agents on
# the SAME 24 val14_sub300 tokens: task0=det, task1=default-WTA(head argmax), task2..7=per-mode 0..5.
# -> canonical oracle(best-of-6)/det/default-WTA/random UNDER SMART, to compare vs IDM-canonical-matched
# (latent +0.119, selection ~random). Does the selection picture persist or SHIFT under realistic agents?
#SBATCH -J smart_sel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 07:30:00
#SBATCH --array=0-7%4
#SBATCH -o /home/patodia.pa/smart_sel-%A_%a.out
#SBATCH -e /home/patodia.pa/smart_sel-%A_%a.err
echo "[ss] START $(date) node=$(hostname) task=$SLURM_ARRAY_TASK_ID"
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
STACK=/home/patodia.pa/smart-stack
RUNS=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk
export NUPLAN_DEVKIT_ROOT=$STACK/nuplan-devkit SMART_ROOT=$STACK/SMART
export NUPLAN_DATA_ROOT=/scratch/patodia.pa/nuplan/val_stage
export NUPLAN_MAPS_ROOT=/scratch/patodia.pa/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/scratch/patodia.pa/av-policy-lab/smart_exp
export NUPLAN_CACHE_ROOT=/scratch/patodia.pa/av-policy-lab/smart_cache
export PYTHONNOUSERSITE=1

# per-task config
t=$SLURM_ARRAY_TASK_ID
if [ "$t" -eq 0 ]; then NAME=det;    HEAD=det; CKPT=$RUNS/det_route_seed0/best.pt; unset WTA_MODE_INDEX
elif [ "$t" -eq 1 ]; then NAME=dwta; HEAD=wta; CKPT=$RUNS/wta_route_seed0/best.pt; unset WTA_MODE_INDEX
else NAME=mode$((t-2)); HEAD=wta; CKPT=$RUNS/wta_route_seed0/best.pt; export WTA_MODE_INDEX=$((t-2)); fi
echo "[ss] config NAME=$NAME HEAD=$HEAD CKPT=$CKPT WTA_MODE_INDEX=${WTA_MODE_INDEX:-unset}"

# same 24 tokens (first 24 of val14_sub300) for every config
TOKS=$($C run -n nuplan_smart python -c "
import json
t=json.load(open('/scratch/patodia.pa/av-policy-lab/eval_tokens/val14_sub300.json'))
t=t['tokens'] if isinstance(t,dict) else t
print('['+','.join(t[:24])+']')
")
cd "$STACK"
PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$STACK/waymo_stub:/home/patodia.pa/av-policy-lab/nuplan" \
$C run -n nuplan_smart python nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_smart_reactive_agents \
  planner=policy_planner planner.policy_planner.device=cuda \
  planner.policy_planner.head_type=$HEAD planner.policy_planner.ckpt_path=$CKPT \
  scenario_builder=nuplan 'scenario_builder.db_files=[/scratch/patodia.pa/nuplan/val_stage/data/cache/val]' \
  scenario_filter=val14_split "scenario_filter.scenario_tokens=$TOKS" \
  scenario_filter.scenario_types=null scenario_filter.limit_total_scenarios=null \
  worker=sequential experiment_name=smart_select_canon/$NAME
echo "[ss] rc=$? DONE $(date)"
