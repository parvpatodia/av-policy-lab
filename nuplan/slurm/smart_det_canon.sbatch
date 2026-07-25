#!/bin/bash
# ADR-088: OUR det planner under SMART reactive agents (his closed_loop_smart_reactive_agents preset =
# CANONICAL CLS), on the first 60 tokens of val14_sub300, as a 12-shard array (SMART ~8min/scenario).
# Pairs with det IDM-canonical (0.252 via canonical_reaggregate) on the SAME tokens -> det IDM-vs-SMART.
#SBATCH -J smart_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-7%4
#SBATCH -o /home/patodia.pa/smart_det-%A_%a.out
#SBATCH -e /home/patodia.pa/smart_det-%A_%a.err
echo "[sd] START $(date) node=$(hostname) task=$SLURM_ARRAY_TASK_ID"
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
STACK=/home/patodia.pa/smart-stack
export NUPLAN_DEVKIT_ROOT=$STACK/nuplan-devkit SMART_ROOT=$STACK/SMART
export NUPLAN_DATA_ROOT=/scratch/patodia.pa/nuplan/val_stage
export NUPLAN_MAPS_ROOT=/scratch/patodia.pa/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/scratch/patodia.pa/av-policy-lab/smart_exp
export NUPLAN_CACHE_ROOT=/scratch/patodia.pa/av-policy-lab/smart_cache
export PYTHONNOUSERSITE=1
NSHARDS=8
# shard token list (strided over first 60 of val14_sub300)
TOKS=$($C run -n nuplan_smart python -c "
import json
t=json.load(open('/scratch/patodia.pa/av-policy-lab/eval_tokens/val14_sub300.json'))
t=t['tokens'] if isinstance(t,dict) else t
sub=t[:60][$SLURM_ARRAY_TASK_ID::$NSHARDS]
print('['+','.join(sub)+']')
")
echo "[sd] shard tokens: $TOKS"
cd "$STACK"
PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$STACK/waymo_stub:/home/patodia.pa/av-policy-lab/nuplan" \
$C run -n nuplan_smart python nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_smart_reactive_agents \
  planner=policy_planner planner.policy_planner.device=cuda \
  scenario_builder=nuplan 'scenario_builder.db_files=[/scratch/patodia.pa/nuplan/val_stage/data/cache/val]' \
  scenario_filter=val14_split "scenario_filter.scenario_tokens=$TOKS" \
  scenario_filter.scenario_types=null scenario_filter.limit_total_scenarios=null \
  worker=sequential experiment_name=smart_det_canon/shard${SLURM_ARRAY_TASK_ID}
echo "[sd] rc=$? DONE $(date)"
