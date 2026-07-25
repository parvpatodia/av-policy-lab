#!/bin/bash
# ISOLATION (ADR-082): our det PolicyPlanner in HIS fork under IDM, SAME 5 val14 tokens, but device=cpu
# (his-fork run 8717075 used device=cuda -> 0.365; our-devkit control 8718009 used device=cpu -> 0.799).
# Splits the two remaining variables: device(cpu vs cuda) vs env(nuplan_smart vs nuplan).
#   jumps to ~0.799 => device=cuda is the bug.  stays ~0.365 => env/torch is the bug.
#SBATCH -J idm_cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /home/patodia.pa/idm_cpu-%j.out
#SBATCH -e /home/patodia.pa/idm_cpu-%j.err
echo "[cpu] START $(date) node=$(hostname)"
set -x
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
STACK=/home/patodia.pa/smart-stack
export NUPLAN_DEVKIT_ROOT=$STACK/nuplan-devkit
export SMART_ROOT=$STACK/SMART
export NUPLAN_DATA_ROOT=/scratch/patodia.pa/nuplan/val_stage
export NUPLAN_MAPS_ROOT=/scratch/patodia.pa/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/scratch/patodia.pa/av-policy-lab/smart_exp
export NUPLAN_CACHE_ROOT=/scratch/patodia.pa/av-policy-lab/smart_cache
export PYTHONPATH=$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$STACK/waymo_stub:/home/patodia.pa/av-policy-lab/nuplan
export PYTHONNOUSERSITE=1
cd "$STACK"
$C run -n nuplan_smart python nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents \
  planner=policy_planner \
  planner.policy_planner.device=cpu \
  scenario_builder=nuplan \
  'scenario_builder.db_files=[/scratch/patodia.pa/nuplan/val_stage/data/cache/val]' \
  scenario_filter=val14_split \
  scenario_filter.limit_total_scenarios=5 \
  worker=sequential \
  experiment_name=policy_idm_cpu
echo "[cpu] rc=$? DONE $(date)"
