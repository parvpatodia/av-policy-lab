#!/bin/bash
# ADR-084 re-dump: his-fork feature dump PINNED to token 33a911fcc99150b1 (match the nuplan dump).
# Earlier limit=1 grabbed a different token (00301) -> invalid diff. Pin scenario_tokens explicitly.
#SBATCH -J dump_smart2
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /home/patodia.pa/dump_smart2-%j.out
#SBATCH -e /home/patodia.pa/dump_smart2-%j.err
echo "[d2] START $(date) node=$(hostname)"; set -x
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
STACK=/home/patodia.pa/smart-stack
export NUPLAN_DEVKIT_ROOT=$STACK/nuplan-devkit SMART_ROOT=$STACK/SMART
export NUPLAN_DATA_ROOT=/scratch/patodia.pa/nuplan/val_stage
export NUPLAN_MAPS_ROOT=/scratch/patodia.pa/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/scratch/patodia.pa/av-policy-lab/smart_exp
export NUPLAN_CACHE_ROOT=/scratch/patodia.pa/av-policy-lab/smart_cache
export PYTHONNOUSERSITE=1
cd "$STACK"
DUMP_FEATS=/home/patodia.pa/feats_smart2.npz \
PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$STACK/waymo_stub:/home/patodia.pa/av-policy-lab/nuplan" \
$C run -n nuplan_smart python nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents planner=policy_planner planner.policy_planner.device=cpu \
  scenario_builder=nuplan 'scenario_builder.db_files=[/scratch/patodia.pa/nuplan/val_stage/data/cache/val]' \
  scenario_filter=val14_split \
  'scenario_filter.scenario_tokens=[33a911fcc99150b1]' \
  'scenario_filter.scenario_types=null' \
  'scenario_filter.limit_total_scenarios=1' \
  worker=sequential experiment_name=dump_smart2
echo "[d2] rc=$? DONE $(date)"; ls -la /home/patodia.pa/feats_smart2.npz 2>&1
