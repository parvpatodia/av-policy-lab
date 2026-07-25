#!/bin/bash
# ADR-084: dump the LIVE SceneFeatureExtractor batch for token 33a911fcc99150b1 (high_lat_acc, index 0)
# under BOTH envs to localize the numpy1.23-vs-2.0 corruption. Same planner, same token, IDM, cpu.
#SBATCH -J dump_feats
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /home/patodia.pa/dump_feats-%j.out
#SBATCH -e /home/patodia.pa/dump_feats-%j.err
echo "[dump] START $(date) node=$(hostname)"
module load anaconda3/2024.06
C=/shared/EL9/explorer/anaconda3/2024.06/bin/conda
REPO=/home/patodia.pa/av-policy-lab
VAL_DB=/scratch/patodia.pa/nuplan/val_stage/data/cache/val
ET=/scratch/patodia.pa/av-policy-lab/eval_tokens
CKPT=/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/det_route_seed0/best.pt
mkdir -p "$ET"
echo '{"tokens": ["33a911fcc99150b1"]}' > "$ET/dump_1tok.json"

# ---- (1) OUR env: nuplan (torch2.8/numpy2.0) via run_cells.py ----
echo "[dump] === nuplan env ==="
cd "$REPO"
export PYTHONNOUSERSITE=1
DUMP_FEATS=/home/patodia.pa/feats_nuplan.npz \
PYTHONPATH="$REPO/nuplan:/home/patodia.pa/tuplan_garage" \
$C run -n nuplan python nuplan/eval/run_cells.py \
    --planner policy --head det --goal route --ckpt "$CKPT" --db-dir "$VAL_DB" \
    --tokens-file "$ET/dump_1tok.json" --reactive 1 --exp-name eval/dump_nuplan
echo "[dump] nuplan rc=$?"

# ---- (2) HIS fork env: nuplan_smart (torch2.1/numpy1.23) via run_simulation.py ----
echo "[dump] === nuplan_smart / his fork ==="
STACK=/home/patodia.pa/smart-stack
export NUPLAN_DEVKIT_ROOT=$STACK/nuplan-devkit SMART_ROOT=$STACK/SMART
export NUPLAN_DATA_ROOT=/scratch/patodia.pa/nuplan/val_stage
export NUPLAN_MAPS_ROOT=/scratch/patodia.pa/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/scratch/patodia.pa/av-policy-lab/smart_exp
export NUPLAN_CACHE_ROOT=/scratch/patodia.pa/av-policy-lab/smart_cache
cd "$STACK"
DUMP_FEATS=/home/patodia.pa/feats_smart.npz \
PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$STACK/waymo_stub:/home/patodia.pa/av-policy-lab/nuplan" \
$C run -n nuplan_smart python nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents planner=policy_planner planner.policy_planner.device=cpu \
  scenario_builder=nuplan 'scenario_builder.db_files=[/scratch/patodia.pa/nuplan/val_stage/data/cache/val]' \
  scenario_filter=val14_split scenario_filter.limit_total_scenarios=1 worker=sequential \
  experiment_name=dump_smart
echo "[dump] smart rc=$?"
echo "[dump] DONE $(date)"; ls -la /home/patodia.pa/feats_*.npz
