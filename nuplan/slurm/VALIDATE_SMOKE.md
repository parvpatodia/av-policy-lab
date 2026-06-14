# F0 — 30-minute interactive smoke validation (DO THIS BEFORE the full extraction)

> Goal: prove the extractor produces correctly-shaped, finite tensors on nuPlan
> MINI — and confirm the version-sensitive devkit calls — in under 30 minutes,
> BEFORE you ever submit the 48h `extract_features.sbatch`. If `--smoke` fails on
> mini, the full run will fail after hours. Cluster: **Explorer**
> (`login.explorer.northeastern.edu`).

## 0. One-time prerequisites
Data was staged mirroring the local layout, and the env vars below were exported in
`~/.bashrc`. The paths the extractor needs:
- nuPlan **MINI** DBs:  `$NUPLAN_DATA_ROOT/mini`  (= `/scratch/$USER/nuplan/data/cache/mini`)
- nuPlan **maps**:      `$NUPLAN_MAPS_ROOT`        (= `/scratch/$USER/nuplan/maps`)
- conda env `nuplan` with the devkit installed (see docs/frontier/F0_IMPLEMENTATION.md).
- This repo checked out under `/home/$USER/av-policy-lab` (code lives in backed-up /home).

Confirm before anything else:
```bash
env | grep NUPLAN                              # DATA_ROOT, MAPS_ROOT, EXP_ROOT all set?
ls "$NUPLAN_DATA_ROOT/mini"/*.db | wc -l        # expect 64
ls "$NUPLAN_MAPS_ROOT"                          # expect us-ma-boston/, sg-one-north/, *.gpkg
```

## 1. Grab a SHORT interactive CPU allocation (no GPU — extraction is CPU-bound)

Smoke is pure CPU. Prefer a short CPU alloc; do **not** burn a GPU slot on it.

```bash
# CPU interactive (preferred for smoke): a few cores, 30 min, on the CPU `short` partition
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=00:30:00 --pty /bin/bash
```

If `short` interactive is congested and you only need a couple of minutes, the
2h GPU interactive partitions also give you a quick shell (the GPU sits unused —
acceptable only for a <2h probe, not the real run):

```bash
# Fallback ONLY (GPU unused): gpu-interactive / gpu-short are 2h max
srun --partition=gpu-interactive --nodes=1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=00:30:00 --pty /bin/bash
```

## 2. Activate the env and run the smoke gate

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nuplan
cd /home/$USER/av-policy-lab

python -u nuplan/features/scene_features.py --smoke \
    --n-scenarios 5 \
    --data-root "$NUPLAN_DATA_ROOT/mini" \
    --map-root  "$NUPLAN_MAPS_ROOT"
```

### What a PASS looks like
- `[smoke] ego-frame transform round-trip OK`
- 5 scenarios loaded; for each, every tensor's shape printed, e.g.
  `ego (20, 8)`, `agents (32, 20, 9)`, `agent_mask (32, 20)`,
  `map_polylines (128, 20, 7)`, `route_polyline (40, 4)`, `traffic_lights (128,)`.
- `valid agents=K/32  map polylines=M/128` with K, M > 0 on urban scenarios.
- `route resolved for R/5 scenarios` — **R should be > 0**. If R == 0, the
  roadblock-id → lane resolution needs the in-env fix (see step 4).
- Final line: `[smoke] PASS — shapes/masks consistent, tensors finite`.

## 3. Run the pure-numpy unit tests (no devkit needed; ~seconds)

```bash
cd /home/$USER/av-policy-lab
python -m pytest tests/test_scene_features.py -q
```
These cover the transform round-trip, padding/masking to 32, polyline resampling,
and normalization invertibility. They must be green before the full run.

## 4. If smoke FAILS — the version-skew checklist (each fixable in <15 min)
See `docs/frontier/F0_IMPLEMENTATION.md` for the full ledger. The usual suspects:
1. **Scenario builder kwargs** (`NuPlanScenarioBuilder` / `ScenarioFilter`) — the
   most version-sensitive call. Adjust in `_build_mini_scenarios`.
2. **route resolved == 0** — `get_route_roadblock_ids()` returns ROADBLOCK ids;
   `_roadblock_interior_lanes` tries `ROADBLOCK` then `ROADBLOCK_CONNECTOR`. If both
   miss on your map version, print the ids and check which layer `get_map_object`
   accepts.
3. **Traffic-light ids all -1** — confirm `TrafficLightStatusData.lane_connector_id`
   matches `LaneConnector.id` formatting (int vs str) on your devkit.

## 5. Only after smoke + unit tests are green: submit the full run
```bash
# default is the MINI split you have staged (DATA_SPLIT=mini); see extract_features.sbatch
sbatch nuplan/slurm/extract_features.sbatch          # mini split, /scratch -> /scratch
# or a costed pilot first:
LIMIT=2000 sbatch nuplan/slurm/extract_features.sbatch
squeue -u $USER                                      # watch it
tail -f /scratch/$USER/av-policy-lab/logs/f0-extract-*.out
```
