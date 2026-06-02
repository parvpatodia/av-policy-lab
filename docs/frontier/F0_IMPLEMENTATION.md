# F0 — Vectorized Scene-Feature Extractor (implementation + honesty ledger)

> Status: code complete; **devkit-touching paths NOT executed by the author** (no
> nuPlan env here). The pure-numpy core was written runnable-by-inspection and must
> be run in-env. Every nuPlan-devkit call is tagged `[VERIFIED]` or `[UNVERIFIED]`
> below — nothing here was fabricated. Builds on `docs/frontier/STAGE_2_data.md`.

## What F0 produces

`nuplan/features/scene_features.py` extracts, per `(scenario, iteration)` sample,
an **ego-centric** (ego at origin, facing +x) dict of named tensors and saves lists
of samples as `.pt` shards (default under a `/scratch` path).

| Tensor | Shape | Contents |
|---|---|---|
| `ego` | `(20, 8)` | last 2 s @ 10 Hz: `[x, y, sin θ, cos θ, vx, vy, ax, ay]` |
| `agents` | `(32, 20, 9)` | nearest ≤32 vehicles/peds/bikes, 2 s each: `[x, y, sin, cos, vx, vy, type-onehot(3)]` |
| `agent_mask` | `(32, 20)` | per-step validity (pad/truncate to 32) |
| `map_polylines` | `(128, 20, 7)` | lane + lane-connector polylines resampled to 20 pts: `[x, y, dir_x, dir_y, type-onehot(3)]` |
| `map_mask` | `(128,)` | polyline validity |
| `crosswalks` | `(16, 20, 2)` | crosswalk polygon boundaries, ego-frame, resampled |
| `crosswalk_mask` | `(16,)` | crosswalk validity |
| `route_polyline` | `(40, 4)` | on-route lane sequence: `[x, y, dir_x, dir_y]` |
| `route_mask` | `(40,)` | route validity |
| `traffic_lights` | `(128,)` | per map-polyline TL status id (`0=green,1=yellow,2=red,3=unknown`, `-1`=not signal-controlled) |

All positions are ego-frame metres / 120; velocities / 15; accel / 5; headings as
`(sin, cos)`. Scales live in one place (`Normalizer`) — STAGE_2_data.md §1.

### Divergences from STAGE_2_data.md (intentional, per the F0 brief)
- **20 history steps**, not 21 (brief: "last 2 s @ 10 Hz (20 steps)"). 20 past steps
  incl. current; STAGE_2's `T_h=21` adds one. One-line change in `FeatureConfig`.
- **agent radius 100 m** (STAGE_2 cites 100–120 m). **route reach 120 m** matches STAGE_2 §4.
- Agent channels are `type-onehot` here; STAGE_2 also lists length/width + a learned
  type embedding. F0 keeps the onehot; the embedding is a Stage-3 (model) concern.

## Data layout (as actually staged)

Data was transferred mirroring the local layout, with these env vars exported in
`~/.bashrc` on Explorer:
- `NUPLAN_DATA_ROOT=/scratch/$USER/nuplan/data/cache`  (PARENT of split subdirs)
- `NUPLAN_MAPS_ROOT=/scratch/$USER/nuplan/maps`
- `NUPLAN_EXP_ROOT=/scratch/$USER/nuplan/exp`

So the MINI DBs live at `$NUPLAN_DATA_ROOT/mini` and maps at `$NUPLAN_MAPS_ROOT`.
The commands below reference those env vars (no hardcoded usernames or the
standard `dataset/nuplan-v1.1/splits/...` layout, which is NOT how this is staged).

## Run commands

### 0. Unit tests (no devkit, seconds) — run first
```bash
cd /home/$USER/av-policy-lab
python -m pytest tests/test_scene_features.py -q
```

### 1. Smoke on MINI (the gate) — see nuplan/slurm/VALIDATE_SMOKE.md
```bash
python -u nuplan/features/scene_features.py --smoke --n-scenarios 5 \
    --data-root "$NUPLAN_DATA_ROOT/mini" \
    --map-root  "$NUPLAN_MAPS_ROOT"
```

### 2. Full extraction (CPU, partition=short, 48h, no GPU)
```bash
sbatch nuplan/slurm/extract_features.sbatch                 # mini split (DATA_SPLIT=mini default)
LIMIT=2000 sbatch nuplan/slurm/extract_features.sbatch      # costed pilot first (recommended)
# once the full split is downloaded:
DATA_SPLIT=trainval sbatch nuplan/slurm/extract_features.sbatch
```

## devkit-call ledger (what to confirm in-env)

### [VERIFIED from devkit docs/source] — confirmed against motional/nuplan-devkit master + this repo's working planners.py / map_utils.py
- `AbstractScenario`: `get_number_of_iterations()`, `get_ego_state_at_iteration(i)`,
  `get_ego_past_trajectory(i, time_horizon, num_samples)`,
  `get_tracked_objects_at_iteration(i)`, `get_past_tracked_objects(i, time_horizon, num_samples)`,
  `get_traffic_light_status_at_iteration(i)`, `get_route_roadblock_ids()`, `.map_api`.
  (source: `nuplan/planning/scenario_builder/abstract_scenario.py`)
- `EgoState`: `.rear_axle.x/.y/.heading`,
  `.dynamic_car_state.rear_axle_velocity_2d.x/.y`, `...rear_axle_acceleration_2d.x/.y`.
  (confirmed in this repo's `planners.py` L102–128, which runs in the sim.)
- `DetectionsTracks.tracked_objects` →
  `TrackedObjects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE|PEDESTRIAN|BICYCLE)`.
  (source: `tracked_objects.py`, `tracked_objects_types.py`)
- Agent fields: `.box.center.x/.y/.heading` (StateSE2), `.box.length/.width`,
  `.velocity.x/.y` (StateVector2D), `.tracked_object_type`, `.track_token`.
  (source: `agent_state.py`, `scene_object.py`, `oriented_box.py`)
- `TrafficLightStatusData.status` (IntEnum GREEN=0,YELLOW=1,RED=2,UNKNOWN=3) +
  `.lane_connector_id: int`. (source: `maps_datatypes.py`)
- `AbstractMap.get_proximal_map_objects(point, radius, layers)`,
  `get_map_object(object_id, layer)`. `SemanticMapLayer.LANE/LANE_CONNECTOR/CROSSWALK/ROADBLOCK/ROADBLOCK_CONNECTOR`.
  (source: `abstract_map.py`, `maps_datatypes.py`)
- Lane/LaneConnector: `.baseline_path.discrete_path` (List[StateSE2] w/ `.x/.y`),
  `.outgoing_edges`, `.get_roadblock_id()`, `.id`. CROSSWALK is a `PolygonMapObject`
  with `.polygon`. (source: `abstract_map_objects.py`; baseline_path usage matches
  this repo's `planners.py` `_build_route`.)

### [UNVERIFIED — confirm in env] — re-confirm via `--smoke` on mini
1. **`NuPlanScenarioBuilder` / `ScenarioFilter` constructor kwargs**
   (`scene_features._build_mini_scenarios`). The builder/filter signatures change
   across devkit versions (Hydra vs direct kwargs, `sensor_root`, `db_files`,
   `expand_scenarios`, etc.). **This is the single most likely failure point** — see below.
2. **Roadblock-id → interior lanes** (`_roadblock_interior_lanes`): assumes
   `get_map_object(rb_id, SemanticMapLayer.ROADBLOCK).interior_edges`. Route ids may
   be `ROADBLOCK_CONNECTOR` on some maps; the code tries both. If `route resolved == 0`
   in smoke, this needs the in-env tweak.
3. **`get_route_roadblock_ids()` availability**: some scenario builders / mini logs
   may not populate route ids. Wrapped in try/except → empty route tensor (masked).
4. **Crosswalk `.polygon.exterior.coords`**: assumes shapely polygon. Verify CROSSWALK
   objects expose `.polygon` (PolygonMapObject contract says yes; confirm concrete class).
5. **Velocity/accel frame**: `rear_axle_velocity_2d` is treated as body-frame and
   rotated by per-step heading into the ego frame. If a devkit version stores these in
   world frame, drop the per-step rotation (one spot in `EgoFeatureBuilder.build`).
   Confirm by checking that a straight-driving ego yields `vy ≈ 0` in smoke output.

## The single most likely failure point + how to test it in <15 min

**Devkit version skew on the scenario-builder construction and the tracked-objects /
map API.** Concretely: `_build_mini_scenarios` (builder/filter kwargs) and the
roadblock-id → lane resolution. These are exactly the calls that differ most across
devkit releases, and they are the only ones the author could not execute.

**Test in <15 min:** run `--smoke` on **mini** (step 1 above). It:
- imports the module (catches signature/import errors immediately),
- builds 5 mini scenarios (catches builder/filter kwarg skew — failure #1),
- extracts one sample each and **prints every tensor shape** + asserts shape/mask
  consistency and finiteness (catches map/agent API skew),
- reports `route resolved for R/5` (catches roadblock-id skew — failure #2),
- runs the ego-frame round-trip assertion (catches transform regressions).

If smoke is green, the full 48h run will not fail on API shape — only on scale/time.

## Throughput / shard-size honesty (CPU-bound — may take many hours)

- Extraction is **CPU-bound**: per sample it does SQLite reads (ego/agents/TL),
  `get_proximal_map_objects` (the expensive map query), and numpy geometry. There is
  **no GPU work** — hence `partition=short`, no `--gres`.
- **Rough cost (order-of-magnitude, MUST be re-measured in-env):** map queries
  dominate at ~tens of ms each; expect **~10²–10³ samples/min single-process**. At
  `stride=1`, a ~15 s scenario yields ~130–150 samples. STAGE_2 §7 targets **~1M
  frames** from the full split → at a few hundred samples/min this is **many hours to
  a couple of days** of wall-clock single-process. This is precisely why the job uses
  `partition=short` with the full **48h** window and shards incrementally (resumable).
  NOTE: the MINI split (64 logs) yields far fewer frames — fine for the full pipeline
  dry-run and F1 bring-up; download trainval (`DATA_SPLIT=trainval`) for the ~1M-frame run.
- **Shard size:** one sample's dense tensors are ≈
  `ego 20·8 + agents 32·20·9 + map 128·20·7 + route 40·4 + crosswalks 16·20·2` ≈ **24k
  floats ≈ ~100 KB float32** (plus masks/ints). At 8 scenarios/shard × ~130
  samples/scenario ≈ ~1000 samples ≈ **~100 MB per shard** (very rough). 1M frames ≈
  **~100 GB total** — fits `/scratch` but **not `/home`** (75 GB).
- **Scale up honestly:** parallelize across logs with a SLURM `--array` (sketched in
  the sbatch) rather than one giant process; raise `--stride` to thin temporally
  (e.g. `stride=5` → 2 Hz sampling, 5× fewer frames) if 1M frames is more than needed.

## Storage discipline (Explorer)
- Dataset + shards: `/scratch/$USER/...` (large, fast, **purged monthly, not backed up**).
- Code: `/home/$USER/av-policy-lab` (backed up, 75 GB cap).
- **Keepers:** copy finished shards to `/projects/<group>/` before the first-Tuesday
  scratch purge. (HPC_NORTHEASTERN.md §4.)
