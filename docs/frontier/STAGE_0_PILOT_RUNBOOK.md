# Stage 0 Pilot Runbook — "Does the leaderboard reorder when IDM → SMART?"

**Goal:** Fail-fast de-risk. Take *released* nuPlan planner checkpoints, run them in nuPlan
closed-loop under (a) standard IDM reactive agents and (b) realistic SMART reactive agents,
and check whether the planner ranking / score margins change. This reproduces the core claim of
**arXiv:2510.14677 — "When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift
nuPlan Benchmarks"** (Hagedorn, Donkov, Distelzweig, Condurache).
If the reordering is real, our controlled-pair project is worth building. If not, we kill it.

- **Target cluster:** Northeastern Explorer HPC (`gpu` partition, 8h max wall, single H200 default, `/projects` for data, compute nodes have internet).
- **Author/date:** Stage 0 planning, June 2026.
- **Status legend:** every command is tagged `[VERIFIED from repo]` (grounded in a file I read in the actual upstream repo) or `[UNVERIFIED — confirm in env]` (plausible/standard but not directly confirmed for this cluster or this version).

---

## 0. TL;DR — read this first

1. **The IDM half of this pilot is fully reproducible today.** tuPlan-garage, Diffusion-Planner, and the nuPlan devkit all ship working `run_simulation.py` configs with `closed_loop_reactive_agents` (= IDM) and `closed_loop_nonreactive_agents`. Verified.
2. **The SMART half is the blocker.** The SMART-reactive drop-in from the paper lives at
   `github.com/shgd95/InteractiveClosedLoop` and **as of this writing contains only a README
   saying "Code is under internal approval."** There is no SMART observation/agent code, no
   config, no checkpoint published yet. The exact `observation=...` field name for SMART is
   therefore **UNVERIFIED**. See §6 and §11.
3. **Recommended sequencing:** run the IDM-side pilot now (it is real work and gives the IDM
   baseline numbers you'll need regardless). Gate the SMART side on the `InteractiveClosedLoop`
   release. Do **not** burn cluster hours trying to reconstruct SMART agents from the Waymo
   `rainmaker22/SMART` repo unless we decide to fund a full port (see §6.3 risk).
4. **Single most likely failure point:** devkit-version skew between tuPlan-garage / Diffusion-Planner
   (built on older nuplan-devkit) and whatever devkit the SMART drop-in targets. Test it in <30 min — see §12.

---

## 1. Components and exact sources (all URLs)

| Component | Repo / URL | What we use it for | Verified? |
|---|---|---|---|
| nuPlan devkit | https://github.com/motional/nuplan-devkit | `run_simulation.py`, closed-loop configs, IDM agents, metric aggregator | `[VERIFIED from repo]` |
| tuPlan-garage (PDM-Closed / PDM-Open / Urban Driver) | https://github.com/autonomousvision/tuplan_garage | released planners + checkpoints, Val14 split | `[VERIFIED from repo]` |
| Diffusion-Planner | https://github.com/ZhengYinan-AIR/Diffusion-Planner | released learned planner + HF checkpoint | `[VERIFIED from repo]` |
| Diffusion-Planner checkpoint | https://huggingface.co/ZhengYinan2001/Diffusion-Planner (`args.json`, `model.pth`) | learned planner weights | `[VERIFIED from repo]` |
| SMART (base, Waymo) | https://github.com/rainmaker22/SMART | the underlying SMART model (NeurIPS 2024, arXiv:2405.15677). **Waymo, not nuPlan.** | `[VERIFIED from repo]` |
| SMART-reactive nuPlan drop-in | https://github.com/shgd95/InteractiveClosedLoop | the actual IDM→SMART swap from arXiv:2510.14677 | `[UNVERIFIED — repo currently README-only, code under internal approval]` |
| Paper | https://arxiv.org/abs/2510.14677 | the claim we are reproducing | `[VERIFIED from repo]` |

**Key fact about checkpoints (`[VERIFIED from repo]`):**
- **PDM-Closed is rule-based — it has NO learned checkpoint.** It is a pure planner (verified from
  `pdm_closed_planner.yaml`: it instantiates `PDMClosedPlanner` with IDM-policy proposals, no weights).
  So "released checkpoint" planners with downloadable weights are: **PDM-Open**, **Urban Driver**
  (both from the tuPlan-garage Google Drive), and **Diffusion-Planner** (HuggingFace).
- PDM-Closed is still the most useful single entry to run first because it needs no weights and is the
  reference top-of-leaderboard rule-based planner — fastest path to a green end-to-end run.
- tuPlan-garage trained models (PDM-Open, Urban Driver):
  https://drive.google.com/drive/folders/1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2 `[VERIFIED from repo README]`

---

## 2. Prerequisites — nuPlan dataset (the minimum to run the pilot)

The pilot needs the **Val14** benchmark (1,090 scenarios per arXiv:2510.14677). What that requires:

| Piece | Needed for pilot? | Notes |
|---|---|---|
| **nuPlan maps** (`nuplan-maps-v1.0`) | **YES — mandatory** | All four cities (sg-one-north, us-ma-boston, us-nv-las-vegas-strip, us-pa-pittsburgh-hazelwood). ~16 GB. `[UNVERIFIED size — confirm on download]` |
| **mini split** (`nuplan-v1.1/splits/mini`) | Good for smoke test only | ~small; lets you prove the pipeline before committing to the big download. The mini split does **not** contain the full Val14 scenario set. |
| **trainval split** (`nuplan-v1.1/splits/trainval`) | **YES for the real Val14 run** | This is the large one (multi-hundred-GB). Val14 scenarios are drawn from here via `scenario_filter=val14_split`. `[UNVERIFIED exact size — confirm on download]` |
| test split | No | Test14/Test14-hard are a different benchmark; not needed for the Val14 pilot. |

**Minimum to *prove the pipeline*:** maps + mini split.
**Minimum to *reproduce the paper's Val14 numbers*:** maps + trainval split.

> WHY two-phase: trainval is huge and the download itself can dominate the schedule. Prove the full
> command path on `mini` first (cheap), then download trainval once and run the real Val14 filter.

Dataset access: register at https://www.nuplan.org/ and follow
https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md `[VERIFIED — this is the documented setup doc]`.

Required environment variables (verified from `sim_diffusion_planner_runner.sh` in Diffusion-Planner repo):
```bash
export NUPLAN_DATA_ROOT="/projects/<your_alloc>/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/projects/<your_alloc>/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/projects/<your_alloc>/nuplan/exp"
export NUPLAN_DEVKIT_ROOT="/projects/<your_alloc>/nuplan-devkit"
```
`[VERIFIED from repo — variable names]` / `[UNVERIFIED — confirm your /projects allocation path]`

Expected layout (`[VERIFIED from repo dataset_setup doc structure]`):
```
$NUPLAN_DATA_ROOT/
  maps/                         # nuplan-maps-v1.0.json + per-city map dirs
  nuplan-v1.1/
    splits/
      mini/                     # *.db smoke-test
      trainval/                 # *.db  (Val14 source)
```

---

## 3. Download data to /projects (documented method)

Explorer compute nodes have internet, so download on a **compute node** (not the login node) inside a short interactive job.

```bash
# [UNVERIFIED — confirm partition/account names for your Explorer allocation]
srun --partition=gpu --gres=gpu:0 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash

mkdir -p /projects/<your_alloc>/nuplan/dataset
cd /projects/<your_alloc>/nuplan/dataset
# nuPlan datasets are distributed as the s3/https tarballs described in the devkit dataset_setup doc.
# Follow the exact aws s3 / wget commands from:
#   https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md
# Download MAPS first (mandatory), then mini (smoke test), then trainval (real run).
```
`[UNVERIFIED — confirm exact download URLs/credentials from the dataset_setup doc + your nuplan.org account]`

> Data download (trainval) will likely exceed a single short job. Use `wget -c` / `aws s3 cp` (resumable)
> and re-`srun` to continue. See §9 for the >8h checkpoint/resume framing — the same logic applies to downloads.

---

## 4. Environment setup (conda + repos + checkpoints)

Two **separate** conda envs, because the planner repos pin different stacks and may target different
devkit commits. Keep them isolated so a SMART-side dependency conflict cannot corrupt the IDM-side run.

### 4a. Env A — tuPlan-garage (PDM-Closed / PDM-Open / Urban Driver)
```bash
# [VERIFIED from repo README — clone + pip install -e .]
module load anaconda3 cuda/12.4   # [UNVERIFIED — use the modules your Explorer exposes]
conda create -n nuplan python=3.9 -y     # [VERIFIED — tuplan_garage setup.py requires python>=3.9]
conda activate nuplan

cd /projects/<your_alloc>
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
pip install -r requirements.txt          # [VERIFIED — standard devkit install]

cd /projects/<your_alloc>
git clone https://github.com/autonomousvision/tuplan_garage.git
cd tuplan_garage
pip install -e .                         # [VERIFIED from repo README]

echo 'export NUPLAN_DEVKIT_ROOT="/projects/<your_alloc>/nuplan-devkit/"' >> ~/.bashrc   # [VERIFIED from repo README]
```
> **Pin the commits.** Record `git rev-parse HEAD` for BOTH `nuplan-devkit` and `tuplan_garage`
> into the experiment log. tuPlan-garage is known to track an *older* devkit; if `pip install -e .`
> of the devkit master breaks tuPlan-garage imports, check out the devkit tag the tuPlan-garage README/issues
> reference and reinstall. `[UNVERIFIED — exact compatible devkit commit not pinned in tuplan_garage setup.py;
> confirm in env, see §11 risk]`

### 4b. Env B — Diffusion-Planner
```bash
# [VERIFIED from repo README]
conda create -n diffusion_planner python=3.9 -y
conda activate diffusion_planner

cd /projects/<your_alloc>
# reuse the same nuplan-devkit clone OR clone its own copy; the README installs devkit into this env too
cd nuplan-devkit && pip install -e . && pip install -r requirements.txt && cd ..

git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git
cd Diffusion-Planner
pip install -e .
pip install -r requirements_torch.txt    # [VERIFIED from repo: torch==2.0.0+cu118, torchvision==0.15.1+cu118, pytorch_lightning==2.0.1, timm==1.0.10, mmengine]
```
> **CUDA/torch:** Diffusion-Planner pins **torch 2.0.0 + CUDA 11.8** (`requirements_torch.txt`,
> `[VERIFIED from repo]`). An H200 is Hopper (sm_90). torch 2.0.0+cu118 wheels **may not ship sm_90
> kernels** → high risk of "no kernel image available" on H200. Plan to bump to a cu12x torch build
> that supports sm_90 (e.g. torch 2.2+/2.4+ cu121) and re-test the model loads. `[UNVERIFIED — confirm
> torch/H200 compatibility in env; this is a likely-to-bite item, see §11]`

### 4c. Checkpoints
```bash
# Diffusion-Planner checkpoint  [VERIFIED from repo README]
conda activate diffusion_planner
cd /projects/<your_alloc>/Diffusion-Planner
mkdir -p checkpoints
wget -P ./checkpoints https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/args.json
wget -P ./checkpoints https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/model.pth

# tuPlan-garage PDM-Open / Urban Driver checkpoints  [VERIFIED — Google Drive folder from repo README]
#   https://drive.google.com/drive/folders/1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2
#   Download via gdown on a compute node:
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2 -O /projects/<your_alloc>/checkpoints/tuplan_garage
# PDM-Closed needs NO checkpoint (rule-based). [VERIFIED from pdm_closed_planner.yaml]
```

---

## 5. The exact `run_simulation` commands (IDM side) — VERIFIED

### 5a. The IDM integration point (this is the heart of the "(a) IDM" arm)
`[VERIFIED from repo]` — In nuplan-devkit, the closed-loop reactive experiment selects IDM agents here:

`nuplan/planning/script/experiments/simulation/closed_loop_reactive_agents.yaml`:
```yaml
# @package _global_
job_name: closed_loop_reactive_agents
defaults:
  - override /observation: idm_agents_observation     # <-- THIS is the agent model
  - override /ego_controller: two_stage_controller
  - override /planner: simple_planner
  - override /simulation_metric: simulation_closed_loop_reactive_agents
  - override /metric_aggregator:
      - closed_loop_reactive_agents_weighted_average
```
And `observation/idm_agents_observation.yaml` instantiates:
```yaml
_target_: nuplan.planning.simulation.observation.idm_agents.IDMAgents
target_velocity: 10
min_gap_to_lead_agent: 1.0
headway_time: 1.5
accel_max: 1.0
decel_max: 2.0
...
```
For contrast, `closed_loop_nonreactive_agents.yaml` uses `override /observation: box_observation`
(log-replay boxes — the "non-reactive" arm). `[VERIFIED from repo]`

**So the swap axis is the `observation` config group.** IDM = `idm_agents_observation`. SMART (when released)
must register a sibling `observation=smart_agents_observation` whose `_target_` is a SMART subclass of
`nuplan.planning.simulation.observation.abstract_observation.AbstractObservation`
(interface verified: it requires `reset`, `initialize`, `get_observation`, `update_observation`).

### 5b. PDM-Closed under IDM reactive agents — fastest first run `[VERIFIED from repo script]`
From `tuplan_garage/scripts/simulation/sim_pdm_closed.sh`:
```bash
conda activate nuplan
SPLIT=val14_split
CHALLENGE=closed_loop_reactive_agents   # IDM. Options: open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=$CHALLENGE \
  planner=pdm_closed_planner \
  scenario_filter=$SPLIT \
  scenario_builder=nuplan \
  hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```
Run the **non-reactive** counterpart by setting `CHALLENGE=closed_loop_nonreactive_agents`.
This gives you CLS-R (IDM reactive) and CLS-NR (log-replay) for PDM-Closed. `[VERIFIED from repo]`

### 5c. PDM-Open (released checkpoint) under IDM reactive `[VERIFIED config field from repo]`
`pdm_open_planner.yaml` exposes `checkpoint_path: ???` (must be provided). So:
```bash
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents \
  planner=pdm_open_planner \
  planner.pdm_open_planner.checkpoint_path=/projects/<your_alloc>/checkpoints/tuplan_garage/pdm_open.ckpt \
  scenario_filter=val14_split \
  scenario_builder=nuplan \
  hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```
`[VERIFIED — checkpoint_path is a real config key]` / `[UNVERIFIED — exact .ckpt filename inside the Drive folder; confirm after gdown]`

### 5d. Diffusion-Planner (released checkpoint) under IDM reactive `[VERIFIED from repo script]`
From `sim_diffusion_planner_runner.sh`, set `CHALLENGE="closed_loop_reactive_agents"`, `SPLIT="val14"`:
```bash
conda activate diffusion_planner
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents \
  planner=diffusion_planner \
  planner.diffusion_planner.config.args_file=./checkpoints/args.json \
  planner.diffusion_planner.ckpt_path=./checkpoints/model.pth \
  scenario_builder=nuplan \
  scenario_filter=val14 \
  worker=ray_distributed \
  worker.threads_per_node=128 \
  distributed_mode='SINGLE_NODE' \
  number_of_gpus_allocated_per_simulation=0.15 \
  enable_simulation_progress_bar=true \
  hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```
> Note the Diffusion-Planner builder name differs by split: `val14` → `scenario_builder=nuplan`;
> `test14-*` → `scenario_builder=nuplan_challenge`. `[VERIFIED from repo script logic]`
> On a single H200, drop `CUDA_VISIBLE_DEVICES` to `0` and raise `number_of_gpus_allocated_per_simulation`
> (e.g. 0.5–1.0) since the script's default 0.15 assumes 8 GPUs. `[UNVERIFIED — tune in env]`

---

## 6. The IDM → SMART swap (arm b) — INTEGRATION POINT + honest status

### 6.1 What the paper says `[VERIFIED from paper text]`
arXiv:2510.14677 (HTML v1) states the SMART agents are "packaged as a drop-in reactive background for
nuPlan, so users can select SMART in the simulator configuration exactly like the standard IDM agents,"
and "we release the SMART agents as a drop-in alternative to IDM at https://github.com/shgd95/InteractiveClosedLoop."

### 6.2 What is actually published right now `[VERIFIED from repo]`
`github.com/shgd95/InteractiveClosedLoop` contains **only `README.md`**, text:
> "Code is under internal approval. Star/watch this repo to get notified when the initial release lands."

There is **no SMART observation class, no config, no checkpoint** published. Therefore:

- The **exact config field** for the swap is **`[UNVERIFIED]`**. Based on the devkit architecture
  (§5a) it will almost certainly be a new entry in the `observation` config group, used as
  `+simulation=closed_loop_reactive_agents observation=smart_agents_observation` (overriding the
  default `idm_agents_observation`), where the SMART YAML's `_target_` points at a SMART subclass of
  `AbstractObservation`. **This is the predicted integration point, not a verified one.**

### 6.3 Files to inspect the moment `InteractiveClosedLoop` is released
1. Any YAML under a `.../config/simulation/observation/` path → the new `observation=` name + `_target_`.
2. The Python class implementing `AbstractObservation` (`get_observation` / `update_observation`) →
   that is the SMART agent engine. Confirm which `update_observation` signature/devkit it expects.
3. Any `requirements.txt` / `setup.py` → the SMART model deps and **which nuplan-devkit version/commit**
   they target. This is the field most likely to clash with the tuPlan-garage / Diffusion-Planner envs.
4. The SMART checkpoint pointer (likely HuggingFace or Drive) + a tokenizer/cluster file
   (base SMART uses `smart/tokens/cluster_frame_5_2048.pkl`, `[VERIFIED exists in rainmaker22/SMART]`).
5. Any nuPlan↔Waymo coordinate/agent-state adapter — base SMART is Waymo-trained; the nuPlan drop-in
   must adapt agent state representation. This adapter is the real engineering content.

### 6.4 Do NOT do this in Stage 0
Do not attempt to port `rainmaker22/SMART` (Waymo, NeurIPS '24) into nuPlan ourselves to fake the SMART
arm. That is a multi-week build (coordinate frames, tokenizer, agent-state I/O, devkit observation glue)
and is exactly the engineering the paper authors are gating behind "internal approval." If we need the
SMART arm before they release, that becomes its own scoped project — not a Stage 0 pilot step.

---

## 7. The metric to read + how to parse it `[VERIFIED from repo config]`

The headline number is the **PDM closed-loop score** (a.k.a. nuPlan closed-loop score), reported separately for:
- **CLS-R** = closed-loop score, **reactive** agents (this is the IDM arm; SMART arm when released).
- **CLS-NR** = closed-loop score, **non-reactive** (log-replay boxes) — your control.

How it's computed (`[VERIFIED from paper]`): per-scenario weighted combination of soft metrics
(route progress, time-to-collision within bounds, comfort/jerk) with **hard multipliers that zero the
score** on at-fault collision or drivable-area violation. Range 0–100.

Where it lands on disk (`[VERIFIED — devkit uses a metric aggregator]`): each run writes to
`$NUPLAN_EXP_ROOT/<experiment_uid>/`. The aggregator selected by the experiment is
`closed_loop_reactive_agents_weighted_average` (reactive) or
`closed_loop_nonreactive_agents_weighted_average` (non-reactive) — verified from the experiment YAMLs.
The aggregated score is written as a **parquet** under the run's `aggregator_metric/` directory, and
per-scenario metrics under `metrics/`.

Parse it:
```bash
# [UNVERIFIED exact filename — confirm the parquet name in $NUPLAN_EXP_ROOT after first run]
python - <<'PY'
import pandas as pd, glob, os
root = os.environ["NUPLAN_EXP_ROOT"]
for f in glob.glob(f"{root}/**/aggregator_metric/*.parquet", recursive=True):
    df = pd.read_parquet(f)
    # the final-row "scenario_type == 'final_score'" (or the all-scenarios row) holds the headline score
    print(f, df.tail(3).to_string())
PY
```
Easiest GUI alternative: launch **nuBoard** (`run_nuboard.py`, used in Diffusion-Planner's
`run_nuboard.ipynb`, `[VERIFIED from repo README]`) and read the closed-loop score off the dashboard.

---

## 8. Explorer sbatch / srun templates

### 8a. Interactive smoke test (single H200, short)
```bash
# [UNVERIFIED — confirm partition/account/gres syntax for Explorer]
srun --partition=gpu --gres=gpu:h200:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash
conda activate nuplan
# run §5b PDM-Closed on the MINI split first (override scenario_filter to a tiny built-in filter)
```

### 8b. Batch job — one planner × one agent-mode on Val14
```bash
#!/bin/bash
#SBATCH --job-name=s0_pdmclosed_idm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00          # gpu partition hard cap = 8h
#SBATCH --output=/projects/<your_alloc>/logs/%x_%j.out
# [UNVERIFIED — confirm Explorer SBATCH directives/account]

source ~/.bashrc
module load anaconda3 cuda/12.4
conda activate nuplan
export NUPLAN_DATA_ROOT=/projects/<your_alloc>/nuplan/dataset
export NUPLAN_MAPS_ROOT=/projects/<your_alloc>/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/projects/<your_alloc>/nuplan/exp
export NUPLAN_DEVKIT_ROOT=/projects/<your_alloc>/nuplan-devkit/

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents \
  planner=pdm_closed_planner \
  scenario_filter=val14_split \
  scenario_builder=nuplan \
  hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```
Submit one job per cell of the matrix (planner × {reactive, nonreactive}).

---

## 9. Checkpoint / resume when a step exceeds 8h

The full Val14 (1,090 scenarios) closed-loop sim for a slow learned planner can exceed 8h on one GPU.
Strategy (`[VERIFIED — nuPlan scenario_filter supports sharding]` / `[UNVERIFIED exact flag names — confirm]`):

1. **Shard by scenario** instead of resuming mid-run. nuPlan's `scenario_filter` supports
   `limit_total_scenarios` and you can run disjoint subsets, then aggregate. Run e.g. 4 shards of
   ~270 scenarios, each well under 8h, as 4 SLURM jobs writing to **distinct `experiment_uid`** dirs.
2. After all shards finish, run the metric aggregator over the union (or average the per-shard
   weighted scores, weighting by scenario count). `[UNVERIFIED — confirm aggregation method matches the
   official weighted_average; safest is to re-run the aggregator pointed at all shard metric dirs]`
3. For data downloads >8h: use resumable `wget -c` / `aws s3 cp` and re-`srun` (§3).

> WHY shard rather than checkpoint: nuPlan closed-loop sim is embarrassingly parallel across scenarios
> and has no native mid-simulation checkpoint. Sharding is the idiomatic way to fit the 8h cap.

---

## 10. DECISION CRITERION — proceed vs kill

Build this 3×2 table (rows = planners with released checkpoints/rule-based; cols = agent model):

| Planner | CLS-R **IDM** | CLS-R **SMART** | Δ (SMART−IDM) |
|---|---|---|---|
| PDM-Closed (rule-based) | ... | ... | ... |
| PDM-Open (ckpt) | ... | ... | ... |
| Diffusion-Planner (ckpt) | ... | ... | ... |

**CONFIRMS the premise (→ proceed to controlled-pair build):**
- The **ranking changes** between the IDM column and the SMART column (e.g. a learned planner that
  beats PDM-Closed under IDM loses to it under SMART, or vice versa), **OR**
- The **margins compress/expand by a large amount** — paper's own framing: under SMART "nearly all
  scores deteriorate" and the gap between rule-based and learned planners narrows/reorders. A Δ that
  changes the *order* of any two planners, or shrinks a >10-point IDM gap to within noise, confirms it.
- Concretely: if rank-correlation (Spearman) between the IDM column and SMART column is well below 1.0
  across our small set, the premise holds and a controlled planner pair is worth building.

**KILLS the premise (→ do NOT build the controlled pair, or rescope):**
- The SMART column is a near-monotone shift of the IDM column (all planners drop ~uniformly) with
  **no rank changes and preserved margins**. If SMART just lowers everyone equally, IDM was a fine
  proxy and a controlled pair adds little.

**Interpretation for our project:** our thesis is that agent realism changes which planner "wins."
Any reordering or margin collapse = the evaluation substrate matters = our controlled-pair experiment
has signal. No reordering = the expensive controlled build is not justified by this evidence; rescope
to a cheaper question.

> Stat hygiene: with only 3 planners the ranking signal is weak. If the IDM run is cheap, add Urban
> Driver (4th released checkpoint) to strengthen the rank test before deciding.

---

## 11. Verification status summary

| Item | Status |
|---|---|
| tuPlan-garage install + `run_simulation.py` invocation | `[VERIFIED from repo README + scripts/simulation/sim_pdm_closed.sh]` |
| PDM-Closed is rule-based (no checkpoint) | `[VERIFIED from pdm_closed_planner.yaml]` |
| PDM-Open `checkpoint_path` config key | `[VERIFIED from pdm_open_planner.yaml]` |
| tuPlan-garage checkpoint Google Drive | `[VERIFIED from repo README]` (exact filenames `[UNVERIFIED]`) |
| Diffusion-Planner install, checkpoint URLs, sim command | `[VERIFIED from repo README + sim_diffusion_planner_runner.sh + requirements_torch.txt]` |
| IDM = `observation: idm_agents_observation`, swap axis = `observation` group | `[VERIFIED from devkit closed_loop_reactive_agents.yaml + idm_agents_observation.yaml]` |
| `AbstractObservation` interface SMART must implement | `[VERIFIED from devkit abstract_observation.py]` |
| Val14 = 1,090 scenarios; CLS-R/CLS-NR definitions | `[VERIFIED from arXiv:2510.14677]` |
| **SMART-reactive drop-in code/config/checkpoint** | `[UNVERIFIED — shgd95/InteractiveClosedLoop is README-only; "code under internal approval"]` |
| **Exact SMART `observation=` field name** | `[UNVERIFIED — predicted smart_agents_observation, confirm on release]` |
| nuplan-devkit commit compatible with tuPlan-garage | `[UNVERIFIED — not pinned in setup.py; confirm in env]` |
| torch 2.0.0+cu118 vs H200 (sm_90) | `[UNVERIFIED — likely needs cu12x torch bump; confirm in env]` |
| Explorer SBATCH/srun directives, partition/account, module names, /projects path | `[UNVERIFIED — confirm against Explorer docs/your allocation]` |
| Dataset sizes, exact download URLs/credentials | `[UNVERIFIED — confirm from dataset_setup.md + nuplan.org account]` |
| Aggregator parquet filename/parse | `[UNVERIFIED — confirm in $NUPLAN_EXP_ROOT after first run]` |

---

## 12. Single most likely failure point + the <30-min test

**Most likely failure point: nuplan-devkit version skew.** Three codebases (tuPlan-garage,
Diffusion-Planner, and the future SMART drop-in) each pin/assume a *different* nuplan-devkit revision,
and the devkit's `AbstractObservation.update_observation` signature and Hydra config layout have drifted
across versions. If the SMART drop-in targets a devkit commit incompatible with the one tuPlan-garage
needs, you cannot run both planners against the same agent backend in one env — which is the entire
point of the pilot. Secondary but related: torch 2.0.0+cu118 not having H200 (sm_90) kernels.

**Test it first, in <30 min of cluster time (before any data download / big run):**
```bash
# [UNVERIFIED commands — but each is a <1-min import check]
srun --partition=gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash

# (1) devkit + tuPlan-garage import together?
conda activate nuplan
python -c "import nuplan; print('devkit', nuplan.__file__)"
python -c "import tuplan_garage; from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner; print('tuplan OK')"
python -c "from nuplan.planning.simulation.observation.idm_agents import IDMAgents; print('IDM OK')"
( cd /projects/<your_alloc>/nuplan-devkit && git rev-parse HEAD )
( cd /projects/<your_alloc>/tuplan_garage && git rev-parse HEAD )

# (2) Diffusion-Planner torch actually runs a CUDA op on H200?
conda activate diffusion_planner
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
x=torch.randn(8,8,device='cuda'); print('matmul ok', (x@x).sum().item())"

# (3) dry-run the smallest possible sim: PDM-Closed on a tiny scenario filter (NOT val14) on MINI split.
#     If this writes a score parquet to $NUPLAN_EXP_ROOT, the whole IDM pipeline is proven.
```
If (1) and (3) pass, the IDM half of the pilot is real and you can commit to the trainval download.
If (2) fails, bump torch to a cu12x/sm_90 build before touching Diffusion-Planner. If you cannot make
one env satisfy both planners, run each planner in its **own env against the same devkit commit** and
only require that the *agent backend* (IDM, later SMART) is byte-identical across planners — that is the
real invariant the pilot depends on.

---

### Appendix: paper claim being reproduced
arXiv:2510.14677 finds IDM-based closed-loop "overestimates planning performance" (nearly all scores
deteriorate under SMART), while some planners improve in interaction-heavy scenarios, and closed-loop-
trained planners are most stable but degrade abruptly in edge cases. The authors propose SMART-reactive
as a *new standard* nuPlan closed-loop benchmark. Our Stage 0 pilot tests whether that re-ranking is
real on our small released-checkpoint set before we invest in a controlled planner pair.
