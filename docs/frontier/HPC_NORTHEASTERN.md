# Northeastern Research Computing HPC — GPU Training Guide

> Audience: a graduate student (no PI-owned partition) who needs to train deep-learning models on GPUs.
> Compiled June 2026 from official Northeastern Research Computing (RC) sources.
> Every fact is cited inline with a URL. Anything not verifiable from official RC docs is marked **[UNVERIFIED — confirm with RC]**.

---

## 0. CRITICAL CONTEXT: "Discovery" is being retired — the live cluster is "Explorer"

When this task says "Discovery", note that **Northeastern RC has migrated its general HPC service from the old *Discovery* cluster to a new cluster named *Explorer*** (part of the MGHPCC / Massachusetts AI Hub "AI Compute Resource" effort). As of mid-2025 **all public GPUs were moved off Discovery onto Explorer**, and the RC docs site now titles its connection page "Connecting To Explorer."

- Login is now `login.explorer.northeastern.edu` (not Discovery). — [connectingtocluster/mac.md (rc-docs)](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/connectingtocluster/mac.md)
- Open OnDemand is now `https://ood.explorer.northeastern.edu`. — [best-practices/transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md)
- Performant storage was renamed: Discovery `/work` → Explorer `/projects`. — [best-practices/transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md)
- Modules now live under `/shared/EL9/explorer/modulefiles` (Rocky/EL9). — [gpus/gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md)
- The **transfer node is still `xfer.discovery.neu.edu`** (RC kept the old hostname during transition). — [datamanagement/transferringdata.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/datamanagement/transferringdata.md)
- "All Public GPUs Now on Explorer" announcement (Jul 2025). — [rc.northeastern.edu/2025/07/29/all-public-gpus-now-on-explorer/](https://rc.northeastern.edu/2025/07/29/all-public-gpus-now-on-explorer/)
- Scale: Explorer provides **45,000+ CPU cores and 525+ GPUs free of charge to all NU faculty and students**. — [rc.northeastern.edu/compute/](https://rc.northeastern.edu/compute/)

**Bottom line for you:** build your pipeline against **Explorer** (`login.explorer.northeastern.edu`), using SLURM exactly as documented below. The job-script syntax is the same as legacy Discovery; only hostnames, module versions, and the `/work`→`/projects` path changed.

Canonical sources: [rc-docs.northeastern.edu](https://rc-docs.northeastern.edu/en/latest/) · [rc.northeastern.edu](https://rc.northeastern.edu/) · docs source repo [github.com/northeastern-rc/rc-public-documentation](https://github.com/northeastern-rc/rc-public-documentation).

---

## 1. GPU HARDWARE

Confirmed GPU types (from the GRES request syntax in RC docs and the `/compute` "GPU Nodes" table):

| GPU type | GRES name (`--gres=gpu:<name>:<n>`) | Memory/GPU | Status | Source |
|---|---|---|---|---|
| NVIDIA H200 (Hopper) | `h200` | **140 GB** | Public, on `gpu-short`/`gpu`/`multigpu`; **32 chips on Explorer** | [quickstart-h200.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html), [rc.northeastern.edu/2025/04/30/all-about-explorer-and-the-new-h200-gpus/](https://rc.northeastern.edu/2025/04/30/all-about-explorer-and-the-new-h200-gpus/) |
| NVIDIA V100 SXM2 (Volta) | `v100-sxm2` | 16/32 GB **[UNVERIFIED — confirm exact VRAM with RC]** | Public | [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md), [multigpu-partition-access.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/multigpu-partition-access.md) |
| NVIDIA V100 PCIe (Volta) | `v100-pcie` | 16/32 GB **[UNVERIFIED — confirm exact VRAM with RC]** | Public | [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md) |
| NVIDIA A100 (Ampere) | `a100` (verify exact label, e.g. `a100`/`a100-sxm4`) **[UNVERIFIED — confirm exact GRES label with RC]** | 40 GB and/or 80 GB **[UNVERIFIED — confirm 40 vs 80 GB split with RC]** | Public; docs reference A100s explicitly | [multigpu-partition-access.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/multigpu-partition-access.md) |
| Older Kepler (K40m / K80) | `k40m` / `k80` | legacy | Present but RC warns some software won't run on these | [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md) |

Verified facts:
- **H200**: 140 GB HBM, 32 chips on Explorer, available in `gpu-short`, `gpu`, and `multigpu` partitions, plus Open OnDemand (select `h200` in the GPU Type dropdown). — [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html)
- **V100** comes in two flavors with distinct GRES labels: `v100-sxm2` and `v100-pcie`. Both appear verbatim in RC's official `srun`/`sbatch` examples. — [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md)
- **A100** is explicitly referenced ("If you need to use A100s in your workflow, consider testing your code on V100 GPUs… If a job scales well on V100s it will also scale on A100s. This approach conserves A100 resources…"). — [multigpu-partition-access.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/multigpu-partition-access.md)
- **Kepler caution**: "some programs do not work on the older k40m or k80 GPUs." Use `sinfo -p gpu-interactive --Format=nodes,cpus,memory,features,statecompact,nodelist,gres` to see non-Kepler GPUs and their idle state. — [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)

**[UNVERIFIED — confirm with RC]**: exact per-type counts of V100 vs A100 (only the aggregate "525+ GPUs" and "32 H200" are published in prose); exact A100 GRES label and 40 GB vs 80 GB split; presence of any H100. The authoritative live list is the **"GPU Nodes" table on [rc.northeastern.edu/compute/](https://rc.northeastern.edu/compute/)** (use only rows marked *Public*) and the live command `sinfo … --Format=…,gres`.

### Exact request syntax (copy/paste), per official RC examples
```bash
# H200 (interactive):
srun --partition=gpu-interactive --nodes=1 --pty --gres=gpu:h200:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
# V100 SXM2 (interactive):
srun --partition=gpu-interactive --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
# V100 PCIe (batch directive):
#SBATCH --gres=gpu:v100-pcie:1
# Generic single GPU (any type the scheduler picks):
--gres=gpu:1
```
Sources: [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md), [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html).
Note: requesting a *specific* GPU type increases queue wait time depending on availability. — [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md)

---

## 2. GPU PARTITIONS & TIME LIMITS

Verbatim from the official "GPUs on the HPC" table. — [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)

| Partition | `--partition=` | Requires approval? | Time hrs (Default / **Max**) | Max jobs (Running / Submitted) | GPU/job limit | Per-user GPU cap |
|---|---|---|---|---|---|---|
| `gpu-short` | `gpu-short` | No | 1 / **2** | 2 / 4 | 1 | **1** |
| `gpu-interactive` | `gpu-interactive` | No | 1 / **2** | 2 / 4 | 1 | **1** |
| `gpu` | `gpu` | No | 4 / **8** | 4 / 8 | 1 | **4** |
| `multigpu` | `multigpu` | **YES (ServiceNow ticket + scaling test)** | 4 / **24** | 4 / 8 | 8 | **8** |

Key points:
- **Anyone with a cluster account has access to `gpu`, `gpu-short`, `gpu-interactive`** — no PI partition required. — [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)
- On `gpu` / `gpu-interactive`, **requesting more than 1 GPU fails** — single-GPU only unless you have `multigpu` access. You also cannot grab all CPUs on a GPU node (reserved for other GPUs). — [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md), [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html)
- `gpu-short` has **higher scheduling priority** than `gpu` (designed for short jobs / quick turnaround). — [rc.northeastern.edu/2025/01/27/hpc-cluster-partition-improvements/](https://rc.northeastern.edu/2025/01/27/hpc-cluster-partition-improvements/)
- **Longest single GPU job without multigpu access = 8 h** (`gpu`). **Absolute longest GPU job = 24 h** (`multigpu`, approval required). — [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)
- `reservation` partition exists for temporary multi-GPU testing reservations during the `multigpu` application (e.g. `srun -p reservation --reservation=<name> --gres=gpu:v100-sxm2:4 --time=24:00:00`). — [multigpu-partition-access.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/multigpu-partition-access.md)

### CPU partitions (for data prep / transfers / quota checks)
The general-access CPU partitions are `short`, `debug`, etc.; `debug` uses the same hardware as `short` with different time/job/core limits. — [rc.northeastern.edu/partitions/](https://rc.northeastern.edu/partitions/)
For exact CPU-partition max wall-clock (e.g. `short` is commonly 24 h, `long`/`large` longer), see the live table at [rc.northeastern.edu/partitions/](https://rc.northeastern.edu/partitions/) and [rc-docs hardware/partitions](https://rc-docs.northeastern.edu/en/latest/runningjobs/index.html). **[UNVERIFIED — confirm exact CPU-partition wall-clock numbers with RC]** (the live partitions table did not render in search; values not quotable from a primary doc).

---

## 3. QUOTAS / LIMITS / ACCESS TIER

- **General-access tier exists and is free.** Explorer is "free of charge to all Northeastern faculty and students." A student does **not** need a PI to get *GPU access* on the public `gpu`/`gpu-short` partitions — those are open to anyone with a cluster account. — [rc.northeastern.edu/compute/](https://rc.northeastern.edu/compute/), [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)
- **BUT account creation now requires PI/storage-space membership.** Per RC's "Getting Access": *"All Explorer users must have membership in a PI/Staff-owned storage space."* A PI can add you with the `project` command, or you submit a Research Computing Access Request (ServiceNow). If your lab has no storage space, a New Storage Space Request is needed. So in practice **a student needs a PI/sponsor to exist on the cluster, but once on it, GPU access is automatic** (no separate GPU allocation). — [rc.northeastern.edu/getting-access/](https://rc.northeastern.edu/getting-access/), [best-practices/transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md)
- **Concurrent-job / GPU caps** (from the table in §2): single-user is capped at **1 GPU** on short/interactive, **4 GPUs** on `gpu` (4 running / 8 queued jobs), **8 GPUs** on `multigpu` (approval).
- **Scheduling is fair-share, not a hard GPU-hour budget.** Priority drops with recent heavy usage and rises with queue wait time and small job size. There is **no documented monthly/semester GPU-hour cap** for general users. — [understandingqueuing.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/runningjobs/understandingqueuing.md)
- **[UNVERIFIED — confirm with RC]**: any explicit GPU-hour allocation cap, QOS tiers, or burst limits beyond fair-share. None documented publicly as of June 2026.
- **Idle enforcement:** bots monitor login/compute nodes; idle GPU jobs trigger emails to you and RC. Use `seff`, `historical-seff`, and `gpu-logs <jobid>` to audit efficiency. Don't sit on idle GPUs. — [clusterusage.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/clusterusage.md)
- **Graduating?** Continued access requires a sponsored NU account. — [faq.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/faq.md)

---

## 4. STORAGE

| Space | Path | Quota | Backed up? | Purpose | Source |
|---|---|---|---|---|---|
| Home | `/home/<username>` | **~75 GB**, fixed, **cannot be increased** | No (small) | scripts, source, small files; OOD apps run from here | [scratchpurge.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/scratchpurge.md), [rc.northeastern.edu/data-storage-options/](https://rc.northeastern.edu/data-storage-options/) |
| Scratch | `/scratch/<username>` | **20 TB + 20M inodes per user** | **No — purged monthly** | temporary/intermediate job output | [rc.northeastern.edu/data-storage-options/](https://rc.northeastern.edu/data-storage-options/), [scratchpurge.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/scratchpurge.md) |
| Projects | `/projects/<group>` (was `/work` on Discovery) | **up to 35 TB complimentary per PI**, summed across that PI's projects; more is purchasable | Persistent (PI-requested) | long-term datasets, conda envs, containers, results | [rc.northeastern.edu/data-storage-options/](https://rc.northeastern.edu/data-storage-options/), [workquota.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/workquota.md), [transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md) |

- **Scratch purge policy:** `/scratch` is purged during the monthly maintenance window (reported as the **first Tuesday of the month**); files are **not** backed up. Move anything you need to `/projects` (or `/home`) before then, or checkpoint long jobs to survive the purge. — [rc.northeastern.edu/scratch-space-policy/](https://rc.northeastern.edu/scratch-space-policy/), [scratchpurge.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/scratchpurge.md)
- **Check your usage:** from a `short` compute node run `check-quota /home/<user>`, `check-quota /scratch/<user>`, `check-quota /projects/<dir>`. Output shows disk soft/hard limits + inode soft/hard limits. — [homequota.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md)

### Where should a 1–5 TB nuPlan dataset live?
- **`/projects/<yourgroup>/`** — this is the only correct home for a large, persistent dataset. It is performant, persistent, and large (35 TB/PI complimentary easily covers 1–5 TB). You (or your PI) must have a `/projects` space; request via [New Storage Space Request](https://bit.ly/NURC-NewStorage). — [homequota.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md), [workquota.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/workquota.md)
- **NOT `/home`** (75 GB cap) and **NOT permanently in `/scratch`** (monthly purge, no backup). Use `/scratch` only for transient intermediate output, then move keepers to `/projects`. — [scratchpurge.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/scratchpurge.md)

### Data-transfer guidance
- **Dedicated transfer node `xfer.discovery.neu.edu`** — you must transfer to/from the cluster through this node (login/compute nodes won't transfer to your local machine). — [transferringdata.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/datamanagement/transferringdata.md)
- **Globus is "highly recommended" for large datasets** (i.e. for nuPlan). — [transferringdata.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/datamanagement/transferringdata.md), [datamanagement/globus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/datamanagement/globus.md)
- **scp / rsync / sshfs** to `xfer.discovery.neu.edu` are supported; for cluster-to-cluster (directory→directory) copies use a compute node with `--constraint=ib` (InfiniBand) for speed. GUI options: OOD File Explorer, MobaXterm, FileZilla. `rclone` is available (module `rclone/1.72`) for Dropbox/Google Drive. — [transferringdata.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/datamanagement/transferringdata.md)
```bash
# Example: scp a dataset to scratch
scp -r ./nuplan <username>@xfer.discovery.neu.edu:/scratch/<username>/
```

---

## 5. SOFTWARE ENVIRONMENT

- **Modules (Lmod):** `module avail`, `module load`, `module show`. On Explorer all modules are new and live under `/shared/EL9/explorer/modulefiles`. Load `module load explorer` to expose the Explorer module tree. **Remove `module load` lines from `.bashrc`** — Discovery modules cause "MODULE NOT FOUND" / env conflicts on Explorer. — [gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md), [transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md)
- **CUDA:** multiple toolkits via modules, e.g. `cuda/12.1.1`, `cuda/12.3.0`. Load with `module load cuda/12.1.1`. `nvidia-smi` on a GPU node shows driver info. — [gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md)
- **Anaconda/Miniconda:** `module load anaconda3/2024.06` (Anaconda available; Miniconda installable yourself). **Build conda envs in `/projects` (use `--prefix=/projects/<group>/<env>`), NOT `/home`** (home quota). Do not auto-init conda in `.bashrc` (breaks OOD). Clean with `conda clean --all`. — [conda.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/software/packagemanagers/conda.md), [homequota.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md)
- **Containers — Apptainer/Singularity:** Apptainer is installed **system-wide on every node** (no module needed); `apptainer` and `singularity` are interchangeable. Prebuilt images at `/shared/container_repository/explorer/`. `apptainer pull docker://…` from NGC/Docker Hub/Singularity Hub works. Always `-B "/projects:/projects,/scratch:/scratch"` to bind your data, and set `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` into `/projects` (not `/home`). — [apptainer.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/containers/apptainer.md)
- **PyTorch (official RC recipe):**
```bash
srun --partition=gpu-interactive --nodes=1 --gres=gpu:v100-sxm2:1 --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
conda create --name pytorch_env -c conda-forge python=3.12.4 -y
source activate pytorch_env
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.cuda.is_available())'   # expect True
```
TensorFlow recipe (`pip install tensorflow[and-cuda]`) is documented similarly. — [gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md)
- **Internet on compute nodes:** **Yes.** RC's own documented recipes run `pip install … --index-url https://download.pytorch.org/…`, `wget https://repo.anaconda.com/…`, and `apptainer pull docker://…` directly from interactive `srun` compute-node sessions (`gpu-interactive` / `short`), proving outbound internet from compute nodes. Heavy *file transfers* should still go through the `xfer` node. — [gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md), [conda.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/software/packagemanagers/conda.md), [apptainer.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/containers/apptainer.md)

---

## 6. INTERACTIVE vs BATCH

### Interactive GPU session (`srun`)
```bash
# 1 H200, 1 hr, 4GB CPU RAM, 1 core:
srun --partition=gpu-interactive --nodes=1 --pty --gres=gpu:h200:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
# 1 V100-SXM2, 2 hr, 2 cores, 10GB:
srun --partition=gpu-interactive --nodes=1 --gres=gpu:v100-sxm2:1 --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
```
Sources: [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html), [gpujobsubmission.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpujobsubmission.md). (RC docs use `srun`; `salloc` is standard SLURM but `srun --pty` is the documented pattern.)

### Batch GPU job (`sbatch`) — official RC template
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1        # or gpu:h200:1, gpu:v100-sxm2:1, gpu:a100:1
#SBATCH --time=08:00:00               # max 8h on `gpu`; 24h needs `multigpu`
#SBATCH --job-name=gpu_run
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate /projects/<group>/envs/pytorch_env
python train.py
```
Submit with `sbatch my_job.sh`; monitor with `squeue -u <username>`; cancel with `scancel <jobid>`. — [accessinggpus.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/accessinggpus.md), [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html), [interactiveandbatch.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/runningjobs/interactiveandbatch.md)

For runs longer than the wall-clock cap, **checkpoint** and resubmit (RC has a dedicated checkpointing best-practices page). — [best-practices/checkpointing.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/checkpointing.md)

---

## 7. NORTHEASTERN SILICON VALLEY / REGIONAL-CAMPUS ACCESS

- RC documentation describes Explorer access purely by **Northeastern identity** ("all Northeastern faculty and students," login with your "Northeastern username/password"), with **no Boston-campus restriction** stated anywhere in the docs. Access is gated by NU credentials + PI/storage-space membership (§3), not by campus. — [rc.northeastern.edu/compute/](https://rc.northeastern.edu/compute/), [rc.northeastern.edu/getting-access/](https://rc.northeastern.edu/getting-access/), [connectingtocluster/mac.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/connectingtocluster/mac.md)
- It is over the internet via SSH / Open OnDemand, so a student physically at the **Silicon Valley / Oakland** campus connects identically (possibly via NU VPN — see below).
- **[UNVERIFIED — confirm with RC]**: whether any VPN (GlobalProtect) is required from off the Boston network, and whether regional-campus students have any onboarding nuance. The docs don't mention a campus-specific policy. Recommend emailing `rchelp@northeastern.edu` to confirm SVL eligibility/VPN, since you are MS AI @ NEU Silicon Valley.

---

## 8. ACCESS / ONBOARDING

1. **Get an account.** Submit a **Research Computing Access Request** in ServiceNow, OR have your PI add you to their `/projects` space with `project <projectname> add members <username>` (this auto-creates your cluster account). Account creation can take up to ~24 h after sponsor approval; you get an email confirmation. — [rc.northeastern.edu/getting-access/](https://rc.northeastern.edu/getting-access/), [transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md)
2. **PI / sponsor requirement.** You must belong to a PI/Staff-owned storage space. If your lab has none, submit a New Storage Space Request. GPU compute itself needs no separate request (public `gpu` partition is open to all accounts); `multigpu` needs a ServiceNow ticket + scaling test (§2). — [rc.northeastern.edu/getting-access/](https://rc.northeastern.edu/getting-access/), [gpuoverview.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md)
3. **Log in (SSH):**
   ```bash
   ssh <username>@login.explorer.northeastern.edu      # add -Y for X11/GUI
   ```
   Set up passwordless SSH (`ssh-keygen`, `ssh-copy-id`) for OOD GUI apps. — [connectingtocluster/mac.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/connectingtocluster/mac.md)
4. **Open OnDemand (web):** `https://ood.explorer.northeastern.edu` — JupyterLab, RStudio, MATLAB, Desktop, and GPU apps (pick `gpu-short`/`gpu`/`multigpu` + GPU type `h200`). — [transition.md](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/transition.md), [quickstart-h200](https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/quickstart-h200.html), [rc.northeastern.edu/ood/](https://rc.northeastern.edu/ood/)
5. **Help:** `rchelp@northeastern.edu`; support catalog at [rc.northeastern.edu/support](https://rc.northeastern.edu/support).

---

## PRACTICAL BUDGET FOR A SOLO STUDENT

Design your training pipeline assuming a **general-access (no `multigpu`) tier**: realistically you can run **1 GPU at a time on `gpu-short`/`gpu-interactive` (≤2 h)** for dev/debug and **up to 4 concurrent single-GPU `gpu` jobs at ≤8 h each** for real training runs — so plan around a **hard 8-hour per-job wall-clock ceiling and roughly 4 GPUs in flight**, with **no published GPU-hour cap** (you are throttled only by fair-share priority, which dips after heavy use and recovers when you back off). That means architect everything around **checkpoint-and-resume in ≤8 h chunks** rather than one monolithic multi-day run; e.g. a Diffusion Policy training that needs ~30 GPU-hours becomes ~4 sequential 8-h `gpu` jobs (or parallel arrays across the 4-GPU cap), each writing checkpoints to `/projects` so the monthly `/scratch` purge and queue preemption never cost you progress. Single-GPU H200 (140 GB) jobs are plentiful and a good default for large models; only pursue the **`multigpu` partition (8 GPUs, 24 h, approval + a V100 scaling-efficiency test ≥0.5)** if a single GPU genuinely can't hold your model or you've proven near-linear multi-GPU scaling. Keep the 1–5 TB nuPlan dataset in `/projects` (35 TB/PI complimentary), stage hot shards to `/scratch` per job, and transfer via Globus. **[UNVERIFIED — confirm with RC]**: exact CPU-partition wall-clock limits, exact A100/V100 counts and A100 GRES label/VRAM, any H100 presence, and SVL-campus VPN/eligibility — confirm via `rc.northeastern.edu/compute/`, live `sinfo`, and `rchelp@northeastern.edu`.

---

### Source index
- GPU overview & partition limits: https://rc-docs.northeastern.edu/en/latest/gpus/gpuoverview.html · https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/gpus/gpuoverview.md
- GPU access (srun/sbatch, GRES): https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html
- H200 quick start: https://rc-docs.northeastern.edu/en/explorer-main/gpus/quickstart-h200.html
- Multi-GPU partition access: https://rc-docs.northeastern.edu/en/latest/gpus/multigpu-partition-access.html
- GPU job submission (CUDA/PyTorch/TF): https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html
- Interactive & batch: https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html
- Queuing / fair-share: https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html
- Cluster usage / seff / gpu-logs: https://rc-docs.northeastern.edu/en/latest/best-practices/clusterusage.html
- Home quota & conda/apptainer hygiene: https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html
- Projects quota: https://rc-docs.northeastern.edu/en/latest/best-practices/workquota.html
- Scratch purge: https://rc-docs.northeastern.edu/en/latest/best-practices/scratchpurge.html · https://rc.northeastern.edu/scratch-space-policy/
- Transition (Discovery→Explorer): https://rc-docs.northeastern.edu/en/latest/best-practices/transition.html
- Data transfer / Globus: https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html
- Conda: https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html
- Apptainer: https://rc-docs.northeastern.edu/en/latest/containers/apptainer.html
- Connecting (SSH/OOD): https://rc-docs.northeastern.edu/en/latest/connectingtocluster/mac.html
- Compute hardware & scale: https://rc.northeastern.edu/compute/
- Data storage options: https://rc.northeastern.edu/data-storage-options/ · https://rc.northeastern.edu/research-projects-storage-space-policy/
- Getting access: https://rc.northeastern.edu/getting-access/
- Open OnDemand: https://rc.northeastern.edu/ood/
- H200/Explorer announcements: https://rc.northeastern.edu/2025/04/30/all-about-explorer-and-the-new-h200-gpus/ · https://rc.northeastern.edu/2025/07/29/all-public-gpus-now-on-explorer/ · https://rc.northeastern.edu/2025/01/27/hpc-cluster-partition-improvements/
- FAQ: https://rc-docs.northeastern.edu/en/latest/faq.html
