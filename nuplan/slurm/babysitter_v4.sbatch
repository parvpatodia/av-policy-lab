#!/bin/bash
# Self-chaining babysitter for the f5_2x2_v4 matched run.
# WHY: crontab is PAM-disabled on this cluster and the gpu partition caps walltime
# at 8h (JobRequeue=0, so in-job self-requeue is unavailable). Each cell trains the
# full 150 epochs (~30h) and so spans ~4 of the 8h gpu windows. This short CPU job
# (QOS normal, separate from the gpu submit cap) runs the cap-safe resubmitter once,
# then schedules the NEXT generation 30 min later via --begin. SLURM provides the
# cadence; no held allocation, no crontab. Self-terminates when all 12 cells reach
# epoch 149.
#SBATCH --job-name=f5-baby
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/av-policy-lab/logs/baby-%j.out

set -uo pipefail
RESUB=/home/patodia.pa/av-policy-lab/nuplan/slurm/cron_resubmit_v4.sh
SELF=/home/patodia.pa/av-policy-lab/nuplan/slurm/babysitter_v4.sbatch
LOG=/scratch/patodia.pa/av-policy-lab/logs/cron_resubmit_v4.log
BLOG=/scratch/patodia.pa/av-policy-lab/logs/babysitter_v4.log

ts() { date -Is; }

# Stop the chain once the resubmitter has reported every cell complete.
if tail -5 "$LOG" 2>/dev/null | grep -q "ALL 12 cells complete"; then
  echo "[baby $(ts)] all 12 cells complete -> chain ends, no successor" >> "$BLOG"
  exit 0
fi

# Schedule the next generation FIRST so the chain survives even if the resubmit below
# errors. At most one extra (harmless) generation runs after completion.
if sbatch --begin=now+30minutes "$SELF" >/dev/null 2>&1; then
  echo "[baby $(ts)] chained successor (+30m)" >> "$BLOG"
else
  echo "[baby $(ts)] WARN: failed to chain successor" >> "$BLOG"
fi

# Run the cap-safe resubmit logic for this tick.
bash "$RESUB" >> "$BLOG" 2>&1
echo "[baby $(ts)] tick done" >> "$BLOG"
