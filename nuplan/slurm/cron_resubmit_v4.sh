#!/bin/bash
# Cron resubmitter for the f5_2x2_v4 matched training run.
# WHY: this cluster has JobRequeue=0 (in-job self-requeue unavailable) and the gpu QOS
# caps the user at 8 submitted / 4 running. So incomplete cells are resumed from latest.pt
# across the 8h wall by resubmitting, throttled to the cap. Run via cron every ~30 min.
# Idempotent + cap-safe: only submits incomplete cells NOT already in the queue, up to the
# free submit slots. A cell at >= TARGET epochs is skipped (done). Self-heals failed cells.
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH

REPO=/home/patodia.pa/av-policy-lab
SBATCH=$REPO/nuplan/slurm/train_policy_array.sbatch
RUNDIR=/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4
LOG=/scratch/patodia.pa/av-policy-lab/logs/cron_resubmit_v4.log
TARGET=149            # epoch is 0-indexed in metrics.jsonl; 149 == the 150th (final) epoch
SUBMIT_CAP=8          # gpu QOS MaxSubmitJobsPerUser
HEADS=(det det diff diff); GOALS=(route precise route precise)

ts() { date -Is; }

# indices already in the queue (running or pending) for this run.
# WHY -r + %K: under plain %i a pending array range collapses to "JOBID_[lo-hi]",
# which the old sed counted as ONE index -> the script then tried to resubmit cells
# that were already queued (a duplicate would corrupt the shared cell dir; only the
# QOS submit cap blocked it). -r expands each array task to its own line and %K is
# the bare array index, so queued cells are counted exactly and never re-submitted.
mapfile -t QIDX < <(squeue -u "$USER" -h -r -n f5-2x2 -o "%K" 2>/dev/null | grep -oE '^[0-9]+' | sort -un)
queued_n=${#QIDX[@]}
is_queued() { local i; for i in "${QIDX[@]:-}"; do [ "$i" = "$1" ] && return 0; done; return 1; }

slots=$(( SUBMIT_CAP - queued_n ))
if [ "$slots" -le 0 ]; then echo "[$(ts)] no free submit slots (queued=$queued_n)" >> "$LOG"; exit 0; fi

need=()
done_n=0
for idx in 0 1 2 3 4 5 6 7 8 9 10 11; do
  cell=$(( idx % 4 )); seed=$(( idx / 4 ))
  dir="$RUNDIR/${HEADS[$cell]}_${GOALS[$cell]}_seed${seed}"
  ep=$(tail -1 "$dir/metrics.jsonl" 2>/dev/null | grep -oE '"epoch": *[0-9]+' | grep -oE '[0-9]+$')
  ep=${ep:--1}                                   # no metrics yet => -1 (not started)
  if [ "$ep" -ge "$TARGET" ]; then done_n=$((done_n+1)); continue; fi   # done
  if is_queued "$idx"; then continue; fi                                # already active
  need+=("$idx")
done

if [ "$done_n" -eq 12 ]; then echo "[$(ts)] ALL 12 cells complete (>= epoch $TARGET). Disable cron." >> "$LOG"; exit 0; fi
if [ ${#need[@]} -eq 0 ]; then echo "[$(ts)] nothing to submit (queued=$queued_n, done=$done_n)" >> "$LOG"; exit 0; fi

# submit up to `slots` incomplete cells, throttled to 4 concurrent (MaxJobsPerUser)
sub=("${need[@]:0:$slots}")
csv=$(IFS=,; echo "${sub[*]}")
cd "$REPO" || exit 1
jid=$(sbatch --parsable --array="${csv}%4" "$SBATCH" 2>>"$LOG")
echo "[$(ts)] submitted array=$csv -> job $jid (queued was $queued_n, slots $slots, done $done_n)" >> "$LOG"
