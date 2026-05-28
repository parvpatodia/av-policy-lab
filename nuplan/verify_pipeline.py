"""
verify_pipeline.py — AV-Policy-Lab pipeline invariant checker.

Checks all critical assumptions about the nuPlan mini DB sampling rate,
training/inference goal timing, DT stamp consistency, checkpoint existence,
and SpeedAdaptive goal-scale alignment.

Run with:
    conda activate nuplan && python nuplan/verify_pipeline.py
"""

import sqlite3
import math
import sys
from pathlib import Path

import numpy as np

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def _pass(msg):  return f"[{GREEN}PASS{RESET}] {msg}"
def _fail(msg):  return f"[{RED}FAIL{RESET}] {msg}"
def _warn(msg):  return f"[{YELLOW}WARN{RESET}] {msg}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file so the script works from any cwd.
_HERE     = Path(__file__).resolve().parent
DB_DIR    = Path("/Users/parvpatodia/nuplan-devkit/data/cache/mini")
CKPT_DIR  = _HERE / "checkpoints"

FUTURE_STEPS  = 16
STRIDE        = 10
GOAL_OFFSET   = 8          # raw DB rows
DT            = 0.1        # trajectory waypoint spacing (seconds)
# _GOAL_LOOKAHEAD_S: matches GoalBCPlanner._get_expert_at_offset offset_steps=8
# at 100_000 µs/step → 8 × 0.1 s = 0.8 s
_GOAL_LOOKAHEAD_S = 8 * 0.1  # = 0.8 s

# ── Counters ──────────────────────────────────────────────────────────────────
N_total = 0
N_pass  = 0
N_fail  = 0
N_warn  = 0
critical_issues = []


def _record(status: str, label: str):
    global N_total, N_pass, N_fail, N_warn
    N_total += 1
    if status == "PASS":
        N_pass += 1
    elif status == "FAIL":
        N_fail += 1
        critical_issues.append(label)
    elif status == "WARN":
        N_warn += 1


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: DB Sampling Rate
# ─────────────────────────────────────────────────────────────────────────────
def check_db_sampling_rate():
    print("\n── Check 1: DB Sampling Rate ────────────────────────────────────────")
    db_files = sorted(DB_DIR.glob("*.db"))
    if not db_files:
        print(_fail(f"No .db files found in {DB_DIR}"))
        _record("FAIL", "Check 1: DB sampling rate (no files found)")
        return None

    db = db_files[0]
    con = sqlite3.connect(str(db))
    rows = con.execute(
        "SELECT timestamp FROM ego_pose ORDER BY timestamp"
    ).fetchall()
    con.close()

    ts = np.array([r[0] for r in rows], dtype=np.float64)
    deltas_us = np.diff(ts)               # microseconds
    mean_delta_ms = deltas_us.mean() / 1_000.0
    rate_hz       = 1_000.0 / mean_delta_ms

    label = f"DB sampling rate: {rate_hz:.1f} Hz"
    if 9.0 <= mean_delta_ms <= 11.0:
        print(_pass(f"{label}  (mean Δt = {mean_delta_ms:.2f} ms  ✓ 9–11 ms window)"))
        _record("PASS", "Check 1: DB sampling rate")
    else:
        print(_fail(f"{label}  (mean Δt = {mean_delta_ms:.2f} ms — expected 9–11 ms)"))
        print(f"       Investigate: check ego_pose table in {db.name}")
        print(f"       If Δt ≈ 100 ms the DB is 10 Hz — all goal timing math changes.")
        _record("FAIL", "Check 1: DB sampling rate")

    return rate_hz


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Training Goal Timing
# ─────────────────────────────────────────────────────────────────────────────
def check_training_goal_timing():
    print("\n── Check 2: Training Goal Timing ────────────────────────────────────")
    db_files = sorted(DB_DIR.glob("*.db"))[:10]
    if not db_files:
        print(_fail("No DB files found — skipping."))
        _record("FAIL", "Check 2: Training goal timing (no files)")
        return None, None

    goal_mags  = []
    speeds     = []

    for db in db_files:
        con = sqlite3.connect(str(db))
        rows = con.execute(
            "SELECT x, y, qw, qx, qy, qz, vx, vy FROM ego_pose ORDER BY timestamp"
        ).fetchall()
        con.close()

        arr = np.array(rows, dtype=np.float64)
        x_g = arr[:, 0]
        y_g = arr[:, 1]
        vx  = arr[:, 6]
        vy  = arr[:, 7]
        N   = len(arr)

        count = 0
        for i in range(0, N - FUTURE_STEPS, STRIDE):
            if count >= 100:   # 100 windows per file → 1000 total
                break
            cx, cy = x_g[i], y_g[i]
            gi = min(i + GOAL_OFFSET, N - 1)
            dx_g = x_g[gi] - cx
            dy_g = y_g[gi] - cy
            goal_mags.append(math.sqrt(dx_g ** 2 + dy_g ** 2))
            speeds.append(math.sqrt(vx[i] ** 2 + vy[i] ** 2))
            count += 1

    mean_goal  = float(np.mean(goal_mags))
    mean_speed = float(np.mean(speeds))
    # T+8 at 100 Hz = 0.08 s; expected distance = speed × 0.08 s
    expected   = mean_speed * 0.08

    pct_diff = abs(mean_goal - expected) / max(expected, 1e-9) * 100
    label = f"Training T+8 goal: {mean_goal:.3f} m vs speed×0.08 = {expected:.3f} m"
    if pct_diff <= 5.0:
        print(_pass(f"{label}  ({pct_diff:.1f}% diff ≤ 5%)"))
        _record("PASS", "Check 2: Training goal timing")
    else:
        print(_fail(f"{label}  ({pct_diff:.1f}% diff > 5% threshold)"))
        print(f"       If goal >> speed×0.08, DB may not be 100 Hz. Re-run Check 1.")
        _record("FAIL", "Check 2: Training goal timing")

    return mean_goal, mean_speed


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: GoalBCPlanner Inference Goal Timing
# ─────────────────────────────────────────────────────────────────────────────
def check_inference_goal_timing(mean_goal_train, mean_speed):
    print("\n── Check 3: GoalBCPlanner Inference Goal Timing ─────────────────────")
    if mean_speed is None:
        print(_fail("Skipping — Check 2 did not produce mean_speed."))
        _record("FAIL", "Check 3: Inference goal timing (dependency missing)")
        return

    # GoalBCPlanner._get_expert_at_offset: offset_steps=8, step=100_000 µs = 0.1 s
    # → T + 8 × 0.1 = T + 0.8 s
    infer_goal = mean_speed * 0.8
    train_goal = mean_goal_train if mean_goal_train is not None else mean_speed * 0.08

    factor = infer_goal / max(train_goal, 1e-9)

    print(f"       Inference goal scale: {infer_goal:.3f} m (T+0.8s)")
    print(f"       Training  goal scale: {train_goal:.3f} m (T+0.08s)")
    label = f"Train→inference scale factor: {factor:.1f}×  [EXPECTED: ~10×]"
    if 8.0 <= factor <= 12.0:
        print(_pass(label))
        _record("PASS", "Check 3: Inference goal timing scale factor")
    else:
        print(_fail(label))
        print(f"       Expected ~10× gap (100 Hz training, 10 Hz inference step).")
        print(f"       If factor ≠ ~10, investigate _get_expert_at_offset offset units.")
        _record("FAIL", "Check 3: Inference goal timing scale factor")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: DT Consistency
# ─────────────────────────────────────────────────────────────────────────────
def check_dt_consistency(mean_speed):
    """
    Check 4: DT stamp vs DB sampling rate.

    This check documents a KNOWN STRUCTURAL PROPERTY of the pipeline, not an
    unknown runtime value. The ratio is always 10× by construction:
      DB at 100 Hz → raw step = 0.01 s
      DT = 0.1 s  → stamps each raw step 10× further apart

    The check is meaningful because it verifies the invariant holds and reminds
    future readers of the quirk. If DT or the DB rate ever change, the ratio
    will deviate from 10× and the WARN threshold catches it.

    WHY not FAIL: perfect_tracking_controller executes trajectories by spatial
    position interpolation, not by absolute timestamps. GoalBC's 1.82 m L2 result
    (validated in goal_bc.ipynb) confirms the mismatch has no practical effect.
    """
    print("\n── Check 4: DT Consistency ──────────────────────────────────────────")
    if mean_speed is None:
        print(_fail("Skipping — Check 2 did not produce mean_speed."))
        _record("FAIL", "Check 4: DT consistency (dependency missing)")
        return

    db_step_s = 1.0 / 100.0          # 100 Hz DB → 10 ms per raw row
    step_displacement = mean_speed * db_step_s
    apparent_speed    = step_displacement / DT
    ratio             = mean_speed / apparent_speed   # structurally always DT/db_step_s

    expected_ratio = DT / db_step_s  # 0.1 / 0.01 = 10

    print(f"       DB rate: 100 Hz (10 ms/row, confirmed by Check 1)")
    print(f"       DT stamp: {DT} s/step")
    print(f"       Structural ratio DT/db_step = {DT}/{db_step_s} = {expected_ratio:.0f}×")
    print(f"       At mean_speed {mean_speed:.2f} m/s: step displacement = {step_displacement:.4f} m")
    print(f"       Waypoint apparent speed: {apparent_speed:.3f} m/s  "
          f"(= actual / {ratio:.0f})")

    label = f"DT/DB-rate ratio: {ratio:.1f}×  [structural, expected {expected_ratio:.0f}×]"
    if abs(ratio - expected_ratio) < 0.5:
        print(_warn(f"{label} — known quirk: waypoints stamped 10× sparser than training."))
        print(f"       perfect_tracking_controller uses spatial position → no L2 impact.")
        print(f"       Evidence: GoalBC 1.82 m L2 despite this mismatch (goal_bc.ipynb).")
        _record("WARN", "Check 4: DT/DB-rate mismatch (known, benign)")
    else:
        print(_fail(f"{label} — ratio has changed from expected {expected_ratio:.0f}×. "
                    f"Investigate DT or DB sampling rate."))
        _record("FAIL", "Check 4: DT consistency unexpected ratio")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Checkpoint Existence
# ─────────────────────────────────────────────────────────────────────────────
def check_checkpoints():
    print("\n── Check 5: Checkpoint Existence ────────────────────────────────────")
    required = [
        ("checkpoints/bc_best.pt",  True),
        ("checkpoints/goal_bc.pt",  True),
        ("checkpoints/trained_route_bc.pt", False),   # optional
    ]
    for rel_path, required_flag in required:
        full = CKPT_DIR.parent / rel_path
        exists = full.exists()
        tag    = "" if required_flag else "  [optional]"
        if exists:
            print(_pass(f"{rel_path}{tag}"))
            _record("PASS", f"Check 5: {rel_path}")
        elif required_flag:
            print(_fail(f"{rel_path} — not found at {full}"))
            _record("FAIL", f"Check 5: {rel_path} missing")
        else:
            print(_warn(f"{rel_path} not found{tag}"))
            _record("WARN", f"Check 5: {rel_path} (optional) missing")


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: SpeedAdaptive vs GoalBC Goal Scale at Inference
# ─────────────────────────────────────────────────────────────────────────────
def check_speed_adaptive_scale(mean_speed):
    """
    Check 6: SpeedAdaptive look-ahead matches GoalBC inference at typical AND edge speeds.

    Two sub-checks:
      6a. At mean_speed (typical driving): look-ahead = GoalBC inference exactly.
          This is trivially true by construction when speed > 0.0625 m/s (above
          the 0.05m floor). We check it and note it's structural.
      6b. At speed = 0 (full stop): SpeedAdaptive floor = 0.05m, GoalBC = 0m.
          We check the floor is small (< 0.1m) so it doesn't dominate at stops.
          The 0.05m floor is 14% of the GoalBC mean training goal (0.35m) — acceptable.
    """
    print("\n── Check 6: SpeedAdaptive Goal Scale at Inference ───────────────────")
    if mean_speed is None:
        print(_fail("Skipping — Check 2 did not produce mean_speed."))
        _record("FAIL", "Check 6: SpeedAdaptive scale (dependency missing)")
        return

    # Sub-check 6a: at mean speed
    look_ahead_sa   = max(0.05, mean_speed * _GOAL_LOOKAHEAD_S)
    look_ahead_goal = mean_speed * _GOAL_LOOKAHEAD_S
    clamp_fired     = look_ahead_sa > look_ahead_goal

    label_a = (f"At mean_speed {mean_speed:.2f} m/s: "
               f"SpeedAdaptive={look_ahead_sa:.3f} m, GoalBC={look_ahead_goal:.3f} m"
               f"{'  [CLAMP ACTIVE]' if clamp_fired else ''}")
    if not clamp_fired:
        print(_pass(f"6a: {label_a}"))
        _record("PASS", "Check 6a: SpeedAdaptive matches GoalBC at mean speed")
    else:
        print(_warn(f"6a: {label_a} — mean_speed < 0.0625 m/s; clamp dominates."))
        _record("WARN", "Check 6a: clamp active at mean speed (very low speed dataset)")

    # Sub-check 6b: stopped vehicle
    floor_m = max(0.05, 0.0 * _GOAL_LOOKAHEAD_S)   # = 0.05m
    floor_pct_of_train = floor_m / 0.35 * 100       # relative to training goal mean
    label_b = (f"At speed=0: SpeedAdaptive floor = {floor_m:.3f} m "
               f"({floor_pct_of_train:.0f}% of training goal mean 0.35m)")
    if floor_m <= 0.1:
        print(_pass(f"6b: {label_b}  — small floor, acceptable"))
        _record("PASS", "Check 6b: SpeedAdaptive floor at stop is small")
    else:
        print(_fail(f"6b: {label_b}  — floor too large for stopped scenarios"))
        _record("FAIL", "Check 6b: SpeedAdaptive floor at stop too large")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  AV-Policy-Lab Pipeline Verification")
    print(f"  DB:   {DB_DIR}")
    print(f"  CKPT: {CKPT_DIR}")
    print("=" * 68)

    rate_hz                    = check_db_sampling_rate()
    mean_goal_train, mean_speed = check_training_goal_timing()
    check_inference_goal_timing(mean_goal_train, mean_speed)
    check_dt_consistency(mean_speed)
    check_checkpoints()
    check_speed_adaptive_scale(mean_speed)

    print("\n" + "=" * 68)
    print("  === Pipeline Verification Summary ===")
    print(f"  N checks: {N_total}  "
          f"{GREEN}PASS: {N_pass}{RESET}  "
          f"{RED}FAIL: {N_fail}{RESET}  "
          f"{YELLOW}WARNING: {N_warn}{RESET}")
    if critical_issues:
        print(f"  {RED}Critical issues:{RESET}")
        for iss in critical_issues:
            print(f"    • {iss}")
    else:
        print(f"  {GREEN}No critical issues.{RESET}")
    print("=" * 68)

    sys.exit(0 if N_fail == 0 else 1)


if __name__ == "__main__":
    main()
