"""
Post-extraction shard verifier and optional merger.

After all 16 SLURM array tasks finish, run this to:
  1. Verify every shard has the correct F0 tensor shapes and dtypes.
  2. Print a summary (total samples, tasks complete, any failures).
  3. Optionally copy all shards into a single flat directory for training.

Usage:
    python nuplan/slurm/merge_shards.py                     # verify only
    python nuplan/slurm/merge_shards.py --merge-to /scratch/$USER/.../f0_merged

This script is stdlib + torch only (no nuPlan devkit required).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Expected F0 tensor spec (must match FeatureConfig in scene_features.py)
_EXPECTED: Dict[str, Tuple] = {
    "ego":           ((20, 8),   torch.float32),
    "agents":        ((32, 20, 9), torch.float32),
    "agent_mask":    ((32, 20),  torch.bool),
    "map_polylines": ((128, 20, 7), torch.float32),
    "map_mask":      ((128,),    torch.bool),
    "crosswalks":    ((16, 20, 2), torch.float32),
    "crosswalk_mask":((16,),     torch.bool),
    "route_polyline":((40, 4),   torch.float32),
    "route_mask":    ((40,),     torch.bool),
    "traffic_lights":((128,),    torch.int64),
}


def _verify_shard(path: Path) -> Tuple[int, List[str]]:
    """Load one shard, check each sample against the F0 spec.
    Returns (n_samples_ok, list_of_error_strings)."""
    try:
        samples = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return 0, [f"  LOAD ERROR: {exc}"]

    if not isinstance(samples, list):
        return 0, ["  not a list"]

    errors: List[str] = []
    ok = 0
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"  sample {i}: not a dict")
            continue
        for key, (expected_shape, expected_dtype) in _EXPECTED.items():
            if key not in sample:
                errors.append(f"  sample {i}: missing key '{key}'")
                continue
            t = sample[key]
            if not isinstance(t, torch.Tensor):
                errors.append(f"  sample {i}['{key}']: not a tensor")
                continue
            if tuple(t.shape) != expected_shape:
                errors.append(
                    f"  sample {i}['{key}']: shape {tuple(t.shape)} != {expected_shape}"
                )
                continue
            if t.dtype != expected_dtype:
                errors.append(
                    f"  sample {i}['{key}']: dtype {t.dtype} != {expected_dtype}"
                )
                continue
        if not errors or errors[-1].startswith(f"  sample {i}"):
            # Only increment if no new error was added for this sample
            if not any(e.startswith(f"  sample {i}") for e in errors):
                ok += 1
        else:
            ok += 1  # errors list already has this sample's issue
    # recount cleanly
    ok = sum(
        1 for i, s in enumerate(samples)
        if not any(f"sample {i}" in e for e in errors)
    )
    return ok, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify + optionally merge F0 shards")
    ap.add_argument(
        "--base-dir",
        default="/scratch/patodia.pa/av-policy-lab/features/f0",
        help="Base directory containing task_NNNN/ sub-dirs (or flat .pt files)",
    )
    ap.add_argument(
        "--num-tasks",
        type=int,
        default=16,
        help="Expected number of task sub-dirs (default 16)",
    )
    ap.add_argument(
        "--merge-to",
        default=None,
        help="If set, copy all verified shards into this flat directory",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any shard has errors",
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"ERROR: base dir not found: {base}")
        sys.exit(1)

    # Collect task sub-dirs
    task_dirs = sorted(base.glob("task_*/"))
    flat_shards = sorted(base.glob("scene_shard_*.pt"))  # pre-array shards

    print(f"Base dir    : {base}")
    print(f"Task sub-dirs found : {len(task_dirs)} / {args.num_tasks} expected")
    print(f"Flat shards in base : {len(flat_shards)}")
    print()

    all_shards: List[Path] = flat_shards[:]
    for td in task_dirs:
        all_shards.extend(sorted(td.glob("scene_shard_*.pt")))

    if not all_shards:
        print("No .pt shard files found. Has extraction run yet?")
        sys.exit(1)

    total_samples = 0
    total_errors: List[str] = []
    failed_shards: List[Path] = []

    print(f"Verifying {len(all_shards)} shards ...")
    for shard in all_shards:
        n_ok, errs = _verify_shard(shard)
        total_samples += n_ok
        if errs:
            print(f"  FAIL  {shard.relative_to(base)}  ({n_ok} ok)")
            for e in errs[:3]:  # cap output
                print(e)
            if len(errs) > 3:
                print(f"  ... and {len(errs)-3} more")
            total_errors.extend(errs)
            failed_shards.append(shard)
        else:
            print(f"  OK    {shard.relative_to(base)}  ({n_ok} samples)")

    print()
    print("=" * 60)
    print(f"Total samples verified : {total_samples:,}")
    print(f"Total shards           : {len(all_shards)}")
    print(f"Failed shards          : {len(failed_shards)}")
    missing_tasks = args.num_tasks - len(task_dirs)
    if missing_tasks > 0:
        print(f"Missing task dirs      : {missing_tasks}  (those tasks may still be running)")
    print("=" * 60)

    if args.merge_to and not failed_shards:
        dest = Path(args.merge_to)
        dest.mkdir(parents=True, exist_ok=True)
        idx = 0
        for shard in all_shards:
            new_name = dest / f"scene_shard_{idx:05d}.pt"
            shutil.copy2(shard, new_name)
            idx += 1
        print(f"Merged {len(all_shards)} shards into {dest}")
    elif args.merge_to and failed_shards:
        print("Merge skipped: fix failed shards first.")

    if args.strict and (failed_shards or missing_tasks > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
