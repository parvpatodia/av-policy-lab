"""Sharded dataset over F0 v2 feature shards (the labeled dataset).

Format (written by SceneFeatureExtractor.save_shard):
    shard = {"samples": [dict of tensors], "config": {...}}
Each sample holds the encoder inputs (pre-normalized by the F0 builders)
plus the training target `ego_future` (16, 3) in RAW meters/radians.

WHY IterableDataset (not map-style): a map-style dataset needs a global
__len__, which would require loading all ~700 shards at init just to count
samples. Streaming shards with two-level shuffling (shard order, then
within-shard sample order) is the standard recipe for sharded data and
costs one shard in memory at a time.

WHY FUTURE_SCALE: inputs arrive ~unit-scale from F0, but ego_future is raw
meters (up to ~100 m at 8 s). Dividing by 10 brings the regression target
to ~unit range so both heads train at sane loss magnitudes; use
unscale_future() before computing metric-space (meter) errors.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

FUTURE_SCALE = 10.0
# WHY scale heading by pi: raw radians span [-pi, pi] while scaled xy spans
# ~[-1, 1]; unscaled heading dominates the 3-channel MSE ~10:1 per unit of
# displacement error. Biases both heads identically but wastes capacity.
HEADING_SCALE = math.pi


def scale_future(fut: torch.Tensor) -> torch.Tensor:
    out = fut.clone()
    out[..., :2] = out[..., :2] / FUTURE_SCALE
    out[..., 2:3] = out[..., 2:3] / HEADING_SCALE
    return out


def unscale_future(fut: torch.Tensor) -> torch.Tensor:
    out = fut.clone()
    out[..., :2] = out[..., :2] * FUTURE_SCALE
    out[..., 2:3] = out[..., 2:3] * HEADING_SCALE
    return out


class F0ShardDataset(IterableDataset):
    """Streams samples from scene_shard_*.pt files under root/task_*/ (or flat)."""

    def __init__(
        self,
        root: str | Path,
        shuffle: bool = True,
        seed: int = 0,
        split: str = "all",
        val_stride: int = 10,
    ):
        super().__init__()
        self.root = Path(root)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.shards: List[Path] = sorted(self.root.glob("task_*/scene_shard_*.pt")) or sorted(
            self.root.glob("scene_shard_*.pt")
        )
        if not self.shards:
            raise FileNotFoundError(f"no scene_shard_*.pt under {self.root}")
        # WHY shard-level split (not sample-level): scenarios within a shard
        # come from the same DB files; splitting by shard keeps near-duplicate
        # frames of one scenario out of both sides. Sorted order makes the
        # split deterministic across runs and machines.
        if split == "train":
            self.shards = [s for i, s in enumerate(self.shards) if i % val_stride != 0]
        elif split == "val":
            self.shards = [s for i, s in enumerate(self.shards) if i % val_stride == 0]
        elif split != "all":
            raise ValueError(f"split must be all|train|val, got {split!r}")

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so the shuffle order differs across epochs."""
        self.epoch = epoch

    def _iter_shards(self) -> List[Path]:
        shards = list(self.shards)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shards)
        # WHY worker sharding by slice: each DataLoader worker gets a disjoint
        # subset of shards, so no sample is seen twice in one epoch.
        info = get_worker_info()
        if info is not None:
            shards = shards[info.id :: info.num_workers]
        return shards

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for shard_path in self._iter_shards():
            data = torch.load(str(shard_path), map_location="cpu", weights_only=False)
            samples = data["samples"] if isinstance(data, dict) else data
            order = list(range(len(samples)))
            if self.shuffle:
                random.Random((self.seed + self.epoch) ^ hash(shard_path.name)).shuffle(order)
            for i in order:
                s = samples[i]
                if "ego_future" not in s:
                    raise KeyError(
                        f"{shard_path.name} sample {i} lacks ego_future — "
                        "this is a v1 (unlabeled) shard; point root at features/f0_v2"
                    )
                out = dict(s)
                out["ego_future"] = scale_future(s["ego_future"].float())
                yield out


def make_loader(
    root: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    split: str = "all",
) -> torch.utils.data.DataLoader:
    ds = F0ShardDataset(root, shuffle=shuffle, seed=seed, split=split)
    # WHY drop_last only when training: val must score every sample, and a
    # partial final batch is harmless there.
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, drop_last=shuffle
    )
