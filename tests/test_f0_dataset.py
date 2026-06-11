"""Tests for the F0 v2 shard dataset. Builds synthetic shards in tmp dirs."""
import torch
import pytest

from models.f0_dataset import (
    F0ShardDataset,
    FUTURE_SCALE,
    make_loader,
    scale_future,
    unscale_future,
)


def _sample(v: float):
    return {
        "ego": torch.full((20, 8), v),
        "agents": torch.full((32, 20, 9), v),
        "agent_mask": torch.ones(32, 20, dtype=torch.bool),
        "map_polylines": torch.full((128, 20, 7), v),
        "map_mask": torch.ones(128, dtype=torch.bool),
        "crosswalks": torch.full((16, 20, 2), v),
        "crosswalk_mask": torch.ones(16, dtype=torch.bool),
        "route_polyline": torch.full((40, 4), v),
        "route_mask": torch.ones(40, dtype=torch.bool),
        "traffic_lights": torch.zeros(128, dtype=torch.int64),
        "ego_future": torch.full((16, 3), v),
    }


@pytest.fixture()
def shard_root(tmp_path):
    """3 task dirs x 2 shards x 5 samples, each sample tagged by a unique value."""
    v = 0.0
    for t in range(3):
        d = tmp_path / f"task_{t:04d}"
        d.mkdir()
        for s in range(2):
            samples = []
            for _ in range(5):
                samples.append(_sample(v))
                v += 1.0
            torch.save({"samples": samples, "config": {}}, d / f"scene_shard_{s:05d}.pt")
    return tmp_path


def test_iterates_all_samples_once(shard_root):
    ds = F0ShardDataset(shard_root, shuffle=False)
    ids = [s["ego"][0, 0].item() for s in ds]
    assert len(ids) == 30 and len(set(ids)) == 30


def test_future_scaling_roundtrip(shard_root):
    ds = F0ShardDataset(shard_root, shuffle=False)
    s = next(iter(ds))
    raw = s["ego"][0, 0].item()  # inputs untouched -> raw tag value
    assert s["ego_future"][0, 0].item() == pytest.approx(raw / FUTURE_SCALE)
    assert s["ego_future"][0, 2].item() == pytest.approx(raw)  # heading unscaled
    x = torch.randn(16, 3)
    assert torch.allclose(unscale_future(scale_future(x)), x, atol=1e-6)


def test_shuffle_changes_order_but_not_content(shard_root):
    a = [s["ego"][0, 0].item() for s in F0ShardDataset(shard_root, shuffle=True, seed=1)]
    b = [s["ego"][0, 0].item() for s in F0ShardDataset(shard_root, shuffle=False)]
    assert sorted(a) == sorted(b) and a != b


def test_epoch_reshuffles(shard_root):
    ds = F0ShardDataset(shard_root, shuffle=True, seed=2)
    e0 = [s["ego"][0, 0].item() for s in ds]
    ds.set_epoch(1)
    e1 = [s["ego"][0, 0].item() for s in ds]
    assert sorted(e0) == sorted(e1) and e0 != e1


def test_loader_batches(shard_root):
    loader = make_loader(shard_root, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    assert batch["ego"].shape == (8, 20, 8)
    assert batch["ego_future"].shape == (8, 16, 3)
    assert batch["agent_mask"].dtype == torch.bool


def test_rejects_unlabeled_v1_shards(tmp_path):
    s = _sample(0.0)
    s.pop("ego_future")
    d = tmp_path / "task_0000"
    d.mkdir()
    torch.save({"samples": [s], "config": {}}, d / "scene_shard_00000.pt")
    with pytest.raises(KeyError, match="f0_v2"):
        list(F0ShardDataset(tmp_path, shuffle=False))
