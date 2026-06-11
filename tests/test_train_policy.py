"""Tests for the F5 training loop: smoke runs, checkpointing, bit-exact resume."""
import json

import torch
import pytest

from training.train_policy import parse_args, run_name, train

B_KEYS = {
    "ego": (20, 8), "agents": (32, 20, 9), "map_polylines": (128, 20, 7),
    "crosswalks": (16, 20, 2), "route_polyline": (40, 4),
}


def _sample(g):
    s = {k: torch.randn(*shape, generator=g) for k, shape in B_KEYS.items()}
    s["agent_mask"] = torch.ones(32, 20, dtype=torch.bool)
    s["map_mask"] = torch.ones(128, dtype=torch.bool)
    s["crosswalk_mask"] = torch.ones(16, dtype=torch.bool)
    s["route_mask"] = torch.ones(40, dtype=torch.bool)
    s["traffic_lights"] = torch.zeros(128, dtype=torch.int64)
    # smooth-ish future so the task is learnable in a few steps
    base = torch.linspace(0, 5, 16).unsqueeze(1) * torch.tensor([[1.0, 0.2]])
    s["ego_future"] = torch.cat(
        [base + 0.1 * torch.randn(16, 2, generator=g), torch.zeros(16, 1)], dim=1)
    # v2 shards carry string identifiers; the loop must not choke on them
    s["scenario_token"] = "tok_abc"
    s["scenario_type"] = "test_type"
    return s


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """4 shards x 8 samples; val_stride=2 puts 2 shards in each split."""
    root = tmp_path_factory.mktemp("f0_v2")
    g = torch.Generator().manual_seed(0)
    d = root / "task_0000"
    d.mkdir()
    for s_idx in range(4):
        samples = [_sample(g) for _ in range(8)]
        torch.save({"samples": samples, "config": {}}, d / f"scene_shard_{s_idx:05d}.pt")
    return root


def _args(data_root, ckpt_dir, head, goal, epochs):
    return parse_args([
        "--head", head, "--goal", goal,
        "--data-root", str(data_root), "--ckpt-dir", str(ckpt_dir),
        "--epochs", str(epochs), "--batch-size", "8",
        "--lr", "1e-3", "--warmup-steps", "1",
        "--val-stride", "2", "--patience", "999",
        "--val-k", "2", "--ddim-steps", "4", "--device", "cpu",
    ])


@pytest.mark.parametrize("head,goal", [
    ("det", "route"), ("det", "precise"), ("diff", "route"), ("diff", "precise"),
])
def test_all_four_cells_smoke(data_root, tmp_path, head, goal):
    args = _args(data_root, tmp_path, head, goal, epochs=2)
    res = train(args)
    out = tmp_path / run_name(args)
    assert (out / "latest.pt").exists() and (out / "best.pt").exists()
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(torch.isfinite(torch.tensor(r["train_loss"])) for r in rows)
    assert rows[0]["n_val"] == 16  # 2 val shards x 8 samples, none dropped
    assert res["best_minADE"] < float("inf")


def test_det_loss_decreases(data_root, tmp_path):
    args = _args(data_root, tmp_path, "det", "precise", epochs=6)
    train(args)
    rows = [json.loads(l) for l in
            (tmp_path / run_name(args) / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"] * 0.7


def test_resume_is_bit_exact(data_root, tmp_path):
    """Train 4 epochs straight vs 2 + resume to 4: identical final weights.
    This is the property that makes the 8 h GPU window a non-event."""
    a_dir, b_dir = tmp_path / "straight", tmp_path / "resumed"
    train(_args(data_root, a_dir, "diff", "precise", epochs=4))
    train(_args(data_root, b_dir, "diff", "precise", epochs=2))
    train(_args(data_root, b_dir, "diff", "precise", epochs=4))  # resumes from latest.pt
    name = "diff_precise_seed0"
    ck_a = torch.load(a_dir / name / "latest.pt", weights_only=False)
    ck_b = torch.load(b_dir / name / "latest.pt", weights_only=False)
    assert ck_a["epoch"] == ck_b["epoch"] == 3
    for part in ("encoder", "head"):
        for k in ck_a[part]:
            assert torch.equal(ck_a[part][k], ck_b[part][k]), f"{part}.{k} diverged"


def test_resume_skips_finished_run(data_root, tmp_path, capsys):
    args = _args(data_root, tmp_path, "det", "route", epochs=2)
    train(args)
    res = train(args)  # epochs already done -> zero additional epochs
    assert res["epochs_run"] == 0
