"""CLS-selection (ADR-018): pick the final checkpoint per cell by closed-loop CLS on a
frozen 50-scenario probe set (DISJOINT from the eval manifest, so selection never sees
the test set). The heavy per-checkpoint eval is run by cls_select_array.sbatch (one task
per cell x candidate-epoch via run_cells.py); THIS module reads the resulting aggregator
parquets and selects argmax CLS per cell.

Usage:
  python cls_select.py --test <exp_dir>          # read one aggregator -> final_score (cheap check)
  python cls_select.py --select <base> --cells ... --epochs ...   # pick best epoch per cell
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path


def read_final_cls(exp_dir: str) -> float | None:
    """Overall closed-loop score for one eval run = the 'final_score' row's `score`."""
    import pandas as pd
    pqs = glob.glob(f"{exp_dir}/**/aggregator_metric/*.parquet", recursive=True)
    if not pqs:
        return None
    df = pd.read_parquet(sorted(pqs)[-1])
    row = df[df["scenario"] == "final_score"]
    if len(row) == 0:
        row = df[df.get("scenario_type") == "final_score"]
    return float(row["score"].iloc[0]) if len(row) else None


def select(base: Path, cells: list[str], epochs: list[int]) -> dict:
    """For each cell, read CLS at each candidate epoch, pick argmax. base/<cell>_e<N>/eval."""
    out = {}
    for cell in cells:
        table = []
        for e in epochs:
            cls = read_final_cls(str(base / f"{cell}_e{e:03d}"))
            if cls is not None:
                table.append((e, cls))
        if not table:
            out[cell] = {"best_epoch": None, "reason": "no_eval_outputs", "table": []}
            continue
        best_e, best_cls = max(table, key=lambda t: t[1])
        out[cell] = {"best_epoch": best_e, "best_cls": best_cls, "table": table}
        print(f"{cell}: best epoch {best_e} (CLS {best_cls:.4f})  from {table}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", help="exp_dir to read a single final_score from")
    ap.add_argument("--select", help="base dir holding <cell>_e<NNN> eval outputs")
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--epochs", nargs="*", type=int, default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.test:
        print("final_score CLS =", read_final_cls(a.test))
        return
    res = select(Path(a.select), a.cells, a.epochs)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
