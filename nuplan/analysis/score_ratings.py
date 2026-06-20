"""Score Parv's blind ratings against F4 -> external-validity result.

Run after Parv returns f4_ratings.csv (blind_id,rating):
    ./.venv/bin/python score_ratings.py sheet/f4_ratings.csv

Reports Spearman(rating, F4) with a permutation p-value, per-band mean rating
(should increase zero<low<med<high if F4 is valid), and flags the scenes where
human and F4 most disagree (the diagnostic cases, e.g. does S_inter over-fire
on converging same-direction traffic).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else 0.0


def band(f):
    if f == 0: return "zero"
    if f <= 1/3: return "low"
    if f <= 2/3: return "med"
    return "high"


def main(csv_path):
    key = json.load(open(HERE / "sheet" / "answer_key.json"))
    ratings = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["rating"].strip():
                ratings[row["blind_id"]] = int(row["rating"])
    pairs = [(key[b]["f4"], ratings[b], b) for b in ratings if b in key]
    if len(pairs) < 5:
        print("too few ratings"); return
    f4 = np.array([p[0] for p in pairs]); rt = np.array([p[1] for p in pairs])
    rho = spearman(f4, rt)
    # permutation p (one-sided rho>0)
    rng = np.random.default_rng(0)
    null = [spearman(f4, rng.permutation(rt)) for _ in range(10000)]
    p = float((np.sum(np.array(null) >= rho) + 1) / (len(null) + 1))
    print(f"n={len(pairs)}  Spearman(F4, human) = {rho:+.3f}  perm p(1-sided) = {p:.4f}")
    print("\nper-band mean human rating (expect increasing):")
    for bn in ("zero", "low", "med", "high"):
        vals = [r for f, r, _ in pairs if band(f) == bn]
        if vals:
            print(f"  {bn:5s} n={len(vals):2d}  mean={np.mean(vals):.2f}")
    # largest disagreements (rank-normalized)
    fr = np.argsort(np.argsort(f4)) / (len(f4) - 1)
    hr = np.argsort(np.argsort(rt)) / (len(rt) - 1)
    diff = fr - hr
    order = np.argsort(-np.abs(diff))
    print("\ntop disagreements (F4 rank - human rank):")
    for i in order[:6]:
        _, _, b = pairs[i]
        print(f"  {b}  F4={key[b]['f4']:.2f} human={rt[i]}  "
              f"type={key[b]['scenario_type']}  (rankΔ={diff[i]:+.2f})")
    if rho >= 0.5 and p < 0.05:
        print("\nVERDICT: F4 externally validated (rho>=0.5, significant).")
    elif rho >= 0.3:
        print("\nVERDICT: weak-moderate validation; inspect disagreements above.")
    else:
        print("\nVERDICT: F4 does NOT match human ambiguity; rethink the metric.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "sheet" / "f4_ratings.csv")
