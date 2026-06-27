"""Unsaturated-subset robustness (closes ADR-033's open question): is the H1 null a
CLS-CEILING artifact, or robust to removing the ceiling?

Re-run the headline route-minus-precise F4-slope contrast on HARD / unsaturated
subsets (low grand-mean CLS = real headroom) with the pre-registered inference
(scenario_type fixed effects + wild-cluster bootstrap). Also the direct H1 prediction
on the top-F4 quartile: is mean Delta_route > mean Delta_precise where the hypothesis
says multimodality should pay off?

WHY slicing by mean-CLS is defensible: the slice variable is the grand level
M=(det_r+diff_r+det_p+diff_p)/4; the estimand is the contrast
C=(diff_r-det_r)-(diff_p-det_p). M (a sum) and C (a difference-of-differences) are
~orthogonal, so conditioning on M does not select on C's sign. Headroom per slice is
reported so the reader can see the ceiling really is removed. (A pure outcome-on-itself
slice would be biased; this is not that.)
"""
import argparse, json
import numpy as np
from analyze_moderation_v2 import read_cell_metric, latest_aggregator, wild_cluster_test


def load4(root, col="score"):
    cells = {}
    for c in ["det_route", "diff_route", "det_precise", "diff_precise"]:
        vals, types = read_cell_metric(latest_aggregator(f"{root}/{c}"), col)
        cells[c] = (vals, types)
    return cells


def build(root, f4):
    cells = load4(root)
    toks = set(f4)
    for c in cells:
        toks &= set(cells[c][0])
    toks = sorted(toks)
    dr = np.array([cells["diff_route"][0][t] - cells["det_route"][0][t] for t in toks])
    dp = np.array([cells["diff_precise"][0][t] - cells["det_precise"][0][t] for t in toks])
    C = dr - dp
    M = np.array([(cells["det_route"][0][t] + cells["diff_route"][0][t]
                   + cells["det_precise"][0][t] + cells["diff_precise"][0][t]) / 4 for t in toks])
    X = np.array([f4[t] for t in toks])
    g = np.array([cells["det_route"][1].get(t, "?") for t in toks])
    return toks, dr, dp, C, M, X, g


def slice_report(name, mask, C, M, X, g, B):
    n = int(mask.sum())
    if n < 30:
        return {"slice": name, "n": n, "note": "insufficient"}
    ceil = float(np.mean(M[mask] >= 0.99))
    wc = wild_cluster_test(X[mask], C[mask], g[mask], B=B)
    return {"slice": name, "n": n, "frac_M_at_ceiling": round(ceil, 3),
            "mean_abs_contrast": round(float(np.mean(np.abs(C[mask]))), 4),
            "beta1": round(wc["beta1"], 4), "cluster_se": round(wc["cluster_se"], 4),
            "t": round(wc["t"], 2), "p_wcr_twosided": round(wc["p_wcr_twosided"], 3),
            "n_clusters": wc["n_clusters"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f4-scores", required=True)
    ap.add_argument("--root", required=True)   # merged_rX_full
    ap.add_argument("--tag", default="r?")
    ap.add_argument("--bootstrap", type=int, default=4999)
    ap.add_argument("--out")
    a = ap.parse_args()
    f4_all = json.load(open(a.f4_scores))
    f4 = {t: r["f4"] for t, r in f4_all.items() if r.get("f4") is not None}
    toks, dr, dp, C, M, X, g = build(a.root, f4)

    out = {"tag": a.tag, "n_total": len(toks), "slices": []}
    q50, q25 = np.quantile(M, 0.50), np.quantile(M, 0.25)
    f4_q75 = np.quantile(X, 0.75)
    for name, mask in [
        ("ALL", np.ones(len(toks), bool)),
        ("unsat_M<0.99", M < 0.99),
        ("hard_bottom50pct_M", M <= q50),
        ("hard_bottom25pct_M", M <= q25),
        ("highF4_top25pct", X >= f4_q75),
    ]:
        rep = slice_report(name, mask, C, M, X, g, a.bootstrap)
        out["slices"].append(rep)
        print(f"[{a.tag}] {rep.get('slice'):20s} n={rep.get('n'):4} "
              f"ceil={rep.get('frac_M_at_ceiling','-')} "
              f"beta1={rep.get('beta1','-')} p_wcr={rep.get('p_wcr_twosided','-')} "
              f"|mean|C|={rep.get('mean_abs_contrast','-')}")

    # direct H1 prediction at top-F4 quartile: paired mean Delta_route vs Delta_precise
    hi = X >= f4_q75
    rng = np.random.default_rng(0)
    diff_pred = dr[hi] - dp[hi]            # = C at high F4; H1 predicts mean > 0
    b = diff_pred[rng.integers(0, hi.sum(), size=(a.bootstrap, hi.sum()))].mean(1)
    ci = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    out["highF4_paired"] = {
        "n": int(hi.sum()),
        "mean_delta_route": round(float(dr[hi].mean()), 4),
        "mean_delta_precise": round(float(dp[hi].mean()), 4),
        "mean_route_minus_precise": round(float(diff_pred.mean()), 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "p_onesided_gt0": round(float((b <= 0).mean()), 3),
    }
    print(f"[{a.tag}] HIGH-F4 paired: mean(Delta_route)-mean(Delta_precise)="
          f"{out['highF4_paired']['mean_route_minus_precise']:+.4f} "
          f"CI95 {out['highF4_paired']['ci95']} (H1 predicts >0)")
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(out, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
