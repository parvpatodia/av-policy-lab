# ADR-093 helper: canonical CLS re-aggregation for the SMART reactive-agents sweep.
# The SMART exp lives under NUPLAN_EXP_ROOT/exp (not sim_results/eval), so canonical_reaggregate.analyze
# cannot read it directly. This reuses the VALIDATED canonical_from_metrics_dir (same formula, xcheck-exact)
# and combines the 8 configs (det, dwta, mode0..5) into oracle(best-of-6)/det/default-WTA/random.
# Supports UNION across multiple exp roots so smart_select_canon (0:24) + smart_select_canon_ext (24:72) -> n=72.
# WHY: keeps the canonical formula in one place; only the exp-root layout differs from the IDM path.
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_reaggregate import canonical_from_metrics_dir

SMART_EXP = "/scratch/patodia.pa/av-policy-lab/smart_exp/exp"

def collect_config(exp_names, config):
    """union per-token canonical CLS for one config across one or more exp roots (disjoint tokens)."""
    d = {}
    for exp in exp_names:
        base = os.path.join(SMART_EXP, exp, config)
        mdirs = sorted(glob.glob(os.path.join(base, "**", "metrics"), recursive=True))
        if mdirs:
            d.update(canonical_from_metrics_dir(mdirs[-1]))
    return d

def analyze(label, exp_names):
    modes = {m: collect_config(exp_names, "mode%d" % m) for m in range(6)}
    det = collect_config(exp_names, "det")
    dwta = collect_config(exp_names, "dwta")
    common = set(modes[0])
    for m in range(1, 6):
        common &= set(modes[m])
    toks = sorted(t for t in common if t in det and t in dwta)
    print("[%s] exp=%s" % (label, exp_names))
    print("per-mode token counts:", {m: len(modes[m]) for m in range(6)})
    print("tokens oracle+det+dwta:", len(toks))
    if not toks:
        print("no overlap"); return
    oracle = {t: max(modes[m][t] for m in range(6)) for t in toks}
    pm = {m: float(np.mean([modes[m][t] for t in toks])) for m in range(6)}
    o = float(np.mean([oracle[t] for t in toks])); dv = float(np.mean([det[t] for t in toks]))
    wv = float(np.mean([dwta[t] for t in toks])); rand = float(np.mean(list(pm.values())))
    print("per-mode means:", {m: round(v, 4) for m, v in pm.items()})
    print("oracle %.4f  det %.4f  default-WTA %.4f  random %.4f  (n=%d)" % (o, dv, wv, rand, len(toks)))
    print("latent value (oracle-det)          %+.4f" % (o - dv))
    print("selection (default-WTA - random)   %+.4f" % (wv - rand))
    print("frac scenes some mode beats det:   %.3f" % np.mean([oracle[t] > det[t] for t in toks]))

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2:])
