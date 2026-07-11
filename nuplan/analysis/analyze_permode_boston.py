"""Boston de-risk selection analysis (ADR-063 core, no selector).
Collects per-mode closed-loop CLS (WTA modes 0..5) from the Boston per-mode eval, forms the
best-of-modes ORACLE and the random-mode baseline (mean of per-mode means), and compares to the
Boston-trained det and the DEFAULT-WTA (head-argmax / imitation score) on the SAME tokens. Reports
the ADR-058 core gaps -- latent value (oracle-det), default-WTA vs random (mechanism) -- WITHOUT
requiring a learned selector (that is deferred; analyze_permode.py requires sel and would return empty).
Usage: analyze_permode_boston.py [permode_exp] [zoo_exp]
"""
import sys, glob
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric

SC = "/scratch/patodia.pa/av-policy-lab"
PERMODE = f"{SC}/sim_results/eval/{sys.argv[1] if len(sys.argv) > 1 else 'permode_boston_r1'}"
ZOO = f"{SC}/sim_results/eval/{sys.argv[2] if len(sys.argv) > 2 else 'boston_zoo_r1'}"


def collect(pattern):
    d = {}
    for dd in sorted(glob.glob(pattern)):
        a = latest_aggregator(dd)
        if not a:
            continue
        try:
            v = read_cell_metric(a, "score")[0]
        except Exception:
            continue
        d.update({t: float(x) for t, x in v.items()})
    return d


modes = {m: collect(f"{PERMODE}/mode{m}_shard*") for m in range(6)}
present = {m: len(modes[m]) for m in range(6)}
print("per-mode token counts:", present)
common = set(modes[0])
for m in range(1, 6):
    common &= set(modes[m])
common = sorted(common)
print("tokens with all 6 modes:", len(common))
if not common:
    print("no complete-mode tokens yet"); sys.exit(0)

oracle = {t: max(modes[m][t] for m in range(6)) for t in common}
det = collect(f"{ZOO}/det_route_shard*")
wta = collect(f"{ZOO}/wta_route_shard*")   # default-WTA (head argmax / imitation score)
print("det tokens:", len(det), " default-WTA tokens:", len(wta))

# core comparison needs oracle + det + default-WTA on the SAME tokens (NO sel requirement)
toks = [t for t in common if t in det and t in wta]
print("tokens with oracle+det+defaultWTA:", len(toks))
if not toks:
    print("waiting on det/default-WTA overlap"); sys.exit(0)

permode_means = {m: float(np.mean([modes[m][t] for t in toks])) for m in range(6)}
o = np.mean([oracle[t] for t in toks])
dv = np.mean([det[t] for t in toks])
wv = np.mean([wta[t] for t in toks])
rand = float(np.mean(list(permode_means.values())))   # random-mode baseline
print("per-mode means:", {m: round(v, 4) for m, v in permode_means.items()})
print(f"oracle {o:.4f}  det {dv:.4f}  default-WTA {wv:.4f}  random {rand:.4f}  (n={len(toks)})")
print(f"latent value   (oracle-det)        {o - dv:+.4f}")
print(f"default-WTA vs det                 {wv - dv:+.4f}")
print(f"default-WTA vs random (mechanism)  {wv - rand:+.4f}   (ADR-058 mini: -0.025, i.e. below random)")
print(f"frac scenes some mode beats det:   {np.mean([oracle[t] > det[t] for t in toks]):.3f}")
print("--- compare to ADR-058/059 (mini): oracle 0.868, det 0.810, default-WTA 0.705, random 0.730 ---")
