"""Selection-bottleneck analysis. Collects per-mode closed-loop CLS (WTA modes 0..5),
forms the best-of-modes oracle per token, and compares to det / selector / WTA on the
SAME tokens. Reports: latent value (oracle-det), realized (sel-det), and the unrealized
selection gap (oracle-sel). Usage: analyze_permode.py [permode_exp] [zoo_exp]"""
import sys, glob
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric

SC = "/scratch/patodia.pa/av-policy-lab"
PERMODE = f"{SC}/sim_results/eval/{sys.argv[1] if len(sys.argv)>1 else 'permode_val14_r1'}"
ZOO = f"{SC}/sim_results/eval/{sys.argv[2] if len(sys.argv)>2 else 'val14_zoo_r1'}"

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
common = set(modes[0])
for m in range(1, 6):
    common &= set(modes[m])
common = sorted(common)
print("tokens with all 6 modes:", len(common))
if not common:
    print("no complete-mode tokens yet"); sys.exit(0)

oracle = {t: max(modes[m][t] for m in range(6)) for t in common}
det = collect(f"{ZOO}/det_route_shard*"); sel = collect(f"{ZOO}/sel_route_shard*"); wta = collect(f"{ZOO}/wta_route_shard*")
# restrict to tokens present in oracle AND det AND sel
toks = [t for t in common if t in det and t in sel]
print("tokens with oracle+det+sel:", len(toks))
if not toks:
    print("waiting on det/sel overlap"); sys.exit(0)
o = np.mean([oracle[t] for t in toks]); dv = np.mean([det[t] for t in toks])
sv = np.mean([sel[t] for t in toks]); wv = np.mean([wta[t] for t in toks if t in wta])
print("per-mode means:", {m: round(float(np.mean([modes[m][t] for t in toks])), 4) for m in range(6)})
print(f"oracle {o:.4f}  det {dv:.4f}  sel {sv:.4f}  wta {wv:.4f}  (n={len(toks)})")
print(f"latent value  (oracle-det) {o-dv:+.4f}")
print(f"realized      (sel-det)    {sv-dv:+.4f}")
print(f"UNREALIZED selection gap (oracle-sel) {o-sv:+.4f}")
print(f"frac scenes some mode beats det: {np.mean([oracle[t] > det[t] for t in toks]):.3f}")
