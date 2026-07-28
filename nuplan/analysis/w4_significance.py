# ADR-094 (Wave 4) significance: per-scenario bootstrap CIs + paired IDM-vs-SMART on the mode-selection
# bottleneck. Reuses the VALIDATED canonical_from_metrics_dir. latent_t = oracle_t(best of 6 modes) - det_t;
# selection_t = defaultWTA_t - random_t (random_t = per-token mean over the 6 modes). Bootstrap over tokens.
# WHY: turns the point estimates (ADR-092/093) into CIs and tests SMART>IDM on MATCHED scenarios.
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_reaggregate import canonical_from_metrics_dir

SIM = "/scratch/patodia.pa/av-policy-lab/sim_results/eval"
SMART = "/scratch/patodia.pa/av-policy-lab/smart_exp/exp"
RNG = np.random.default_rng(0)

def _collect(pattern):
    d = {}
    for cell in sorted(glob.glob(pattern)):
        mdirs = glob.glob(os.path.join(cell, "**", "metrics"), recursive=True)
        if mdirs:
            d.update(canonical_from_metrics_dir(sorted(mdirs)[-1]))
    return d

def idm_seed(seed):
    if seed == 0:
        P, Z = SIM + "/permode_boston_matched", SIM + "/boston_zoo_matched_r1"
    else:
        P = SIM + "/permode_boston_matched_seed%d" % seed
        Z = SIM + "/boston_zoo_matched_seed%d_r1" % seed
    modes = {m: _collect("%s/mode%d_shard*" % (P, m)) for m in range(6)}
    return modes, _collect(Z + "/det_route_shard*"), _collect(Z + "/wta_route_shard*")

def smart_all():
    def cfg(name):
        d = {}
        for root in ("smart_select_canon", "smart_select_canon_ext"):
            md = glob.glob("%s/%s/%s/**/metrics" % (SMART, root, name), recursive=True)
            if md:
                d.update(canonical_from_metrics_dir(sorted(md)[-1]))
        return d
    return {m: cfg("mode%d" % m) for m in range(6)}, cfg("det"), cfg("dwta")

def per_token(modes, det, dwta):
    toks = set(modes[0])
    for m in range(1, 6):
        toks &= set(modes[m])
    toks = sorted(t for t in toks if t in det and t in dwta)
    lat = {t: max(modes[m][t] for m in range(6)) - det[t] for t in toks}
    sel = {t: dwta[t] - float(np.mean([modes[m][t] for m in range(6)])) for t in toks}
    return toks, lat, sel

def boot_ci(vals, nb=10000):
    a = np.asarray(vals, float)
    bs = RNG.choice(a, size=(nb, len(a)), replace=True).mean(axis=1)
    return a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def boot_ci_paired(d, nb=10000):
    a = np.asarray(d, float)
    bs = RNG.choice(a, size=(nb, len(a)), replace=True).mean(axis=1)
    return a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(np.mean(a > 0))

print("=" * 78)
print("WAVE 4 SIGNIFICANCE  (bootstrap 10k, 95% CI over scenarios)")
print("=" * 78)

idm = {}
for s in (0, 1, 2):
    modes, det, dwta = idm_seed(s)
    toks, lat, sel = per_token(modes, det, dwta)
    idm[s] = (toks, lat, sel)
    lm, ll, lh = boot_ci([lat[t] for t in toks])
    sm, sl, sh = boot_ci([sel[t] for t in toks])
    print("IDM seed%d  n=%d  latent %.4f [%.4f, %.4f]   selection %.4f [%.4f, %.4f]"
          % (s, len(toks), lm, ll, lh, sm, sl, sh))

# pooled IDM across seeds (stack per-token values from all 3 seeds)
pool_lat = [v for s in (0, 1, 2) for v in idm[s][1].values()]
pool_sel = [v for s in (0, 1, 2) for v in idm[s][2].values()]
lm, ll, lh = boot_ci(pool_lat); sm, sl, sh = boot_ci(pool_sel)
print("IDM POOLED n=%d  latent %.4f [%.4f, %.4f]   selection %.4f [%.4f, %.4f]"
      % (len(pool_lat), lm, ll, lh, sm, sl, sh))

sm_modes, sm_det, sm_dwta = smart_all()
st_toks, st_lat, st_sel = per_token(sm_modes, sm_det, sm_dwta)
lm, ll, lh = boot_ci([st_lat[t] for t in st_toks])
sm2, sl, sh = boot_ci([st_sel[t] for t in st_toks])
print("SMART seed0 n=%d  latent %.4f [%.4f, %.4f]   selection %.4f [%.4f, %.4f]"
      % (len(st_toks), lm, ll, lh, sm2, sl, sh))

# paired SMART vs IDM(seed0) on shared tokens
shared = sorted(set(st_toks) & set(idm[0][0]))
d_lat = [st_lat[t] - idm[0][1][t] for t in shared]
d_sel = [st_sel[t] - idm[0][2][t] for t in shared]
m, lo, hi, frac = boot_ci_paired(d_lat)
print("-" * 78)
print("PAIRED SMART-IDM(seed0) latent  n=%d  delta %.4f [%.4f, %.4f]  frac(SMART>IDM)=%.3f"
      % (len(shared), m, lo, hi, frac))
m, lo, hi, frac = boot_ci_paired(d_sel)
print("PAIRED SMART-IDM(seed0) selection n=%d delta %.4f [%.4f, %.4f]  frac>0=%.3f"
      % (len(shared), m, lo, hi, frac))
