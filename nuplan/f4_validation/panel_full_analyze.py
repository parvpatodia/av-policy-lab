"""Signal B full analysis (n=80): 4-persona panel vs F4 + components.
Reliability (Cronbach a, ICC(2,k), mean pairwise) + convergence with bootstrap CI.
Saves merged per-persona ratings for provenance.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/panel_full_analyze.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

REPO = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo')
MAN = REPO / 'data/scene_renders/validation_set/manifest_PRIVATE.json'
RDIR = REPO / 'data/panel_ratings_n80'
RNG = np.random.default_rng(0)

cautious = {"val_00":0.55,"val_01":0.2,"val_02":0.15,"val_03":0.2,"val_04":0.6,"val_05":0.55,"val_06":0.25,"val_07":0.15,"val_08":0.3,"val_09":0.45,"val_10":0.35,"val_11":0.2,"val_12":0.6,"val_13":0.65,"val_14":0.4,"val_15":0.55,"val_16":0.3,"val_17":0.4,"val_18":0.45,"val_19":0.2,"val_20":0.8,"val_21":0.45,"val_22":0.35,"val_23":0.4,"val_24":0.2,"val_25":0.5,"val_26":0.5,"val_27":0.75,"val_28":0.65,"val_29":0.5,"val_30":0.3,"val_31":0.45,"val_32":0.3,"val_33":0.5,"val_34":0.7,"val_35":0.6,"val_36":0.6,"val_37":0.65,"val_38":0.45,"val_39":0.55,
 "val_40":0.75,"val_41":0.4,"val_42":0.2,"val_43":0.45,"val_44":0.5,"val_45":0.3,"val_46":0.5,"val_47":0.8,"val_48":0.4,"val_49":0.5,"val_50":0.45,"val_51":0.4,"val_52":0.65,"val_53":0.55,"val_54":0.2,"val_55":0.3,"val_56":0.5,"val_57":0.75,"val_58":0.6,"val_59":0.3,"val_60":0.2,"val_61":0.55,"val_62":0.35,"val_63":0.35,"val_64":0.45,"val_65":0.6,"val_66":0.7,"val_67":0.8,"val_68":0.5,"val_69":0.6,"val_70":0.6,"val_71":0.3,"val_72":0.35,"val_73":0.45,"val_74":0.55,"val_75":0.65,"val_76":0.4,"val_77":0.7,"val_78":0.8,"val_79":0.1}
assertive = {"val_00":0.35,"val_01":0.15,"val_02":0.15,"val_03":0.2,"val_04":0.2,"val_05":0.4,"val_06":0.45,"val_07":0.2,"val_08":0.2,"val_09":0.3,"val_10":0.35,"val_11":0.15,"val_12":0.45,"val_13":0.45,"val_14":0.55,"val_15":0.35,"val_16":0.2,"val_17":0.45,"val_18":0.3,"val_19":0.2,"val_20":0.7,"val_21":0.45,"val_22":0.35,"val_23":0.5,"val_24":0.2,"val_25":0.55,"val_26":0.35,"val_27":0.8,"val_28":0.55,"val_29":0.35,"val_30":0.15,"val_31":0.5,"val_32":0.3,"val_33":0.45,"val_34":0.4,"val_35":0.7,"val_36":0.6,"val_37":0.6,"val_38":0.2,"val_39":0.5,
 "val_40":0.3,"val_41":0.35,"val_42":0.15,"val_43":0.4,"val_44":0.7,"val_45":0.2,"val_46":0.35,"val_47":0.7,"val_48":0.4,"val_49":0.5,"val_50":0.55,"val_51":0.35,"val_52":0.65,"val_53":0.6,"val_54":0.3,"val_55":0.2,"val_56":0.7,"val_57":0.85,"val_58":0.7,"val_59":0.15,"val_60":0.2,"val_61":0.55,"val_62":0.5,"val_63":0.15,"val_64":0.55,"val_65":0.55,"val_66":0.85,"val_67":0.5,"val_68":0.35,"val_69":0.5,"val_70":0.6,"val_71":0.2,"val_72":0.45,"val_73":0.35,"val_74":0.15,"val_75":0.7,"val_76":0.3,"val_77":0.6,"val_78":0.85,"val_79":0.1}
safety = {"val_00":0.4,"val_01":0.2,"val_02":0.2,"val_03":0.1,"val_04":0.2,"val_05":0.45,"val_06":0.55,"val_07":0.2,"val_08":0.3,"val_09":0.5,"val_10":0.3,"val_11":0.3,"val_12":0.55,"val_13":0.7,"val_14":0.65,"val_15":0.4,"val_16":0.3,"val_17":0.4,"val_18":0.35,"val_19":0.3,"val_20":0.85,"val_21":0.45,"val_22":0.35,"val_23":0.4,"val_24":0.2,"val_25":0.5,"val_26":0.4,"val_27":0.75,"val_28":0.55,"val_29":0.4,"val_30":0.1,"val_31":0.45,"val_32":0.35,"val_33":0.5,"val_34":0.7,"val_35":0.65,"val_36":0.45,"val_37":0.55,"val_38":0.15,"val_39":0.5,
 "val_40":0.5,"val_41":0.4,"val_42":0.1,"val_43":0.15,"val_44":0.3,"val_45":0.2,"val_46":0.3,"val_47":0.35,"val_48":0.6,"val_49":0.45,"val_50":0.5,"val_51":0.75,"val_52":0.65,"val_53":0.6,"val_54":0.35,"val_55":0.35,"val_56":0.45,"val_57":0.8,"val_58":0.5,"val_59":0.25,"val_60":0.3,"val_61":0.5,"val_62":0.5,"val_63":0.4,"val_64":0.45,"val_65":0.55,"val_66":0.5,"val_67":0.45,"val_68":0.35,"val_69":0.5,"val_70":0.65,"val_71":0.15,"val_72":0.4,"val_73":0.4,"val_74":0.15,"val_75":0.65,"val_76":0.4,"val_77":0.75,"val_78":0.5,"val_79":0.05}
av_eng = {"val_00":0.7,"val_01":0.1,"val_02":0.1,"val_03":0.4,"val_04":0.7,"val_05":0.6,"val_06":0.3,"val_07":0.1,"val_08":0.4,"val_09":0.6,"val_10":0.2,"val_11":0.1,"val_12":0.7,"val_13":0.5,"val_14":0.4,"val_15":0.7,"val_16":0.4,"val_17":0.3,"val_18":0.2,"val_19":0.5,"val_20":0.8,"val_21":0.3,"val_22":0.2,"val_23":0.4,"val_24":0.1,"val_25":0.6,"val_26":0.7,"val_27":0.8,"val_28":0.7,"val_29":0.7,"val_30":0.2,"val_31":0.5,"val_32":0.3,"val_33":0.6,"val_34":0.7,"val_35":0.8,"val_36":0.6,"val_37":0.5,"val_38":0.4,"val_39":0.6,
 "val_40":0.8,"val_41":0.6,"val_42":0.15,"val_43":0.45,"val_44":0.3,"val_45":0.2,"val_46":0.55,"val_47":0.75,"val_48":0.4,"val_49":0.5,"val_50":0.45,"val_51":0.65,"val_52":0.6,"val_53":0.6,"val_54":0.25,"val_55":0.35,"val_56":0.4,"val_57":0.7,"val_58":0.5,"val_59":0.25,"val_60":0.2,"val_61":0.65,"val_62":0.6,"val_63":0.4,"val_64":0.5,"val_65":0.6,"val_66":0.7,"val_67":0.5,"val_68":0.35,"val_69":0.65,"val_70":0.65,"val_71":0.2,"val_72":0.3,"val_73":0.45,"val_74":0.2,"val_75":0.65,"val_76":0.3,"val_77":0.65,"val_78":0.7,"val_79":0.05}

RATERS = {'cautious': cautious, 'assertive': assertive, 'safety': safety, 'av_eng': av_eng}


def icc_2k(M):
    # ICC(2,k) two-way random, average measures
    n, k = M.shape
    gm = M.mean()
    ms_r = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)


def main():
    RDIR.mkdir(parents=True, exist_ok=True)
    for nm, r in RATERS.items():
        assert len(r) == 80, f'{nm} has {len(r)}'
        (RDIR / f'{nm}.json').write_text(json.dumps(r))
    man = json.loads(MAN.read_text())
    scenes = sorted(man.keys(), key=lambda s: int(s.split('_')[1].split('.')[0]))
    stem = {s: s.replace('.png', '') for s in scenes}
    raters = list(RATERS)
    M = np.array([[RATERS[r][stem[s]] for r in raters] for s in scenes])  # (80,4)
    print(f'panel n={len(scenes)} x {len(raters)} personas')

    # reliability
    var_r = M.var(0, ddof=1).sum(); tot = M.sum(1); alpha = (4/3)*(1 - var_r/tot.var(ddof=1))
    pair_p = [np.corrcoef(M[:,i],M[:,j])[0,1] for i in range(4) for j in range(i+1,4)]
    pair_s = [spearmanr(M[:,i],M[:,j])[0] for i in range(4) for j in range(i+1,4)]
    print(f'  Cronbach a={alpha:.3f}  ICC(2,k)={icc_2k(M):.3f}  '
          f'mean pairwise Pearson={np.mean(pair_p):.3f} Spearman={np.mean(pair_s):.3f}')

    panel = M.mean(1)
    f4 = np.array([man[s]['f4'] for s in scenes])
    si = np.array([man[s]['s_inter'] for s in scenes])
    sb = np.array([man[s].get('s_branch') or 0.0 for s in scenes])

    def boot_rho(x):
        out=[]
        for _ in range(3000):
            idx=RNG.integers(0,len(x),len(x))
            out.append(spearmanr(panel[idx],x[idx])[0])
        return np.percentile(out,[2.5,97.5])
    print('\n  convergence (Spearman + bootstrap 95% CI, n=80):')
    for lab,x in [('F4 combined',f4),('S_inter',si),('S_branch',sb)]:
        rho,p=spearmanr(panel,x); lo,hi=boot_rho(x)
        print(f'    panel vs {lab:12s}: rho={rho:.3f}  CI[{lo:.3f},{hi:.3f}]  p={p:.2e}')

    g0=panel[f4<1e-6]; g1=panel[f4>=1e-6]
    U,p=mannwhitneyu(g1,g0,alternative='greater'); rbc=2*U/(len(g1)*len(g0))-1
    print(f'\n  group contrast: f4=0 (n={len(g0)}) mean {g0.mean():.2f} vs f4>0 (n={len(g1)}) '
          f'mean {g1.mean():.2f}; MW p={p:.2e} rank-biserial={rbc:.3f}')
    print('\n  panel mean by band:')
    for b in ['zero','mid','hi_branch','hi_inter']:
        vs=[panel[i] for i,s in enumerate(scenes) if man[s]['band']==b]
        print(f'    {b:10s} n={len(vs)} mean {np.mean(vs):.2f}')


if __name__ == '__main__':
    main()
