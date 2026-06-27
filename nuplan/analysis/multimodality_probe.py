"""#1b open-loop multimodality probe. Does the trained diffusion HEAD produce a
multimodal trajectory distribution per scene, or do samples collapse? If they
collapse, diff~=det is a head/training matter; if multimodal, the closed-loop
MEDOID selector (serving/policy_planner.py) is what kills it at deployment.

Per scene: sample K trajectories from the diff head (EMA), measure endpoint
dispersion (mean pairwise L2, meters) and MODE COUNT (union-find clusters of the
K endpoints at eps~lane-width). Reports the distribution over scenes. Route head
only (no goal needed offline)."""
import argparse, json
import numpy as np, torch
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import DiffusionHead, HeadConfig, CosineSchedule
from models.samplers import ddim_sample
from models.f0_dataset import unscale_future

ENCODER_KEYS = ("ego","agents","agent_mask","map_polylines","map_mask",
                "crosswalks","crosswalk_mask","route_polyline","route_mask","traffic_lights")

def load_scenes(shard, n):
    d = torch.load(shard, map_location="cpu", weights_only=False)
    samples = d["samples"][:n]
    return {k: torch.stack([s[k] for s in samples]) for k in ENCODER_KEYS}  # keep dtypes

def n_modes(ep, eps):  # ep (K,2) numpy -> cluster count via union-find
    K=len(ep); par=list(range(K))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    D=np.linalg.norm(ep[:,None,:]-ep[None,:,:],axis=-1)
    for i in range(K):
        for j in range(i+1,K):
            if D[i,j]<eps: par[f(i)]=f(j)
    return len({f(i) for i in range(K)})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", default="/scratch/patodia.pa/av-policy-lab/features/f0/scene_shard_00000.pt")
    ap.add_argument("--n-scenes", type=int, default=256)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--eps", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ema", action="store_true")
    a=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc=SceneEncoder(SceneEncoderConfig()).to(dev).eval()
    head=DiffusionHead(HeadConfig()).to(dev).eval()
    sched=CosineSchedule().to(dev)
    ck=torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    used_ema=False; ema=ck.get("ema")
    if ema and not a.no_ema:
        sh=ema["shadow"]
        for mod,pre in ((enc,"encoder."),(head,"head.")):
            sd={k[len(pre):]:v for k,v in sh.items() if k.startswith(pre)}
            mod.load_state_dict(sd, strict=False); used_ema=True
    batch={k:v.to(dev) for k,v in load_scenes(a.shard, a.n_scenes).items()}
    with torch.no_grad():
        mem=enc(batch)
        gen=torch.Generator(device=dev).manual_seed(0)
        samp=ddim_sample(head, sched, mem, goal=None, num_samples=a.K, num_steps=20, generator=gen)
        samp=unscale_future(samp)              # (B,K,H,3) meters
    end=samp[...,-1,:2].float().cpu().numpy()  # (B,K,2) endpoints
    disp=[]; modes=[]; med_off=[]
    for b in range(end.shape[0]):
        ep=end[b]; D=np.linalg.norm(ep[:,None,:]-ep[None,:,:],axis=-1)
        disp.append(D[np.triu_indices(a.K,1)].mean())
        modes.append(n_modes(ep,a.eps))
        cen=ep.mean(0); med=ep[D.sum(1).argmin()]; med_off.append(np.linalg.norm(med-cen))
    meandisp=float(np.linalg.norm(end,axis=-1).mean())  # mean endpoint displacement (ego frame), context for dispersion
    disp=np.array(disp); modes=np.array(modes)
    res={"ckpt":a.ckpt,"device":str(dev),"used_ema":used_ema,"n_scenes":int(end.shape[0]),"K":a.K,"eps_m":a.eps,
         "endpoint_dispersion_m":{"p10":float(np.percentile(disp,10)),"median":float(np.median(disp)),
             "mean":float(disp.mean()),"p90":float(np.percentile(disp,90)),"max":float(disp.max())},
         "modes":{"frac_ge2":float((modes>=2).mean()),"frac_ge3":float((modes>=3).mean()),
             "mean":float(modes.mean()),"max":int(modes.max())},
         "median_medoid_offset_from_centroid_m":float(np.median(med_off)),
         "mean_endpoint_displacement_m":meandisp,
         "dispersion_over_displacement_median":(float(np.median(disp)/meandisp) if meandisp>0 else None)}
    print(json.dumps(res,indent=2))
    if a.out: json.dump(res,open(a.out,"w"),indent=2)

if __name__=="__main__": main()
