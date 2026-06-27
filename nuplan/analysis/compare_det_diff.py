"""Capstone for #1: are the deterministic and diffusion route policies the SAME
function? Run both on identical scenes (det = single forward; diff = medoid of K
samples, exactly as deployed in serving/policy_planner.py) and measure the
det-vs-diff trajectory ADE + endpoint L2 (meters and as fraction of displacement).
If the gap << lane width, the moderation experiment compared a policy to a near-copy
of itself -> full mechanistic explanation for the diff~=det CLS null."""
import argparse, json
import numpy as np, torch
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import DeterministicHead, DiffusionHead, HeadConfig, CosineSchedule
from models.samplers import ddim_sample
from models.f0_dataset import unscale_future
EK=("ego","agents","agent_mask","map_polylines","map_mask","crosswalks","crosswalk_mask","route_polyline","route_mask","traffic_lights")
def load_scenes(shard,n):
    d=torch.load(shard,map_location="cpu",weights_only=False); s=d["samples"][:n]
    return {k:torch.stack([x[k] for x in s]) for k in EK}
def load_cell(ckpt, head_cls, dev):
    enc=SceneEncoder(SceneEncoderConfig()).to(dev).eval(); head=head_cls(HeadConfig()).to(dev).eval()
    ck=torch.load(ckpt,map_location=dev,weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    ema=ck.get("ema")
    if ema:
        sh=ema["shadow"]
        for mod,pre in ((enc,"encoder."),(head,"head.")):
            sd={k[len(pre):]:v for k,v in sh.items() if k.startswith(pre)}; mod.load_state_dict(sd,strict=False)
    return enc,head
def medoid(samp):  # (B,K,H,3)->(B,H,3)
    xy=samp[...,:2]; d=(xy.unsqueeze(1)-xy.unsqueeze(2)).norm(dim=-1).mean(-1)  # (B,K,K)
    idx=d.sum(-1).argmin(-1)  # (B,)
    return samp[torch.arange(samp.shape[0]),idx]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--det-ckpt",required=True); ap.add_argument("--diff-ckpt",required=True)
    ap.add_argument("--shard",default="/scratch/patodia.pa/av-policy-lab/features/f0/scene_shard_00000.pt")
    ap.add_argument("--n-scenes",type=int,default=2000); ap.add_argument("--K",type=int,default=8)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); dev=torch.device("cpu")
    sched=CosineSchedule().to(dev)
    denc,dhead=load_cell(a.det_ckpt,DeterministicHead,dev)
    fenc,fhead=load_cell(a.diff_ckpt,DiffusionHead,dev)
    batch={k:v.to(dev) for k,v in load_scenes(a.shard,a.n_scenes).items()}
    with torch.no_grad():
        det=unscale_future(dhead(denc(batch),goal=None))                 # (B,H,3)
        gen=torch.Generator(device=dev).manual_seed(0)
        samp=ddim_sample(fhead,sched,fenc(batch),goal=None,num_samples=a.K,num_steps=20,generator=gen)
        diff=unscale_future(medoid(samp))                                # (B,H,3)
    det=det[...,:2].cpu().numpy(); diff=diff[...,:2].cpu().numpy()        # (B,H,2)
    ade=np.linalg.norm(det-diff,axis=-1).mean(-1)                         # (B,) mean over horizon
    endp=np.linalg.norm(det[:,-1]-diff[:,-1],axis=-1)                     # (B,) endpoint
    disp=np.linalg.norm(det[:,-1],axis=-1).mean()                        # mean displacement
    res={"det_ckpt":a.det_ckpt,"diff_ckpt":a.diff_ckpt,"n":int(len(ade)),"K":a.K,
         "det_vs_diff_ADE_m":{"median":float(np.median(ade)),"mean":float(ade.mean()),"p90":float(np.percentile(ade,90)),"max":float(ade.max())},
         "det_vs_diff_endpoint_m":{"median":float(np.median(endp)),"mean":float(endp.mean()),"p90":float(np.percentile(endp,90))},
         "mean_displacement_m":float(disp),"ADE_over_displacement_median":float(np.median(ade)/disp)}
    print(json.dumps(res,indent=2))
    if a.out: json.dump(res,open(a.out,"w"),indent=2)
if __name__=="__main__": main()
