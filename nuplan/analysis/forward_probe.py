# ADR-084 probe: run our det encoder+head forward on ONE fixed pre-extracted f0 sample (bypasses
# SceneFeatureExtractor) under whatever env invokes this. Compare stdout across nuplan vs nuplan_smart.
# probe output DIFFERS across envs => model forward (torch) is version-sensitive.
# probe output IDENTICAL across envs => model is stable => corruption is in live feature extraction (numpy).
import sys, torch, numpy as np, hashlib
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
from serving.policy_planner import PolicyPlanner, ENCODER_KEYS
from models.f0_dataset import unscale_future

CKPT = "/scratch/patodia.pa/av-policy-lab/runs/boston_derisk/det_route_seed0/best.pt"
FEAT = "/scratch/patodia.pa/av-policy-lab/features/f0_boston/task_0000/scene_shard_00000.pt"

torch.manual_seed(0); np.random.seed(0)
p = PolicyPlanner(ckpt_path=CKPT, head_type="det", goal_mode="route", device="cpu")
p.initialize(None)  # loads encoder+head+ema weights; arg unused for loading

d = torch.load(FEAT, map_location="cpu", weights_only=False)["samples"][0]
batch = {k: d[k].unsqueeze(0) for k in ENCODER_KEYS}

def sig(x):
    if torch.is_tensor(x): return float(x.double().sum().item())
    if isinstance(x, dict): return sum(sig(v) for v in x.values() if torch.is_tensor(v) or isinstance(v,(dict,list,tuple)))
    if isinstance(x, (list, tuple)): return sum(sig(v) for v in x if torch.is_tensor(v) or isinstance(v,(dict,list,tuple)))
    return 0.0

with torch.no_grad():
    mem = p._encoder(batch)
    traj = p._head(mem, goal=None)[0]
pred = unscale_future(traj).cpu().numpy()

print(f"ENV torch={torch.__version__} numpy={np.__version__}")
print(f"encoder_mem_sig= {sig(mem):.6f}")
print(f"head_traj_sig=   {float(traj.double().sum().item()):.6f}")
print(f"pred shape={pred.shape} mean={pred.mean():.6f} std={pred.std():.6f}")
print(f"pred_first3= {np.round(pred[:3],4).tolist()}")
print(f"pred_md5= {hashlib.md5(np.ascontiguousarray(pred.astype(np.float32)).tobytes()).hexdigest()}")
