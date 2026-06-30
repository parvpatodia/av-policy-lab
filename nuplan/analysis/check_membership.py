"""Check how many tokens from a token-file exist in a nuPlan DB dir (via the builder
+ ScenarioFilter scenario_tokens). Args: <db_dir> <token_json>."""
import sys, json, time
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
from features.scene_features import _build_mini_scenarios
DB = sys.argv[1]
tokfile = sys.argv[2]
MAP = "/scratch/patodia.pa/nuplan/maps"
toks = json.load(open(tokfile))["tokens"]
t0 = time.time()
scs = _build_mini_scenarios(DB, MAP, 10_000_000, scenario_tokens=toks)
found = {s.token for s in scs}
print(f"db={DB.rstrip('/').split('/')[-1]} requested={len(toks)} found={len(found)} elapsed_s={round(time.time()-t0)}")
