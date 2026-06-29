"""Phase-1 infra: build a token index for the f0_v3 feature cache so we can
measure TRUE coverage of the canonical Val14 / Test14-hard splits without
re-scanning the 43GB cache each time. Scans every shard once, collects the
scenario_token of every cached sample, writes a sorted token list to JSON."""
import glob, json, time
import torch

SC = "/scratch/patodia.pa/av-policy-lab"
shards = sorted(glob.glob(f"{SC}/features/f0_v3/task_*/scene_shard_*.pt"))
print(f"shards: {len(shards)}", flush=True)
toks = set(); t0 = time.time()
for i, sp in enumerate(shards):
    try:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for s in d.get("samples", []):
            t = s.get("scenario_token")
            if t:
                toks.add(t)
        del d
    except Exception as e:
        print("ERR", sp, repr(e), flush=True)
    if i % 25 == 0:
        print(f"{i}/{len(shards)} shards | {len(toks)} tokens | {time.time()-t0:.0f}s", flush=True)
out = f"{SC}/eval_tokens/f0v3_token_index.json"
json.dump({"n": len(toks), "tokens": sorted(toks)}, open(out, "w"))
print("WROTE", out, "n_tokens", len(toks), "elapsed", f"{time.time()-t0:.0f}s", flush=True)
