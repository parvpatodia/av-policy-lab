import re, json, sys
v = set(re.findall(r"[0-9a-f]{16}", open(sys.argv[1]).read()))
h = set(re.findall(r"[0-9a-f]{16}", open(sys.argv[2]).read()))
outdir = sys.argv[3]
json.dump({"tokens": sorted(v)}, open(f"{outdir}/val14.json", "w"))
json.dump({"tokens": sorted(h)}, open(f"{outdir}/test14hard.json", "w"))
json.dump({"tokens": sorted(v | h)}, open(f"{outdir}/canonical_val14_test14hard.json", "w"))
print("val14", len(v), "test14hard", len(h), "union", len(v | h))
