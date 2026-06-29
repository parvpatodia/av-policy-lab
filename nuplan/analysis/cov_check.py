import sys, re, json, os
hexre = re.compile(r"[0-9a-f]{16}")
def toks(p):
    try:
        return set(hexre.findall(open(p).read()))
    except Exception:
        return set()
val = toks(sys.argv[1]); hard = toks(sys.argv[2]); mp = sys.argv[3]
try:
    m = json.load(open(mp))
except Exception as e:
    print("manifest_load_error", e); m = None
def cached_from(m):
    if isinstance(m, list):
        out = set()
        for x in m:
            if isinstance(x, str):
                out.add(x)
            elif isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, str) and hexre.fullmatch(v):
                        out.add(v)
        return out
    if isinstance(m, dict):
        if isinstance(m.get("tokens"), list):
            return set(m["tokens"])
        ks = set(k for k in m.keys() if hexre.fullmatch(k))
        if ks:
            return ks
        out = set()
        for v in m.values():
            if isinstance(v, list):
                out |= {x for x in v if isinstance(x, str) and hexre.fullmatch(x)}
        return out
    return set()
cached = {c for c in cached_from(m) if hexre.fullmatch(c)}
print("manifest", os.path.basename(mp), "type", type(m).__name__, "manifest_tokens", len(cached))
print("val14_total", len(val), "covered_by_manifest", len(val & cached), "missing", len(val - cached))
print("test14hard_total", len(hard), "covered_by_manifest", len(hard & cached), "missing", len(hard - cached))
