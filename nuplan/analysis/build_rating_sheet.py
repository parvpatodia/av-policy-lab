"""Build a BLIND F4 ambiguity-rating sheet from local f0_v2 shards.

External validation of F4: Parv rates each scene 1-5 on "how many genuinely
different reasonable things could the car do here", WITHOUT seeing F4. We then
correlate his ratings with F4 (Spearman). Independent judgment is the point,
so F4 is hidden and scenes are shown in randomized order.

Stimulus (honest, minimal): top-down ego-frame view of lanes, crosswalks,
other agents (by type, with heading), the ego (at origin, pointing up), and a
COARSE intent arrow (quantized left/straight/right from the route endpoint) so
"reasonable options" is well-defined without revealing the exact maneuver.

Outputs (in ./sheet/):
  img/<blind_id>.png        one render per scenario
  rating_sheet.html         the thing Parv opens; rate + export CSV
  answer_key.json           blind_id -> token -> F4 (NOT for Parv until after)
"""
from __future__ import annotations

import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/Users/parvpatodia/av_assets")
SHARDS = sorted(glob.glob(str(ROOT / "f0_v2" / "task_*" / "scene_shard_*.pt")))
F4 = json.load(open(ROOT / "f4_rating" / "f4_scores_v11.json"))
OUT = ROOT / "f4_rating" / "sheet"
POS = 120.0
VEL = 15.0
N_PER_BAND = 13          # ~52 total, balanced across 4 bands
VIEW_M = 50.0            # half-window in meters
SEED = 20260614


def select_tokens() -> list:
    scored = {t: r for t, r in F4.items()
              if r.get("f4") is not None and not r.get("excluded")}
    bands = {
        "zero": [t for t, r in scored.items() if r["f4"] == 0.0],
        "low":  [t for t, r in scored.items() if 0.0 < r["f4"] <= 1 / 3],
        "med":  [t for t, r in scored.items() if 1 / 3 < r["f4"] <= 2 / 3],
        "high": [t for t, r in scored.items() if r["f4"] > 2 / 3],
    }
    rng = random.Random(SEED)
    picked = []
    for name, pool in bands.items():
        pool = sorted(pool)
        rng.shuffle(pool)
        picked.extend(pool[:N_PER_BAND])
    rng.shuffle(picked)                      # randomize presentation order
    return picked


def coarse_intent(route_xy: np.ndarray) -> str:
    """left/straight/right from the route endpoint angle (ego frame, +x fwd)."""
    if len(route_xy) < 2:
        return "unknown"
    end = route_xy[-1]
    ang = math.degrees(math.atan2(end[1], end[0]))   # +y left, +x forward
    if ang > 20:
        return "left"
    if ang < -20:
        return "right"
    return "straight"


def render(sample: dict, blind_id: str, out_png: Path) -> str:
    g = lambda k: np.asarray(sample[k], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)

    # lanes
    mp = g("map_polylines"); mm = np.asarray(sample["map_mask"], dtype=bool)
    for i in np.flatnonzero(mm):
        pts = mp[i, :, :2] * POS
        ax.plot(pts[:, 0], pts[:, 1], color="0.78", lw=1.0, zorder=1)
    # crosswalks
    cw = g("crosswalks"); cwm = np.asarray(sample["crosswalk_mask"], dtype=bool)
    for i in np.flatnonzero(cwm):
        pts = cw[i] * POS
        ax.plot(pts[:, 0], pts[:, 1], color="#c8a23a", lw=1.2, ls="--", zorder=2)
    # agents (last valid step), color by type onehot dims 6:9 [veh,ped,cyc]
    ag = g("agents"); am = np.asarray(sample["agent_mask"], dtype=bool)
    for j in range(ag.shape[0]):
        valid = np.flatnonzero(am[j])
        if len(valid) == 0:
            continue
        k = valid[-1]
        x, y = ag[j, k, 0] * POS, ag[j, k, 1] * POS
        if abs(x) > VIEW_M or abs(y) > VIEW_M:
            continue
        onehot = ag[j, k, 6:9]
        kind = int(np.argmax(onehot)) if onehot.any() else 0
        color = {0: "#c0392b", 1: "#8e44ad", 2: "#16a085"}[kind]  # veh/ped/cyc
        ax.scatter([x], [y], s=46, c=color, zorder=4, edgecolors="white", linewidths=0.6)
        # velocity arrow (1.5 s lookahead, meters): length encodes speed and
        # direction, so parallel traffic vs crossing traffic is visually obvious
        vx, vy = ag[j, k, 4] * VEL, ag[j, k, 5] * VEL
        spd = math.hypot(vx, vy)
        if spd > 0.5:
            L = min(spd * 3.0, 28.0)        # 3 s lookahead so crossings show
            ux, uy = vx / spd, vy / spd
            ax.annotate("", xy=(x + L * ux, y + L * uy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
                        zorder=4)
    # ego nominal path: the lane/route corridor ahead (model-free input, NOT
    # the expert future), drawn faint so crossing conflicts are perceptible.
    rt = g("route_polyline")[:, :2] * POS
    rmask = np.asarray(sample["route_mask"], dtype=bool)
    rpath = rt[rmask]
    intent = coarse_intent(rpath) if rmask.any() else "unknown"
    if len(rpath) >= 2:
        ax.plot(rpath[:, 0], rpath[:, 1], color="#1e8449", lw=2.6,
                alpha=0.55, zorder=3)
        ax.annotate("", xy=rpath[min(len(rpath) - 1, 8)], xytext=rpath[0],
                    arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=2.0),
                    zorder=3)
    # ego at origin, pointing +x (drawn pointing up via axis orientation)
    ax.add_patch(Rectangle((-1.0, -2.4), 2.0, 4.8, color="#2c3e90",
                           zorder=5, alpha=0.9))
    ax.scatter([0], [0], s=10, c="white", zorder=6)

    ax.set_xlim(-VIEW_M, VIEW_M); ax.set_ylim(-VIEW_M, VIEW_M)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{blind_id}   intent: {intent}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return intent


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    targets = {t: f"S{idx:02d}" for idx, t in enumerate(select_tokens())}
    remaining = dict(targets)
    key = {}
    print(f"selected {len(targets)} scenarios across 4 F4 bands; scanning shards...")
    for n, sp in enumerate(SHARDS):
        if not remaining:
            break
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for s in data["samples"]:
            tok = s.get("scenario_token")
            if tok in remaining:
                bid = remaining.pop(tok)
                intent = render(s, bid, OUT / "img" / f"{bid}.png")
                key[bid] = {"token": tok, "f4": F4[tok]["f4"],
                            "scenario_type": F4[tok]["scenario_type"],
                            "intent": intent}
        if n % 50 == 0:
            print(f"  shard {n}/{len(SHARDS)}, found {len(targets) - len(remaining)}/{len(targets)}")
    json.dump(key, open(OUT / "answer_key.json", "w"), indent=2)
    write_html(sorted(key), OUT / "rating_sheet.html")
    if remaining:
        print(f"WARNING: {len(remaining)} tokens not found in local shards")
    print(f"done: {len(key)} rendered -> {OUT}/rating_sheet.html")


def write_html(blind_ids: list, path: Path):
    cards = "\n".join(f'''
    <div class="card">
      <img src="img/{b}.png" loading="lazy">
      <div class="rate" data-id="{b}">
        {"".join(f'<label><input type="radio" name="{b}" value="{v}">{v}</label>' for v in range(1,6))}
      </div>
    </div>''' for b in blind_ids)
    html = f'''<!doctype html><meta charset="utf-8">
<title>F4 blind ambiguity rating</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1a1a1a}}
 h1{{font-size:20px}} .q{{background:#f3f3ef;padding:14px;border-radius:8px;line-height:1.6}}
 .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:18px}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:8px}}
 .card img{{width:100%;border-radius:4px}}
 .rate{{display:flex;gap:12px;justify-content:center;padding:6px}}
 .rate label{{font-size:15px}} button{{font-size:15px;padding:10px 18px;margin-top:20px}}
 .legend span{{display:inline-block;margin-right:14px}}
</style>
<h1>F4 blind ambiguity rating</h1>
<div class="q">
 <b>One question per scene.</b> Given the car wants to go in the green-arrow
 direction (its intent), how many genuinely different reasonable trajectories
 could it take right now?<br>
 <b>1</b> = only one sane action (e.g. stopped at a red light, empty single lane).
 <b>5</b> = a real fork where multiple choices are all defensible (e.g. yield
 or go at an unprotected turn, pick a gap among crossing traffic).<br>
 Rate on the situation as shown. Do not overthink; first instinct is fine.
 <div class="legend" style="margin-top:8px">
  <span style="color:#2c3e90">&#9632; ego (you), points up</span>
  <span style="color:#c0392b">&#9679; vehicle</span>
  <span style="color:#8e44ad">&#9679; pedestrian</span>
  <span style="color:#16a085">&#9679; cyclist</span>
  <span style="color:#1e8449">&#8594; intended direction</span>
 </div>
</div>
<div class="grid">{cards}</div>
<button onclick="exp()">Download my ratings (CSV)</button>
<script>
function exp(){{
 let rows=[["blind_id","rating"]];
 document.querySelectorAll(".rate").forEach(d=>{{
   let id=d.dataset.id, sel=d.querySelector("input:checked");
   rows.push([id, sel?sel.value:""]);
 }});
 let csv=rows.map(r=>r.join(",")).join("\\n");
 let a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));
 a.download="f4_ratings.csv"; a.click();
}}
</script>'''
    path.write_text(html)


if __name__ == "__main__":
    main()
