"""Generate a self-contained HTML rating tool for the Signal B human study.
Embeds the human tasklist (so no fetch/CORS issues under file://) and references
val_*.png by relative path. Open the HTML from inside the validation_set folder.

    /opt/anaconda3/envs/nuplan/bin/python nuplan/f4_validation/make_human_tool.py
"""
import json
from pathlib import Path

VAL = Path('/Users/parvpatodia/Desktop/diffusion-policy-zoo/data/scene_renders/validation_set')
tasklist = json.loads((VAL / 'human_tasklist.json').read_text())
items = [{'id': t['display_id'], 'file': t['file']} for t in tasklist]  # is_repeat hidden

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>F4 ambiguity rating</title>
<style>
 body{margin:0;font-family:-apple-system,Helvetica,Arial,sans-serif;background:#0e1116;color:#e6e6e6}
 #wrap{display:flex;height:100vh}
 #left{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:12px}
 #img{max-height:78vh;max-width:96%;border:1px solid #2a2f3a;border-radius:6px;background:#000}
 #right{width:330px;padding:18px;background:#141923;overflow:auto;border-left:1px solid #222}
 h2{font-size:16px;margin:6px 0}
 .rub{font-size:13px;line-height:1.45;color:#cdd3dd}
 .rub b{color:#fff}
 #bar{height:6px;background:#222;border-radius:3px;margin:10px 0}
 #fill{height:6px;background:#3d7bd9;border-radius:3px;width:0%}
 .btns{display:flex;flex-direction:column;gap:8px;margin-top:14px}
 button.rate{padding:11px;border:0;border-radius:6px;color:#fff;font-size:14px;cursor:pointer;text-align:left}
 .b0{background:#2e7d4f}.b1{background:#5a8f3c}.b2{background:#9a8a2e}.b3{background:#b5652e}.b4{background:#c0392b}
 .nav{display:flex;justify-content:space-between;margin-top:14px}
 .nav button{padding:8px 14px;background:#28303d;color:#fff;border:0;border-radius:6px;cursor:pointer}
 #dl{margin-top:14px;width:100%;padding:12px;background:#3d7bd9;border:0;border-radius:6px;color:#fff;font-size:15px;cursor:pointer;display:none}
 .muted{color:#8b93a1;font-size:12px}
 .cur{color:#9ec5ff;font-weight:600}
</style></head><body><div id="wrap">
 <div id="left"><img id="img" src=""><div class="muted" id="counter"></div></div>
 <div id="right">
  <h2>Rate the EGO's decision ambiguity</h2>
  <div id="bar"><div id="fill"></div></div>
  <div class="rub">
   The <b style="color:#6fa8ff">blue</b> car is you (pointing up). Rate how AMBIGUOUS your
   immediate driving DECISION is.<br><br>
   <b>Ambiguity</b> = a real decision where a competent driver could reasonably choose
   differently: <b>yield-or-go</b> (cross/merge timing vs conflicting agents) or
   <b>which-way</b> (several plausible routes at a junction).<br><br>
   <b>Busyness is NOT ambiguity.</b> Dense same-direction traffic where you just follow
   your lane is LOW even with many cars. Use the velocity arrows to tell crossing from
   parallel traffic.<br><br>
   <span class="muted">Legend: blue=you, orange=vehicles, green=pedestrians, arrows=velocity,
   gray=lanes, red/green lines=traffic-light connectors, tan hatched=crosswalk.</span>
  </div>
  <div class="btns">
   <button class="rate b0" data-v="0">[1] 0.00 - clear, one obvious action</button>
   <button class="rate b1" data-v="0.25">[2] 0.25 - mild</button>
   <button class="rate b2" data-v="0.5">[3] 0.50 - moderate</button>
   <button class="rate b3" data-v="0.75">[4] 0.75 - genuine yield-or-go / route choice</button>
   <button class="rate b4" data-v="1">[5] 1.00 - strong (multiple conflicts / routes)</button>
  </div>
  <div class="nav"><button id="prev">&larr; back</button>
   <span class="cur" id="pos"></span>
   <button id="next">skip &rarr;</button></div>
  <button id="dl">Download ratings CSV</button>
  <div class="muted" id="saved" style="margin-top:8px"></div>
 </div></div>
<script>
 const ITEMS=__ITEMS__;
 const KEY='f4_human_ratings';
 let R=JSON.parse(localStorage.getItem(KEY)||'{}');
 let i=0;
 const img=document.getElementById('img');
 function show(){
  const it=ITEMS[i]; img.src=it.file;
  document.getElementById('counter').textContent=it.id;
  document.getElementById('pos').textContent=(i+1)+' / '+ITEMS.length+(R[it.id]!=null?'  ('+R[it.id]+')':'');
  const done=Object.keys(R).length;
  document.getElementById('fill').style.width=(100*done/ITEMS.length)+'%';
  document.getElementById('saved').textContent=done+' / '+ITEMS.length+' rated'+(done==ITEMS.length?'  - ready to download':'');
  document.getElementById('dl').style.display=done==ITEMS.length?'block':'none';
 }
 function rate(v){ R[ITEMS[i].id]=v; localStorage.setItem(KEY,JSON.stringify(R));
   if(i<ITEMS.length-1)i++; show(); }
 document.querySelectorAll('.rate').forEach(b=>b.onclick=()=>rate(parseFloat(b.dataset.v)));
 document.getElementById('prev').onclick=()=>{if(i>0)i--;show();};
 document.getElementById('next').onclick=()=>{if(i<ITEMS.length-1)i++;show();};
 document.onkeydown=e=>{const m={'1':0,'2':0.25,'3':0.5,'4':0.75,'5':1};
   if(e.key in m)rate(m[e.key]);
   else if(e.key=='ArrowLeft'){if(i>0)i--;show();}
   else if(e.key=='ArrowRight'){if(i<ITEMS.length-1)i++;show();}};
 document.getElementById('dl').onclick=()=>{
   let csv='display_id,rating\\n'; ITEMS.forEach(it=>{csv+=it.id+','+(R[it.id]??'')+'\\n';});
   const a=document.createElement('a');
   a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
   a.download='human_ratings.csv'; a.click();
 };
 show();
</script></body></html>"""

out = VAL / 'rate_scenes.html'
out.write_text(HTML.replace('__ITEMS__', json.dumps(items)))
print(f'wrote {out}\nopen with:  open "{out}"')
print(f'{len(items)} items; ratings autosave to browser localStorage; CSV download when all rated')
