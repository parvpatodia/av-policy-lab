# F4 blind ambiguity rating (Parv, ~20 min)

Purpose: external validation of F4. You rate scenes blind to the F4 score; we
correlate. This is the independent check a reviewer asks for.

## Do this
1. On the laptop, open in a browser:
   ~/av_assets/f4_rating/sheet/rating_sheet.html
2. 52 top-down scenes. For each, one question: given the green intended path,
   how many genuinely different reasonable trajectories could the car take now?
   1 = only one sane action. 5 = a real fork (yield-or-go, pick a gap).
   Legend on the page. First instinct; ~20 s each.
3. Click "Download my ratings (CSV)" -> f4_ratings.csv. Put it in
   ~/av_assets/f4_rating/sheet/ .

## What happens next (automated)
   ./.venv/bin/python ~/av_assets/f4_rating/score_ratings.py sheet/f4_ratings.csv
gives Spearman(F4, your ratings), per-band means, and the top disagreements.
rho>=0.5 significant = F4 externally validated. Disagreements are diagnostic:
the open question is whether S_inter over-fires on converging same-direction
traffic (some high-F4 scenes may look like ordinary traffic to you). Honest
either way.

## Notes
- F4 is hidden; scenes are in randomized order; the answer key is separate.
- Stimulus shows lanes, crosswalks (dashed), other agents with 3 s motion
  arrows (red veh / purple ped / teal cyc), ego (blue, points up), and the
  route corridor path (green). It deliberately does NOT show the expert's
  actual future, so you judge the situation, not the recorded outcome.
