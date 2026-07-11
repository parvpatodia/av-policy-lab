# Boston de-risk retrain — selection-bottleneck reproduction on real train data

AV-policy-lab. Consolidates ADR-060..072. Status: near-complete (oracle-vs-det pending det zoo 8281301).
Question: does the mini-trained selection-bottleneck result (ADR-058/059) reproduce when models are
trained on REAL nuPlan train-split data instead of mini? Scoped to one city (Boston) as a capital-
efficient gate before committing the full bos+pitt+singapore retrain (Parv-approved, ADR-061).

## Setup
- Data: 64-log Boston subset (matched to mini's 64 logs), f0 extraction identical recipe to f0_v3
  (num_scenarios_per_type 20, stride 5, perturb 0.5) -> ~110k samples (ADR-062/064).
- Models (ADR-063, core only; selector deferred): det = train_policy.py --head det --goal route, 150
  epochs (minADE 0.070); wta = wta_derisk_train.py, 6 modes, 4000 steps.
- Eval: Val14 sub300 (SAME 300 tokens as ADR-058), IDM reactive, PDM-style CLS. Per-mode via
  WTA_MODE_INDEX 0..5 (oracle + random); default-WTA = head argmax (imitation score).

## Results (n=300)
| quantity | Boston | mini (ADR-058/059) |
|---|---|---|
| oracle (best-of-6) | 0.678 | 0.868 |
| det | [PENDING det zoo] | 0.810 |
| default-WTA (imitation score) | 0.647 | 0.705 |
| random mode | 0.628 | 0.730 |
| default-WTA - random (mechanism) | +0.019 (ABOVE) | -0.025 (BELOW) |
| oracle - default-WTA | +0.031 | +0.163 |
| oracle - det (latent value) | [PENDING] | +0.058 |
| cl_corr pearson(open-loop proxy, closed-loop CLS) | 0.071 | 0.11 |

## Findings
1. MECHANISM REPRODUCES (clean). cl_corr = 0.071 (spearman ~0), ~ mini 0.11: the open-loop proxy is
   ~orthogonal to closed-loop CLS on real data too. This is the recipe/budget-INDEPENDENT test and the
   novel claim (feedforward can't recover closed-loop mode-value). It HOLDS on real Boston data.
2. GAP MAGNITUDES SHRINK. Oracle headroom over default-WTA collapses (+0.031 vs mini +0.163); modes
   cluster tightly (per-mode means 0.60-0.64). The mini "imitation selection worse than random" FLIPS
   to slightly-above-random (+0.019).
3. oracle-vs-det (the PRIMARY latent-value claim): PENDING (det zoo running).

## Confounds (do not read the magnitudes as a clean reproduction)
- Recipe: mini per-mode/default-WTA used rl_long (RL-trained scorer); Boston used plain wta_derisk_train
  (no RL scorer). The default-WTA-vs-random comparison is confounded by score-head kind.
- Budget: det ~228k optimizer steps (~14.6M sample-views) vs wta 4000 steps (256k) = ~57x. det is far
  better optimized, so a small/negative oracle-det would be partly a budget artifact, not proof the
  modes lack latent value.

## Honest bottom line (draft, pending oracle-det)
The NOVEL mechanistic claim (feedforward unpredictability of closed-loop mode-value) reproduces cleanly
on real training data. The dramatic mini MAGNITUDES (huge oracle headroom, imitation-selection-worse-
than-random) do NOT cleanly reproduce, but under two confounds (recipe + ~57x budget) that make the
Boston-vs-mini magnitude comparison unfair. So the de-risk neither confirms nor refutes the gap size on
real data; it confirms the mechanism and shows the gap is recipe/budget-sensitive.

## Gate decision (for Parv, after oracle-det lands)
Options for the full bos+pitt+singapore retrain:
 A. Full retrain with MATCHED recipe + budget for det/wta/selector (clean magnitude test) — most work,
    the only design that yields a defensible gap number.
 B. Proceed with the mechanism as the headline (cl_corr reproduces) + report the gap as recipe-sensitive
    — cheaper, honest, workshop-ceiling.
 C. Reconsider scope given the weak magnitudes.
Recommendation TBD on the oracle-det number.
