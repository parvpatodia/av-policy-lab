# What the Foremost Minds Say Is Conceptually Missing in Learned AV Planning — And a Brutal Pressure-Test of Our Thesis

> Research-taste scout report. Date: June 2026.
> Author: frontier-upgrade scout pass. Every non-obvious claim is cited with a URL.
> **Reading guide:** Section 1–3 = what the best people independently say is broken. Section 4 = brutal pressure-test of our 2×2 thesis. Section 5 = the ranked deliverable.
>
> **Epistemic flags used below:** `[VERIFIED]` = read from the primary source / paper text or its abstract. `[SECONDHAND]` = paraphrased from a search-engine summary of the source, not the source text directly — treat the *gist* as reliable, exact wording as not. `[INFERENCE]` = my synthesis, not a claim any single author makes.

---

## 0. TL;DR for the impatient

1. The single most-repeated conceptual complaint from the best people is **not** "we need better architectures." It is: **we cannot currently measure planner quality, because the metric we optimize (open-loop trajectory matching) is negatively correlated with the thing we care about (closed-loop driving), and the simulator we validate in is populated by agents that don't react.** Geiger/Chitta, the BEV-Planner authors, the Copycat-problem authors, and the SMART/nuPlan-R authors all independently land here. This is the field's load-bearing crack.

2. The **root cause** of the open-loop/closed-loop gap is over-determined, and the leaders do **not** agree on a single culprit. The strongest causal story (de Haan/Jayaraman/Levine → the 2025 "Copycat" paper) is **causal confusion via the ego-state shortcut**, which is a *distinct* mechanism from vanilla covariate shift, though the two compound. Treating "covariate shift" and "causal confusion" as the same thing is itself one of the misconceptions.

3. On **multimodality**: almost everyone *asserts* "driving is multimodal" and almost no one rigorously *decomposes the source*. The literature implicitly contains three different multimodalities — route/intention ambiguity, agent-interaction outcomes, and human-demonstrator inconsistency — and routinely conflates them. **This conflation is a genuine, under-exploited gap.** Our thesis lives exactly here, which is good news. The bad news is in Section 4.

4. The IL paradigm **is** under question at the top. Levine, Hotz, Waymo Research, and the CaRL/Copycat lines are all converging on "**BC is the wrong objective; you need closed-loop signal (RL, RL-fine-tuning, or world-model rollouts) somewhere in the loop.**" Diffusion is treated as an *expressivity* fix, not a *causality* or *distribution-shift* fix — and that distinction is fatal to a naive "diffusion helps" framing.

---

## 1. The deep, unsolved CONCEPTUAL problems (ranked by independent convergence)

I rank by **how many distinct top groups raise the same issue without citing each other as the reason**. That is the signal of a real conceptual fault line, not a fashion.

### Rank 1 — "We cannot measure planner quality." (Evaluation is not just noisy, it is *anti-correlated*.)
This is the most independently-converged-upon claim in the field.

- Dauner, Hallgarten, Geiger, Chitta, *Parting with Misconceptions about Learning-based Vehicle Motion Planning* (CoRL 2023): open-loop and closed-loop "are **fundamentally misaligned and should be addressed independently**," and they find a **negative correlation** — learned planners win ego-forecasting but lose closed-loop, rule-based planners do the reverse; a ~20-year-old rule-based baseline (PDM) beats all learned SOTA in closed-loop. `[VERIFIED]` https://arxiv.org/abs/2306.07962
- Li et al., *Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?* (CVPR 2024): on nuScenes, an MLP on **ego status alone** matches camera-based SOTA on L2/collision; perturbing the image barely changes the plan, perturbing ego velocity changes it a lot. Their recommendation is not a new model but: **"the development of more appropriate datasets and metrics represents a more critical and urgent challenge."** `[VERIFIED]` https://arxiv.org/html/2312.03031v2
- The 2025–2026 IDM-critique cluster (Section 3) says the *closed-loop* side of the benchmark is itself broken because the background agents are unrealistic — so even "the good metric" was being measured in a rigged world. `[VERIFIED]` https://arxiv.org/abs/2510.14677

**Why it's conceptual, not engineering:** the problem isn't that L2/ADE is imprecise; it's that **the quantity being measured is the wrong quantity, and optimizing it actively selects for the ego-extrapolation shortcut.** You cannot out-engineer a metric that rewards the failure mode.

### Rank 2 — Causal confusion / the ego-state shortcut (a *specific* mechanism, not generic distribution shift)
- de Haan, Jayaraman, Levine, *Causal Confusion in Imitation Learning* (NeurIPS 2019): behavior cloning is **non-causal** — it learns *correlates* of the expert action. The signature symptom is the **"causal misidentification" paradox: more information yields worse closed-loop performance**, because the policy latches onto a nuisance correlate (e.g., its own past motion / a brake indicator) that is present in demos but not causally driving the right action. `[VERIFIED]` https://arxiv.org/abs/1905.11979
- This is exactly what the AV "ego-status shortcut" is: the policy reads its own velocity/history and extrapolates, because in 73.9% of nuScenes the future *is* an extrapolation of the present. `[VERIFIED]` https://arxiv.org/html/2312.03031v2 The *Parting with Misconceptions* paper independently observes the same fingerprint: **adding ego-history to PDM-Open *lowers* closed-loop score** (CLS 53 → 50). `[VERIFIED]` https://arxiv.org/html/2306.07962v1
- The bridge paper, Wang/Zhou et al. (Apr 2025), *Exposing the Copycat Problem of Imitation-based Planner: A Novel Closed-Loop Simulator, Causal Benchmark and Joint IL-RL Baseline*, states the link explicitly: **"open-loop training tends to cause causal confusion during closed-loop testing,"** and that minimal-pose input / SDE-style noise can mitigate the shortcut. `[SECONDHAND — read from search summary, not paper text]` https://arxiv.org/abs/2504.14709

**Why this matters for us:** "covariate shift" and "causal confusion" are **different failure modes** that both produce open-loop/closed-loop divergence. Covariate shift = right model, wrong (drifted) states. Causal confusion = wrong model that *looks* right on the training distribution because it exploits a non-causal correlate. **Conflating them is itself a misconception the leaders are at pains to separate.** `[INFERENCE, grounded in the two papers above]`

### Rank 3 — The BC objective is the wrong objective (IL paradigm under question)
- Levine has long argued the missing ingredient is **counterfactual reasoning**: offline RL "must be able to reason about counterfactuals — what will happen if we take a different action," whereas BC cannot, because it never sees the consequence of leaving the expert distribution. `[VERIFIED, quote paraphrased from his offline-RL essay]` https://medium.com/@sergey.levine/decisions-from-data-how-offline-reinforcement-learning-will-change-how-we-use-ml-24d98cb069b0
- Hotz/comma.ai (2023 blog, *Imitation Learning*) is the field-engineer version: pure IL "could drive for ~10 seconds before error accumulated" and drifts; the practical fix was to add **explicit lane-position prediction as an unbiased correction** (i.e., bolt on structure BC can't learn) and to gather data from ~100k human lane changes for the lane-change "feature." `[SECONDHAND — search summary of the blog; primary blog fetch was blocked this session]` https://geohot.github.io/blog/jekyll/update/2023/11/18/imitation-learning.html
- The fundamental-limits theory line (Block/Foster et al.) shows **error compounds exponentially in horizon** for autoregressive next-action BC even on stable systems — a hard impossibility result, not an engineering nuisance. `[SECONDHAND]` https://arxiv.org/abs/2102.12948
- Empirically converging: a 2025 Waymo-Open-Motion study reports a CQL (offline-RL) agent at **~3.2× success and ~7.4× lower collision** vs the strongest BC baseline. `[SECONDHAND]` https://arxiv.org/html/2508.07029v1 Waymo Research's own RL-fine-tuning of sim agents improves collision/off-road over pure IL. `[VERIFIED]` https://waymo.com/research/improving-agent-behaviors-with-rl-fine-tuning-for-autonomous-driving/ And on nuPlan, **CaRL (closed-loop RL) ranks #1 under realistic SMART agents**, attributed to "exposure of the policy to its actions in closed-loop rollouts during training." `[VERIFIED]` https://arxiv.org/abs/2510.14677

### Rank 4 — Simulator/world-model validity (the sim-to-eval gap, not just sim-to-real)
- The whole point of Wayve's GAIA world-model line is that you cannot validate a learned driver in a hand-built sim; you need a *learned* sim. GAIA-3 (Dec 2025) is pitched explicitly "to accelerate the **evaluation and validation**" of the driving AI. `[VERIFIED]` https://wayve.ai/press/wayve-launches-gaia3/
- Levine's adjacent provocation (imbue podcast, 2023): **"simulation is doomed to succeed"** — sim lets you brute-force a number that may not transfer, which is the meta-version of the metric-gaming critique. `[SECONDHAND — podcast title/summary]` https://imbue.com/podcast/2023-03-01-podcast-episode-28-sergey-levine/
- Waymo's Sim Agents Challenge quantifies the residual unrealism: even 2024-SOTA sim agents **collide in 5–6% and go off-road in 6–12%** of scenarios — i.e., the "realistic" agents we'd validate against still aren't. `[SECONDHAND]` https://openreview.net/pdf?id=5FnttJZQFn

### Rank 5 — Multimodality is asserted, not understood (the gap our thesis targets)
See Section 2(b) — promoted to its own treatment because it is the crux of our contribution.

---

## 2. Interrogating the three questions the prompt flags

### (a) Open-loop/closed-loop misalignment — what is the ROOT CAUSE?
**There is no single root cause, and the honest answer is a stack of three:**

1. **Benchmark/metric design** (the proximate cause). Open-loop ADE/L2 on logged human data rewards trajectory *matching*, which is dominated by the trivially-predictable straight-line majority of the data. `[VERIFIED]` https://arxiv.org/html/2312.03031v2
2. **Causal confusion via the ego-state shortcut** (the mechanism that *exploits* the bad metric). The policy keys on its own state because that correlate maximizes the metric, and that exact correlate is what kills it in closed loop. `[VERIFIED]` https://arxiv.org/abs/1905.11979 + https://arxiv.org/html/2306.07962v1
3. **Covariate shift / compounding error** (what turns a small closed-loop error into off-road). `[VERIFIED]` general IL; quantified theoretically https://arxiv.org/abs/2102.12948

The *Parting with Misconceptions* authors deliberately decline to name one culprit — they call it "fundamental task misalignment." `[VERIFIED]` https://arxiv.org/html/2306.07962v1 The Copycat paper is the one source that explicitly fuses (1)+(2): open-loop training → causal confusion → closed-loop failure. `[SECONDHAND]` https://arxiv.org/abs/2504.14709 **Human-demonstration inconsistency is the *least* discussed of the candidate causes in the AV-planning literature** — a gap worth noting.

### (b) Multimodality — does anyone ask WHERE it comes from?
**Mostly no.** The dominant pattern is to *assert* "driving is multimodal" and reach for an expressive generator (diffusion, GMM, anchors). The Diffusion Policy paper itself frames the entire value proposition as "gracefully handling **multimodal action distributions**" that unimodal-Gaussian BC mode-averages or mode-collapses on. `[VERIFIED]` https://arxiv.org/abs/2303.04137 The AV diffusion papers (Diffusion-Planner, DiffVLA, AnchDrive, etc.) inherit this framing wholesale without decomposing the source. `[VERIFIED, from abstracts]` https://arxiv.org/abs/2501.15564

But the *components* of a source-decomposition **do exist, scattered and unintegrated**:
- **Route/intention ambiguity:** the goal-conditioning literature shows a single endpoint goal "resolves strategic ambiguity," i.e., much apparent multimodality is just *unspecified route*. Condition on the route and that mode structure largely collapses. `[SECONDHAND]` https://arxiv.org/pdf/2602.03376
- **Agent-interaction outcomes:** the game-theoretic line (Bayesian-game planners) treats multimodality as arising from *incomplete information about other agents' intentions* — i.e., interaction-driven. `[SECONDHAND]` https://arxiv.org/html/2409.13993v1
- **The prediction/planning asymmetry** is the sharpest statement near our thesis: *prediction* "forecasts surrounding agent motion under **unknown intentions**, producing multimodal distributions, while planning assumes **known ego objectives** and generates deterministic trajectories." `[SECONDHAND]` https://arxiv.org/pdf/2602.03376

**Implication for us, stated plainly:** the field has the *ingredients* to argue multimodality is interaction-driven vs route-driven, but **no one has run the controlled experiment that isolates them.** That is the white space. **However** — the prediction/planning-asymmetry quote is also a *threat* to our thesis: a serious reviewer can argue that *once the ego goal/route is given, the ego's own plan is nearly deterministic*, and the residual multimodality is the **other agents' problem**, not the ego planner's. If that's true, route-conditioning would suppress multimodality more than agent realism re-introduces it. We must pre-empt this (Section 4).

### (c) Benchmark/simulator validity beyond the IDM critique
The IDM critique is now a *cluster*, and it has matured past "IDM is passive":
- *When Planners Meet Reality* (Oct 2025, the SMART-agents paper, **2510.14677** in our thesis): swapping IDM→SMART, **nearly all scores deteriorate**; IL planners drop ~−5.17 avg on Val14, hybrid −3.25, rule-based −2.0. Crucially the ranking is **non-uniform** — some planners *improve* in multi-lane/turn/lane-change scenarios, and closed-loop-trained methods (CaRL) become most stable. IDM "cannot react to vehicles in adjacent lanes," manufacturing artificial collisions. `[VERIFIED]` https://arxiv.org/abs/2510.14677
- nuPlan-R (Nov 2025, **2511.10403**): independent, replaces IDM with **noise-decoupled diffusion-based reactive agents** + interaction-aware agent selection; same thesis — IDM "lack[s] behavioral diversity and fail[s] to capture realistic human interactions." `[VERIFIED]` https://arxiv.org/abs/2511.10403
- The deeper validity worry (Waymo WOSAC): even *learned* sim agents still collide/go off-road at single-digit-to-low-double-digit rates, so "realistic agents" is a moving target, not a solved input. `[SECONDHAND]` https://openreview.net/pdf?id=5FnttJZQFn

**Net:** the leaders' answer to "can we even measure planner quality today?" is **"not reliably, and the closed-loop fix (realistic agents) is itself only partially trustworthy."** This is *favorable* to a thesis that explicitly tests robustness *across* agent models rather than picking one.

### (d) Is the IL paradigm itself in question?
Yes — but with a precise nuance the leaders insist on: **the issue is the *training signal*, not the *generative form*.** Moving BC→diffusion changes the *form* (you can now represent multiple modes) but not the *signal* (still open-loop log-likelihood of human actions, still non-causal, still no counterfactual). The consensus prescription is to inject closed-loop signal: offline RL (CQL), RL fine-tuning (Waymo, PlannerRFT), closed-loop RL training (CaRL), or world-model rollouts (Wayve GAIA). `[VERIFIED across the cited sources]` **A reviewer who has internalized this will read "we compare deterministic-MLP vs diffusion" as *comparing two points on the wrong axis* unless we explicitly say we are isolating the expressivity axis on purpose.**

---

## 3. Source map (who said what, for citation hygiene)

| Source | Conceptual claim we use | Flag |
|---|---|---|
| Dauner/Geiger/Chitta 2023, *Parting w/ Misconceptions* | OL/CL negatively correlated; centerline-only best OL; ego-history hurts CL; IDM agents "unrealistically passive" | VERIFIED https://arxiv.org/abs/2306.07962 |
| Li et al. CVPR'24, *Is Ego Status All You Need* | ego-state shortcut; 73.9% straight; metric rewards the shortcut; fix the dataset/metric | VERIFIED https://arxiv.org/html/2312.03031v2 |
| de Haan/Jayaraman/Levine 2019, *Causal Confusion* | BC is non-causal; more info → worse; "causal misidentification" | VERIFIED https://arxiv.org/abs/1905.11979 |
| Wang et al. 2025, *Copycat Problem* | OL training → causal confusion in CL; minimal-pose mitigates; joint IL-RL | SECONDHAND https://arxiv.org/abs/2504.14709 |
| Hu et al. 2025, *When Planners Meet Reality (SMART)* | IDM overestimates; rankings shift; CL-trained planners most stable | VERIFIED https://arxiv.org/abs/2510.14677 |
| nuPlan-R 2025 | independent IDM critique w/ diffusion reactive agents | VERIFIED https://arxiv.org/abs/2511.10403 |
| Chi/Song et al. 2023, *Diffusion Policy* | diffusion = fix for *mode-averaging* of unimodal BC (expressivity) | VERIFIED https://arxiv.org/abs/2303.04137 |
| Zheng et al. 2025, *Diffusion-Planner* (ICLR Oral) | diffusion planner SOTA CL on nuPlan; joint pred+plan | VERIFIED abstract https://arxiv.org/abs/2501.15564 |
| Levine, offline-RL essay / imbue podcast | counterfactuals; "simulation is doomed to succeed" | SECONDHAND https://medium.com/@sergey.levine/decisions-from-data-how-offline-reinforcement-learning-will-change-how-we-use-ml-24d98cb069b0 ; https://imbue.com/podcast/2023-03-01-podcast-episode-28-sergey-levine/ |
| Hotz/comma.ai 2023 blog | pure IL drifts in ~10s; bolt on structure; data-scale lane changes | SECONDHAND https://geohot.github.io/blog/jekyll/update/2023/11/18/imitation-learning.html |
| Wayve GAIA-3 | learned world model for evaluation/validation | VERIFIED https://wayve.ai/press/wayve-launches-gaia3/ |
| Waymo WOSAC / RL-fine-tuning | learned sim agents still collide 5–6%, off-road 6–12%; RL-FT helps | SECONDHAND https://openreview.net/pdf?id=5FnttJZQFn ; VERIFIED https://waymo.com/research/improving-agent-behaviors-with-rl-fine-tuning-for-autonomous-driving/ |
| Block/Foster et al. | BC error compounds exponentially in horizon (impossibility) | SECONDHAND https://arxiv.org/abs/2102.12948 |
| Goal-conditioned transformer / Bayesian-game lines | multimodality sources: route-intention vs interaction; pred/plan asymmetry | SECONDHAND https://arxiv.org/pdf/2602.03376 ; https://arxiv.org/html/2409.13993v1 |

---

## 4. BRUTAL PRESSURE-TEST OF OUR THESIS

**Our thesis (restated):** a controlled 2×2 — {precise-goal vs route-conditioned goal} × {deterministic-MLP vs diffusion} — in nuPlan closed-loop under BOTH IDM and SMART agents, to isolate (i) *when* multimodal/diffusion planning helps and (ii) whether that answer is an artifact of unrealistic sim agents. Deeper claim: driving multimodality is **largely interaction-driven**, so it is *suppressed* under IDM (→ null diffusion result) and should *re-emerge* under SMART.

### (a) Is this a gap the leaders care about, or a niche curiosity?
**It is adjacent to a gap they care about a lot, but as currently framed it is one reframing away from being a curiosity.** Two honest readings:

- **Generous reading (why it matters):** Every Rank-1 through Rank-5 problem above is about *the eval substrate determining the conclusion*. Our design's spine — "**does the answer to 'does multimodality help' flip when you fix the simulator?**" — is a direct, falsifiable instance of the field's #1 conceptual anxiety (we can't measure planner quality). The SMART and nuPlan-R authors *just* showed rankings flip; **no one has yet shown that a specific architectural conclusion (diffusion vs deterministic) is an artifact of the agent model.** That is a clean, citable, novel claim if it holds. `[INFERENCE, grounded in 2510.14677 + 2501.15564]`

- **Brutal reading (why it risks being niche):** "Diffusion vs MLP" is a *form* axis, and the leaders have explicitly said form is not the bottleneck — *signal* is (Section 2d). If our result is "diffusion ≈ MLP under IDM, diffusion > MLP under SMART," a sharp reviewer says: *"You've shown realistic agents create interaction-multimodality that an expressive head can exploit — fine, but that's a property of your simulator + the well-known mode-averaging of Gaussian BC, not a new insight about planning."* The 2×2 is then a clean ablation, not a discovery.

### (b) The single biggest threat to the thesis (must address head-on)
The **prediction/planning asymmetry**: *the ego's own multimodality may be small once its goal/route is given; the multimodality lives in the other agents.* `[SECONDHAND]` https://arxiv.org/pdf/2602.03376 If that's right, then:
- **Route-conditioning suppresses ego multimodality directly** (you removed route ambiguity).
- **SMART realism injects multimodality into the *agents*, not necessarily into the *ego's optimal plan*** — the ego may still have one good response to a richer world.
So the predicted "re-emergence under SMART" could fail not because the thesis is wrong but because **the ego planner's response to interaction is often still unimodal-but-context-dependent** (one correct yield/merge), which a *deterministic context-conditioned* MLP can also represent. Diffusion only wins if interaction creates genuine *equivalent-cost multi-mode* decisions for the ego (e.g., "merge ahead OR behind, both fine"), and those are rarer than the word "multimodal" implies.

**This is survivable, and turning it into a measured quantity is what makes the work significant** (see (c)).

### (c) What would make this intellectually significant vs a clean ablation
A clean ablation reports *which cell wins*. Significant work explains *why* and *predicts the flip in advance with a mechanism*. Concretely, the upgrade is to **measure the multimodality, don't just assume it**:

1. **Instrument the source of multimodality, per-scenario.** Define and log an **interaction-induced-multimodality score** — e.g., entropy/number of distinct cost-comparable ego modes — computed under IDM vs SMART for the *same* scene. If your central claim is "multimodality is interaction-driven," **you must show the multimodality itself rises from IDM→SMART**, independent of which planner you run. That single curve is the thesis; the 2×2 then *explains* it. `[INFERENCE]`
2. **Make a falsifiable, pre-registered prediction:** "diffusion's CL margin over MLP correlates with the per-scenario interaction-multimodality score, and that score is ≈0 under IDM." If the correlation holds, you've shown the diffusion benefit is *causally* an interaction-multimodality benefit — that's a mechanism, not an ablation.
3. **Add the route-conditioning as a *multimodality knob*, not just a 4th cell.** Frame: precise-goal *removes route multimodality*, SMART *adds interaction multimodality*. The 2×2 then cleanly **double-dissociates** the two sources — which is exactly the decomposition the field has *never* run (Section 2b). That double-dissociation is the intellectually load-bearing result.
4. **Pre-empt the "form not signal" critique** by explicitly scoping: you are *not* claiming diffusion fixes IL; you are using diffusion as an *instrument to read out* whether exploitable multimodal structure exists in the optimal-plan distribution. That reframing converts the weakness in 4(a) into the contribution.

**Verdict:** As a "does diffusion help" study → niche. As a **"controlled double-dissociation of the *source* of planning multimodality, using the simulator-agent model as the independent variable and diffusion-vs-deterministic as a multimodality read-out"** → genuinely interesting, solo-feasible, inference-cheap, and it plants a flag in the field's #1 anxiety (eval validity). The pivot is from *"which planner wins"* to *"where does the multimodality come from, and can we make the published conclusion flip on demand by changing only the agents."*

### (d) Feasibility/honesty checks
- **Solo-feasible & inference-cheap:** yes — nuPlan + drop-in SMART agents (released per 2510.14677) + a small MLP and a small diffusion head. No training of world models, no RL. `[VERIFIED that SMART agents are released as a drop-in]` https://arxiv.org/abs/2510.14677
- **A null result is still publishable** *iff* you instrumented multimodality (point 1): "interaction-multimodality does rise under SMART, yet diffusion still doesn't beat MLP" is a *more* interesting paper (it would say ego multimodality is not the ego planner's problem — the asymmetry view) than a positive ablation.
- **Watch-out:** confounds. SMART changes *many* things (collisions, off-road, comfort), not just multimodality. Without point-1 instrumentation, a diffusion>MLP gap under SMART is uninterpretable. **This is the make-or-break methodological requirement.**

---

## 5. THE DELIVERABLE — 3 most intellectually significant, solo-feasible, inference-cheap research questions (June 2026)

Ranked by significance × feasibility × novelty.

### Q1 — "Is published architectural superiority an artifact of the simulator's agents?" *(our thesis, upgraded)*
**Question:** Holding the planner fixed, does the *conclusion* "diffusion > deterministic for planning" flip purely as a function of the background-agent model (IDM vs SMART), and does that flip track a directly-measured *interaction-induced-multimodality* score?
**Why significant:** It operationalizes the field's #1 anxiety (we can't measure planner quality) into a falsifiable claim about *meta-validity* — that a peer-reviewed architectural conclusion can be manufactured by an eval choice. That is a stronger, more transferable statement than "diffusion helps." Builds directly on 2510.14677 / 2511.10403 but goes beyond "rankings shift" to "*a published mechanism-claim shifts.*"
**Feasibility:** High. Drop-in SMART agents, small models, no RL. The one hard part is defining the multimodality metric — but that *is* the contribution.
**Cite-anchor:** https://arxiv.org/abs/2510.14677 ; https://arxiv.org/abs/2306.07962 ; https://arxiv.org/abs/2303.04137

### Q2 — "Double-dissociating the sources of planning multimodality: route ambiguity vs agent interaction."
**Question:** Using {precise-goal vs route-conditioned} × {IDM vs SMART} as two orthogonal knobs that *remove route multimodality* and *add interaction multimodality* respectively, can we show a clean double-dissociation — diffusion's benefit vanishes under (precise-goal ∧ IDM) and is maximal under (route-free ∧ SMART)?
**Why significant:** The field *asserts* "driving is multimodal" without ever decomposing the source (Section 2b). A double-dissociation is the textbook causal-isolation design and **no AV-planning paper has run it.** It also directly tests the prediction/planning-asymmetry hypothesis (https://arxiv.org/pdf/2602.03376): if the asymmetry view is right, the (route-free ∧ SMART) cell *still* won't favor diffusion — itself a publishable, surprising result.
**Feasibility:** High — it is the *same* compute as Q1, just analyzed as a 2×2 factorial with the multimodality readout. This is the most efficient significance-per-GPU-hour question on the list.
**Cite-anchor:** https://arxiv.org/pdf/2602.03376 ; https://arxiv.org/html/2409.13993v1 ; https://arxiv.org/html/2312.03031v2

### Q3 — "Does the ego-state shortcut survive realistic agents? A causal-confusion stress test under SMART."
**Question:** Re-run the *Is-Ego-Status / ego-history-hurts-CL* finding (https://arxiv.org/html/2306.07962v1, https://arxiv.org/html/2312.03031v2) but under SMART agents: does feeding ego-history still *help* open-loop while *hurting* closed-loop, and does the size of that gap *grow* under realistic agents (because the shortcut is exactly what fails in interaction)? Compare deterministic vs diffusion heads as a probe of whether expressive heads resist or amplify the shortcut.
**Why significant:** It connects the two deepest conceptual threads (causal confusion ↔ eval validity) and tests whether the most famous AV-IL pathology (ego-state shortcut) is *worse* than reported because it was measured against passive agents. If the shortcut's closed-loop penalty grows under SMART, that is a clean, mechanistic indictment of the entire open-loop-trained-then-deployed pipeline — and a natural lead-in to "you need closed-loop signal" (the Rank-3 consensus).
**Feasibility:** High — it's an input-ablation (with/without ego-history; minimal-pose à la Copycat) crossed with the agent model. No new model classes beyond what Q1/Q2 already build.
**Cite-anchor:** https://arxiv.org/abs/1905.11979 ; https://arxiv.org/abs/2504.14709 ; https://arxiv.org/html/2306.07962v1

> **Reasoning across all three:** they are the same experimental rig (nuPlan + SMART drop-in + small MLP + small diffusion head + an *instrumented multimodality metric*), reused three ways. Q1 makes the meta-validity claim, Q2 isolates the *source* of multimodality (the conceptual white space), Q3 ties it to the causal-confusion mechanism. Doing Q2's instrumentation is the unlock that turns the whole program from "clean ablation" into "we can manufacture the field's published conclusions by changing only the simulator's agents — here's the mechanism." That sentence is the paper.

---

### Appendix: claims I could NOT fully verify this session (treat with caution)
- The exact wording of George Hotz's 2023 blog (primary fetch was blocked; relied on search summary). `[SECONDHAND]`
- The Copycat-problem paper's precise phrasing linking open-loop training to causal confusion (read from search summary, not paper body). `[SECONDHAND]`
- The CQL "3.2×/7.4×" numbers (single secondary source; not cross-checked against the paper's tables). `[SECONDHAND]`
- Block/Foster exponential-compounding result attribution (theory line is real; exact paper-to-claim mapping not re-derived). `[SECONDHAND]`
- The "prediction/planning asymmetry" quote comes from a goal-conditioned-transformer preprint's framing, not from a Levine/Geiger-tier source — it is a *useful framing to attack*, not an authority. `[SECONDHAND]`
