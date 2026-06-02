# LIT_DIFFUSION.md — Diffusion / Generative Trajectory Planning: A Critical Literature Review

**Scope:** Diffusion and generative trajectory planning for autonomous driving and sequential decision-making.
**Author:** av-policy-lab frontier review
**Date:** June 2026
**Posture:** Brutally honest. Every claim carries a URL. Anything I could not verify from a primary source (paywall, blocked PDF, review text not surfaceable via search) is explicitly flagged **[UNVERIFIED]** or **[PARTIAL]**.

---

## 0. Method / Verification Caveats (read this first)

This review was assembled under two hard tooling constraints that the reader must weigh:

1. **Full-text PDF fetch was unavailable.** I could not render arXiv/CVF/OpenReview PDFs directly. Numbers, recipes, and quotes below come from (a) abstracts and HTML landing pages, (b) author project pages and GitHub READMEs, (c) third-party summaries (themoonlight.io, emergentmind, alphaXiv, Medium reviews), and (d) follow-up papers that cite and re-tabulate the originals. Where a number comes only from a secondary source it is marked **[2ndary]**.
2. **OpenReview review text could not be extracted verbatim.** Search surfaced the forum URLs but not the body of individual reviews/rebuttals. So the "MISTAKES / CRITICISMS" sections below are reconstructed from: authors' own stated limitations, follow-up papers that explicitly position against the original's weakness, and known community critiques — **not** from reading the literal reviewer comments. Each such section is flagged **[criticism reconstructed, not from verbatim reviews]**. This is the single biggest verification gap in this document. Confirm against the live OpenReview threads before citing reviewer claims in a paper.

A second-order warning: searches in June 2026 surface several very recent arXiv IDs (e.g., 2512.xxxxx, 2601.xxxxx, 2602.xxxxx, 2603.xxxxx). I have treated these as real where multiple independent hits agree, but I have **not** been able to open any of them, so treat all 2026 follow-ups as **[UNVERIFIED — title/abstract only]**.

---

## 1. Diffusion Planner (ICLR 2025 Oral) — *Diffusion-Based Planning for Autonomous Driving with Flexible Guidance*

- arXiv: https://arxiv.org/abs/2501.15564
- HTML: https://arxiv.org/html/2501.15564v2
- ICLR proceedings PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/5c0cd43063f087eb32e572d040a19cd2-Paper-Conference.pdf
- OpenReview: https://openreview.net/forum?id=wM2sfVgMDH
- Code: https://github.com/ZhengYinan-AIR/Diffusion-Planner
- Project: https://zhengyinan-air.github.io/Diffusion-Planner/

**1.1 Recipe**
- **Task framing.** Joint prediction + planning under one model: it diffuses the *future trajectories of the ego vehicle and neighboring agents jointly*, so the planner produces cooperative behavior rather than ego-only. (https://arxiv.org/abs/2501.15564)
- **Conditioning.** Encoded scene context — ego history/state, neighbor agents, lanes/map (vectorized), and navigation/route information — fed to a transformer. Goal/route enters as part of the conditioning context, not as a hard endpoint. (https://arxiv.org/html/2501.15564v2)
- **Denoiser architecture.** Transformer (DiT-style), **not** a U-Net. The model "learns the gradient of the trajectory score function" and uses transformer blocks to fuse conditional scene context with the noisy trajectory tokens. (https://arxiv.org/abs/2501.15564)
- **Representation.** Multi-agent future trajectories (states over a horizon), modeled jointly and permutation-aware across agents.
- **x0 vs eps.** Reported as learning the **score / data-prediction** form of the trajectory distribution (score-function gradient), enabling classifier guidance at sampling. Exact x0-vs-eps parameterization **[UNVERIFIED — not confirmable from HTML/abstract; check §3 / code]**.
- **Steps.** Few-step sampling using a fast ODE solver (DPM-Solver family). Exact train-T and sampling-step count **[PARTIAL — code uses DPM-Solver; precise step count not verified from text]**.
- **Multimodality mechanism.** Generative diffusion over the joint trajectory + **flexible classifier guidance** at inference: differentiable cost terms (safety, comfort, speed/target) are added as guidance gradients, combinable without retraining. This is the paper's headline contribution — "flexible guidance." (https://arxiv.org/html/2501.15564v2)

**1.2 Results (nuPlan closed-loop, NR = non-reactive, R = reactive) [2ndary, re-tabulated from search of the proceedings PDF]**

| Method | Val14 NR/R | Test14 NR/R | Test14-hard NR/R |
|---|---|---|---|
| PDM-Open | 53.5 / 54.2 | 52.8 / 57.2 | 33.5 / 35.8 |
| UrbanDriver | 68.6 / 64.1 | 51.8 / 67.2 | 50.4 / 50.0 |
| PlanTF | 84.7 / 77.0 | 85.6 / 79.6 | 69.7 / 61.6 |
| PLUTO (w/o refine) | 88.9 / 78.1 | 89.9 / 78.6 | 70.0 / 59.7 |
| **Diffusion Planner** | **89.9 / 82.8** | **89.2 / 82.9** | **76.0 / 69.2** |

Headline claim: SOTA closed-loop on nuPlan **without rule-based post-processing/refinement**, plus transfer to a 200-hour delivery-vehicle dataset. (https://arxiv.org/abs/2501.15564)

**1.3 MISTAKES / CRITICISMS / LIMITATIONS** *[criticism reconstructed, not from verbatim reviews — OpenReview body not extractable]*
- The gains over PLUTO-w/o-refine on the easy splits (Val14/Test14) are ~1 point — i.e., within the range where seeding, ego-extrapolation handling, and metric noise matter. The *large* margin is on Test14-hard (+6 NR, +9.5 R). A fair reading: diffusion's advantage shows up under distribution shift / hard scenarios, not on the bulk benchmark. The paper frames this as a clean SOTA win; the honest story is more conditional.
- "No rule-based refinement" is a real selling point, but PLUTO-*with*-refine and PDM-Closed hybrids remain extremely strong on nuPlan; the claim is "competitive without rules," not "beats every rule-augmented system everywhere." **[verify against full table]**
- Classifier guidance is a double-edged sword: each guidance term is a hand-designed differentiable cost. The "flexibility" is also a tuning surface, and guidance strength trades safety against imitation fidelity. The paper does not (visibly, from what I could read) provide a robustness sweep over guidance weights at test time. **[UNVERIFIED]**
- Sampling cost vs. a one-shot regression planner is higher per frame even with DPM-Solver — a recurring objection to diffusion planners (see §10 efficiency follow-ups). The paper argues few-step sampling mitigates this. **[PARTIAL]**

**1.4 The GAP they leave open**
They demonstrate that *adding* diffusion + guidance helps on nuPlan, but they do **not** isolate *when* the multimodal generative head is actually necessary versus when a deterministic head conditioned on the same rich context would match it. Their own framing ("imitation learning struggles with multi-modal behavior") is asserted and then supported by a global benchmark win — not by a controlled ablation that turns multimodality on/off while holding conditioning fixed. This is the seed of the cross-paper gap (§9).

---

## 2. DiffusionDrive (CVPR 2025 Highlight) — *Truncated Diffusion Model for End-to-End Autonomous Driving*

- arXiv: https://arxiv.org/abs/2411.15139 / HTML https://arxiv.org/html/2411.15139
- CVPR PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.pdf
- OpenReview (CVPR'25 forum): https://openreview.net/forum?id=sh7vDLo5EY
- Code: https://github.com/hustvl/DiffusionDrive

**2.1 Recipe**
- **Task framing.** End-to-end (sensor → trajectory) on NAVSIM, built on a Transfuser-style perception backbone.
- **Conditioning.** BEV/perception scene features from the E2E backbone (ResNet-34 default), fused via a **cascade diffusion decoder with sparse deformable attention** to the scene context. (https://arxiv.org/html/2411.15139)
- **Denoiser architecture.** Transformer-based cascade decoder (not U-Net), interacting iteratively with conditional features.
- **Representation.** Multi-mode driving *action/trajectory* anchors.
- **The key trick — truncated diffusion.** Instead of denoising from pure Gaussian noise, they place **prior multi-mode anchors** (offline k-means trajectory clusters) and **truncate the diffusion schedule** (~50/1000), so the model learns to denoise from an *anchored Gaussian* to the multi-mode action distribution. (https://arxiv.org/abs/2411.15139)
- **Steps.** **2 denoising steps at inference** (≈10× fewer than vanilla diffusion policy); 45 FPS at ResNet-34. Top-1 scoring trajectory selected for evaluation. **[2ndary]**
- **Multimodality mechanism.** The anchor set *is* the multimodality prior; diffusion refines each anchor. This is closer to "anchored mixture refinement" than free-form generation.

**2.2 Results [2ndary]**
- **88.1 PDMS on NAVSIM** with ResNet-34 at 45 FPS. Claims ~+3.5 PDMS and ~+64% mode-diversity score over a vanilla diffusion policy baseline, with 10× fewer steps. (https://arxiv.org/abs/2411.15139, https://www.emergentmind.com/papers/2411.15139)
- Benchmark is **open-loop / pseudo-closed-loop NAVSIM PDMS**, *not* nuPlan reactive closed-loop. Important scope difference vs. Diffusion Planner.

**2.3 MISTAKES / CRITICISMS / LIMITATIONS** *[criticism reconstructed; partly from authors' own follow-up DiffusionDriveV2]*
- **Anchors are static.** They come from offline k-means and are not adapted online; richer/continuous anchor sets could give better coverage. Explicitly acknowledged in the DiffusionDriveV2 follow-up. (https://arxiv.org/abs/2512.07745 **[UNVERIFIED — 2026]**, https://github.com/hustvl/DiffusionDriveV2)
- **Mode collapse under IL.** "DiffusionDrive generates trajectories with excellent multimodality, yet constrained by imitation learning, it also produces numerous colliding ones as most negative modes lack supervision during training." So the multimodality is partly *uncalibrated* — diverse but not all safe. (paraphrased from DiffusionDriveV2 motivation; **[2ndary]**)
- **Evaluation scope.** NAVSIM PDMS/EPDMS only — not full reactive closed-loop. Truncation is an engineering acceleration, not evidence that diffusion's multimodality is what wins; the anchors may carry most of the multimodal signal.
- **Top-1 selection.** Evaluating only the top-1 scored mode means the reported metric does not directly reward multimodality — it rewards "best-mode" quality. This undercuts "diffusion handles multimodality" as the explanation for the score.

**2.4 The GAP**
They never disentangle "anchors give multimodality" from "diffusion refinement gives quality." A baseline of *anchors + deterministic regression refinement* (no diffusion) is the obvious missing control. They show diffusion-vs-vanilla-diffusion, not diffusion-vs-deterministic-given-the-same-anchors. (The TransfuserDP ablation in the paper goes the other direction — adds diffusion to Transfuser — see §9.)

---

## 3. MotionDiffuser (CVPR 2023) — *Controllable Multi-Agent Motion Prediction using Diffusion*

- arXiv: https://arxiv.org/abs/2306.03083
- CVF: https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.html
- Waymo: https://waymo.com/research/motiondiffuser-controllable-multi-agent-motion-prediction-using-diffusion/

**3.1 Recipe**
- **Task.** *Prediction* (not ego planning) — joint future distribution over multiple agents on Waymo Open Motion Dataset (WOMD).
- **Conditioning.** Scene context via an existing backbone (encoder-agnostic; "combined with existing backbone architectures"). Permutation-invariant over agents.
- **Denoiser.** Operates on a **PCA-compressed trajectory latent**, enabling exact sample log-prob and efficient computation. Simple predictor, **single L2 loss**, **no trajectory anchors**. (https://arxiv.org/abs/2306.03083)
- **Representation.** Compressed (PCA) joint trajectory representation.
- **Multimodality mechanism.** Free-form diffusion over the joint distribution + a **general constrained-sampling framework**: differentiable cost functions enforce rules/physical priors or craft scenarios at sampling time (controllability is the headline).
- **x0/eps & steps.** Not verified from accessible text. **[UNVERIFIED]**

**3.2 Results**
- SOTA multi-agent motion prediction on WOMD at publication (joint metrics). Specific minADE/minFDE/overlap numbers **[UNVERIFIED — full table behind PDF]**.

**3.3 CRITICISMS / LIMITATIONS** *[reconstructed]*
- It is a **predictor, not a closed-loop planner** — no ego control, no closed-loop simulation. Using it as a planner requires bolting on selection/guidance.
- PCA latent assumes trajectories live near a low-dim linear subspace; fine for short horizons, questionable for long-horizon or highly nonlinear maneuvers.
- Controllability via differentiable costs inherits the brittleness of hand-designed guidance (shared with Diffuser/Diffusion Planner).

**3.4 GAP**
No closed-loop evaluation; no isolation of when the joint multimodal model beats a marginal/deterministic predictor under matched context.

---

## 4. Diffuser (Janner et al., ICML 2022) — *Planning with Diffusion for Flexible Behavior Synthesis*

- arXiv: https://arxiv.org/abs/2205.09991
- PMLR: https://proceedings.mlr.press/v162/janner22a.html
- Code: https://github.com/jannerm/diffuser
- Project: https://diffusion-planning.github.io/

**4.1 Recipe**
- **Task.** Offline-RL / trajectory optimization (D4RL: Maze2D, locomotion). One diffusion model serves as **both world model and planner**.
- **Representation.** Trajectory as a 2D array — columns are state-action pairs across timesteps; the model denoises the whole trajectory block.
- **Denoiser.** **1D temporal U-Net**, ~6 repeated residual blocks (temporal convolutions). (https://arxiv.org/html/2401.02644v1 [secondary description])
- **Conditioning / planning.** **Classifier guidance with a learned reward/value gradient** steers sampling toward high-return trajectories; **inpainting** of start/goal states implements goal-conditioning and constraints. eps-prediction DDPM.
- **Steps.** Many denoising steps (hundreds, DDPM-era); this is the source of its latency reputation.
- **Multimodality mechanism.** Free-form trajectory diffusion; guidance composes constraints at test time ("flexible behavior synthesis").

**4.2 Results**
- Strong long-horizon / sparse-reward results (notably Maze2D) and test-time compositionality; competitive on D4RL locomotion. (Diffuser ≈ 75.3 avg locomotion as re-tabulated by Decision Diffuser, https://arxiv.org/html/2211.15657)

**4.3 CRITICISMS / LIMITATIONS** *[reconstructed; partly authors-acknowledged]*
- **Slow.** Hundreds of denoising steps → low planning frequency; this is the motivation for an entire follow-up line (DiffuserLite https://arxiv.org/html/2401.15443v5, hierarchical diffusion https://proceedings.iclr.cc/paper_files/paper/2024/file/46027e3de0db3617a911f1a647def3bf-Paper-Conference.pdf).
- **Reward-guidance brittleness / artifacts.** Reward-guided sampling can produce unreliable trajectories ("artifacts"), flagged as unsuitable for safety-critical deployment by follow-ups (e.g., RefiningDiffuser NeurIPS'23 https://proceedings.neurips.cc/paper_files/paper/2023/file/4c5722bad9759216474df8fc46c97af2-Paper-Conference.pdf).
- Underperforms when both a large receptive field and strong generalization are needed. **[2ndary]**

**4.4 GAP**
Diffuser argues multimodality/flexibility qualitatively (Maze2D pictures of diverse paths). It does not run a controlled "diffusion vs deterministic planner, conditioning held fixed, modality of the task varied" study. Maze2D *is* essentially a multimodal-goal showcase, but it is a demonstration, not an isolation.

---

## 5. Decision Diffuser (ICLR 2023 Oral) — *Is Conditional Generative Modeling all you need for Decision-Making?*

- arXiv: https://arxiv.org/abs/2211.15657 / HTML https://arxiv.org/html/2211.15657
- OpenReview: https://openreview.net/forum?id=sP1fo2K9DFG
- ICLR 2023 **Oral**: https://iclr.cc/virtual/2023/oral/12696
- Project: https://anuragajay.github.io/decision-diffuser/

**5.1 Recipe**
- **Task.** Offline RL via conditional generative modeling — *no* value function, *no* dynamic programming.
- **Representation.** Diffuses **state sequences only** (actions recovered via an inverse dynamics model) — a deliberate departure from Diffuser's state-action blocks.
- **Denoiser.** Temporal U-Net (Diffuser-style).
- **Conditioning.** **Classifier-free guidance** on **return** (return-conditioned), plus conditioning on constraints/skills. Low-temperature sampling at inference, hypothesized to implicitly do "DP-like" behavior selection. (https://arxiv.org/html/2211.15657)
- **Steps.** DDPM-style multi-step. **[exact count UNVERIFIED]**
- **Multimodality mechanism.** Conditional generation + classifier-free guidance; conditioning vector selects the behavior mode.

**5.2 Results [2ndary, from the paper's own tables]**
- D4RL locomotion avg **81.8**, beating CQL 77.6, IQL 77, DT 74.7, TT 78.9, MoReL 72.9, Diffuser 75.3. Larger gains on long-horizon Kitchen and Kuka block-stacking. (https://arxiv.org/html/2211.15657)

**5.3 CRITICISMS / LIMITATIONS** *[criticism reconstructed, not from verbatim reviews]*
- The provocative title ("all you need") invites the obvious rebuttal: it works on D4RL but inherits diffusion's slow sampling and offline-only nature; **online fine-tuning is left to future work** (authors' own framing).
- **Stitching debate.** Whether return-conditioned generation truly "stitches" sub-optimal trajectories the way DP does was contested; a NeurIPS'25 line (Generative Trajectory Stitching via Diffusion Composition, https://arxiv.org/abs/2503.05153) exists precisely because stitching in diffusion planners is non-trivial.
- Inverse-dynamics action recovery adds an error source absent in action-space diffusion.
- I could **not** read the literal ICLR reviews; the above is from the paper, the title's framing, and follow-up positioning. **[verify on OpenReview]**

**5.4 GAP**
Same structural gap: return-conditioning *is* a modality knob, but they vary it to maximize return, never to test "does the generative head beat a deterministic return-conditioned regressor when the task is unimodal vs multimodal."

---

## 6. Diffusion Policy (Chi et al., RSS 2023) — *Visuomotor Policy Learning via Action Diffusion*

- arXiv: https://arxiv.org/abs/2303.04137 / HTML https://arxiv.org/html/2303.04137v5
- IJRR version: https://journals.sagepub.com/doi/full/10.1177/02783649241273668
- Project: https://diffusion-policy.cs.columbia.edu/

**6.1 Recipe**
- **Task.** Visuomotor robot manipulation; receding-horizon **action chunking** (predict ~16 actions, execute ~8).
- **Conditioning.** Observation features condition the denoiser. Two variants:
  - **CNN/U-Net** with **FiLM** conditioning of obs features into every conv block (channel-wise).
  - **Transformer-DDPM**: noisy actions as input tokens, diffusion-step sinusoidal embedding as first token, obs as cross-attention context.
- **Representation.** Action sequence (chunk) diffusion.
- **x0/eps & steps.** **eps-prediction**; DDPM **K=100** for training-quality, **DDIM K≈10** at inference. (https://arxiv.org/html/2303.04137v5)
- **Multimodality mechanism.** Free-form action-distribution diffusion — the canonical demonstration that diffusion captures multimodal demonstrations (the Push-T multimodal-decision showcase).

**6.2 Results**
- Large average gains over prior IL/BC baselines (LSTM-GMM, IBC, BET) across simulated + real tasks; stable training, little task-specific tuning. (https://arxiv.org/html/2303.04137v5)

**6.3 CRITICISMS / LIMITATIONS** *[authors-acknowledged + community]*
- **Inference latency / reactivity.** Higher cost than LSTM-GMM; action chunking partially hides it but caps control rate and hurts highly-dynamic tasks. An entire 2025-26 line addresses this (Real-time Iteration Scheme https://arxiv.org/pdf/2508.05396; Real-Time Chunking flow policies https://arxiv.org/html/2506.07339; Delay-Aware DP https://www.arxiv.org/pdf/2512.07697 **[UNVERIFIED-2026]**).
- **Idle-time / horizon trade-off**: too-long horizon → sluggish reaction; too-short → loses multimodal consistency.
- Offline IL only; multimodality benefit is strongest when demonstrations are themselves multimodal — on single-mode tasks the advantage over a good regressor shrinks (widely observed, not rigorously isolated in the paper). **[2ndary]**

**6.4 GAP**
Push-T qualitatively shows multimodality helps at decision points, but the paper does not parametrically vary task multimodality to draw the boundary where diffusion stops beating a unimodal regressor.

---

## 7. 2025-2026 nuPlan / AD diffusion-planning landscape (newer work discovered)

All entries here are **[UNVERIFIED — title/abstract/search only]** unless noted; I could not open the PDFs. Treat IDs ≥ 2509 with extra caution.

- **CarPlanner (CVPR 2025)** — autoregressive RL planner; *first* RL planner to beat IL+rule SOTA on nuPlan. Notable because it is the **non-diffusion** competitor that questions whether diffusion's multimodality is the thing that matters, or whether RL + selection does just as well. https://arxiv.org/abs/2502.19908
- **Efficient Virtuoso (Sep 2025)** — latent **DiT**, PCA latent, **goal-conditioned** single-agent trajectories on WOMD; minADE 0.2541. **Crucially, it ablates endpoint-goal vs. multi-step sparse-route conditioning** and concludes a sparse route is needed for high-fidelity tactical execution while an endpoint resolves strategic ambiguity. This is the **closest published thing to a goal-modality ablation** I found. https://arxiv.org/abs/2509.03658
- **Discrete Contrastive Learning for Diffusion Policies in AD (Mar 2025)** — contrastive objective to fix uncalibrated/colliding modes. https://arxiv.org/html/2503.05229
- **BridgeDrive (Sep 2025)** — diffusion-*bridge* policy for closed-loop planning. https://arxiv.org/html/2509.23589
- **FlowDrive (Sep 2025)** — flow-matching + data balancing for trajectory planning. https://arxiv.org/html/2509.21961v2
- **DiffVLA (May 2025)** — vision-language-guided diffusion planning. https://arxiv.org/html/2505.19381v4
- **CoPlanner (Sep 2025)** — contingency-aware diffusion, interactive planning. https://arxiv.org/html/2509.17080
- **Drive As You Like (Aug 2025)** — multi-head diffusion, strategy-level planning. https://arxiv.org/pdf/2508.16947
- **DiffRefiner (Nov 2025)** — coarse-to-fine diffusion refinement E2E. https://arxiv.org/html/2511.17150v1
- **LAP: Fast Latent Diffusion Planner (Dec 2025)** — latency-focused. https://arxiv.org/pdf/2512.00470
- **PlannerRFT (Jan 2026)** — closed-loop, sample-efficient RL fine-tuning of diffusion planners. https://arxiv.org/html/2601.12901v1
- **ReflexDiffusion (Jan 2026)** — reflection-enhanced trajectory planning. https://arxiv.org/pdf/2601.09377
- **RAPiD (Feb 2026)** — real-time *deterministic* trajectory planning via diffusion behavior priors (note: "deterministic" framing — relevant to our axis). https://arxiv.org/pdf/2602.07339
- **DiffusionDriveV2 (Dec 2025)** — RL-constrained truncated diffusion; fixes DiffusionDrive's uncalibrated modes via GRPO. https://arxiv.org/abs/2512.07745
- **PC-Diffuser (Mar 2026)** — CBF safety-filtered diffusion planner. https://arxiv.org/html/2603.10330
- **Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop AD (Mar 2026, U-Tokyo/TIER IV)** — decomposes a diffusion planner into ONNX modules, native C++ DPM-Solver++, ROS2/Autoware integration. **Deployment/latency benchmark, NOT a mechanistic when-does-multimodality-help study.** https://arxiv.org/abs/2603.01023

---

## 8. Per-paper summary table

| Paper | Domain | Denoiser | Cond. | Multimodality mechanism | Eval regime | Isolates *when* multimodality wins? |
|---|---|---|---|---|---|---|
| Diffusion Planner '25 | nuPlan AD | Transformer/DiT | scene+route, classifier guidance | free-form joint diffusion + guidance | **closed-loop** nuPlan | No |
| DiffusionDrive '25 | NAVSIM E2E | Transformer cascade | BEV feats | **anchors** + truncated diffusion | NAVSIM PDMS (~open-loop) | No (anchors confound) |
| MotionDiffuser '23 | WOMD predict | (PCA latent) | scene | free-form joint + constrained sampling | open-loop prediction | No |
| Diffuser '22 | D4RL RL | 1D temporal U-Net | inpaint + reward guidance | free-form trajectory diffusion | offline RL | No (qualitative) |
| Decision Diffuser '23 | D4RL RL | temporal U-Net | **return**, classifier-free | conditional gen + CFG | offline RL | No |
| Diffusion Policy '23 | robot manip | U-Net / Transformer | obs (FiLM/cross-attn) | free-form action diffusion | sim+real IL | No (qualitative, Push-T) |
| Efficient Virtuoso '25 | WOMD plan | latent DiT | **goal vs route (ablated!)** | latent diffusion | open-loop minADE | **Partially** — goal-modality ablated, but no det. baseline |

---

## 9. CROSS-PAPER GAP ANALYSIS

### 9.1 The common unanswered question
Across **every** paper above, the argument structure is identical:

> "Human/agent behavior is multimodal → deterministic regression mode-averages → diffusion captures multimodality → [global benchmark win]."

The first three steps are asserted as a chain; the fourth is the only thing measured. **No paper in this set rigorously isolates *when* a multimodal/diffusion head beats a deterministic baseline by a controlled ablation that varies the conditioning's modality while holding everything else fixed.** What we have instead:

- **Global benchmark wins** (Diffusion Planner on nuPlan, DiffusionDrive on NAVSIM, Decision Diffuser on D4RL, Diffusion Policy on manip). These conflate architecture, capacity, conditioning, training recipe, and selection.
- **Confounded "multimodality" ablations.** DiffusionDrive's gain is entangled with anchors (the anchors may carry most of the multimodal signal; the top-1 selection metric doesn't even reward multimodality). Its TransfuserDP study *adds* diffusion to a deterministic planner and shows improvement — but does **not** vary task/goal modality, so it shows "diffusion ≥ MLP here," not "diffusion > MLP *because the situation is multimodal*."
- **Qualitative multimodality demos** (Diffuser Maze2D, Diffusion Policy Push-T). Pictures of diverse trajectories, not a controlled boundary.
- **The closest real attempt — Efficient Virtuoso (2509.03658)** — actually varies the *conditioning modality* (endpoint goal vs sparse route) and reports that conditioning richness changes fidelity. But (a) it's **open-loop** (minADE on WOMD), (b) single-agent prediction-style, and (c) it does **not cross** the goal-modality axis with a **deterministic-vs-diffusion** axis. So it answers "does richer conditioning help diffusion?" not "does diffusion's multimodality help *only when the goal is ambiguous*, and does it collapse to a regressor's performance once the goal is precise?"

So the precise question — **"as you tighten the conditioning from route-level (ambiguous, multimodal) to precise-goal (near-unimodal), does the diffusion planner's advantage over a matched deterministic MLP shrink to zero?"** — is, as far as I can verify, **unanswered** in the AD diffusion-planning literature as of June 2026.

### 9.2 Why has this controlled isolation been left behind?
Honest diagnosis, in order of explanatory weight:

1. **Benchmark culture rewards a single SOTA number, not a mechanism curve.** nuPlan/NAVSIM/WOMD leaderboards are scalar. A 2x2 mechanism study produces a *crossing curve* ("diffusion wins under route conditioning, ties under precise goal"), which does not slot into a "we got +X PDMS" headline. Reviewers at CVPR/ICLR reward the headline. The incentive gradient points away from mechanism.
2. **Incentive to attribute the win to the fashionable component.** "Diffusion handles multimodality" is the narrative that justifies the architecture. Running the ablation that might show "a deterministic MLP with the same conditioning matches us when the goal is precise" *weakens your own paper's framing*. There is a structural disincentive to run the experiment that could deflate your contribution.
3. **Conditioning modality is rarely a controllable knob in these benchmarks.** nuPlan gives you a route/mission; NAVSIM gives a navigation command; WOMD gives intent buckets. None ships a clean "precise-goal vs route-only" toggle, so you'd have to *construct* the conditioning variants yourself and re-train both a diffusion and a deterministic model under each — 4 training runs minimum, plus closed-loop eval (which is the expensive part on nuPlan). That's real compute, for a result that, per (1)-(2), doesn't help the leaderboard.
4. **Closed-loop eval cost.** The mechanism question is most interesting *closed-loop* (does multimodality matter when the sim reacts?), and closed-loop nuPlan eval is heavy. The cheap papers do open-loop; the open-loop papers can't see the closed-loop multimodality payoff. Efficient Virtuoso is open-loop precisely because that's tractable.
5. **Deterministic baselines are often "strawmanned."** When a deterministic baseline appears, it's frequently a *different, weaker* architecture (e.g., PDM-Open, a regression head) rather than the *same* model with the generative head swapped for an MLP. So even the comparisons that exist don't hold capacity/conditioning fixed.

Net: this is **not** a fundamental scientific obstacle — it's an incentive + cost + benchmark-design problem. The experiment is straightforward; the field's reward structure simply doesn't pay for it.

---

## 10. What I could NOT verify (explicit flags)
- **All OpenReview reviewer/rebuttal text** for Diffusion Planner (wM2sfVgMDH), DiffusionDrive (sh7vDLo5EY), Decision Diffuser (sP1fo2K9DFG). Forum URLs confirmed; **review bodies not extractable** with available tooling. Every §x.3 is "criticism reconstructed," not verbatim reviews.
- **Exact x0-vs-eps parameterization and exact diffusion-step / sampling-step counts** for Diffusion Planner and MotionDiffuser (HTML/abstract didn't state; check code/PDF).
- **Full numeric tables** (MotionDiffuser WOMD metrics; full Diffusion Planner table with refine/no-refine variants) — only partially re-tabulated from secondary search hits, marked **[2ndary]**.
- **Every 2026 follow-up** (PlannerRFT, ReflexDiffusion, RAPiD, PC-Diffuser, DiffusionDriveV2, LAP, the U-Tokyo modular benchmark) — title/abstract only; PDFs not opened. Do not cite their internal numbers without opening them.

---

## 11. Verdict — Is our 2×2 (precise-goal vs route-goal × MLP vs diffusion) novel?

**PARTIALLY — leaning toward genuinely novel for the closed-loop nuPlan setting.**

Reasoning:
- **What is NOT novel:** the *individual* axes. (a) Diffusion-vs-deterministic comparisons exist (DiffusionDrive's TransfuserDP; deterministic baselines like PDM-Open/PlanTF on nuPlan; RAPiD's "deterministic diffusion prior"). (b) Goal-vs-route conditioning has been ablated once — **Efficient Virtuoso (arXiv 2509.03658)** explicitly compares endpoint-goal vs sparse-route conditioning and finds conditioning richness matters. So neither axis alone is new.
- **What IS novel:** the **crossing of the two axes as a controlled mechanism test** — holding model capacity and scene encoder fixed, varying *only* (i) the head (deterministic MLP vs diffusion) and (ii) the conditioning modality (precise-goal ≈ unimodal vs route ≈ multimodal), to measure *whether the diffusion advantage is conditional on goal ambiguity*. I found **no paper** that runs this 2×2, and specifically none that does it **closed-loop on nuPlan**. Efficient Virtuoso has one of the two axes but is open-loop, single-agent, and never swaps in a deterministic head.
- **Why "partially" not "fully":** the conceptual hypothesis ("diffusion only helps when the situation is multimodal; tighten the goal and it collapses to a regressor") is *implicit folklore* in the field — it's the unstated null that everyone's narrative assumes but no one tests. Building a 2×2 to confirm/refute it is a clean, citable contribution, but it's an *isolation/confirmation* of a suspected effect rather than a brand-new phenomenon. That is exactly the kind of "mechanism over leaderboard" study §9.2 explains has been left behind — which is *why* it's worth doing.

**Closest prior work:** Efficient Virtuoso (https://arxiv.org/abs/2509.03658) — same goal-modality axis, but missing the deterministic-vs-diffusion crossing and the closed-loop nuPlan setting. Position our 2×2 explicitly against it: "Efficient Virtuoso varies conditioning modality for a diffusion model open-loop; we additionally vary the head (MLP vs diffusion) and evaluate closed-loop on nuPlan, isolating *when* multimodality is necessary rather than assuming it always is."

**Recommended framing for the paper:** lead with the mechanism question, not a SOTA claim. The contribution is the *crossing experiment and the boundary it draws*, not another point on the nuPlan leaderboard.

---

*Sources (primary URLs):*
- Diffusion Planner: https://arxiv.org/abs/2501.15564 · https://openreview.net/forum?id=wM2sfVgMDH · https://github.com/ZhengYinan-AIR/Diffusion-Planner
- DiffusionDrive: https://arxiv.org/abs/2411.15139 · https://openreview.net/forum?id=sh7vDLo5EY · https://github.com/hustvl/DiffusionDrive
- MotionDiffuser: https://arxiv.org/abs/2306.03083
- Diffuser: https://arxiv.org/abs/2205.09991 · https://proceedings.mlr.press/v162/janner22a.html
- Decision Diffuser: https://arxiv.org/abs/2211.15657 · https://openreview.net/forum?id=sP1fo2K9DFG · https://iclr.cc/virtual/2023/oral/12696
- Diffusion Policy: https://arxiv.org/abs/2303.04137 · https://diffusion-policy.cs.columbia.edu/
- Efficient Virtuoso: https://arxiv.org/abs/2509.03658
- CarPlanner: https://arxiv.org/abs/2502.19908
- Open-Source Modular Diffusion Benchmark: https://arxiv.org/abs/2603.01023
- PLUTO: https://arxiv.org/html/2404.14327v1 · Diffusion-ES: https://arxiv.org/pdf/2402.06559
