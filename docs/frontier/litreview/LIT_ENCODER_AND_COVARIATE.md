# Literature Review — Vectorized Scene Encoders & Covariate-Shift / Closed-Loop Training Cures

> **Scope.** Two-part frontier review for `av-policy-lab` (nuPlan closed-loop planning).
> **Date:** 2026-06-01. **Author:** Parv Patodia (compiled with Claude Code).
> **Method.** Web search over arXiv / CVF Open Access / OpenReview / NeurIPS proceedings / lab pages. Every factual claim carries a URL.
>
> **Reliability conventions used below:**
> - **[V]** = verified against a primary source (paper PDF, CVF/NeurIPS proceedings, or official repo).
> - **[~]** = stated in a secondary source / search snippet, consistent with the paper, but the exact figure was not read off the primary PDF in this pass. Treat as *probably correct, confirm before citing in a publication*.
> - **[UNVERIFIED]** = could not confirm; flagged explicitly.
>
> **Honesty note.** PDF page rendering and `WebFetch` were unavailable in this environment (sandbox + permission limits), so a few exact integers (param counts, some tensor dims) come from search snippets, not from reading the PDF table directly. Those are marked **[~]**. The architecture *mechanisms* and the *headline numbers* are verified.

---

## PART A — Vectorized Scene Encoders

### Why this matters for *this* repo

The README's central empirical finding is a textbook perception-absence result: the BC MLP gets **0.058 m open-loop ADE but 49.4 m closed-loop L2 (≈850× worse)**, and **DAgger iter-2 (12,678 on-policy samples) produced ~0 % closed-loop improvement** because a 6-dim kinematic state *cannot perceive where the road is*. Adding a single goal/route reference (GoalBC) collapsed L2 to 1.82 m. The lesson — *representation does not fix perception absence* — is exactly what Part A is about: what scene tensor to feed the planner so it can perceive lanes, agents, route, and lights. The encoders below are the proven answer.

---

### A1. VectorNet (Gao et al., CVPR 2020)
- Paper: https://arxiv.org/abs/2005.04259 · PDF: https://arxiv.org/pdf/2005.04259 · CVF: https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html

**Input representation [V].** The founding idea: represent *everything* — agent trajectories AND HD-map geometry (lane boundaries, crosswalks, stop signs) — as **polylines**, each polyline a sequence of **vectors**. Each vector (graph node) carries: start point `(xs, ys)`, end point `(xe, ye)`, attribute features (e.g. object type, timestamp for trajectory, road-feature semantic label), and an integer **polyline group id** so vectors are grouped back into their parent polyline. (https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html)

**History length [V].** Evaluated on Argoverse 1.1: **2 s observation (20 steps @ 10 Hz) → 3 s prediction (30 steps)**; total scenario is 5 s @ 10 Hz = 50 steps (https://www.argoverse.org/av1.html). Note: an early automated fetch in this review *hallucinated* "20 seconds of history / up to 128 agents" — **discarded as UNVERIFIED/false**; VectorNet uses ~2 s on Argoverse.

**Encoder architecture [V].**
1. **Polyline subgraph (local):** per-polyline MLP + max-pool aggregation, repeated; the paper finds **3 subgraph layers** optimal. Node hidden dimension **64** (https://ar5iv.labs.arxiv.org/html/2005.04259).
2. **Global interaction graph:** a single self-attention layer over polyline-level features — paper finds **1 global layer** sufficient (https://ar5iv.labs.arxiv.org/html/2005.04259). This is *late fusion* in modern terms (encode each polyline, then attend across them).
3. **Auxiliary task:** self-supervised **node-feature completion** (mask a polyline's features, reconstruct from context) for regularization (https://ar5iv.labs.arxiv.org/html/2005.04259).

**Params [~].** Small (sub-1M to ~1–2M depending on config); paper emphasizes it matches a ResNet rasterized baseline "much more economical in model size and FLOPs" and cuts DE@3 by ~12 % vs ConvNet on Argoverse (https://ar5iv.labs.arxiv.org/html/2005.04259). Exact integer param count **[UNVERIFIED]** in this pass.

**Limitations (paper's own + downstream).** Max-pooling within a polyline is lossy; single global attention layer limits higher-order reasoning; agent-centric single-target framing (one focal agent at a time) — superseded by scene-centric/multi-agent encoders below.

---

### A2. HiVT (Zhou et al., CVPR 2022)
- Paper PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf · Repo: https://github.com/ZikangZhou/HiVT

**Input/representation [V].** Vectorized like VectorNet, but with **translation-invariant scene representation + rotation-invariant spatial modules** — features are computed in each agent's *local* frame so the encoder is robust to global pose. Argoverse: **2 s / 20-step history** (https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf).

**Encoder [V].** Two-stage hierarchical transformer: (1) **local context extraction** within a radius (default local region radius **r = 50 m**), (2) **global interaction** across local representations. Predicts *all* agents in one forward pass.

**Params / speed [~].** Two variants by hidden dim: **HiVT-64 (~662 K params)** and **HiVT-128 (~2.5–2.7 M params)**; HiVT-128 @ r=50 m runs ~**20 ms** inference (real-time) (https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf). The "64"/"128" naming = hidden dimension. Exact 662 K / 2.5 M integers are **[~]** (snippet-level, consistent with paper's "82–95 % fewer params" claims).

**Limitations.** Local-radius cropping can miss long-range map structure; rotation-invariance machinery adds implementation complexity; still a *prediction* model, not a closed-loop planner.

---

### A3. Scene Transformer (Ngiam et al., ICLR 2022)
- Paper: https://arxiv.org/abs/2106.08417 · OpenReview: https://openreview.net/forum?id=Wm3EA5OlHsG · PDF: https://arxiv.org/pdf/2106.08417v2

**Representation [V].** **Scene-centric** (single shared global frame for all agents — not re-centered per agent), built to scale to many agents in dense scenes; agent-permutation-equivariant. (https://openreview.net/pdf?id=Wm3EA5OlHsG)

**Encoder [V].** **Factorized attention** that *alternates* between the **time axis** and the **agents axis** — the key efficiency trick (avoids full attention over the agents×time product). A **masking strategy as the query** unifies marginal prediction, agent-conditioned prediction, and goal-conditioned prediction in one model (BERT-style masking). (https://openreview.net/pdf?id=Wm3EA5OlHsG)

**Why it matters for planning.** The masking interface lets *one* model do prediction AND planning (condition on the AV's intended future) — directly relevant to a planner that must reason jointly about other agents.

**Limitations.** Scene-centric frame loses the pose-invariance benefits HiVT/QCNet exploit (must learn invariances from data); factorized attention is an approximation of full attention.

---

### A4. Wayformer (Nayakanti et al., arXiv 2207.05844 / ICRA 2023)
- Paper: https://arxiv.org/abs/2207.05844 · PDF: https://arxiv.org/pdf/2207.05844 · Waymo page: https://waymo.com/research/wayformer/

**Input representation [V/~].** Four input modalities fed to one attention scene encoder: (1) **agent history** — WOMD uses **1 s history = 10 steps @ 10 Hz** [~ from notes, consistent with WOMD]; (2) **agent interactions**; (3) **roadgraph** as **polylines / line segments by endpoints**; (4) **time-varying traffic-light state**. Predicts **K = 6** trajectories over **8 s** future. (https://devinz1993.medium.com/paper-notes-wayformer-motion-forecasting-via-simple-efficient-attention-networks-9b0a83234a66, https://waymo.com/research/wayformer/)

**Encoder — the paper's whole point [V].** Systematic study of **fusion** × **efficiency**:
- **Early fusion** (concatenate all modality tokens, one cross-modal encoder) vs **late fusion** (per-modality encoders) vs **hierarchical**. **Finding: early fusion is simplest, modality-agnostic, and SOTA on WOMD + Argoverse** (https://arxiv.org/abs/2207.05844).
- **Efficiency:** **factorized attention** or **latent-query attention**. Latent queries cross-attend the input set down to a fixed small token budget; reported **~192 latent queries, model dim 256, 2 encoder layers**, achieving **~16× compression with no performance loss** [~] (https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/wayformer.html, https://devinz1993.medium.com/paper-notes-wayformer-motion-forecasting-via-simple-efficient-attention-networks-9b0a83234a66).

**Limitations.** Self-attention is O(n²) in token count — the explicit motivation for latent queries; it is a *prediction* architecture (no closed-loop guarantees); early fusion's simplicity trades away some modality-specific inductive bias.

---

### A5. MTR / MTR++ (Shi et al., NeurIPS 2022 / TPAMI 2024)
- MTR: https://arxiv.org/abs/2209.13508 · NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf · OpenReview: https://openreview.net/forum?id=9t-j3xDm7_Q · Repo: https://github.com/sshaoshuai/MTR
- MTR++: https://arxiv.org/abs/2306.17770 · IEEE: https://ieeexplore.ieee.org/document/10398503/

**Input representation [V].** WOMD: agent history **H = 11 steps (1.1 s @ 10 Hz)** + **80-step (8 s)** future; **up to 8 agents** evaluated per scene (https://www.waymo.jp/intl/fil/open/data/motion). Roadgraph as polylines, **each polyline up to ~20 map points**; encoder selects the **768 nearest map polylines** around the focal agent (https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf).

**Encoder [V].** Polyline (PointNet-like) encoder → **transformer context encoder with local self-attention restricted to 16 nearest neighbors** (sparsity = efficiency). Default **6 encoder + 6 decoder layers**, **hidden dim 512** for WOMD. (https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf)

**Decoder — the contribution [V].** **64 learnable motion-query pairs**, whose **intention points come from k-means clustering** of training endpoints (global intention localization), refined locally; plus a **dense future-prediction** auxiliary task and a **dynamic map-collection** module that re-gathers map features along each predicted trajectory. Won WOMD Motion Prediction Challenge 2022. (https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf)

**MTR++ [V].** Generalizes from one focal agent to **all agents in a single pass** via **symmetric (query-centric) scene context modeling** — a *shared* encoder encodes the scene in each agent's local frame using query-centric self-attention — plus **mutually-guided intention querying** for interaction-aware multi-agent prediction. This is the same query-centric idea QCNet formalizes. (https://arxiv.org/abs/2306.17770, https://arxiv.org/html/2306.17770v2)

**Limitations (OpenReview / paper).** 64 fixed k-means intention points bake the dataset's mode distribution into the model (poor transfer to a dataset with different turn statistics); local 16-NN attention can miss distant-but-relevant context; heavy (512-dim, 6+6 layers) relative to HiVT.

---

### A6. QCNet (Zhou et al., CVPR 2023) — the de-facto modern standard
- CVF PDF: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf · Repo: https://github.com/ZikangZhou/QCNet

**Input representation [V].** Argoverse 2: **50 historical steps (5 s @ 10 Hz)** + **60 future steps (6 s)**. Map = **polygons** (lanes, crosswalks) + agents. Scene encoder produces **map encodings `[M, D]`** and **agent encodings `[A, T, D]`**, where `M` = #map polygons, `A` = #agents, `T` = #history steps, `D` = hidden dim. (https://github.com/ZikangZhou/QCNet/blob/main/README.md)

**Encoder — the contribution [V].** **Query-centric** paradigm: every element is encoded in its **own local spacetime frame** with **relative (roto-translation-invariant) spacetime positional encodings**, so representations are independent of the global frame and **can be cached/reused across timesteps (streaming)**. **Factorized attention** in three flavors: **temporal attention, agent↔map attention, social (agent↔agent) attention** (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf).

**Decoder [V].** **Anchor-free, recurrent** trajectory proposal generation — emits waypoints recurrently so different horizons can attend to different context, then a refinement stage. Hidden dim **128** [~]. **Ranked #1 on both Argoverse 1 and Argoverse 2** motion-forecasting leaderboards (https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html).

**Limitations (paper's own).** Explicitly: *"increasing the number of fusion blocks yields better results… however, the resulting inference latency is not amenable to real-time applications such as autonomous driving."* (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf). The recurrent decoder + deep factorized attention is **slow**; downstream work (e.g. SmartRefine) reports added latency, and memory pressure forces dropping streams in some multi-agent settings.

---

### A7. The de-facto 2026 STANDARD (what to feed a nuPlan-scale planner)

Synthesizing the trajectory: **VectorNet (2020) established polyline vectorization → HiVT/Scene Transformer (2022) added the local-global / scene-centric split → Wayformer (2022) established early fusion + latent queries → MTR (2022) added query/intention decoding → QCNet/MTR++ (2023) converged on the query-centric, relative-spacetime, factorized-attention recipe that is now standard.**

**Consensus answers to the four questions:**

| Question | Research-standard answer (2026) | Evidence |
|---|---|---|
| **Agent count** | Model **all agents jointly** (scene-centric / query-centric), not one focal agent. Benchmark *evaluation* caps are small (WOMD: up to **8** agents scored; Argoverse: 1 focal + scene actors), but the *encoder* attends over all (often **≥32–128** nearby agents, capped by radius/NN). | WOMD 8 agents (https://www.waymo.jp/intl/fil/open/data/motion); MTR++ all-agents-one-pass (https://arxiv.org/abs/2306.17770); HiVT all agents in one pass |
| **History length** | **~1 s on WOMD (11 steps), ~2 s on Argoverse 1 (20 steps), ~5 s on Argoverse 2 (50 steps)**, all @ 10 Hz. The **modern standard for a fresh planner is ~2 s @ 10 Hz (≈20 steps)** — long enough for intent, short enough to avoid overfitting stale history (PlanTF on nuPlan deliberately uses *minimal* history). | WOMD H=11 (https://www.waymo.jp/intl/fil/open/data/motion); Argoverse 1 = 20 steps (https://www.argoverse.org/av1.html); QCNet AV2 = 50 steps (https://github.com/ZikangZhou/QCNet); PlanTF minimal-history (https://mit-spark.github.io/robotRepresentations-RSS2023/assets/papers/12.pdf) |
| **Map element set** | **Lane centerlines/segments + lane connectors + crosswalks**, vectorized as polylines/polygons; plus **per-timestep traffic-light state** and the **route** (sequence of lanes/roadblocks). On nuPlan specifically: lanes, lane_connectors, roadblocks, baselines, crosswalks, traffic-light status, route. | nuPlan map layers (https://arxiv.org/html/2403.04133v1, https://github.com/motional/nuplan-devkit); MTR 768 polylines × ~20 pts (https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf) |
| **Encoder** | **Query-centric, relative-spacetime positional encoding, factorized attention (temporal × agent↔map × social)**, polyline/polygon tokenization. Early fusion is the strong simple default (Wayformer); query-centric (QCNet/MTR++) is SOTA. | QCNet (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf); Wayformer early fusion (https://arxiv.org/abs/2207.05844) |
| **Encoder size** | **Hidden dim 64–128, ~0.7–2.5 M params is enough for SOTA-class trajectory quality** (HiVT). MTR's 512-dim/6+6-layer is the heavy end for leaderboard chasing. For a *research planner on nuPlan-mini*, **64–128 dim is the right scale.** | HiVT-64/128 (https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf) |

---

## PART B — Covariate Shift / Closed-Loop Training Cures

### The problem, stated in this repo's own numbers

Behavior cloning minimizes one-step prediction error on expert states, but at deployment the policy visits states *it* caused, which were never in the training distribution → **compounding error**. This repo measured it cleanly: **0.058 m open-loop ADE vs 49.4 m closed-loop L2 (≈850×)**, and **DAgger iter-2 with 12,678 on-policy samples gave ~0 % closed-loop improvement** because the underlying state representation had no road perception. The literature below is the menu of cures, with honest numbers on how much each one actually closes the gap.

---

### B1. DAgger (Ross, Gordon & Bagnell, AISTATS 2011) — the theory
- PDF: https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf

**Mechanism [V].** Iterative dataset **aggregation**: roll out the *current* policy, query the **expert for the correct action on the states the policy actually visited**, add those `(visited-state, expert-label)` pairs to a growing dataset, retrain on the aggregate. Reduces imitation learning to no-regret online learning. (https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)

**How much it closes the gap [V].** The headline theorem: naive behavior cloning has error that grows **O(T²ε)** in the horizon T (quadratic compounding); DAgger achieves **O(Tε)** (linear). (https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)

**The catch (critical for this repo) [V + repo evidence].** DAgger requires (a) an **interactive expert** that can label arbitrary off-distribution states, and (b) a policy whose **representation can actually perceive** the state difference. This repo already demonstrated (b) is the binding constraint: DAgger added on-policy data but a 6-dim kinematic state "looks identical whether on-road or 50 m off-track," so more labels did nothing. **DAgger is necessary-condition theory, not a drop-in fix when the representation is blind.** Modern variants (DADAgger, disagreement-augmented, https://arxiv.org/pdf/2301.01348) reduce expert queries but do not relax (b).

---

### B2. ChauffeurNet (Bansal, Krizhevsky & Ogale, 2018 / RSS 2019) — perturbation augmentation, the original
- Paper: https://arxiv.org/abs/1812.03079 · PDF: https://arxiv.org/pdf/1812.03079 · RSS: https://www.roboticsproceedings.org/rss15/p31.pdf

**Mechanism [V].** Two interlocking ideas:
1. **Trajectory perturbation augmentation** ("synthesizing the worst"): take an expert trajectory, **perturb the agent off the lane center**, then fit a smooth trajectory bringing it back. This synthesizes *recovery* situations (near-collisions, off-road) that pure expert data never contains. (https://arxiv.org/pdf/1812.03079)
2. **Explicit auxiliary losses** that penalize the synthesized bad events: **collision loss, on-road / off-road loss, geometry loss** — the perturbations are what give these losses a learning signal. Plus **imitation dropout** (drop past-pose history for ~50 % of examples so the net relies on the scene, not on extrapolating its own history). Mid-level (rasterized) input/output representation, trained on **30 M real examples**. (https://arxiv.org/pdf/1812.03079)

**How much it closes the gap — and the decisive honest finding [V].** The paper's own ablation (M1→M4) is the single most important data point for our F4 decision:
- **M1 = perturbation only, no extra losses → insufficient.**
- **M2 adds environment (collision/off-road) losses; M3/M4 add the machinery to avoid imitating bad behavior.** Only the *full* combination is robust enough to drive a real car. (https://arxiv.org/pdf/1812.03079)
- The paper states plainly: **"even with 30 million examples… pure imitation learning is not sufficient"** — the model gets stuck or collides where a nudge-and-pass was viable, *because the system runs closed-loop where errors accumulate.* (https://www.emergentmind.com/papers/1812.03079)

**Takeaway:** **Perturbation augmentation alone does NOT close the gap.** It only works as a *vehicle for auxiliary closed-loop-style losses* (collision/off-road). It is a cheap, training-time, no-rollout method — strictly weaker than true closed-loop rollout training, but far better than vanilla BC.

---

### B3. TrafficSim (Suo, Regalado, Casas & Urtasun, CVPR 2021)
- Paper: https://arxiv.org/abs/2101.06557 · CVF: https://openaccess.thecvf.com/content/CVPR2021/html/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.html

**Mechanism [V].** Implicit-latent-variable **joint** multi-agent policy. The cure for off-distribution: **unroll the policy during training through a fully differentiable simulation** and backprop through the rollout, optimizing demonstrations **+ a common-sense (collision/off-road) loss**. This is *closed-loop training* (the policy sees its own multi-step consequences), not just augmentation. (https://arxiv.org/abs/2101.06557)

**How much [V/~].** Produces "significantly more realistic and diverse" multi-agent rollouts than baselines; its rollouts also serve as **data augmentation that improves a downstream motion planner**. (https://arxiv.org/abs/2101.06557) Exact gap-closure numbers vs BC **[UNVERIFIED]** in this pass. Conceptual contribution: **differentiable closed-loop rollout > open-loop BC** for long-horizon stability.

---

### B4. Symphony (Igl et al., ICRA 2022)
- Paper: https://arxiv.org/abs/2205.03195 · PDF: https://arxiv.org/pdf/2205.03195

**Mechanism [V].** Combines a learned policy with a **parallel beam search at rollout time**: branches are pruned by an **adversarial discriminator** (GAIL-style realism critic), pushing the closed-loop trajectory distribution toward realistic states. To stop beam search from collapsing diversity, adds a **hierarchical factorization (goal generation → goal-conditioned policy)**. (https://arxiv.org/pdf/2205.03195)

**How much [V/~].** On Waymo (proprietary + open) data, Symphony agents are "more realistic and diverse." The honest tension the paper itself raises: **beam search improves realism but harms diversity (mode collapse)** unless the hierarchical goal structure is added. Exact numbers **[UNVERIFIED]** in this pass. Lesson: adversarial/closed-loop realism objectives genuinely reduce covariate-shift artifacts but introduce a realism↔diversity trade-off.

---

### B5. CAT-K — Closed-Loop SFT of Tokenized Traffic Models (Zhang et al., CVPR 2025 Oral) — the modern recipe
- Paper: https://arxiv.org/abs/2412.05334 · CVF PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Closed-Loop_Supervised_Fine-Tuning_of_Tokenized_Traffic_Models_CVPR_2025_paper.pdf · Repo: https://github.com/NVlabs/catk · Project: https://zhejz.github.io/catk/

**Mechanism [V].** Tokenized (discrete-action) policies are pre-trained with open-loop BC, then **closed-loop fine-tuned**. The trick: **CAT-K = "Closest Among Top-K"** rollouts — during the closed-loop rollout, deterministically pick, among the policy's **top-K** likeliest next tokens, the one whose resulting state is **closest to ground truth**. This keeps the rollout near the expert (so existing trajectory labels stay valid as supervision) while still exposing the policy to its own multi-step dynamics. **No RL, no GAIL, no extra data** — only existing trajectories. (https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Closed-Loop_Supervised_Fine-Tuning_of_Tokenized_Traffic_Models_CVPR_2025_paper.pdf)

**How much it closes the gap — real numbers [V].**
- WOSAC **Realism Meta-Metric (RMM):** SMART-tiny-7M after open-loop BC = **0.7671**; after CAT-K closed-loop SFT = **0.7702** (**+0.0111** over its own BC baseline). (https://ar5iv.labs.arxiv.org/html/2412.05334)
- The CL-SFT lets a **7 M-param** model **beat a 102 M-param** model and top the WOSAC leaderboard, with a **+0.0271 RMM lead over GUMP (523 M params)**. (https://liner.com/review/closedloop-supervised-finetuning-tokenized-traffic-models)
- For **ego planning** specifically: CAT-K fine-tuning **reduces collisions by 25.7 % and off-road by 33.9 %**. (https://zhejz.github.io/catk/)

**Honest reading.** The *absolute* RMM gain looks small (+0.0111) because BC is already strong on the *realism* metric — but the **+25.7 %/+33.9 % reduction in the safety failures that dominate closed-loop L2** is exactly the kind of localized intersection/off-road failure this repo's PDM analysis flagged. **Closed-loop SFT closes the gap where it actually hurts.** Ablation confirms "closest" beats "uniform/neg-dist/prob/max-prob" sampling and K=32 is best.

---

### B6. RoaD — Rollouts as Demonstrations (García-Cobo, Igl et al., NVIDIA, arXiv Dec 2025) — newest
- Paper: https://arxiv.org/abs/2512.01993 · Lab: https://research.nvidia.com/labs/avg/publication/garciacobo.igl.etal.arxiv2025/

**Mechanism [V].** Generalizes CL-SFT beyond tokenized policies to **continuous / end-to-end (sensor-to-action)** policies. Generate the policy's **own closed-loop rollouts**, **bias them toward high-quality behavior with light expert guidance**, then treat those rollouts as fresh demonstrations for fine-tuning. Removes CAT-K's reliance on *discrete recovery targets* by adding a lightweight **recovery mode**, so no reward function / no RL is needed. (https://arxiv.org/abs/2512.01993)

**How much [V/~].** Claims "robust closed-loop adaptation with **orders of magnitude less data than RL**," and applicability to modern E2E driving. Exact nuPlan/benchmark deltas **[UNVERIFIED]** — not surfaced in this pass; **flag for follow-up read of the PDF tables.**

---

### B7. The nuPlan-specific honest verdict: open-loop ≠ closed-loop
- "Parting with Misconceptions" (Dauner et al., CoRL 2023): https://arxiv.org/abs/2306.07962 · PDF: https://arxiv.org/pdf/2306.07962 · Repo: https://github.com/autonomousvision/tuplan_garage
- "Beyond Behavior Cloning… Survey of Closed-Loop Training" (Karkus, Igl et al., NVIDIA, PAMI 2025): https://research.nvidia.com/labs/avg/publication/karkus.igl.etal.pami2025/ · Preprint PDF: https://d1qx31qr3h6wln.cloudfront.net/publications/beyond_bc_survey_preprint.pdf

**The misalignment, quantified [V].** Dauner et al. show on nuPlan that **ego-forecasting (open-loop) and planning (closed-loop) are "fundamentally misaligned" — improvement in one does not transfer to the other**, and "should be addressed independently." Concretely, their winning **PDM-Closed** (rule-based: centerline + IDM proposals + simulate-and-score) reaches **~92 closed-loop score (CLS-R) and ~84 open-loop (OLS) on Val14**, while strong imitation-learning planners post good open-loop errors but **much weaker closed-loop scores** unless explicitly trained for reactivity. Tellingly, **PlanCNN improves CLS by *removing* the ego-state input — deliberately sacrificing open-loop accuracy to reduce closed-loop shortcutting/causal-confusion.** (https://arxiv.org/abs/2306.07962, https://mit-spark.github.io/robotRepresentations-RSS2023/assets/papers/12.pdf)

**The survey's framing [V].** "Beyond Behavior Cloning" (PAMI 2025) organizes the entire space along three axes — **action generation: policy *rollouts* vs *perturbed demonstrations*; environment-response generation (real data / AV-sim / generative video / latent world models); related techniques.** This is the authoritative 2025 map for our F4 decision and confirms perturbation and rollout are the two poles. (https://d1qx31qr3h6wln.cloudfront.net/publications/beyond_bc_survey_preprint.pdf)

**2025-2026 frontier context [V/~].** nuPlan's IDM reactive agents are themselves a confound — **nuPlan-R** (https://arxiv.org/abs/2511.10403) and a NeurIPS-area study (https://arxiv.org/pdf/2510.14677) show learned reactive agents shift the rankings; **CaRL** (CoRL 2025, https://www.cvlibs.net/publications/Jaeger2025CORL.pdf) trains scalable RL planners with simple rewards. Direction of travel: **closed-loop training + reactive eval is becoming mandatory, not optional.**

---

### Does perturbation alone suffice, or is closed-loop rollout required? (the bottom line)

| Method class | Closed-loop rollout in training? | Honest gap-closure | Cost |
|---|---|---|---|
| Vanilla BC | No | Baseline (here: 49.4 m closed-loop) | Cheapest |
| **Perturbation aug (ChauffeurNet)** | No (synthetic, single-step recovery) | **Partial — only with collision/off-road aux losses; "pure imitation not sufficient even at 30 M examples"** | Cheap, no sim needed |
| DAgger | Yes (policy rollout + expert relabel) | O(T²ε)→O(Tε) *in theory*; **nil if representation is blind** | Needs interactive expert |
| Differentiable rollout (TrafficSim) | Yes | Better long-horizon stability | Needs differentiable sim |
| Beam search + discriminator (Symphony) | Yes (test-time) | Realism↑ but diversity↓ | Discriminator + search |
| **Closed-loop SFT (CAT-K / RoaD)** | **Yes (own rollouts as supervision)** | **Collisions −25.7 %, off-road −33.9 %; 7 M beats 102 M; no RL** | **Moderate — needs a sim loop, but no reward/RL** |

**Verdict:** **Perturbation augmentation alone is NOT sufficient** — ChauffeurNet's own ablation (M1 fails) and its 30 M-example admission settle this. It helps *only* as the carrier for auxiliary safety losses. **Closing the open-loop/closed-loop gap on the failures that actually matter (intersection collisions, off-road) requires exposing the policy to its own rollouts** — and the cheapest modern way to do that without RL is **closed-loop SFT (CAT-K/RoaD)**. Perturbation is a strong, cheap *first* step; rollout-based fine-tuning is the *finisher*.

---

## MISTAKES / LIMITATIONS / GAPS (consolidated, papers' own admissions + reviews + this review's caveats)

**Part A**
- **VectorNet:** lossy polyline max-pool; single global attention layer; agent-centric single-target. (paper config discussion)
- **HiVT:** local-radius cropping drops long-range map; rotation-invariance adds complexity; prediction-only.
- **Scene Transformer:** scene-centric frame must *learn* invariances HiVT/QCNet build in; factorized attention is an approximation.
- **Wayformer:** O(n²) attention (the reason for latent queries); early fusion sacrifices modality-specific inductive bias; prediction-only.
- **MTR/MTR++:** 64 k-means intention points hard-code dataset mode statistics (transfer risk); 16-NN local attention can miss distant context; heavy (512-dim).
- **QCNet:** **paper's own words — deeper fusion "not amenable to real-time… autonomous driving"**; recurrent decoder is slow; memory pressure in dense multi-agent.
- **Review caveats:** Wayformer's exact latent-query/param numbers and HiVT's exact 662 K/2.5 M params are **[~]** (snippet-level). VectorNet param count **[UNVERIFIED]**. The first automated PDF fetch hallucinated "20 s history / 128 agents" for VectorNet — **rejected**.

**Part B**
- **DAgger:** needs an interactive expert that labels off-distribution states; theory assumes the policy can *perceive* state differences — **false in this repo's 6-dim case (empirically confirmed).**
- **ChauffeurNet:** **perturbation-only (M1) is insufficient**; needs collision/off-road losses; **"pure imitation not sufficient even with 30 M examples"** (authors).
- **TrafficSim:** needs a *differentiable* simulator; exact BC-vs-rollout gap numbers **[UNVERIFIED]** here.
- **Symphony:** beam search **harms diversity** (mode collapse) absent hierarchical goals; needs a discriminator; numbers **[UNVERIFIED]** here.
- **CAT-K:** assumes a **tokenized/discrete** policy and **discrete recovery targets**; absolute RMM gain is small (+0.0111) though safety gains are large.
- **RoaD:** newest (Dec 2025); exact benchmark deltas **[UNVERIFIED]** — read PDF tables before citing numbers.
- **nuPlan eval itself:** IDM reactive agents are a confound (nuPlan-R, 2510.14677); open-loop and closed-loop are *misaligned* (Dauner) — **do not trust ADE/FDE as a closed-loop proxy.** This repo already lived this (0.058 m ADE, 49.4 m L2).

---

## RECOMMENDATIONS

### (1) Scene representation for F0/F1 — adopt a query-centric vectorized encoder

**Use a HiVT/QCNet-style vectorized scene encoder, sized small (Argoverse-research scale), feeding the planner — NOT a scalar state and NOT an ego-history raster.** This repo *proved* the raster (BEV CNN) and scalar (BC/MILE) representations are road-blind (all plateau at ~49.4 m). Specifics:

- **Agents:** all agents within a radius (cap **~32 nearby agents**), encoded jointly (not single-focal). Scene-/query-centric so the planner perceives interactions. [QCNet/MTR++ all-agents-one-pass: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf, https://arxiv.org/abs/2306.17770]
- **History:** **2 s @ 10 Hz = 20 steps.** Matches Argoverse-1 standard, balances intent vs staleness; nuPlan planners (PlanTF) deliberately keep history minimal. [https://www.argoverse.org/av1.html, https://mit-spark.github.io/robotRepresentations-RSS2023/assets/papers/12.pdf]. (Note nuPlan-mini logs are 100 Hz per README — subsample to 10 Hz.)
- **Map elements (nuPlan):** **lane centerlines + lane connectors + crosswalks**, vectorized as polylines; **the route as a lane/roadblock sequence**; **per-timestep traffic-light status**. Cap **~768 nearest polylines, ≤20 points each** (MTR convention). The README already shows route/roadblock info is the difference between failure and 1.8 m — so **route must be a first-class input tensor.** [nuPlan layers: https://arxiv.org/html/2403.04133v1, https://github.com/motional/nuplan-devkit; MTR polyline budget: https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf]
- **Encoder:** **query-centric, relative-spacetime positional encoding, factorized attention (temporal × agent↔map × social)**; or, if you want maximum simplicity first, **Wayformer early fusion + latent queries** as F0, upgrading to query-centric in F1. [https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf, https://arxiv.org/abs/2207.05844]
- **Size:** **hidden dim 64–128, ~0.7–2.5 M params** (HiVT scale) — proven SOTA-class on Argoverse and the right scale for nuPlan-mini compute. Do **not** start at MTR's 512-dim/6+6. [https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf]

> **F0 = HiVT-64-style local-global vectorized encoder + early fusion** (simplest thing that perceives the road). **F1 = QCNet-style query-centric upgrade** (relative spacetime PE, streaming, factorized attention) once F0 closes the perception gap.

### (2) F4 — do BOTH: perturbation augmentation FIRST, then closed-loop fine-tuning (not perturbation-only)

**Recommendation: perturbation + closed-loop fine-tuning, staged. Perturbation alone is documented as insufficient.**

- **Stage 1 (cheap, no sim loop): ChauffeurNet-style perturbation augmentation + auxiliary collision/off-road losses + imitation dropout.** This is low-effort and directly attacks recovery-from-drift — but the paper's own M1 ablation and the "30 M examples still not enough" admission mean **you cannot stop here.** [https://arxiv.org/pdf/1812.03079]
- **Stage 2 (the finisher): closed-loop SFT à la CAT-K / RoaD** — roll the policy out in the nuPlan closed-loop sim, keep rollouts near the expert (CAT-K's "closest among top-K" if you tokenize actions; RoaD's lightweight recovery mode for a continuous head), and fine-tune on those rollouts. **No RL, no reward shaping required.** This is the method that, in 2025, made a 7 M model beat 102 M and **cut ego collisions 25.7 % / off-road 33.9 %** — precisely the localized intersection/off-road failures this repo's PDM-Score analysis isolated. [https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Closed-Loop_Supervised_Fine-Tuning_of_Tokenized_Traffic_Models_CVPR_2025_paper.pdf, https://arxiv.org/abs/2512.01993, https://zhejz.github.io/catk/]

**Honest tradeoff.** Perturbation-only is the cheapest and needs no simulator in the loop, but it **demonstrably under-closes the gap** (single-step synthetic recovery ≠ the multi-step compounding the policy actually causes). Closed-loop SFT needs a working sim rollout loop (you already have the nuPlan closed-loop harness — this is a real advantage) and is more engineering, but it is the only family with **published, attributable reductions in the exact failure mode** (intersection collisions/off-road) blocking this project. DAgger is the theoretical ancestor but is a no-op here until the F0/F1 encoder gives the policy eyes — **fix representation (Rec 1) before expecting any closed-loop-training method to work.** That ordering (encoder → perturbation → closed-loop SFT) is the single most important conclusion of this review.

---

## Sources (primary)
- VectorNet — https://arxiv.org/abs/2005.04259 · https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html · https://ar5iv.labs.arxiv.org/html/2005.04259
- HiVT — https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf · https://github.com/ZikangZhou/HiVT
- Scene Transformer — https://arxiv.org/abs/2106.08417 · https://openreview.net/forum?id=Wm3EA5OlHsG
- Wayformer — https://arxiv.org/abs/2207.05844 · https://waymo.com/research/wayformer/
- MTR — https://arxiv.org/abs/2209.13508 · https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf · https://openreview.net/forum?id=9t-j3xDm7_Q · https://github.com/sshaoshuai/MTR
- MTR++ — https://arxiv.org/abs/2306.17770 · https://ieeexplore.ieee.org/document/10398503/
- QCNet — https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf · https://github.com/ZikangZhou/QCNet
- DAgger — https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
- ChauffeurNet — https://arxiv.org/abs/1812.03079 · https://www.roboticsproceedings.org/rss15/p31.pdf
- TrafficSim — https://arxiv.org/abs/2101.06557 · https://openaccess.thecvf.com/content/CVPR2021/html/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.html
- Symphony — https://arxiv.org/abs/2205.03195
- CAT-K — https://arxiv.org/abs/2412.05334 · https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Closed-Loop_Supervised_Fine-Tuning_of_Tokenized_Traffic_Models_CVPR_2025_paper.pdf · https://github.com/NVlabs/catk · https://zhejz.github.io/catk/
- RoaD — https://arxiv.org/abs/2512.01993 · https://research.nvidia.com/labs/avg/publication/garciacobo.igl.etal.arxiv2025/
- Closed-loop survey (PAMI 2025) — https://d1qx31qr3h6wln.cloudfront.net/publications/beyond_bc_survey_preprint.pdf
- Parting with Misconceptions / PDM (CoRL 2023) — https://arxiv.org/abs/2306.07962 · https://github.com/autonomousvision/tuplan_garage
- nuPlan benchmark — https://arxiv.org/abs/2106.11810 · https://arxiv.org/html/2403.04133v1 · https://github.com/motional/nuplan-devkit
- nuPlan-R / reactive-agents context — https://arxiv.org/abs/2511.10403 · https://arxiv.org/pdf/2510.14677 · CaRL https://www.cvlibs.net/publications/Jaeger2025CORL.pdf
- WOMD / Argoverse specs — https://www.waymo.jp/intl/fil/open/data/motion · https://www.argoverse.org/av1.html
