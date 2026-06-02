# STAGE 1 — SOTA LITERATURE & GAP ANALYSIS

> Status: DESIGN/ANALYSIS ONLY. No experimental results are produced here. All numbers attributed to external papers are quoted from those papers; numbers attributed to `av-policy-lab` are from the repo's existing reports (README.md, PROGRESS.md, docs/PHASE3D_EVALUATION_REPORT.md).

## 0. Where av-policy-lab is today (ground truth from the repo)

From `nuplan/train_diffusion_policy.py`, `README.md`, `PROGRESS.md`, `docs/PHASE3D_*`:

- **State:** ego-frame kinematics `[sin(yaw), cos(yaw), vx, vy, ax, ay]` (6-dim).
- **Conditioning (diffusion):** 10-dim `[6 kinematics ‖ dx_near, dy_near, dx_far, dy_far]`.
- **Trajectory target:** 16 steps × (x,y,...) flattened to a **48-dim vector**.
- **Denoiser:** MLP `Linear(122→256)→ReLU→256→256→Linear(256→48)`, ~175K params. Conditioning is **concatenated**.
- **Diffusion:** cosine schedule, T=100 train, **DDIM η=0, 10 steps**, K=8 candidate trajectories.
- **Data:** nuPlan **mini** (64 logs), ~327K sliding windows; eval on **30 scenarios**.
- **Headline result:** DiffusionPolicy 27.59m L2 vs deterministic MLP 27.55m, **Wilcoxon p=0.79** — statistically identical. (No scene perception; collisions/off-road structurally inevitable.)

This is a faithful, well-instrumented research sandbox — but it sits roughly two generations behind the SOTA recipes below on every axis.

---

## 1. Diffusion Planner (ICLR 2025 Oral, arXiv:2501.15564)

REF: arXiv:2501.15564 ; code: github.com/ZhengYinan-AIR/Diffusion-Planner

- **Input:** ego state; up to **M neighboring agents within 100 m**; vectorized map elements; route. **2 s history** of past agent trajectories.
- **Encoder:** **MLP-Mixer** over sparse per-entity vectors → **vanilla transformer encoder** to aggregate the scene into conditioning tokens. Built on **DiT (Diffusion Transformer)**.
- **Decoder (key idea):** a **single diffusion model jointly generates the ego plan AND the future trajectories of neighbor agents** — prediction and planning are one denoising process. This is what gives it genuine multi-modality and interaction-awareness.
- **Conditioning mechanism:** **multi-head cross-attention** from the noisy trajectory tokens to the encoded scene tokens, plus a **classifier-guidance** mechanism for test-time controllability (e.g., comfort, target speed) without retraining.
- **Noise schedule:** VP (variance-preserving), `σt=(1−t)β_min + t·β_max`. Sampler: **DPM-Solver++, ~10–15 steps**.
- **Loss:** predicts the **clean trajectory x0** (x0-parameterization): `‖μθ(x_t, t, C) − x0‖²`.
- **Training scale:** **~1M scenarios** from full nuPlan, **500 epochs, batch 2048, 8× A100**.
- **Results (PDM-score, higher is better):**
  - Val14 non-reactive: **89.87** (PDM-Closed 92.84).
  - Test14-hard reactive: **69.22** (PDM-Closed 75.19).

**Lesson for us:** cross-attention conditioning + joint ego/agent denoising + x0-parameterization + transformer (not MLP) + DPM-Solver. The conditioning is *rich and underspecified*, so diffusion's multimodality is actually exercised — the opposite of our 10-dim near+far goal.

**Compute-gap reality check (E9):** Diffusion Planner's budget is **1M scenarios × 500 epochs × batch 2048 × 8 A100**. Our honest target (Stage 2/4) is **~1M frames × ~100 epochs × batch ~128 × ≤4 A100** — roughly a **25–50× smaller compute budget**. We therefore do NOT expect to match its absolute numbers; we aim for the *controlled scientific result* (Stage 5), not a leaderboard win. Stating this up front prevents over-promising.

---

## 2. PLUTO (arXiv:2404.14327; jchengai/pluto)

REF: arXiv:2404.14327

- **Input:** **2 s history**, **8 s planning horizon**, map + agents within **120 m**. Vectorized map: `N_P` polylines × `n_p` points, **8 channels/point** (relative position, lane-boundary info). Agent-type and traffic-signal **learnable attribute embeddings**; **Fourier positional embedding** of `(p, θ)`.
- **Encoder:** `L_enc` transformer encoder layers, hidden dim `D`; neighbor attention-based feature aggregation.
- **Decoder (longitudinal-lateral factorization):** queries = `N_R reference lines × N_L longitudinal queries`. **Factorized lateral/longitudinal self-attention** cuts cost from `O(N_R²N_L²)` to `O(N_R²N_L + N_R N_L²)`. Outputs a *set* of multimodal trajectories with scores.
- **Losses:** (1) imitation = **smooth-L1** on trajectory + **cross-entropy** on mode scores; (2) **drivable-area auxiliary loss** via an **ESDF** with `N_c` covering circles; (3) **triplet contrastive loss (CIL)** with temperature; (4) **agent prediction** smooth-L1.
- **Augmentation (the crux):**
  - **Positives (T⁺):** ego **state perturbation** + non-interactive agent dropout.
  - **Negatives (T⁻):** leading-agent dropout/insertion, interactive-agent dropout, **traffic-light inversion**.
  - Contrastive objective pulls scene embedding toward T⁺, pushes from T⁻ → teaches *why* a behavior is correct, not just to copy it.
- **Training scale:** **1M frames**, all scenario types.
- **Results (closed-loop score):** Val14 **NR-CLS ≈ 93 / R-CLS ≈ 87** (vs PDM-Closed 93/93); Test14-hard **≈ 84 / 76** (vs PDM-Closed 89/76). First IL planner to match/surpass PDM on Val14 non-reactive. *(Exact decimals to be re-confirmed from the results table at implementation time; secondary-source rounding noted.)*

**Lesson for us:** the reference-line query decoder makes multimodality explicit; contrastive learning + rich augmentation is the covariate-shift cure; auxiliary drivable/collision losses inject map awareness directly into the objective.

---

## 3. PlanTF (ICRA 2024, arXiv:2309.10443; jchengai/planTF)

REF: arXiv:2309.10443

- **Input:** agents, map, ego — **separately encoded then concatenated**. Best ego config = **state6** (position, heading, velocity, acceleration, **steering angle**).
- **State Dropout Encoder (SDE):** a **learnable query cross-attends** over the ego state-token set; **dropout 0.75** on state tokens forces the model not to over-rely on instantaneous ego state (the main compounding-error driver).
- **Augmentation:** **state perturbation applied with probability 0.5**; the paper's central empirical finding is that **perturbation only helps with correct feature normalization**. (Exact lateral/longitudinal/heading magnitudes are set in the released config — must be read from code at implementation time; do NOT fabricate them.)
- **Encoder:** stack of standard transformer encoder layers.
- **Training:** batch 128, 25 epochs, LR 1e-3 → 0 cosine, 1M frames over 75 scenario types.
- **Results:** Val14 **NR-CLS 84.83 / R-CLS 76.78**; Test14-hard **NR-CLS 72.68 / R-CLS 61.70**.

**Lesson for us:** this paper *is* the direct answer to our root cause #3. State perturbation + state-dropout + normalization is a cheap, architecture-light covariate-shift fix that we can adopt immediately.

---

## 4. CaRL (CoRL 2025, arXiv:2504.17838; autonomousvision/CaRL)

REF: arXiv:2504.17838

- **Paradigm:** privileged-planning **RL (PPO)**, not IL — so it does **not suffer compounding error** by construction.
- **Reward:** deliberately **simple — primarily route completion**, with infractions terminating the episode or multiplicatively shrinking route completion (mirrors the PDM score's multiplicative penalty structure).
- **Scale:** PPO to **500M samples on nuPlan** (300M in CARLA), single 8-GPU node.
- **Results:** Val14 **91.3 non-reactive / 90.6 reactive**; CARLA longest6 v2 64 DS.

**Lesson for us:** RL is the long-term answer to covariate shift and the long tail, but **500M samples / 8 GPUs** puts a *from-scratch* CaRL reproduction out of scope for a solo student in 3 months. We treat CaRL as the **stretch/closed-loop-finetune** direction, not the core.

---

## 5. Scene encoders — Wayformer (arXiv:2207.05844) & VectorNet (CVPR 2020, arXiv:2005.04259)

- **VectorNet:** represents each map/agent entity as a **polyline of vectors**; a local subgraph GNN encodes each polyline, then a **global interaction graph** (self-attention) over polyline features. Origin of "vectorized" scene representation. REF: arXiv:2005.04259
- **Wayformer:** homogeneous **attention scene encoder** + transformer cross-attention decoder. Key findings: **early fusion** (concatenate all modality tokens, let attention sort it out) is modality-agnostic and SOTA; **latent-query attention** compresses a large token set to a small latent set for **2–16× speedups** with no quality loss. Decoder = learned queries cross-attending the scene encoding to emit trajectories. REF: arXiv:2207.05844

**Lesson for us:** adopt **vectorized entities + early-fusion transformer + latent-query bottleneck** as the scene encoder. This is the standard, proven recipe and is far cheaper to get right than rasterized BEV.

---

## 6. GAP TABLE — av-policy-lab vs SOTA, every axis

| Axis | av-policy-lab (now) | Diffusion Planner | PLUTO | PlanTF | Target for us |
|---|---|---|---|---|---|
| Scene perception | **None** | agents+map+route, 100m | agents+map, 120m | agents+map+ego | agents+map+route (vectorized) |
| Ego input | 6-dim kinematic | full state | state + Fourier PE | state6 (+steer) | state-history, ego-frame |
| Agents | 0 | M @100m, 2s hist | N_A @120m, 2s | yes | **N=32, 2s history** |
| Map | 0 | vectorized polylines | N_P×n_p×8ch | vectorized | vectorized polylines |
| Conditioning→traj | concat, **near-unimodal** | **cross-attn, multimodal joint** | reference-line queries | concat | **cross-attn + route-region goal** |
| Decoder over time | **flattened 48-vec** | DiT temporal tokens | factorized L/L queries | transformer | **temporal (1D-conv-UNet or transformer)** |
| Diffusion param. | ε-pred MLP | **x0-pred DiT** | n/a (regression) | n/a | x0-pred temporal transformer |
| Sampler | DDIM 10 | DPM-Solver++ ~10–15 | n/a | n/a | DPM-Solver++ |
| Aux losses | none | guidance | **drivable/collision + contrastive** | — | drivable + collision aux |
| Covariate-shift fix | bigger models (wrong) | joint pred | **perturb + contrastive + dropout** | **perturb + SDE + norm** | **perturb + SDE + DAgger** |
| Train scale | mini, 64 logs, 327K win | 1M scenarios, 8×A100 | 1M frames | 1M frames | full nuPlan, ~1M frames, A100 |
| Eval | 30 scenarios | Val14 + Test14-hard | Val14 + Test14-hard | Val14 + Test14-hard | **Val14 (1118) + Test14-hard** |
| Stat protocol | Wilcoxon (good!) | PDM decomposition | PDM decomposition | PDM decomposition | paired tests + PDM decomp + CIs |

**Strength to keep:** the repo's **statistical rigor** (Wilcoxon, CIs, per-scenario stratification) already exceeds what most planning papers report. Preserve and extend it.

---

## 7. Minimal set of changes that close the largest gaps (priority-ordered)

1. **Add a vectorized scene encoder** (Wayformer-style: vectorized agents+map+route, early-fusion transformer, latent queries). Fixes root cause #2 (NO SCENE) — the single biggest gap; without it, collision/off-road failures are unavoidable.
2. **Make conditioning genuinely multimodal:** replace the precise near+far goal with a **route-region / lane-set goal**, so `p(traj|cond)` is multimodal and the diffusion-vs-deterministic test is finally *fair*. Fixes root cause #1.
3. **Temporal decoder:** stop flattening 16×3→48; denoise over the time axis (1D-conv-UNet or transformer). Fixes root cause #5.
4. **Cross-attention conditioning + x0-parameterization + DPM-Solver++**, replacing concat-MLP-ε-DDIM. (Diffusion Planner recipe.)
5. **Ego-state perturbation + state-dropout + normalization (PlanTF), then DAgger / contrastive (PLUTO).** Fixes root cause #3.
6. **Scale to full nuPlan + Val14 / Test14-hard on HPC A100s.** Fixes root cause #4.
7. **Auxiliary drivable-area + collision losses** (PLUTO) to inject map constraints into the objective.

CaRL-style RL is explicitly **out of core scope** (500M-sample budget) — flagged as a stretch goal for closed-loop fine-tuning only.

---

## HANDOFF TO NEXT STAGE (Stage 2 — Data & Representation)

Decisions Stage 2 must honor:
- **Adopt vectorized representation** (not BEV raster): agents + map polylines + route, ego-centric frame.
- **History 2 s, planning horizon 8 s** (nuPlan-standard; matches all three IL SOTA planners) — note this changes our current 16-step (1.6s) horizon; Stage 2 must reconcile horizon/step-count.
- **Agents N = 32 within ~100–120 m**; map polylines with ~8 channels/point.
- **The goal representation must become a route-region / lane-set, not precise near+far points** — Stage 2 designs the exact tensor for this (this is what makes multimodality real).
- **Ego-state perturbation augmentation is mandatory** and Stage 2 must specify its exact distribution + correction target (and read PlanTF/PLUTO configs for real magnitudes — do not invent).
- **Splits by log/geography, not random windows**, to prevent leakage at full scale.
- **Move mini → full nuPlan + Val14 (1118) / Test14-hard.**
- Target training scale ≈ **1M frames** (matches PLUTO/PlanTF; 10× less than Diffusion Planner's 1M-scenario × 500-epoch budget, which is honest for a solo student).

## Open items / honesty flags
- PLUTO exact decimal Val14/Test14-hard numbers and PlanTF perturbation magnitudes must be re-read from the papers' result tables / released configs at implementation time. Cited values here are from the papers/secondary sources and rounding is noted; **do not treat as final until code-verified**.
