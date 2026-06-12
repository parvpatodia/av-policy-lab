# STAGE 2 — DATA & REPRESENTATION
> NOTE (2026-06-11): tensor shapes in this document may lag the code. The canonical shapes are the asserts in nuplan/features/scene_features.py (_assert_sample_consistency) plus the F0 v2 additions: ego_future (16,3) and per-sample scenario identifiers.


> Status: DESIGN ONLY. Tensor shapes, splits, and augmentation are specifications to implement; no data has been processed. Builds on Stage 1 handoff.

## 1. Coordinate frame & normalization

- **Ego-centric frame at t=0** (current step): origin at ego rear-axle, x = ego heading, y = left. WHY: translation/rotation invariance is standard in all SOTA (VectorNet/Wayformer/PLUTO) and removes a nuisance dimension the model would otherwise waste capacity on.
- **Per-feature standardization** to ~unit scale: positions / max-radius (120 m), velocities / 15 m/s, accel / 5 m/s², headings encoded as `(sin, cos)`. WHY: PlanTF's central empirical result — perturbation augmentation only works *with correct normalization* (NR-CLS jumped 71.86→85.99 in their ablation when re-normalized). Normalization is not cosmetic; it is load-bearing for root cause #3. REF: arXiv:2309.10443

## 2. Input tensors (exact shapes)

Let `T_h = 21` history steps (2.0 s @ 10 Hz incl. current), `D = 128` model dim.

| Tensor | Shape | Contents |
|---|---|---|
| `ego_history` | `[T_h=21, 7]` | x, y, sin θ, cos θ, v, a, steering |
| `agents` | `[N=32, T_h=21, 8]` | x, y, sin θ, cos θ, vx, vy, length, width (+ type embedding lookup) |
| `agent_mask` | `[N=32, T_h]` | validity (padding mask for <32 agents) |
| `map_polylines` | `[P=128, S=20, 8]` | per point: x, y, dir_x, dir_y, type-onehot(... ), left/right boundary flag |
| `map_mask` | `[P=128, S=20]` | validity |
| `route_polyline` | `[R=40, 4]` | ordered route centerline pts: x, y, dir_x, dir_y (coarse — see §4) |
| `traffic_lights` | `[P]` | per-lane signal state embedding id |

- **N=32 agents within 100 m**, nearest-first by distance, zero-padded + masked. WHY: matches Diffusion Planner (M@100m) / PLUTO (120m) order of magnitude; 32 covers dense urban nuPlan scenes while staying cheap.
- **P=128 map polylines, S=20 points each, 8 channels.** WHY: mirrors PLUTO's `N_P × n_p × 8-channel` vectorized map. REF: arXiv:2404.14327
- **Type/attribute embeddings** (agent type, lane type, TL state) as learnable lookups, à la PLUTO. Fourier positional embedding of `(p, θ)` optional.

## 3. Output target

- **Planning horizon 8 s @ 2 Hz = 16 waypoints** OR **8 s @ 10 Hz = 80** (decided in Stage 3 §4: H=16 @ 2 Hz). Target `[H, 3]` = (x, y, heading) in ego frame.
- WHY 8 s: nuPlan-standard planning horizon; matches all three IL SOTA planners. **This changes the repo's current 1.6 s / 48-dim flat target** — the temporal decoder (Stage 3) consumes `[H,3]` as a sequence, not a flat vector. This directly retires root cause #5.

## 4. The multimodal goal representation (fixes root cause #1)

**Replace** the current precise `(dx_near, dy_near, dx_far, dy_far)` 4-dim goal — which nearly fully specifies one trajectory and forces diffusion to collapse to the conditional mean — **with a route-region goal:**

- `route_polyline [R=40, 4]`: a **coarse** ordered centerline of the *roadblock-level route* (which lanes are legal), NOT a precise future position. At a junction with multiple legal successors, the route admits **multiple lane-level continuations**.
- Optionally a **lane-set goal mask**: the set of candidate goal lanes reachable in 8 s, without committing to one.

WHY: this makes `p(traj | conditioning)` genuinely **multimodal** (multiple legal junction paths, gap-accept vs gap-reject, lane-change vs stay). Only then is "diffusion vs deterministic MLP" a *fair test* — the entire Phase-3d null result was caused by conditioning that left no modality for diffusion to model. This is the single most important representational change in the whole upgrade.

**Route-region goal construction algorithm (E1 fix — reproducible):**
1. Read nuPlan `route_roadblock_ids` for the scenario (already used by the repo's `RoadblockRouteMapBC`).
2. Walk the **lane graph**: from the ego's current roadblock, take all successor lanes/roadblocks along the route up to an 8 s reach (≈ `v·8` arc length, capped at 120 m).
3. **Resample** the union of legal lane centerlines to `R=40` points at fixed arc-length spacing → `route_polyline [40,4]`. At a junction with K legal successor lanes, include **all K** (do not pick one) → the goal is a *set of legal continuations*, which is precisely what creates multimodality.
4. The **lane-set goal mask** marks which of the candidate goal lanes are reachable; the K diffusion candidates (Stage 3) are anchored one-per-goal-lane.
WHY a set, not a point: a single point re-introduces the unimodal collapse; the *set* leaves the choice of which path to the model, which is what diffusion is for.

## 5. Ego-state perturbation augmentation (fixes root cause #3)

Applied **with probability p=0.5** per sample (PlanTF setting). REF: arXiv:2309.10443

- **Perturb the ego state at t=0** (and recent history) by a small rigid offset:
  - lateral `Δy ~ N(0, σ_lat²)`, longitudinal `Δx ~ N(0, σ_lon²)`, heading `Δθ ~ N(0, σ_θ²)`.
  - **Starting magnitudes** (to be confirmed against PlanTF/PLUTO released configs — flagged, not fabricated): `σ_lat ≈ 0.5–1.0 m`, `σ_lon ≈ 0.5 m`, `σ_θ ≈ 0.05–0.1 rad`. **Sweep these in Stage 4.**
- **History re-fit (E2 fix):** after offsetting the ego pose at t=0, **re-fit a kinematically feasible smooth history** that ends at the perturbed t=0 state (so the history→t=0 seam is consistent), rather than leaving a discontinuous jump. The *future* is left untouched.
- **Correction target = the ORIGINAL expert future trajectory** (unchanged), expressed in the perturbed ego frame. WHY: PlanTF/nuPlan finding — supplying a *re-solved corrected* guiding future *hurts*; using the true expert future as supervision keeps the data distribution intact and still teaches recovery, because the model now sees off-nominal start states mapped to expert recovery. (Counterintuitive but empirically established; arXiv:2309.10443.)
- **Re-express the whole scene in the perturbed frame (E3 fix):** agents, map, and route inputs are all recomputed in the perturbed ego-centric frame so inputs and supervision agree; **re-normalize after** (see §1) — the make-or-break detail.
- WHY this over "bigger models": covariate shift is a *distribution* problem (train sees only expert states, test sees drifted states). Perturbation injects drifted start states at train time; capacity does nothing for it. This is the corrected diagnosis from the brief.

**Later (Stage 4) contrastive/DAgger augmentation (PLUTO):** positives = perturbation + non-interactive agent dropout; negatives = leading-agent insertion/dropout, traffic-light inversion. Deferred to training stage.

## 6. Splits (prevent leakage)

- **Split by log (and by city/geography), never by random window.** WHY: sliding windows from the same log share scene/agents/map; random splitting leaks near-duplicate frames into val and inflates open-loop metrics (the repo's current 0.058 m ADE is almost certainly leakage-inflated). Log-level splitting is what nuPlan's official Train/Val/Test14 splits do, with strict temporal+spatial separation.
- **Geographic holdout:** hold out at least one city (e.g., Singapore) entirely for cross-domain generalization checks.
- **Use the official nuPlan splits + Val14 / Test14-hard** for the headline numbers so results are comparable to the literature.

## 7. Scale: mini → full nuPlan + Val14 / Test14-hard

- **Train:** sample **~1M frames** from full nuPlan train split across all scenario types (matches PLUTO & PlanTF; an honest 10× under Diffusion Planner's 1M-scenario×500-epoch budget). WHY: 1M frames is the demonstrated sufficient scale for SOTA IL planners and is feasible on the available A100s.
- **Val14:** **1,118 scenarios** for the main closed-loop number (reactive + non-reactive).
- **Test14-hard:** the long-tail split (≈272 scenarios = the 20% lowest-scoring per type under PDM-Closed). WHY: Val14 skews easy; Test14-hard is where the scene encoder + augmentation actually have to earn their keep.
- This replaces mini (64 logs) / 30 scenarios, which the brief correctly identifies as having ≈zero statistical power (root cause #4).

## HANDOFF TO NEXT STAGE (Stage 3 — Architecture)

- Encoder input is the **vectorized tensor set in §2** (ego, agents, map, route, masks) — design a Wayformer-style early-fusion transformer over these, with latent-query compression.
- Conditioning for the diffusion decoder is the **encoded scene tokens** + the **route-region goal (§4)** — NOT a precise point. Cross-attention is required (Stage 3 justifies).
- Decoder denoises a **`[H,3]` temporal sequence** (H from §3), not a flat 48-vector.
- Augmentation hooks (§5) feed the training loop in Stage 4; perturbation magnitudes are a **Stage-4 sweep**.
- Splits/scale in §6–§7 define the data volumes Stage 4's SLURM pipeline must stage to the cluster.

## Honesty flags
- Exact perturbation σ values and Val14/Test14-hard reference numbers must be code-verified, not taken from memory.
- 1M-frame extraction + caching from full nuPlan is itself a multi-day engineering task on HPC (SQLite → cached tensors); costed in Stage 4 / final plan.
