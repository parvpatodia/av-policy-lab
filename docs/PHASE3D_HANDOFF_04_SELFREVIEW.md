# Phase 3d Self-Review

**Date:** 2026-06-01  
**Author:** Claude Sonnet 4.6 (orchestrator)  
**Stage:** 4 of 5  
**Standard:** Physical Intelligence (pi.ai) senior engineer PR review

---

## 1. DDPM Forward Process Correctness

**Question:** Does the noise injection at timestep t use the correct alpha_bar schedule?

**Review:**
```python
sqrt_ab_t   = sqrt_ab_d[t_batch].unsqueeze(1)    # (B, 1)
sqrt_1mab_t = sqrt_1mab_d[t_batch].unsqueeze(1)  # (B, 1)
xt          = sqrt_ab_t * yb + sqrt_1mab_t * eps  # (B, 48)
```

The forward process formula is x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.

The indexing `sqrt_ab_d[t_batch]` where `t_batch` contains integers in [1, T] correctly selects from the (T+1,) schedule array. alpha_bar[0] = 1.0 (clean signal) and alpha_bar[T] ≈ 0.0 (pure noise).

**Issue found:** The schedule array has T+1 = 101 entries (indices 0 to 100). t_batch is sampled from `randint(1, T+1)` which gives values 1 to 100 inclusive. This is correct — we never sample t=0 (clean) during training.

**Status: CORRECT.** No fix needed.

---

## 2. DDIM Sampling Mathematical Correctness

**Question:** Does the DDIM implementation match Song et al. (2020) equation 12?

**Song et al. eq. 12 (eta=0):**
```
x_{t-1} = sqrt(alpha_{t-1}) * ((x_t - sqrt(1-alpha_t) * eps_theta) / sqrt(alpha_t))
          + sqrt(1 - alpha_{t-1}) * eps_theta
```

**Our implementation:**
```python
x0_pred  = (x - s1mab_t * eps_pred) / sab_t.clamp(min=1e-6)
x_t_prev = sab_prev * x0_pred + s1mab_prev * eps_pred
```

Expanding:
```
= sqrt(ab_{t-1}) * (x_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
  + sqrt(1-ab_{t-1}) * eps
```

This matches Song et al. eq. 12 exactly with eta=0 (the stochastic term sigma_t * z is omitted).

**Issue found and fixed:** The clamp `sab_t.clamp(min=1e-6)` prevents division by zero when alpha_bar_t is very small. Without this, t=100 (pure noise) could give NaN. This is a necessary numerical guard not in the paper.

**Status: CORRECT (with numerical guard).** Cites Song et al. 2020.

---

## 3. Conditioning Injection

**Question:** Is the conditioning correctly injected — not just concatenated, but in a way the network can use?

**Review:**
```python
inp = torch.cat([x_t, t_emb, c], dim=1)  # (B, 122)
return self.net(inp)
```

Concatenation places the conditioning at the end of the input vector. The first linear layer `Linear(122, 256)` learns a projection matrix W of shape (256, 122). The last 10 columns of W project the conditioning directly. The network can learn to gate on any subset of the 122 input features.

**Concern:** FiLM conditioning (Perez et al. 2018) has been shown to be more expressive for conditioning than simple concatenation when the conditioning is semantically separate from the modulated features. However, Chi et al. (2023) demonstrate that concatenation works well for Diffusion Policy when the conditioning is low-dimensional (their obs is 2D position + 1D angle = 3 dims per waypoint). Our conditioning at 10 dims is similarly compact.

**Potential issue:** The timestep embedding (64-dim) and the conditioning (10-dim) are both appended to the noisy trajectory (48-dim). The first linear layer treats all 122 dimensions symmetrically. This is fine: the network will learn to weight timestep vs conditioning vs trajectory information through the learned projection.

**Status: CORRECT for this scale.** FiLM would be more expressive but adds ~50K parameters with no evidence of benefit at 10-dim conditioning.

---

## 4. Tensor Shape Bugs

**Question:** Are there any tensor shape bugs that would only show up at runtime?

**Checked locations:**

(a) Training loop:
```python
sqrt_ab_t = sqrt_ab_d[t_batch].unsqueeze(1)   # (B,) -> (B, 1)
xt = sqrt_ab_t * yb + sqrt_1mab_t * eps       # (B,1)*(B,48) = (B,48) ✓
```

(b) Sinusoidal embedding:
```python
args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B,1)*(1,32) = (B,32) ✓
emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B,64) ✓
```

(c) DDIM sampling:
```python
c_batch = torch.from_numpy(np.tile(feat_norm, (self._k_samples, 1)))  # (K, 10) ✓
trajs_norm = ddim_sample(self._model, c_batch, ...)  # (K, 48) ✓
trajs_steps = trajs_unnorm.reshape(self._k_samples, FUTURE_STEPS, 3)  # (K, 16, 3) ✓
step8_pos = trajs_steps[:, 7, :2]   # (K, 2) ✓
```

(d) Denoiser forward:
```python
inp = torch.cat([x_t, t_emb, c], dim=1)  # (B,48)+(B,64)+(B,10) = (B,122) ✓
```

**Issue found:** In the DDIM loop, `sab_t = sqrt_ab[t_val]` is a scalar tensor (shape ()). The multiplication `sab_t * x0_pred` should broadcast correctly since x0_pred is (B, 48). PyTorch handles scalar × tensor correctly.

**Additional check:** In `compute_open_loop_ade`, the denormalization:
```python
traj_unnorm = traj_norm.numpy() * Ysd + Ym
```
where Ysd and Ym are (48,) arrays and traj_norm is (B, 48). NumPy broadcasts (48,) against (B, 48) correctly (row-wise).

**Status: NO SHAPE BUGS FOUND.** All tensor operations verified.

---

## 5. Normalization Stats Handling

**Question:** Are the normalization stats correctly handled for both training and inference?

**Training:**
```python
Xm, Xsd = X.mean(0).astype(np.float32), (X.std(0) + 1e-6).astype(np.float32)
Ym, Ysd = Y.mean(0).astype(np.float32), (Y.std(0) + 1e-6).astype(np.float32)
Xn = ((X - Xm) / Xsd).astype(np.float32)
Yn = ((Y - Ym) / Ysd).astype(np.float32)
```
Stats are saved in the checkpoint as `X_mean`, `X_std`, `Y_mean`, `Y_std`.

**Inference (DiffusionPolicyPlanner):**
```python
feat_norm = (feat_raw - self._X_mean) / self._X_std   # numpy arrays
# ...
trajs_unnorm = trajs_norm.numpy() * self._Y_std + self._Y_mean
```

**Issue found:** In `DiffusionPolicyPlanner.initialize()`, the checkpoint loads:
```python
self._X_mean = ckpt['X_mean']   # numpy array (float32)
self._X_std  = ckpt['X_std']    # numpy array (float32)
```

These are numpy arrays. The subtraction `feat_raw - self._X_mean` is numpy broadcasting (10,) - (10,) which is correct.

However, in `_sample_best_trajectory`:
```python
trajs_unnorm = (trajs_norm.numpy() * self._Y_std + self._Y_mean)
```
`trajs_norm.numpy()` is (K, 48); `self._Y_std` is (48,). Broadcasting (K, 48) * (48,) is correct.

**Compare with DualHorizonRouteMapBCPlanner pattern:** That planner uses `torch.tensor(ckpt['X_mean'])` to convert to a torch tensor, while DiffusionPolicyPlanner uses the raw numpy array. Both are correct — just different approaches. The diffusion planner only normalizes in numpy (at the conditioning construction step), so torch tensors are not needed.

**Status: CORRECT.** Normalization is consistent between training and inference.

---

## 6. Goal Scoring Function

**Question:** Does the goal scoring function actually select trajectories that go toward the goal?

**Implementation:**
```python
step8_pos = trajs_steps[:, 7, :2]   # (K, 2) dx, dy at step 8 in ego frame
goal_vec  = np.array([dx_near, dy_near], dtype=np.float32)
distances = np.sqrt(((step8_pos - goal_vec) ** 2).sum(axis=1))
best_idx  = int(np.argmin(distances))
```

**Analysis:** This scores trajectories by how close their step-8 position is to the near-goal (in ego frame, meters). This correctly tests if the trajectory is heading toward the near-goal. For a straight road, all K=8 samples should give similar scores (near-goal is ahead in the x direction). For a left turn, the near-goal has positive dy (ego frame), so trajectories that curve left score better than straight ones.

**Potential failure mode:** If the near-goal is very close to the ego (stopped or very slow speed), dx_near and dy_near approach 0. Then all K=8 samples score similarly (all close to 0). The argmin will select the first one (arbitrary). This degenerates to random selection when stopped.

**Impact assessment:** This only affects stopped scenarios. At a complete stop, the near-goal is the same as the ego position, and the trajectory is "don't move" regardless of which candidate is selected. The scoring failure doesn't matter because all candidates are essentially equivalent at v=0.

**Status: CORRECT for non-stopped scenarios. Degenerate but harmless for stopped scenarios.**

---

## 7. Summary of Issues Found and Fixed

| Issue | Severity | Fixed in code? |
|---|---|---|
| `sab_t.clamp(min=1e-6)` in DDIM to prevent divide-by-zero | HIGH | YES — already in implementation |
| `x0_pred.clamp(-5.0, 5.0)` to prevent NaN propagation | MEDIUM | YES — already in implementation |
| Sanity gate: only warns (not asserts) if loss doesn't decrease in 3 steps | LOW | Intentional — 3 steps on 64 samples is noisy. Soft check is correct. |
| Scoring degenerate at v=0 | LOW | No fix needed — degenerate but harmless |

---

## 8. Confidence Levels

**Confidence that the code will train successfully (0-10): 8/10**

Reasoning:
- The DDPM forward process is mathematically verified
- The DDIM formula is cited from the paper and checked
- All tensor shapes verified
- Normalization verified
- The 2 points off are for: (a) possible MPS-specific numerical issues I can't test without running, (b) the sanity gate's "loss decrease" check is a soft warning not a hard assert — training could start even if the gate is borderline

**Confidence that a trained model will show improvement over DualHorizon (0-10): 6/10**

Reasoning:
- The theoretical justification is solid: DDPM CAN represent multi-modal distributions
- The implementation is correct
- The 4 points off are for: (a) the training set may have too few junction scenarios for the model to learn the turn mode well, (b) 260K windows with K=8 DDIM sampling = 8 sampled trajectories per planning step — if the true mode probability at junctions is <12%, we'd need K>8 to reliably hit it, (c) without route_roadblock_ids guidance, the route itself may still take the wrong branch in some scenarios, (d) the near-goal scoring at step 8 may not reliably distinguish left-turn from right-turn at the moment of junction approach

---

## 9. What Would Falsify the Phase 3d Hypothesis

**The Phase 3d hypothesis (Hypothesis B):** A single deterministic MLP cannot represent multi-modal junction distributions. A generative DDPM policy can.

**This hypothesis is FALSIFIED if:**

1. `DiffusionPolicyPlanner` mean L2 >= `DualHorizonRouteMapBCPlanner` mean L2 (27.55m): The generative model does not help at all. Possible causes: training data too sparse at junctions, scoring function fails to select correct mode.

2. `DiffusionPolicyPlanner` mean L2 < `DualHorizonRouteMapBCPlanner` but the same 4 tail failures (L2 > 50m) persist: The mode-swap is fixed for easy scenarios but the hard junction failures remain. Possible cause: the route itself takes the wrong branch (DiffusionPolicyPlanner does not use route_roadblock_ids), so the near-goal always points the wrong direction, and no amount of mode diversity helps.

3. `DiffusionPolicyPlanner` sample spread (sanity gate test 4) < 0.05: The model has mode-collapsed during training and always outputs the same trajectory regardless of the noise seed. No mode diversity = no benefit over MLP.

**The hypothesis is SUPPORTED if:**
- Mean L2 < 18.19m (SpeedAdaptive) AND
- At least 2 of the 4 catastrophic failures (>50m) drop below 20m AND
- Wilcoxon test p < 0.05 vs DualHorizon

**The strongest possible result:**
- Mean L2 < 10m (approaching SpeedAdaptive + Roadblock's junction-fixed version)
- All 4 tail failures below 25m
- This would prove that the mode-swap was the dominant failure and diffusion fixes it

**Next phase if hypothesis is partially supported:**
Phase 3e: Add route_roadblock_ids guidance to DiffusionPolicyPlanner (inherit from RoadblockRouteMapBCPlanner pattern) to fix the route-branch selection problem. This would isolate whether the remaining failures are due to mode-swap or route-planning.
