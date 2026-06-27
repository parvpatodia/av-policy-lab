# SPEC — RL capstone: scene-adaptive multimodal policy (DIVER-style) + unsaturated eval + H1 re-test

## Why RL (from ADR-032/037)
Supervised fixes are exhausted: no-repulsion -> unimodal fan; strong-repulsion -> uniform manufactured
modes (scene-inappropriate, less accurate). The missing ingredient is a per-scene quality signal that
makes diversity ADAPTIVE: spread only where multiple options are good. RL with a reward provides exactly
that (DIVER, arXiv:2507.04049). GRPO's group-relative advantage naturally yields scene-adaptive
multimodality: within a group of K samples for a scene, good trajectories get positive advantage; at a
junction multiple options are good (multimodal), at a trivial scene one is (unimodal).

## The tractability crux: an OPEN-LOOP proxy reward (no per-sample sim)
True reward = closed-loop CLS, too expensive per RL sample. Use a cheap open-loop proxy computed from
scene tensors, REUSING f4_score machinery (agent CTRV rollouts, route corridor, crossing detection):
  R(traj, scene) = w_prog * progress_along_route
                 - w_coll * collision_risk(traj vs agent rollouts)
                 - w_off  * off_route/off-drivable (lateral offset from route corridor)
                 - w_comf * discomfort (jerk / lat-accel)
                 - w_kin  * kinematic_infeasibility
All terms computable from ego candidate traj + the scene's agents/route (no nuPlan sim).

## De-risk ladder (cheapest decisive first; NO multi-week commit until each gate passes)
- GATE-RL-1 (reward validity) — THIS STEP. Build reward_proxy.py; validate ordering: the EXPERT
  (ego_future) trajectory must score HIGH; a collision/off-route/jerky perturbation must score LOW;
  reward correlates with held-out CLS sign where checkable. If the reward can't rank trajectories
  sensibly, RL cannot work. Cheap (CPU, frozen scenes).
- GATE-RL-2 (reward-weighted update moves the policy) — small AWR/GRPO loop on the multi-hypothesis
  head: advantage-weight modes by R; confirm modes shift toward higher reward and that diversity
  becomes SCENE-ADAPTIVE (more spread at decision types than at stationary -- the ADR-037 failure
  inverted). Bounded de-risk run.
- GATE-RL-3 (closed-loop smoke) — run the RL policy through the real sim on a small set; confirm it
  drives + CLS is sane.
- THEN: full RL train (route+precise x seeds, fairness-matched) -> closed-loop eval on the UNSATURATED
  slice (bottom-CLS-quartile + decision types) -> H1 re-test (analyze_moderation_v2).

## Algorithm (tractable, GRPO/AWR on the multi-hypothesis head)
Policy = multi-hypothesis head (M modes + scores). Per scene: sample/take M modes, reward each R_m,
group-relative advantage A_m = (R_m - mean)/std. Update: (a) advantage-weighted regression nudging
modes toward higher reward; (b) score CE toward the reward ranking; (c) a light GT-anchor (best mode
regresses to ego_future) so modes stay realistic. Diversity emerges from the reward, not a fixed
repulsion -> scene-adaptive by construction.

## Honest risks
Reward-proxy fidelity (open-loop != closed-loop); RL convergence/instability; even a good multimodal
policy may not move the SATURATED CLS -> the unsaturated slice (C2) is required for a meaningful H1
re-test. Each gate can end the line with a real finding.
