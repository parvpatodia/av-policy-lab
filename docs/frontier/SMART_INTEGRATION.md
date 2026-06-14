# SMART Reactive-Agent Axis: Integration Status

The third experimental axis replaces IDM background agents with learned,
reactive SMART agents (Hagedorn et al., "When Planners Meet Reality",
arXiv:2510.14677) to test whether the diffusion-vs-deterministic conclusion
survives a realistic simulator.

## Assets in hand (2026-06-13)

Two SMART checkpoints, both for academic use, NEITHER redistributed in this repo.

1. **Bosch nuPlan-trained SMART** (primary). From Steffen Hagedorn via the
   private repo shgd95/checkpoint_smart_nuplan. File epoch=07_1180.ckpt, 82 MB.
   - 7.16M params, 818 tensors, epoch 7, 147,088 steps, PL 2.0.3.
   - ALREADY nuPlan-trained (config dataset="nuplan"). No fine-tuning needed.
   - Architecture (embedded in checkpoint hyper_parameters, so no codebase
     required to read it): hidden_dim 128, num_heads 8, head_dim 16,
     num_historical_steps 11, num_future_steps 80 (8 s @ 10 Hz, 2 Hz tokens),
     token_size 2048, num_map_layers 3, num_agent_layers 6, a2a_radius 60,
     pl2a_radius 30, pl2pl_radius 10, time_span 30, use_intention True,
     num_freq_bands 64.
   - Input featurization (from embedding shapes): agent token MLP input 8-dim
     (separate veh/ped/cyc heads token_emb_veh/ped/cyc), map token MLP input
     22-dim. Layer naming (to_q/to_k/to_v/to_g/to_s, r_pt2pt_emb.freqs) is the
     ORIGINAL SMART repo (rainmaker22/SMART), NOT the catk refactor.
   - Safety: static pickle scan clean (torch, collections, easydict only).

2. **catk SMART-tiny (WOMD)** (fallback). From Zhejun Zhang (NVIDIA),
   pre_bc_E31 + clsft_E9. WOMD-trained, would need nuPlan fine-tuning. Kept
   only as a backup if the Bosch path stalls; strictly worse for our use.

## Decision: wait for the Bosch codebase, do not reverse-engineer now

Steffen expects the full Bosch codebase open-sourced by end of June 2026,
including their nuPlan observation integration (the "drop-in IDM replacement").
That release provides exactly the two things the checkpoint cannot supply by
itself: the matching model class (original SMART) and the AbstractObservation
wrapper that drives nuPlan agents at the model's 2 Hz token rate.

Reverse-engineering the original SMART model + rollout + nuPlan observation
now would be days of work made redundant in ~2 weeks. The core 2x2-under-IDM
experiment is branch-independent of SMART and proceeds first; the SMART axis
slots in on release with the checkpoint already validated and the config known.

## Open items when the codebase lands

1. Instantiate the original-SMART model from the embedded config; load
   epoch=07_1180.ckpt (expect missing=0/unexpected=0, as with catk).
2. Use Bosch's nuPlan observation wrapper (preferred) rather than a hand-built
   one; confirm it consumes per-step DetectionsTracks and replans at 2 Hz.
3. Confirm the map/agent tokenizer + token vocab files ship with the release.
4. Run the eval array with --reactive 1 pointed at the SMART observation,
   same frozen manifest as the IDM run, so CLS deltas are paired per scenario.

## License

Bosch checkpoint: nuPlan-trained, "no Bosch IP" per Hagedorn (originally from
the SMART authors). nuPlan dataset license (non-commercial research). Cite
Hagedorn et al. 2025. Do NOT redistribute weights or commit them to this repo.
catk checkpoint: WOMD terms, no redistribution.
