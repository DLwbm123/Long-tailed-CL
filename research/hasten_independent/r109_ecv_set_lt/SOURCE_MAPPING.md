# Source mapping: ECV-SET-LT engineering v1

These are adaptations of user-supplied ICLR 2027 submission PDFs, not claims of acceptance, novelty, or transferred theoretical guarantees. No third-party PDF or full text is redistributed.

| Source | Original location | Implemented use | Boundary |
|---|---|---|---|
| FedNASC: Noise-Aware Semantic Control for Federated LoRA Adaptation | p.5, Eq.5; discussion of grouped observations and fixed-map assumptions | `state_features.split_half`: group aggregation, odd/even coefficient, four fixed splits, missing evidence | No federated LoRA, multi-source score, reward confidence interval, or clinical patient-independence claim. The real integration deliberately reports this feature unavailable. |
| Reliable Replay through Spatial Coherence in Online Continual Learning (SPHERE) | pp.5–6, Eqs.4–11 | `risk_transport`: signed smoothing, one-marginal column softmax, pullback mask, uniform floor; proposals only | No Sinkhorn, change to inner CE/KD/replay weighting, optimal-memory guarantee, or unbiased without-replacement gradient claim. Geometry and temperature are detached while differentiating the transport value. |
| VETO: Reward-Supervised World Models as Action Verifiers for Sample-Efficient RL | pp.4–5, Eqs.1–3; p.9 dynamics ablation | `Verifier`: real dynamic targets plus return targets, with equal-capacity DIRECT_RETURN | No robot latent simulator, imagined reward substitution, or uncorrected top-k filtering before on-policy updates. |
| CoEvoST: A Dual Evolution Agent for Continual Spatiotemporal Forecasting | p.6, Eqs.6–10 | Full checkpoint branches, actual outcome comparison, near-optimal set probability supervision | No LLM, field-weighted DPO, future-label access, or automatic adoption guarantee. |

The implementation reuses the R107 feasible ZERO/BALANCED constructor and feature-token memory, and R108's device-independent version of the same R107 classifier. Exact source-file identities are in CONTRACT_LOCK.json. ZERO means a uniform legal next action, with fixed coverage and completion constraints; it is not a uniform distribution over all legal final sets.

The method-specific contribution remains a hypothesis: whether class-conditioned risk, actual intervention dynamics, and delayed utility improve finite-memory control. Engineering correctness does not establish this hypothesis. Full original-paper algorithms have not been reproduced here.

