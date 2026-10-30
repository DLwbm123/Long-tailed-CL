# R84: frozen-representation, independently learned decision boundaries

Frozen before software, remote probe, encoding or fit. Mainline mechanism diagnosis; not a new RL method or an improved comparator. R82/R83 gates remain failed. User authorized evidence-led continuation on 2026-10-30.

## Evidence and single hypothesis
R82 exposed ineffective tiny-class allocation. R83 repaired that actuator, but 36/36 forward protected-risk utility gains did not translate into reliable BA gains. Class1/2/12 own-priority actions never increased fullQ correct counts. This motivates separating representation movement from decision-boundary estimation, not claiming that all representations are sufficient. Original detached NCM fixes each normal to a support mean; a single support has no initial own-prototype tangent attraction. Independent classifier normals remove that particular geometric constraint without new images, semantic prototypes, augmentation or visual updates. The single-support argument does not cover all classes or prove causation.

Prior fixed ridge work (D56) showed a large resubstitution/generalization gap. Thus successful S fitting is explicitly not evidence of success here. This is a different, anchored multinomial discriminative objective on the fixed R81 states, not a repetition of the old ridge result. If it fails on disjoint Q, do not tune its strength or choose a different head on these Q results.

Primary literature: Kang et al., Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR2020, https://openreview.net/pdf?id=r1gRTCVFvB supports testing separation, not efficacy in this medical/CIL setting. This is a known classifier family, not novelty or an original-paper reproduction.

## Fixed data and interventions
Use all six sealed R81 uniform source stages, HK74002, original arrived classes only. Encode TRAIN embeddings at each exact source delta once; no visual gradients, visual optimizer or controller. R81 AdamW stays read-only and its schema/step recorded, not resumed. Original R82 component folds define S/Q; same Qa/Qb metadata remains only a descriptive split. No future-stage data at earlier stages, no VAL/TEST, no new random split, no patient/independent-seed claims.

For each of 6 stages x 2 folds, form normalized S mean prototypes P and base logits 20*z*P.T. Three arms:
- NCM: original logits, no fitted parameters.
- natural: logits 20*z*P.T + z*V.T + b; minimize sum_i [n/N * CE_i] + .5*(||V||_F^2+||b||^2).
- balanced (primary): same parameterization, minimize sum_i [CE_i / N_yi] + the same quadratic anchor.
Both objectives have total data mass n, same S/images/features and zero residual initialization. Unit anchor is one fixed, auditable prior, not a tuned standard or claimed optimal value. No +log train-count term: these heads directly fit the declared target classifier; natural is the matched prior control. No residual normalization or logit temperature search. Natural and balanced CPU fits use identical deterministic FP64 L-BFGS settings (max300 iterations/max600 evaluations, history10, strong-Wolfe, grad tolerance1e-7). Save actual optimizer iteration/evaluation counts and final gradient residual; require maxabs gradient <=1e-5. Nonconvergence is engineering failure, no retuning.

All 24 fitted heads (12 per fit arm) and all six frozen feature sets are sealed before any Q scoring. Fit S only; never Qa/Qb feedback or label-based model selection. Save private features, heads, optimizer states and class correct/counts; public only safe aggregates/source. No existing 756-branch selector sweep.

## Predeclared necessary decision gate
Balanced minus NCM AND balanced minus natural: global stage-equal macro BA and tail BA must all be >0, and at least5/6 stages must have all four positive fold-mean gains. Additionally fullQ class1/2/12 each must have nonnegative stage/fold-mean recall change versus NCM, at least two strictly positive. Report every class and both components/folds, ceiling/headroom and all Qa/Qb directions without choosing a direction. Fold1 class1 lacks dual-Q component coverage: fullQ retained, no invented slice reliability. Stage/fold means are conditional repeated development, not independent replication.

This gate only supports proceeding to a matched classifier-policy study; it cannot establish GRPO benefit. If balanced is useful but no RL benefit established, attribute it to supervised boundary learning. Future RL would require CE/natural, uniform and applicable shuffled/lookahead controls and original SFT final/tail gates. No new policy launches as part of R84.

## Resource and stopping budget
New independent FIRST/deadline stored before all copy/probe/CPU/native; 20h includes queue, repair, encoding, fits, analysis and cleanup, no reset. Only physical GPU0 frozen native tail / GPU1 frozen prefix, CUDA_VISIBLE_DEVICES0,1; all GPUreserved<12GiB. Admission128GiB MemAvailable and14GiB free each. Exclusive original memory lock; RAM group29GB/worker28GB, 2s watchdog, CPU4,400000CPU-seconds,file1GiB. Native engineering uses only two arrived TRAIN images, zero optimizer/backward/VAL, followed by all six stages; no short efficacy pilot.

Predicted new science: six FP32 embedding sets <80MB, metadata and private three-arm scores <40MB, 24 heads/CPU optimizer states <40MB, own source/logs <20MB. Shared native/base weights read-only symlink, no copied large weights. Prefix197x768 FP32 RAM up to6821 rows~4.13GB, cache23GB, root/remote prefix spill0, no pixel disk cache. Conservative new disk hardcap2,000,000,000B including all artifacts, 512MiB early-stop margin, recent protected~27.2GB+2GB+margin<50GB; actual admission records fresh retained size and NFS probe. Existing remoteaggregate21GB and combined29GB upper bounds only tightened. All head/features/science protected; only declared prefix RAM/index cleaned. On failure audit original clock/cost/fit/score/seal boundary first, no blind overwrite or partial resume.

Saved verification, analysis and safe public delivery follow completion; exact remote commit/anonymous report200/source+local receipts required. No private class risk/margin, perexample identity, features, weights, raw logs or private paths public.
