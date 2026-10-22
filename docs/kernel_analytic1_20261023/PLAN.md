# NIH: fixed reference coordinates and patient/class-balanced analytical prototypes

## Falsifiable full hypothesis

Frozen native complete visual features may retain useful information without the learned local compression and repeated stochastic replay updates used in the closed runs. The new mechanism combines (1) an immutable real base-patient landmark coordinate system, (2) patient/class-mass sufficient statistics from every actually arrived image, and (3) joint all-seen convex analytical decision prototypes. It changes the historical state and learning objective, not a post-hoc refit of a closed model. No result is known. Source cannot prove which preceding component caused preceding failures.

DAP (OpenReview Tsqun8N1mn), methods3.2/3.3 equations6/7/11/12 and implementation details4.1 motivate stable reference coordinates plus joint class-weighted analytical prototypes. This is a bounded NIH adaptation: no base encoder fine-tuning, synthetic Gaussian examples/mean replacement or inherited separability theorem, no claim of faithful DAP reproduction or new TMI novelty. Chosen landmarks2048 and median-distance width/ridge1e-4 are frozen task adaptations, not the paper's reported dataset-specific hyperparameters and not selected with validation.

## Representation and historical information

Read fresh legal images through existing reader/GPUFP16 full pretrained ViT; concatenate CLS and FP32 patchmean, L2 normalize1536. No backbone optimization, old cache/features/weights from experiments, extra dataset download, test/Etest or future-class initialization. Choose2048 distinct real base patients deterministically by metadata hash, one image each; guarantee all base classes represented. Retain their features as fixed landmarks. This is explicitly larger base exemplar permission than the earlier32/class memory; all controls have the same current-data/landmark/statistic permission. Never call it exemplar-free or equivalent to old A/B.

Kernel: exp(-squared distance / sigma), sigma=lower median non-diagonal pairwise squared landmark distance (base only). Subtract the per-coordinate mean over landmark kernel rows, L2 normalize each mapped row. Linear control uses the complete native1536 normalized feature. Append constant1 and divide bysqrt2 for all arms. Reference coordinates, bandwidth and center never change after base; no nonlinear moment transport. No augmentation/dropout at fitting or deployment.

For class c retain exact FP64 G_c=mean_patient(mean_image(phi phi^T)) and m_c=mean_patient(mean_image(phi)), patient count and image count. Every arrived image participates; current descriptors are released after online independent NumPy check. These are actual fixed-function statistics, not a generative replacement or estimate of a changing nonlinear feature. Retain a bounded8-row mapping proof per class/kind for audit, plus64-row stage score probes. No historical raw images/full-feature tables retained. Retain one linear and one kernel moment family; both kernel arms share the exact same stored statistics.

## Fair strong controls and exact objective

- LINEAR_BALANCED: complete native feature, joint patient/class-balanced analytical solution. Strong direct control, not the weak previous learned compression. Feature/head dimension1537 includes intercept.
- KERNEL_NATURAL: same2049-dimensional landmark coordinates, same exact patient-normalized class statistics and ridge, classes weighted P_c/sum P_c. Patients receive equal global mass, classes keep natural patient imbalance.
- KERNEL_BALANCED primary: same coordinates/statistics/solver, each class receives equal1/K mass, then each patient and each of their images equal within its group.

For masses a_c, minimize sum_c a_c E_patient E_image ||phi(x)W-e_c||² +1e-4||W||². W=(sum_c a_c G_c +lambda I)^(-1)[a_c m_c]_c. Recompute joint seen-class system at each actual arrival from retained class moments; class-mass renormalization is exact. Old examples are not revisited or hallucinated. Shared ridge is strictly positive. GPUFP64 solve residual max<1e-9; no automatic solver/regularization rescue, scans or parameter search. Formal18solves, zero formal optimizer steps. Classification is allseen argmax(phiW), no task-id or calibrated-probability claim.

All have identical legal arriving data and no validation tuning; controls share historical moment/landmark permission and byte-accounting. Linear and kernel active feature/head dimensions and compute differ, explicitly disclosed; no equal-FLOP or equal-active-capacity claim. The matched kernel pair isolates class-mass normalization; the linear comparison tests the complete nonlinear reference-space mechanism and is not a one-factor causal proof. Additional base-landmark retention/full class moments differ from finite replay, so do not attribute cross-experiment gains solely to one component.

## Qualification before bulk

20 real distinct base patients, formal reader/GPUFP16 encoder→GPUFP64 full formal1537/2049 feature/moment/system/solve sizes→base8 then arrival2 class extension→temporary head actual64-row forward/backward/clip/SGD parameter update per arm (discarded, three steps)→save/load→independent CPU NumPy mapping/moments/full normal solve and score. Repeated qualification landmarks only expand tensor size, are disclosed and discarded; scientific landmarks must be2048 unique patients. Reference in simulatedbase8 uses only those8, no simulatedfuture input. Qualification includes actual analytical path, not using discarded SGD as a substitute. PASS required programmatically before any formal bulk encoding.

## Independent acceptance and scientific gates

18 sealed stage heads before opening validation;300val images encoded once. Final CPU NumPy independently reselects landmark metadata, reconstructs bandwidth/centering, full class-moment normal systems and solves, validates18 score groups<=2e-6 and identical predictions, equation residual<1e-9, actual stage probes, moments/counts/seen/patient intersection0 and resource budgets. Full current moments independently recomputed by NumPy while arriving descriptors remain in RAM, then descriptors released; final acceptance uses retained class moments and bounded mapping proofs. It does not re-encode historical patients or claim independent every-image decoding. Acceptance zero new encodings and zero optimization.

All13 primary gates must pass: final>=91/300, six-stage mean>=36.005291005291%,tail7>=25/105,initial>=44/150,later>=41/150;vs each linear-balanced and kernel-natural final+4,mean+1pp,tail4+2/60,later not lower. Any scientific failure closes the recipe; no changing ridge/landmarks/bandwidth/mean/priors/encoder/seed/gates or switching primary to rescue validation scores. Repeated300val is development screening, not clinical test/SOTA.

## Frozen budget and delivery

8h from first remote admission, including all preparation/qualification/waits/failures/fits/independent acceptance, never reset;3e9B hard/2.7e9 prestop, RSS16GiB/GPUreserved8GiB. Actual NFS mount/free/write-read and freeGPU8GiB/host16GiB admission. One independent5s process-group guard, TERM60s before deadline/KILL30s if needed. Estimate retained class linear/kernel Grams ~1.05GB plus landmarks25MB and small head/proofs; class moment files stored once, never copied into all stage packs. Full current descriptors only RAM ~0.42GB, no train disk cache. No scientific retries.

Window cumulative<30e9B, remote27e9prestop and local100MB reserve; include both retired asymmetric/q672 and structured/q673 and all new root/neutralentry/papers/failures. Do not reset accounting by experiment name or erase evidence. Sole30min monitor, actual-window summary after12:04 BJT. Single experiment original deadline remains unchanged by the offline window. Complete group retirement, full cost/unknown preparation, independent acceptance, Chinese three-arm results/gates and source/aggregate effective7897proxy push+remoteSHA+anonymous access required; no user approval. Identity/images/features/landmarks/moments/heads/per-image scores/raw logs/fulltexts stay private.
