# NIH: stable local evidence and patient-mode prototype composition

This is a new complete main-line hypothesis, not a mask or auxiliary-loss rescue of the closed token recipe. No performance is known at registration. Source uses the existing NIH split, class arrival order and repeated 300-image development validation; no test/Etest, future initialization, extra image download or validation-driven selection.

## Mechanism

1. A frozen pretrained full ViT produces 197x768 tokens. On base classes only, learn a shared 768→128 projection, eight competitive local queries and a reconstruction decoder. A temporary cosine base classifier supports discriminative learning. Each image descriptor contains normalized projected CLS and eight normalized local summaries (1152 dimensions). Optimize base CE + patch reconstruction + query diversity for 1000 updates, logical64/micro8, AdamW3e-4/wd1e-4/clip5. Drop 10% complete patch vectors only during this phase. Freeze the complete evidence map before any later class arrives. The decoder is retained for audit but unused at deployment.
2. At actual arrival, image descriptors are averaged within each patient. Fit two deterministic patient modes (8 kmeans iterations plus final assignment). Retain mode counts, local means, diagonal empirical variances, sums/squares and full-descriptor class means. Empty modes have zero mass. Singleton variance is zero before flooring. Keep at most32 metadata-selected patients/class, one immutable descriptor each. No old raw images/tokens or all-patient historical descriptors retained. Old modes and descriptors never refresh; no nonlinear moment transport.
3. All-seen discriminative cosine anchors are combined with class-conditional local mixture energy. Component log Gaussian includes its volume term; class-conditioned attention averages part evidence, then logsumexp mixes modes and divides by128. Because attention depends on the query, this is a density-derived energy, not a normalized generative likelihood or calibrated probability. Primary variance is n/(n+16)*empirical +16/(n+16)*fixed base class-balanced within-patient variance, floor1e-4. Means are not replaced. Two global mixing logits [0,-2] are learned with the head. Center alignment .05 and head-geometry .01 act on decision anchors, not a theorem-backed collapse of patient modes.

## Fair strong controls

GLOBAL uses the same learned full1152 descriptor, head initialization, memory, class-mode allocation and training batches, with cosine scores. UNSHRUNK uses the exact same local/global composition with empirical variance instead of shrinkage. CALIBRATED is primary. All allocate the same head/mixing parameters and retained-statistic formats; GLOBAL mixing is inactive, so active parameters/operations differ and equal FLOPs are not claimed. The evidence map is shared and learned once. This isolates incremental inference composition; it does not isolate the benefit of learning the evidence map itself.

All use all-seen CE + identical anchor penalties, AdamW3e-4/wd1e-4/clip5, 2000 base +5x500 later updates, optimizer reset each stage. Each batch draws8 seen classes uniformly without replacement,8 patients/images per class, one undropped descriptor view. Identical actual batch across arms. Source is old finite memory plus current complete descriptors, without duplicated current-memory rows. Each arm4500 head updates/288000 model rows; shared base learner1000 updates/64000 token forward-backward rows, explicitly extra cost. New heads initialize only from actually arrived patient-equal descriptor means.

## Budget and stops

One original remote admission clock: 8h, never reset by qualification, preparation, failure, waiting or acceptance. 3,000,000,000B hard /2.7GB prestop, host40GiB/GPUreserved8GiB, freehost48GiB/freeGPU8GiB admission. Raw current tokens exist in RAM (~18.2GB base), never whole-table concatenated or disk cached; release before later arrivals. 5s independent process-group guard, TERM60s before deadline/KILL30s if needed. Window roots plus entries of this experiment AND asymmetric_replay count together against30GB/27GB remote prestop; local papers/source/JSON ≤100MB reserved, no deleting old evidence to reset usage. No automatic scientific retry or parameter sweep.

## Mandatory qualification before bulk

20 legal base-class images, real reader→formal GPUFP16 full encoder→GPUFP32 evidence optimizer logical64/micro8 (actual parameter change)→freeze→8+2 simulated arrival/patient modes/memory/head for all3 arms, forward/backward/clip/AdamW each2 updates→save/load→CPUFP64 map/classifier versus independent NumPy attention, energy and objective. Check singleton/empty masses, finite variances and finite-difference head/mixture derivatives. Seven discarded optimizer steps, never science. PASS written before bulk is programmatically required.

## Acceptance and gates

18 head models sealed before opening validation;300 validation images encoded once. CPUFP64 evidence map shared, scores at actual seen prefixes. Independent NumPy map and18 scores ≤2e-6 and all predictions identical; actual-training-batch endpoint objective; memory metadata independent reselection/old descriptors and modes exactly unchanged; patient intersection0; mapping exactly frozen; counts/budget.

At each arrival, a separate NumPy implementation refits patient modes on the complete current descriptor table BEFORE release, records PASS and retains sufficient moment proofs. Final independent acceptance reconstructs moments from those proofs. It does not retain all historical patients, independently re-extract images, or replay every optimization step. Online refit is additional cost, not extra independent patients. No acceptance claim before actual verifier completion.

Primary must pass all13 fixed gates: final≥91/300, six-stage mean≥36.005291005291%, tail7≥25/105, initial≥44/150,later≥41/150; versus each GLOBAL and UNSHRUNK final+4, mean+1pp, tail4+2/60,later not lower. Any failure closes the fixed recipe, no swapping primary, re-fitting rescue or validation tuning. Science failure is distinct from engineering failure.

## Literature and claim scope

Mechanistic inspiration: PDAM competitive local evidence (Desk Rejected submission; no endorsement), OGPCMIL class-conditional evidence aggregation (no gene inputs transferred), DAP patient-count statistical reliability, BASIL anchor alignment, FBCL class geometry. Adaptations differ from these papers; no faithful reproduction, inherited guarantee, TMI novelty or NIH benefit is claimed. The full catalog was metadata-screened; only seven identified method bodies were read, not all ICLR2027 PDFs. See ../iclr27_nih_mechanisms_20261022/REPORT_ZH.md.

Completed run requires full group retirement, independent acceptance, all costs including qualifications/failures/unknown preparation, Chinese three-arm metrics/gates and source/aggregate proxy push + remote SHA + anonymous access. Private identities/images/tokens/features/weights/per-image scores/fulltexts stay private. No user review is required.
