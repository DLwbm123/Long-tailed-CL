# OPSA internal last-block pilot — fixed before actual execution

User requested immediate experiments after implementation. This is an independent, bounded internal-adapter transfer on existing ISIC and HyperKvasir, not an author-code reproduction or completed continual-learning experiment. Implementation source is derived from commit 10acbcd; the unresolved MAPC activation-to-parameter mapping is explicitly LOCAL row-any/down-only, resetting before standard AdamW accumulation. No basin confinement or minority-safety theorem is assumed.

## Data and isolation

Use existing ISIC initial2555, original seed74290 lesion-linked split and support-only F mapper/head for each of the two producers. No original681 development, future classes or test. HK uses original S_old3140/13 classes and disjoint E986/Qgate388; these are query cohorts, not producer folds. Known-group exclusion is enforced across the entire chain; patient independence remains unknown and is not a prerequisite. Only the explicitly authorized existing original images are read with existing canonical 224x224 bicubic/.5 normalization. No images, pretrained weights or environments are copied/downloaded. HK labels1/2 singleton support remain included; repeated sampling does not create independent examples.

All three producers and all learned arms are saved/read back before any query image forward or risk calculation. Training image opens are constrained to current support paths. Only immutable frozen reference F assets may be reused; no old T/P/C/D/CLIP learned initialization. Metadata/immutable F files may contain held-out rows, but training selects support rows before any feature operation. No query-based selection, early stopping, temperature, offset or alpha sweep.

## Internal intervention and controls

Reuse the exact original locked pretrained APART/ViT-B16 backbone and deterministic pointwise dual streams. Freeze all original parameters and buffers. Attach one shared rank16,width768 adapter as a pre-hook immediately BEFORE backbone.blocks[11] (zero-based last block), affecting all tokens in both streams. This is inside the encoder, before final attention/MLP/norm, not a pooled-cache-head substitution. No stochastic backbone training, AMP, gradient checkpoint recomputation or distributed execution. No new activation cache.

Original F is the unchanged reference. Train four arms for each producer:
- H: original F mapper/head, head-only CE control; uses immutable support cached features.
- L: same adapter parameters, token normalization, quadrature and fixed up radius, but both cosines fixed to one; plain AdamW.
- N: full OPSA forward, plain AdamW (no MAPC).
- M: same full OPSA forward plus the disclosed local MAPC rule.

L/N/M have identical initial parameter tensors, original F head initialization, schedules and sampled image rows. H uses the same initial head/orders. Wdown std .1/sqrt(768*16), Vup std1/sqrt(768*16), gamma8,kappa1,delta.25. Up initialization is a local convention. The initial residual is nonzero. A fixed support batch records no-adapter replay and initialized perturbation before training; no efficacy claim from this check.

Each arm: exactly800 updates, batch16, seed74440+producer index. Draw with replacement from class/group-equal/image-within-group weights; identical orders allarms. This is a new explicit sampling contract, not the old eight-distinct-groups recipe. Unweighted minibatch CE is unbiased for that frozen support measure. AdamW adapter LR1.2e-4/head2e-4, wd.01, beta(.9,.999), eps1e-8; 200-step linear warmup then600-step cosine decay; global gradient clipping1.0. Head remains the original mapped 1535+intercept linear readout, trained jointly. Not the paper100epoch or full12-site reproduction. No ridge solve/mapper fit. Three datasets × four800-step arms =9600 updates, including7200 image-adapter updates and2400 cached head updates.

MAPC trigger: any training token in either stream over pi/(2*kappa)-delta marks that down-projection row; only elements with sign(current gradient) != sign(old firstmoment) reset. Other moments and up parameters are unchanged. Record activation maxima/boundary fraction/reset counts, CE trajectory and parameter displacement. Zero MAPC activity is reported, not a reason to retune. No selective rescue of class0.

## Checks, risks, stop

Before training: first32 support images only, fixed existing feature tolerance1e-6 and mapped-score tolerance1e-5 (inherited benchmark replay tolerances); matching original support row labels/groups; frozen backbone versions; trainable count; synthetic gradient/frozen-backbone hook test. Deviations fail without retry. These checks do not redo GGA or relax its retired tolerances.

At completion: all model parameters finite, backbone versions unchanged, full artifact serialized fields readback exact, support score readback <=1e-10 using same in-memory features, all12 heads sealed. Query F scores from canonical replay compared to original sealed F responses (<=1e-5, exact predicted labels for ISIC, exact per-class counts for HK). Allclasses and bothISIC folds/HKcohorts included. Save private responses, publish aggregate only.

Primary M gates: pooledBA >= F+1pp; class1 at least one new net correct and paired H=0; ISIC eachfold BA/class1 no loss; HK eachcohort BA+1pp and class1+1; pooled/cell allclass recall loss<=1pp; M pooledBA strictly above H/L/N, eachcellBA at least H/L/N. All gates required on BOTH datasets. Report paired rescue/harm vsF/H/L/N. No bootstrap or confirmatory success in this pilot; CI NA even if point gates pass. Positive point signal requires a new inference protocol, not automatic promotion. Any point failure STOP_OPSA_INTERNAL_PILOT; no coefficient/LR/step/seed/mapping rescue of this endpoint.

## Resources and delivery

One run, GPU0 only; no per-arm retry/restart. Absolute wall21600s from first actual including checks, alltraining/evaluation; coordinator TERM at21590, KILL5s later. Worker CPU36000s, RSS8GiB, per-file64MiB, total new round including reserve32MiB strictly<512MiB. Original17GB soft/20GB hard window and1,001,040,384 outside reserve retained. Prelaunch NFS capacity/write-read and GPU>=8GiB free. Models use shared immutable backbone; no backbone copies. Expected total new actual storage<256MiB, no new big cache. Budget does not use removal of old artifacts. These small adapter artifacts remain below the old19,058,800 individual cap as well.

Neutral parent/worker argv and GPU display, independent guard, one existing30min heartbeat. Check short launch health then background. Publish source/protocol before actual via proxy, aggregate/cost/failures after actual. Record wait4CPU/wall/coordinator once at exit. No raw images/IDs/features/models/Gram/grad/private logs/third-party material public. Preserve previous stops and cumulative costs.
