# R121: one prospective paired-seed mechanism replication of frozen R117

Frozen before implementation or new model queries, 2026-11-09. R120 has completed, failed its original conjunction (6/10), and been verified/delivered. No further resolution, SFT-schedule, support or protection-threshold variants. This is the previously declared single new-seed replication of R117's positive and negative pattern, not confirmation of an already successful method. R117's original nine gates and failed head gate remain unchanged.

## Data, seed and common initialization

Static HK23 TRAIN6821 components/VAL1698; TEST0. Same manifests/prompt/class mapping/greedy evaluation/headTRAIN>=250/tail<250/tiny<25. No new patients or independent VAL, no claiming continual learning. Use generic Qwen2.5-VL-3B, frozen vision, language q/v LoRA r8 alpha16 at224square/64visualtokens. No loading previous task adapters, optimizer states or TARS table. Existing224 frozen visual features only are shared read-only, no new disk feature copies.

Exactly one new training seed **74006**. Reinitialize every LoRA A by CUDA generator74006/kaiming_uniform(sqrt5) in original adapter order, B all zero; independently replay the first A draw and verify the new As differ from constructor74005 and B/ref generic restoration. Set global torch seed74006; TRAIN natural randperm generator74006. This explicitly overrides the old Engine's hardcoded74005 initialization without altering that historical source. There is no seed selection or automatic second repeat on failure.

Same R114 bootstrap: two complete natural TRAIN permutations, effective batch16/micro2,854updates/13642queries, AdamW lr2e-4/wd0, gradclip1, lr(k)=2e-4*.5*(1+cos(pi*k/854)). Save final common SEED_SFT, freeze reference; no epoch selection. Full TRAIN greedy once yields fresh class error and fixed TARS w=max(1,[(mean(n)/n)*(1-correct/n)]^.75/median_floor1e-8). q=p*w/Z and rho=p/q, Z=sum(p*w). This is replication of the procedure under new randomness, not reuse of old weights or old queries.

## Paired posttraining controls

Query generator11774006 generates256x8 queries by class q then within-class replacement, shared across all arms. Response generator seeds11481000+k, shared across two RL arms at each update. Native engineering seed121000; discard native updates and restore exact generic params/ref before any scientific SFT. No filtering/resampling zero-reward or constant groups.

All three arms independently restore the same SEED_SFT with fresh AdamWlr2e-5/wd0,256updates/G8/clip.2/beta.04/gradclip1/max24/temp1/std+1e-6/micro2:

- SEED_TARS_FULL_IS: PG coefficient rho*w=Z, KL coefficient rho.
- SEED_GRPO_FULL_IS: PG coefficient rho, KL coefficient rho; same query allocation, no TARS in objective.
- SEED_CE_FULL_IS: teacherforced class coefficient Z, no KL, fewer forwards; not a compute/KL-matched proof of RL-objective necessity.

Record exact per-class allwrong/allright/mixed groups, successful responses, query counts/unique components, actual parameter L2/nonzero updates separately from optimizer attempts, KL and behavior-path ratio checks. Fixed-policy expected-objective identities are not identical Adam trajectories or guaranteed full-variance reduction. BF16 sampled versus teacherforcing numeric discrepancy remains disclosed; unchanged same optimization-path logp difference must equal0.

Seal common SFT and all three arms before any new VAL feature reads/generation, then four common greedy VAL readouts. SFT/initialization contribution and incremental RL are separate. No choosing seed, checkpoint, gates, classes or hyperparameters from partial responses or VAL.

## Predeclared readouts and stopping

Report seven-component diagnostic conjunction: main over own SFT macro>=+1pp/tail>=+2/head>=-1; main over matched GRPO macro>0/tail>0; main over CE macro>0/tail>0. Also separately report six directional positive-signal components: macro/tail>0 over ownSFT,GRPO,CE, and whether all six repeat; this directional readout does not erase the seven-component failure or original R117 head failure. If head fails, say the head-harm pattern persists even if positives repeat. If positive directions do not repeat, preserve the failed replication and do not select another seed. Always report all raw metrics/counts, per-class offsets, tiny denominators and original74005 results side-by-side without formal significance from two seeds.

The historical PG_ONLY/naturalTARS controls are not rerun; do not claim reproduction of all original nine gates or the whole R117 matrix. This minimal main/GRPO/CE pairing tests repeatability of recipe-level advantages, not isolated importance-correction causality. No further seed or static variation is automatically authorized by this protocol's outcome; analyze complete evidence under ongoing research authorization.

## Hard resources and real admission

Only pro5000physicalGPU1, neutral visible main.py/controller/worker argv. First-preparation clock in private state, harddeadline8h later (<24h), native/preparation/retries/children share originalclock, no reset. Independent controller checks every2s, stops deadline-120s or cumulative conservative disk>=9GB/RSS>=48GB/free<1GB, with termination escalation. Cumulative all-window new physical strictly<10,000,000,000B across retained R114–R120 roots, this root, local source/delivery/tmp/cache/logs. Baseline prior retained roots5,866,721,280B plus200MB localreserve; new checkpoints/reports/cache expected<100MB, conservative full peak<6.3GB, no model/features downloaded/copied. Filecap256MiB/core dumps0/CPU4/GPUallocator56GiB/startFSfree5GB/hostavailable24GB. Mount/free/write-read probe and current GPU free verified before large work. No my-gpu/otherprocess intervention, deletion of history/failed evidence/models/data, or accounting resets.

Native exactly TRAINrows0,1:224native/manual logits and old224cached features parity<=.05, genuine SFTCE/2RL/weightedCE backwards and optimizer paths, one greedy timing profile, objective/gradient fixture, generic parameters/reference restored, legal zero steps retained. Forecast entire SFT+TRAINfit+3post+4VAL with1.5margin+1800s reserve must fit originalremainingclock before science. Normal costs:4discardednative+854SFT+768post attempts;46,415responses=34native+6821TRAINfit+32768RL+6792VAL;4TRAINencodings=2freshnative+2nativefullforward repeat,0newVAL encodings. Reused6821TRAIN/1698VAL features incur historical encoding cost, not new.

After actual CONTROL_EXIT0, CPU-only saved finalizer checks seed initialization/replay/generic no oldtaskload,854SFT+3x256curves,92classrows,freshTRAINweights/tape/math,actualupdates/rewardgroups,seven gates and directional readout,seals beforeVAL,costs. Safe aggregates/source/report only published by existing explicit7897proxy/allowlist/no-force/remoteSHA/anonymous200 workflow. No identities/images/features/weights/individual predictions/privatepaths/credentials/PDFs. Same existing thirty-minute research task; no new monitor.
