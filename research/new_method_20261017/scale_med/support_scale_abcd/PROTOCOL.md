# A-D support-side scale experiment — frozen 2026-10-22

## Authorization and question

User authorized A-D only, freeze splits/controls/criteria/budget before launch, no E/Qgate tuning, analyse results before any next experiment. This is one bounded independent experiment after the earlier project pause, not a restart of old training or a continuous search. No reliability module, hierarchy arm, loss change, full-support final refit, query evaluation or subsequent experiment. Final reporting and public source/aggregate delivery are in scope.

Question: does one support-selected global NCM probability temperature improve the existing half-mixture under honest group-out evaluation? Run both CLIP ViT-B-32 OpenAI and pinned BiomedCLIP, preserving prior frozen FP32 loaders, image transforms, prompts, adapter, loss and optimizer. It is a controlled readout study, not an independent confirmation of a method.

## Frozen support-only partition

Use only the existing 3,140 support images of 13 classes and their known identity_component metadata. Known class group/image counts are [502,1,4,69,232,323,590,392,202,135,477,209,4]. Assert counts and single-label groups before sealing. No patient-independence claim. Class 1 has one group: include it in all training sets, never in validation, exclude it from the 12-class primary BA and calibration objective. Predictions still cover all 13 classes, including class1 as a possible error.

Four outer folds. For each non-singleton class, sort group IDs, shuffle using Python Random(86420+class_id), assign j modulo4. Within each outer training partition, re-sort remaining groups, shuffle Random(86420+1000*(outer_index+1)+class_id), assign j modulo3. Group membership is indivisible. Every eligible support group is outer-held-out exactly once, and each outer-training eligible group is inner-held-out exactly once. Each 4-group class has 3 groups in outer training and 1 in outer test; each inner training has 2 and inner validation 1. The singleton remains training-only throughout. Both backbones and all seeds share the exact split. PLAN.private.json seals complete support rows/indices; SPLIT_SEAL.json exposes only counts. Entry refuses metadata drift.

Frozen backbones encode support images once each; sharing these frozen, label-independent representations across folds does not fit on held-out labels. Every inner and outer adapter is fresh; every NCM prototype uses that fit's training groups only, averaging within group then across groups and normalizing. Outer held-out labels never select temperature, checkpoint or stopping time. Image I/O guard restricts files to support paths and prohibits E/Qgate/responses artifacts.

## A-D and fitting budget

Seeds74510/74511/74512. For each backbone/seed/outer fold: three inner adapters generate held-out calibration outputs, then one outer adapter trains on that outer training partition. Total2*3*4*4=96 fits and57,600 optimizer updates. Each fit uses600 updates, batch64 sampled with class/group-balanced weights, identity-initialized visual/text512x512 adapters, AdamW lr.001 cosine decay, weight_decay.0001, gradient clip1, existing flat CE+.001 identity penalty. No early stopping, seed selection, ridge solves, text encoder updates or hierarchy loss. Parent prompts remain in the reused flat implementation but their loss coefficient is0. Sampling generator is the named seed for each fit.

A=NCM softmax(10*cosine); B=pure adapter softmax(existing logits); C=.5*(A+B); D=.5*(softmax(NCM_logits/T)+B). A-D share a given fitted model/prototypes; there is no separate training per readout. Adapter temperature and probability mixing weight remain fixed.

T candidates exactly[1/16,1/8,1/4,1/2,1,2,4]. For each backbone/seed/outer fold choose one global T by minimum class/group-balanced NLL of the MIXTURE on concatenated inner held-out predictions. Every eligible class gets1/12 weight, every group equal within class, every image equal within group. Exact ties prefer T nearest1 in log2 space, then lower list index. Persist choice and full seven inner NLL values before outer fitting/scoring. No outer temperature sweep, classwise temperatures, mixture-weight search, outer-label choice or post-result change of grid.

## Metrics and predeclared decisions

Primary: pooled outer OOF class/group-balanced accuracy across12 estimable classes, per seed and their mean. Also report image BA, balanced NLL, all13-class confusion/counts (singleton denominator0/recallnull), all four outer fold results and every selected T. Four-group classes are exploratory, not independent replication. No compare-to-old E/Qgate BA as same-protocol performance.

Report C and D relative to the two branches: NCM-exclusive correct retained, adapter-exclusive retained, both-wrong rescued, rescued/harmed versus adapter, and paired D-C/D-B changes per class. Distinguish net count and changed identity. Private per-group OOF predictions remain remote for analysis; public output only aggregates.

Per backbone, frozen point gate: mean(D-C)>=.0100, D-C>0 for all3 seeds; mean(D-B)>0 and D-B>=0 for all3 seeds. Protection gate: mean recall difference versus BOTH B and C >=0 for classes2 and12, and >=-.0100 for every other estimable class. Class1 protection is NOT ESTIMABLE, never assumed passed or imputed. These are pragmatic screening thresholds, not statistical guarantees.

Uncertainty: 2,000 paired stratified known-group bootstrap replicates, fixed seed86421, resample within class using the same group for all three seeds, average seed differences before sampling. Report percentile2.5/97.5% intervals for D-C and D-B. This is conditional on fixed OOF models; overlapping training folds, unknown patient relationships and very small groups prevent treating it as a full independent generalization CI. Uncertainty gate requires both lower bounds>0. SUPPORT_SCALE_CANDIDATE requires point+protection+uncertainty gates; otherwise NO_QUALIFIED_SCALE_SIGNAL. Both outcomes keep method_success=false. No automatic advancement even if one/both backbones pass. Report backbone outcomes separately; do not select a winner and suppress the other.

## Resource and stop contract

One original wall clock for BOTH backbones:7200 seconds from coordinator launch; TERM at7190, KILL after5 seconds. Failed attempts and retries share this clock; no automatic retries of scientific work. Worker CPU hard limit14400 seconds; RSS guard6GiB. GPU0 only; preflight free memory exceeds8GiB. Existing environment and CLIP checkpoint are read-only reused. Biomed pinned783705670-byte checkpoint streams into bounded RAM using the previously qualified loader; no persistent full checkpoint or new text-model download. Small tokenizer/config assets under this round only. No dependency install.

New storage per server strictly<1,000,000,000 bytes, counting allocated blocks or logical size, whichever greater, plus64MiB safety reserve and temporary neutral entry overhead (under1MiB). Stop new work at cap-reserve. Expected persistent models~202MB logical, support features~13MB, small outputs; conservatively<600MB allocated. Per-file1GiB guard is additional, not a substitute for total cap. Preserve all old artifacts; no cleanup of prior experiment resources. The previous experiment's spent time/storage and negative results remain recorded; this is a new support-only protocol with its own stricter cap, not a retry of that experiment.

Independent coordinator survives chat/SSH and guards wall/RSS/storage. Launch with neutral parent/worker argv and verify ps plus nvidia-smi. Record actual encoding counts, update counts, model receipts, resource exits and source commit. Synthetic self-checks are labeled separately from real fits and are not scientific results. On failure preserve evidence, analyse, and stop; do not silently expand limits or replace protocol.

Before actual run: split seal, synthetic logic checks, public source/protocol commit pushed via proxy with remote ref and anonymous access checked. After completion: analyse and publish source/protocol/aggregate/resource report only, excluding private rows, identities, features, adapters and raw logs. A single scoped30-minute monitor may follow this run to analysis/delivery only and is removed at completion; no next experiment is authorized.
