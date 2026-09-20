# V8: Old-synthetic per-head objective, one controlled H-based fork

## Evidence and question

H (V6) final val BA52.580%, old recall47.526%, current67.743%, fails the old-recall gate by2.309pp. I (V7) exactly preserved old classifier rows, including cancellation of AdamW decay, yet BA fell to50.447% and old recall to44.051%. Thus row freezing alone is not a supported repair here. H/I still show recent-class preference in the main head and near-zero current recall in the few head. Existing real CE includes main, sum and pool-weighted few terms; old-synthetic CE includes only sum. Does providing direct old supervision to each existing head improve old/new discrimination? This is an unverified hypothesis, not a promised positive result.

## Single change relative to H

Candidate J uses the same old Gaussian samples and targets. If m/f are the main/few logits restricted to all seen classes, set

    L_old = [CE(m + f, y_old) + CE(m, y_old) + CE(f, y_old)] / 3

Each CE uses its original batch-mean reduction; multiply this average by H's existing known/current coefficient, S1=2 and S2=3. There is no extra loss weight, bias calibration, new head or module. This is a separate explicitly named objective variant, not a claim that original ConCM-lite was already defined this way. Original APART/ConCM-lite code and completed outputs remain untouched; legacy sum-only dispatch is the default. No coefficient/grid search.

Real-image main CE, main+few CE, pool-weighted few CE, their existing reduction and /3, pool assignment, pull, theta, optimizer, sampling and augmentation are unchanged. The few real term remains pool weighted; J is NOT exact symmetry or a class-balanced objective. Synthetic few CE has no pool gate because the memory does not store a synthetic pool assignment. Do not invent such labels/statistics. No old-row constraint from I is used: all four head tensors are optimized as in H. Future rows retain H optimizer behavior but cannot receive CE gradients. Non-head parameters/buffers stay at each paired C-S0.

Training and diagnostics share the same synthetic CE helper. Log all three raw component losses, their mean, and weighted replay. Fixed probes report actual J gradients plus a clearly labeled sum-only shift reference. The actual common shift probe adds one unit to each head's current logits, hence two to their sum; its expected derivative is (current probability mass main + few + 2*sum)/3 on old targets. No diagnostic changes formal state or RNG.

## Locked matrix and data

Three prescribed seed/order pairs1993/1994/1995. Each forks from its original V2-C S0 by ordinary audited restore; never train from H/I final heads or retrain S0. S1/S2 each exactly10epochs, yielding60epochs and6final checkpoints. The first formal epoch is the throughput pilot and is resumed, not repeated. No early stop or best-epoch selection.

Use unchanged V2 data and pinned AugReg weight: train18,718, val295, test764, actual train imbalance389.963:1, eight classes4+2+2. Use current train images only, preserve original ConCM-lite train-only memory, 4 synthetic samples/old class, cap48. No split reconstruction, quota repair, restored samples or old image replay. Seen-class val only, sorted-ID batch48, main/few/sum. New test predictions0. No val fit, replay or statistics. All parent/code/protocol/manifest/weight locks remain mandatory.

## Engineering and resource gates

Before full launch: a real update; finite losses/gradients; unchanged feature tensors; exact checkpoint restore and next update; wrong parent/seed rejection; actual synthetic sampler/head dispatch and gradients equal the independent explicit three-CE formula; future CE gradients zero; legacy sum-only loss unchanged; diagnostic component weights and actual shared-shift derivative match the objective. Check real stream/exposures against V3 E and H, and per-epoch raw replay equals the three logged CE mean. Keep checkpoint format S0_HEAD_DELTA_V1 with immutable parents and ordinary resume guards.

Use hb01:30154 with neutral process argv, maximum two training workers and measured GPU headroom. Keep at least1GiB disk safety space and preserve every historical checkpoint; no cumulative post-V3 GPU-hour limit or capacity purchase. Record elapsed/GPU-process time, throughput, memory and disk. Stop on data/weight/parent drift, nonfinite values, state mismatch or resource failure, preserving all evidence.

## Comparison and decision

Main success J-C: mean final BA> C, current recall at least+10pp, old recall decline at most5pp, at least2/3 seed pairs improve both BA/current, no additional current zero-recall class at S1/S2. Mechanism contrast J-H; also report J-I/G/F/E using the same val layout. All repetitions and negative results are published on their own branch, no main merge or test use. These are adaptively reused development-val gates, not independent generalization evidence. Patient isolation, clinical label mapping, pretraining exposure and Gaussian/real feature mismatch remain unresolved.

This executor stops after J's complete fixed matrix and report. The authorized hourly monitor may choose another single evidence-based follow-up if fixed criteria fail, but no Full Dynamic, joint training, extra backbone or broad parameter search. On success complete public delivery and pause monitoring. Preserve V1/V2/V3 and V4-V7 conclusions and artifacts.
