# V7: Preserve old classifier rows with stationary S0 features

One autonomous post-V3 follow-up, candidate I, locked before formal training.

## Evidence and hypothesis

H completed 3/3 repetitions and 60 epochs. H-C final val BA +10.732 pp, current recall +64.854 pp, old recall -7.309 pp; retention gate fails by 2.309 pp. H-G old recall +3.747 pp with current recall -4.986 pp shows partial benefit of replay pressure. H fixed-current-train probes retain substantial old-row real gradients, sometimes with very weak replay gradients. This motivates a single constrained-classifier test; it does not establish causality or guarantee improved accuracy.

## Only change relative to H

At each incremental session start, save existing classes' rows in BOTH main/few linear heads, including biases (S1 rows 0:4, S2 rows 0:6). Compute the identical H losses and gradients. Immediately before each AdamW step mask old-row gradients; after the step restore those old rows exactly, including cancellation of decoupled weight decay, and zero their exp_avg/exp_avg_sq (and max_exp_avg_sq if present). The scalar Adam step count remains shared and unchanged. Current and future rows retain H behavior; future rows receive no CE gradients. No new module, trainable parameter, loss or hyperparameter is introduced.

Persist the session anchor and constrained step count inside each compact checkpoint. Ordinary V2 restore first validates the parent, order, seed, code, data, weight and training arguments. Delta resume also validates the row-anchor metadata and exact agreement with stored head rows, then restores optimizer/memory/RNG normally. Never re-anchor from a corrupted resumed state. At every epoch and checkpoint assert old rows remain bitwise unchanged and their Adam moments are zero. Raw diagnostic gradients remain unconstrained counterfactual gradients; parameter updates enforce the constraint.

## Fixed matrix and access boundary

Three independent paired seeds/orders: 1993, 1994, 1995. I forks from each original V2-C S0 using ordinary audited restore, not from H's trained heads. S0 is not retrained. S1 and S2 each have exactly 10 epochs, producing 60 new epochs and six final checkpoints; no early stop or best epoch. A pilot is the first formal epoch and is resumed exactly.

Retain H's fixed S0 non-head tensors/buffers, all-seen real CE, original reduction and /3, pool/few weighting, pull, theta, batch48, augmentation, natural long-tail current train sampling, original ConCM-lite Gaussian statistics and 4 synthetic examples per old class. Effective replay coefficient remains known/current count: S1=2, S2=3. The original frozen data, weights and S0 parent states are untouched.

Use only the V2 locked train and seen-class val. No historical real-image replay or val statistics for fitting. Evaluate sorted-ID batch48 and main/few/sum identically to C/E/F/G/H. New test predictions must be zero. Protocol/manifests/code/parents retain explicit locked SHA checks. Clinical class names, patient isolation and pretraining exposure remain unresolved.

## Gates and decision

Engineering: one actual update; unchanged non-head tensors; future CE gradients zero; current rows update; old rows exact despite AdamW decay and deliberately nonzero optimizer moments; delta roundtrip and next update bitwise equal; incorrect parent, seed and old-row anchor rejected. Measure actual throughput/memory/disk before launching the full paired matrix. At most two GPU workers on hb01:30154, neutral process argv. No cumulative GPU-hour cap after V3; retain at least 1 GiB disk safety headroom and all original artifacts.

Main success I-C: mean final BA > C; current recall gain >=10 pp; old recall loss <=5 pp; at least 2/3 pairs improve both BA/current recall; no additional current zero recall at S1/S2. Mechanism comparison I-H; also report I-G/F/E. Failures are retained and published. These are development-validation gates after adaptive reuse of val, not held-out scientific confirmation. Publish all repetitions and distinguish fact/inference/open questions. This executor stops after this matrix; the hourly monitor may choose another single evidence-based intervention under the user's autonomy authorization if the fixed gates fail. Stop/pause on success after public delivery. Do not launch Full Dynamic, joint training, extra backbones, broad parameter searches or test evaluations.
