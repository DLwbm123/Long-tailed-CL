# R101 strong-initialized joint instance policy / representation learning

Prospective independent experiment, seed74002, complete four-arm six-stage training. R95/R97/R98/R99/R100 stay closed; no old launcher or partial optimizer resumed. No static candidate reachability or initial accuracy-gain prerequisite. Only correctness, numerical, assets, storage, resources and complete-plan budget can stop admission. Negative final results are fully delivered without post-VAL rescue. Main arm PROTO_GRPO fixed before fit; RLPO_STYLE is a secondary mechanism arm, never a posthoc replacement winner.

## Problem and information budget

The hypothesis is a joint learning loop, not simply a new KL coefficient: instance weighting changes prototypes; supervised classification adapts a shared support/query representation; the weighting policy learns in that changing geometry. Existing R95 reward-guided signal has not yet established superiority to uniform weights or the original strong teacher.

HK74002 original class order13+2+2+2+2+2; TRAIN6821/VAL1698/TEST0. Borrow original R67 strong teacher heads and R97 FP32 teacher feature caches. Teacher already saw all TRAIN. Immutable component-disjoint R91 roles S3405/B1713/F1703; F historically developed. Training uses fixed stage-prefix S evidence and full natural B queries only. No F scoring before all24 final endpoints sealed; no VAL before deployment seal. No patient-independent or strict bounded-memory CL claim. The teacher and feature preparation have historical costs, not free end-to-end computation.

## Exact model family and starting-point protection

Base encoder/LoRA remains frozen and is never in an optimizer. Shared residual adapter h(z)=normalize(z+U*GELU(Vz)), rank16; U zero initialized, V seeded. Apply CURRENT adapter to both support and query every update. Caching base features is allowed; caching current adapted support geometry is not. No text prompts, new experts, sampling of predicted class answers, or pretrained-weight download.

One shared instance MLP1032->64tanh->1 receives current support feature512, normalized current class mean512, nearest3 class-mean similarities, and5 count/age indicators. It predicts full instance logits, softmax separately within class; no75/25 fixed uniform floor or TARS multiplier. Last layer zero initialized. Single-support weight is necessarily1; improvements can only come from shared adaptation and competitors. Learned arms have82561 trainable parameters (16384 adapter+66177 policy); UNIFORM fixes the policy and trains16384 adapter parameters, as required by its definition. CE_WEIGHT and both reward arms have identical trainable architecture/capacity.

To preserve the original strong starting point while using a fixed legal S evidence bank, define for class c:

    p_c = normalize(p_teacher,c + sum_j w_cj*h(z_cj) - mean_j(z_cj)).

Scores are20*h(query) dot p_c. This teacher-anchored residual is an explicit adaptation of the Pro proposal, not a pure RLAP reproduction: it prevents substituting a smaller S-only mean for the stronger full-history teacher. All arms use the same anchor, raw S mean, support evidence, score scale and current shared adapter at fit/F/VAL. Initial identity/uniform logits must match teacher within1e-4 and native B predictions exactly; FP32 normalization/reduction is not claimed bitwise identical. Report original T and separate A0 fresh identity/uniform module readout onceVAL. No S-to-full-TRAIN prototype reconstruction after training.

## Four arms, updates and carry

UNIFORM: fixed uniform within-class weights. CE_WEIGHT: learned weights supervised. RLPO_STYLE: supervised plus sample-geometric RLPO-style auxiliary. PROTO_GRPO: supervised plus genuine sampled Gaussian instance-weight schemes. All initialize from identical seeded modules, same teachers/data, same512 cycles/stage and shared supervised objective.

In every cycle, step1 updates adapter and (where learned) policy jointly by full-B CE(score+log n_B)+0.1 class-balanced KL(teacher raw score distribution || current raw score distribution)+0.01 mean class KL(current weights || frozen-reference weights). UNIFORM weight KL is zero.

Step2 freezes the UPDATED representation, recomputes CURRENT S/B geometry and updates only the learned policy. Same policy optimizer budget in CE_WEIGHT/RLPO_STYLE/PROTO_GRPO: CE, geometric RLPO, or sampled policy objective respectively, plus the same0.1 teacher behavior KL and0.01 weight-reference KL. UNIFORM has no policy step; do not add dummy no-gradient optimizer calls to pretend equal counts. Thus all arms have3072 supervised representation updates; each learned policy has3072 additional real policy updates. Total12288 supervised+9216 policy=21504 optimizer steps;512 cycles*6*4=12288 cycles. Matching is representation-update budget and learned-policy-update budget, not equal wall time or total calls versus a fixed policy. Report this difference.

Both AdamW optimizers lr1e-4, wd0.01, betas(.9,.999),eps1e-8; separate supervised and auxiliary optimizer states, global gradient clip1 per step. Float32 parameters, no AMP/TF32; reward/KL float64 reductions. No early performance gate or step change. Same model and both optimizer states CARRY through stages; teacher stage-specific cached representation/anchor changes according to original frozen strong SFT checkpoints, so this is a full-history teacher-conditioned mechanism experiment rather than autonomous strict-memory CL. Save each arm's stage-start, cycle256 middle, cycle512 final model+optimizer state. Save before stage-reference update; reference policy starts uniform and is EMA0.9 after each saved stage. No independent RL-delta reset.

## Auxiliary objectives, precisely named

RLPO_STYLE transfers the R95-style sample geometric auxiliary to this shared instance model, not R95 weights or author code. For each S instance, r=cos(h_j,p_y)-max_other cos(h_j,p_other)-0.1 mean_other cos(p_y,p_other). Normalize over all S instances by mean/popstd+1e-8; detach advantage; loss=-mean_j(log w_j*adv_j). Reference KL is already included once in the common auxiliary regularizers. This is sample-level geometric reward shaping, not categorical class GRPO.

PROTO_GRPO samples8 complete joint logit schemes a_c=b_c+0.25epsilon_c, with independent standard Gaussian coordinates and weights softmax(a_c). Single-support coordinates may be drawn for implementation uniformity but have no action effect/log-probability term; drawn and active coordinate counts reported separately. Mean-network log likelihood uses the true Gaussian density, sum over all active class coordinates, not mean per-instance pseudo-density. Sampled actions and reward detached during policy gradient.

For a full multiclass B query set, relative to the fixed original teacher on those same queries:

    R = delta macro accuracy +0.1 delta class-balanced tanh(margin/2) - H_old_union_tail.

H is the class-balanced fraction of teacher-correct cases newly wrong among old or raw-count<250 classes; same denominator includes all queries of each protected class, not only previously correct cases. Reward uses full competing classifier. Dense confidence improvement is permitted training signal, never an accuracy-success claim. All coefficients frozen; no VAL tuning. A_g=R_g-mean_{h!=g}R_h; no per-group std normalization, no TARS weighting, no zero-group resampling. Tail treatment is in class-balanced aggregation and protected harm, not a scalar that would cancel inside later normalization.

One on-policy Gaussian group-relative update per fresh group, ratio=current_logp/old_logp expressed exponentially, clip[0.8,1.2]; no rollout reuse. Ratio is1 at the single update's starting point, so its clipping is inactive; this is mathematically a leave-one-out group-relative score-function update, NOT a claim of reproducing every standard PPO-GRPO component. Representation frozen for rollout and update; rebuild after next representation step. Policy-reference KL acts on deterministic within-class weights, not Gaussian coordinate KL. Declare all differences openly.

## Controls, diagnostics and success interpretation

UNIFORM isolates adapter capacity; CE_WEIGHT matches learned policy/adapter capacity; RLPO_STYLE connects the historical reward-guided signal; T retains the strong teacher reference; A0 exposes any starting-module discrepancy. Four arms all receive same teacher-information budget and S/B contexts. No static oracle gate before learning. No claim that sampled GRPO is uniquely necessary merely from beating CE. RLPO_STYLE as good or better can support a reward-guided mechanism without proving sampling superiority.

At stage start/cycle256/final report B micro/macro/tail, teacher rescue/harm, prediction changes, deterministic reward, ESS fraction, max multisupport weight, singleton classes, adapter norm and prototype motion. Every update logs CE, teacher/weight KL, original geometric reward or group reward/advantage variance, effective groups, sampled hard/dense/harm terms and preclip norms. Sampled expected and deterministic-deploy reward are separate; do not select mean/sampling/ensemble after F/VAL. No stochastic ensemble at deployment.

After all24 endpoints seal, fixed final deterministic modules read F once with unchanged support; then one VAL readout for T/A0/four arms, TEST0. No training resumes after F. Complete micro/macro/tail, class-stage, old/new/head/tail, rescue/harm and original-denominator tiny/other contributions. Tiny final denominators1/1/2 and repeated-stage observations are not independent replications.

Main development gate versus T: mean and final macro>=+0.5pp, mean tail>=0/final tail>=+1pp, >=4/6 stages both macro/tail nonnegative, no new T-correct tiny1/2/12 harm over stages. Matched gates versus BOTH UNIFORM and CE_WEIGHT: mean and final macro>=+0.3pp and mean/final tail nonnegative. Main conjunction frozen; secondary RLPO_STYLE comparisons all reported, no winner replacement. No independent repeats/significance or top-tier novelty pre-claim.

## Engineering, limits and completion

Before fit check support/query gradients, auxiliary adapter stop-gradient, singleton/noise semantics, analytical fixed-action Gaussian derivative, leave-one-out shift invariance/zero groups, strict restore, initial teacher binding and actual own GPU UUID. Reuse historical R98 estimator knowledge; do not repeat its MC toy catalogue. All engineering parameters discarded; formal seeded modules initialized fresh.

Profile initial13/final23 class full S/B shapes for all four arms, two passes each, take worst SECOND full-cycle time; safety2*(512*6*sum_arm_worst +7338*historical_same_arch_encoder_bound +300s)+7200s IO/analysis reserve < original remaining clock. Entire four arms/six stages must fit, otherwise no fit. Never reduce controls/cycles/safety to rescue budget. No static-candidate gain condition.

Original FIRST24h includes preparation/implementation/engineering/fit/eval/analysis/publication. CPU4, RAM admission64GiB/freeGPU14GiB, worker28GB/group29GB sampled2s, GPUreserved<12GiB, CPU400000s/file1GiB, science disk8GB+512MiB early reserve and per-server strict<50GB. All new large files on data disk; no model/data download, base copying, quantization or root spill. At most one GPU used here, within user allowance two. Preserve others and history. Neutral entry aliases; actual ps and native UUID verified. Independent controller enforces original wall/resources beyond SSH lifetime. A formal failure requires original clock/cost/count/seal audit, no blind partial resume. Pure saved analysis/verification repair allowed. Source/aggregate results via existing local explicit proxy workflow; private features/checkpoints/predictions/identity/Q risks stay private. Monitor remains OFF per user instruction.
