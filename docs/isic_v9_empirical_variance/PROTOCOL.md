# V9: Empirical diagonal variance without the inherited upper cap

## Evidence and hypothesis

V8 J completed all60epochs/6checkpoints and engineering/completion checks, but J-C old recall declined9.369pp and J-H BA declined0.937pp. The per-head replay objective is not retained. H remains the comparison backbone with sum-only replay, fixed C-S0 features and coefficients2/3.

A direct audit of stored training-only memory found that 99.219–99.870% of dimensions in48 final class/head states exceed the inherited variance cap1.0; the cap removes an unweighted mean75.600% of summed diagonal variance. Mean variance per dimension is3.163–5.194. The final-current states are included in48; the full report also separates the36 actual S2 old-class states. Example seed1993 class4 main RMS norm implied by the stored moments changes89.937→75.668 under clipping. These are training-moment calculations, not validation fitting or observed accuracy gains.

Hypothesis: extreme upper clipping makes synthetic old classes too concentrated and easy to separate, weakening retention on real old images. Removing this particular mismatch may help; diagonal independence and non-Gaussian effects remain unresolved. No positive outcome is assumed.

## Single intervention relative to H

Candidate K samples each head's existing class memory as

    x_synthetic = empirical_mean + Normal(0,I) * sqrt(max(empirical_variance,0) + 1e-6)

Only the upper variance cap1.0 is removed. No variance multiplier, searched cap, full covariance, feature normalization, temperature, head norm, new loss or module. Independent main/few noise, original RNG order,4 samples per old class, cap48, target mapping, means/statistics and epsilon are retained. The wrapper temporarily passes None for the original torch.clamp upper bound and restores the legacy attribute in finally. Original APART sampling code remains unchanged.

Exact configuration: synthetic_variance_rule=empirical_unclipped, effective_synthetic_var_max=null. The inherited concm_stage1_var_max=1.0 remains recorded as the legacy setting, explicitly overridden only while K samples. Legacy dispatch remains unchanged for other variants. K has no I row freeze and no J three-head objective: replay is original sum-only CE over all seen classes, multiplied by known/current2 atS1 and3 atS2. Keep H real losses/reductions,/3,pool gate,assignment,pull,theta,optimizer,sampler,augmentation and fixed C-S0 features. All four head tensors update as in H; future classes remain outside CE.

## Matrix and data boundaries

Three locked seed/order pairs1993/1994/1995; fork each original V2-C S0 with ordinary validated restore and inherited memory/RNG. Do not retrainS0 or start from any H/I/J final model. S1/S2 each10epochs,60 total new epochs,six final incremental checkpoints. Formal first epoch is the throughput pilot and is resumed exactly. No best epoch, early stop or cancellation of remaining seeds based on initial performance.

Use unchanged V2 train18,718/val295/test764,eight classes4+2+2,actual training imbalance389.963:1. Fit current train only and original train-only statistics; no old real-image replay,val fitting,quota repair,split remake or restored samples. Evaluate seen val in sorted-ID batch48 and all main/few/sum outputs. New test predictions0. Preserve all original data,weights,checkpoints and historical conclusions.

## Engineering and resource gates

Before full launch: actual real update; finite losses/gradients; frozen feature tensors exact; future CE gradients zero; exact delta restore/next update; wrong parent/seed rejected. Check actual K samples bitwise against an independent empirical-variance formula with identical noise, compare labels and final synthesis RNG to legacy clipping, demonstrate sample differences only from the variance rule, and restore the legacy cap attribute after each call. Diagnostics use the actual sampler; the existing sum-only gradient probe is therefore the actual K replay objective. Verify raw/weighted coefficient2/3 and pair all real input streams/exposures with H/V3E.

K memory itself must match H stored train statistics under the same frozen features and seeds; audit compact checkpoint memory tensors on completion. If this equality fails, investigate rather than treat the run as a one-variable comparison. Keep S0_HEAD_DELTA_V1 parents and ordinary resume guards.

Run only hb01:30154 with neutral main/child argv, measured peak headroom, at most two workers and at least1GiB disk safety reserve. Retain all histories. No post-V3 cumulative GPU-hour budget or capacity purchase. Record throughput,wall/process time,GPU memory/utilization and disk. Block on data/weight/parent drift,nonfinite numbers,unexpected state differences or resource failure, preserving partial results.

## Comparisons and stopping

Main success K-C: mean Final BA improves,current recall gain>=10pp,old recall loss<=5pp,at least2/3 pairs improve BA/current,neither S1 norS2 adds current zero-recall classes. Mechanism K-H; also report K-J/I/G/F/E with identical val layout. Report every pair,stage,class andtail group, confusion/margins,objective and norm diagnostics. All comparisons are development-val after adaptive reuse; they are not independent test generalization evidence. Patient isolation,clinical label mapping and pretraining exposure remain unresolved.

Publish source,locked configs,engineering and aggregate results on a separate public branch; no main merge,private data,predictions,weights or checkpoints. Executor stops after its locked matrix/report. Hourly monitor can select another single evidence-based follow-up if gates fail under existing autonomy; no Full Dynamic,joint training,additional backbone or broad hyperparameter search. On success complete delivery and pause automation.
