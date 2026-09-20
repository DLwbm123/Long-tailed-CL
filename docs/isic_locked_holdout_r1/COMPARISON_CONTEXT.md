# Locked existing-holdout follow-up

This is R1, not V10. The existing 764-image holdout was already evaluated by V2 and its results informed development. This evaluation is therefore a **既有holdout的锁定后续评价**, not a previously untouched independent confirmation or clinical validation. V2's COMPLETE_MIXED_SIGNAL and all V3–V9 negative results remain unchanged.

The primary comparison K−C measures the existing package of changes. K−H isolates the existing controlled variance-rule contrast. K−AugReg-CBRidge and K−DINOv2-CBRidge compare whole systems that differ in S0 adaptation, feature route and classifier. They do not isolate Gaussian-versus-ridge or pretraining causality.

C/H/K use three fixed trained seed/order runs. RA/RD each have one deterministic final analytic fit; old/current partitions and early-stage class sets vary by order. They are not three independent trained ridge repetitions. All comparisons use identical stage-filtered, sample-ID-sorted holdout images and the fixed batch-48 layout; APART batchwise routing is retained.

Class-stratified connected-component paired bootstrap uses 2,000 draws with seed 91001. Component multiplicities weight all images in each sampled group. The primary estimand remains image-within-class recall. All methods, stages and the three fixed models share draws; trained seeds are never resampled. The intervals do not cover adaptive model selection, a training-seed population, unknown patient dependence, or external domain shift. They are descriptive, without post-hoc significance selection.

The five historical development gates are checked mechanically before rounding, without moving thresholds. Tail results are reported separately. Passing these empirical thresholds does not establish clinical noninferiority. Completion is independent of the direction of the scientific result. R1 stops after E3, without more training, test tuning or scheduled monitoring.
