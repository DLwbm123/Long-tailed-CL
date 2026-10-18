# Fixed-budget future-class depth residual pilot v1

Authorized 2026-10-18. One ISIC sequence {0,1,2,3}->{4}->{5}, one seed, block6 CLS of the existing frozen ViT-B/16, residual rank128. No HK, extra layers/ranks/seeds/orders, post-test tuning or automatic extension. Existing exposed ISIC evaluation (586 images,500 known identity groups) is exploratory; patient isolation unknown. Labels0..5 are existing remapped labels, not original diagnostic identifiers.

## Budget and controls (frozen before data fitting)

Per-arm final six-class total persistent logical state cap=19,058,800 bytes, referenced to historical G0 dense float64 train statistics+selected head+normalizer. This run additionally counts 64 scalar/metadata bytes, transform matrices, selected indices and all normalizers. Include one selected deployed head; the other two penalty heads, earlier-stage snapshots and other diagnostic duplicates are separately costed, not hidden as deployed state. All floats64, selected indicesint64. No triangular-storage savings claimed. Fixed seeded global-coordinate permutation74010, regenerate rather than retain an index tensor; no learned dense global projection. Features omitted solely by the byte cap; dimensions are not performance tuned.

|Arm|Rule|global core / selected width|output width excluding intercept|final bytes|
|---|---|---:|---:|---:|
|G|global only|1535|1535|19046472|
|R|global plus128 fixed orthonormal random block6 directions, seed74011|1375|1503|19057488|
|P|global plus128 ordinary block6 weighted PCA directions|1375|1503|19057488|
|C|global plus128 conditional residual PCA directions|1318|1446|19055536|
|S|Head2Toe-style supervised coordinate selection from full global+block6|1534|1534|19040224|

G loses one coordinate because this run counts output normalization and metadata too. Larger ordinary controls retain more dimensions than C: candidate transform cost is paid, not waived. Budget formula and actual tensor accounting must agree at six classes. Each arm keeps dense augmented G=X'WX,H=X'WY; all six H columns are reserved from birth. Cost model excludes common frozen encoder weights but includes extra encoder compute and reports common model bytes separately. No persistent matrix A: only U and V=A U are needed at inference.

## Frozen preprocessing and fitting

Use normalized original1536-D feature caches. Curator process only partitions existing S_new cache into separate class4/5 bundles; it fits nothing and does not extract future intermediate features. Learner may open only current-stage bundle and current-stage raw images; neither whole S_new cache nor future bundle is accessible. No old per-image features/images after their stage. Existing data remain unmodified.

Permanent group holdout seed74006, per class max(1,floor(.2*n_groups)), same for all arms. Never refit on validation. All six ISIC classes must have >=2 groups; otherwise stop. Only initial training rows fit transforms/normalization/feature selection. New-class validation rows are excluded identically but never select anything. Train class/group/image weights use existing core_attribution.weights. Fit base global and block6 weighted mean/scalar RMS. PCA and residual regression use initial-train class/group weights. A=(Gg/4+.01 I)^-1 Ggh/4; fixed .01, no reconstruction-penalty search. C keeps top128 eigenvectors of weighted residual covariance. P uses block6 covariance. R uses fixed QR Gaussian directions. Fixed tie/sign conventions are numerical only, no search. Fit one final weighted mean/scalar RMS per arm on initial train, then freeze.

S is a mechanism control, not a full Head2Toe reproduction: weighted four-class CE with row-group lasso, zero initialization,500 proximal-gradient steps. Step=1/lambda_max(X'WX/4); group penalty=.01*maximum initial nonbias row-gradient norm, no sweep. Intercept unpenalized. Rank coordinates by nonbias weight row norm, tie by index; retain1534, then same ridge as other arms. Record proximal residual, do not claim exact optimizer convergence.

Ridge penalties [.0001,.01,1]. At initial stage G's validation BA selects ONE common penalty for all arms, ties larger penalty. Freeze it for both later stages. Keep the other fixed penalties as sensitivity diagnostics; do not select them using later validation or evaluation. Equal per-arm total feature energy after normalization, unpenalized intercept. Retain only sufficient statistics and frozen transforms between stages.

## Reparameterization check

For C with fixed U,V, [g,hU-gV] and [g,hU] have identical linear hypothesis classes. Residualization alone is not a contribution. Verify actual logits from direct coordinates using the exactly transformed ridge penalty agree (1e-7 absolute, identical argmax) with residual coordinates; also synthetic direct-vs-incremental and congruence-penalty solve equality. Failure is engineering failure. Any scientific benefit must survive ordinary PCA/random/supervised selection controls, not be credited to a new function class or independent information.

## Seals, evaluation, primary gates

Seal transforms/alpha after initial stage, then all three-stage heads before opening any test features or images. Stage1 evaluates labels0..4 against5 output columns, stage2 labels0..5 against6. Base metrics descriptive. Score all3 fixed penalties without changing choices. Main primary metric is mean of stage1 and stage2 all-seen class-balanced image accuracy. Use paired class-stratified known-group bootstrap10000,seed74003; same sampled groups across arms and stages.

Compare C separately against EACH of G,R,P,S; no test-selected comparator is deployed. Require all point gains>=1pp and lower95% percentile bound of minimum paired gain across the four controls>0. Future-class gate: positive class4 stage1 recall gain and positive class5 stage2 recall gain versus each control. Retirement gate: class4 stage2 gain remains positive and decreases by no more than1pp from its stage1 gain versus each control. Report changes in original old-class BA, per-class counts, harm/rescue and tiny-class uncertainty. Sensitivity gate: mean-BA gain positive against each control at >=2 of3 common penalties including the initial-selected one; third penalty cannot lose>1pp against any control. These are exploratory gates, not patient-generalization certificates.

If any scientific gate fails, STOP_NO_EXTENSION after completing this fixed matrix and public report. Do not add any layer/rank/order/seed/grid or rescue rule. Gains only in initial classes fail future transfer; birth gain lost at next stage fails retirement; only MSE improvement does not count. All passing only permits independent research discussion, no automatic confirmation run. Max coordinator wall3600seconds, no automatic scientific retry; preserve failures and costs. GPU0 only. Resource timeout terminates only this run's worker and reports unresolved/failed, never scientific infeasibility.

## Receipts and delivery

Check storage mount/capacity/probe, GPU memory, source commit, process argv neutrality, feature-hook/manual block6 parity, group/path/cache binding, support counts, training-validation isolation, retirement guard rejection, finite scores, exact budget accounting and seal-before-test. Record extraction/transform/selector/ridge/evaluation wall times, persistent bytes, diagnostic snapshots, peak RSS/GPU, raw-cache historical cost excluded. Private identities, image paths, features, predictions, models, logs and server configs stay private. Publish only source, protocol, budget table, aggregate results/costs and Chinese mechanism report through required GitHub proxy. No periodic monitoring requested.
