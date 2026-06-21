# GPA Experiment Checklist

Use this checklist before moving from local to server and before trusting any reported number.

## 1. Dataset sanity

- [ ] CIFAR-100 train split is long-tailed.
- [ ] CIFAR-100 test split remains balanced.
- [ ] `rho = N_min / N_max`, not the inverse.
- [ ] For `rho=0.01`, max train count is about 500 and min train count is about 5.
- [ ] `class_counts.json` is saved.
- [ ] `class_order.json` is saved.
- [ ] Baseline and GPA use the same class counts and class order.
- [ ] Shuffled runs save seed and exact class order.

## 2. Task split sanity

- [ ] Base task has 50 classes in paper-like runs.
- [ ] Remaining 50 classes are split into 5 or 10 incremental tasks.
- [ ] Local debug mode uses fewer classes/epochs but does not overwrite paper-like configs.
- [ ] Evaluation after each task uses all seen classes.
- [ ] Test-time label mapping is correct after class reordering.

## 3. Classifier sanity

- [ ] Classifier output dimension expands when new classes arrive.
- [ ] Old weights are preserved during expansion.
- [ ] New class weights are initialized randomly for no-GPA baseline.
- [ ] New class weights are initialized with normalized prototypes for GPA.
- [ ] Bias initialization is optional.
- [ ] Class id and classifier index are not confused.

## 4. Prototype sanity

- [ ] Frozen prototypes are computed with the previous model.
- [ ] Prototype computation uses `model.eval()`.
- [ ] Prototype computation uses `torch.no_grad()`.
- [ ] Prototype norms are logged.
- [ ] Every current-task class has at least one sample.
- [ ] Prototype tensor dimension matches classifier weight dimension.

## 5. Anchoring sanity

- [ ] Anchor loss is finite and non-negative.
- [ ] Anchor loss only compares normalized vectors.
- [ ] Anchor loss applies to current-task classes in the first implementation.
- [ ] `lambda_gpa` is logged.
- [ ] Turning `--gpa false` removes both prototype initialization and anchor loss.

## 6. Metrics sanity

- [ ] `metrics.jsonl` exists.
- [ ] Accuracy matrix exists.
- [ ] Final accuracy is reported.
- [ ] Average incremental accuracy is reported.
- [ ] Forgetting is reported.
- [ ] Many/medium/few accuracy is reported.
- [ ] Head-tail gap is reported.
- [ ] Per-seed summary is saved.

## 7. Fair comparison sanity

For every comparison:

- [ ] Same seed.
- [ ] Same class order.
- [ ] Same training epochs.
- [ ] Same optimizer.
- [ ] Same augmentation.
- [ ] Same backbone.
- [ ] Same classifier type except GPA initialization/anchoring.
- [ ] Same evaluation protocol.

## 8. Local-to-server gate

Do not start full server runs until:

- [ ] Finetune smoke finishes.
- [ ] Finetune+GPA smoke finishes.
- [ ] Summary script compares both runs.
- [ ] No NaNs.
- [ ] Logs and configs are saved.
- [ ] Class counts look correct.
- [ ] GPA prototype initialization is confirmed in logs.

## 9. Server run minimum set

- [ ] Shuffled 5-task Finetune, seeds 0/1/2.
- [ ] Shuffled 5-task Finetune+GPA, seeds 0/1/2.
- [ ] Shuffled 10-task Finetune, seeds 0/1/2.
- [ ] Shuffled 10-task Finetune+GPA, seeds 0/1/2.

Optional:

- [ ] Ordered 5-task.
- [ ] Ordered 10-task.
- [ ] `lambda_gpa` sweep.
- [ ] GPA bias on/off.
- [ ] Batch anchor vs EMA anchor.

## 10. Before writing results

- [ ] Include mean ± std across seeds.
- [ ] Include per-class group metrics.
- [ ] Include forgetting.
- [ ] Include exact data protocol.
- [ ] Include exact task order protocol.
- [ ] State whether results are exact reproduction or approximate reimplementation.
