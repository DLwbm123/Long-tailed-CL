# AGENTS.md

## Mission

This repository is for reproducing **GPA: Geometric Prototype Alignment** for **long-tailed class-incremental learning (LT-CIL)**, then reusing the verified pipeline on medical long-tailed classification datasets.

The immediate priority is **CIFAR-100-LT reproduction first**. Do not spend time on the medical extension until the local CIFAR-100-LT pipeline is working end-to-end and the implementation shows the expected GPA-vs-baseline trend.

## Working mode

Use Plan-first development.

Before making non-trivial changes:
1. Inspect the repository structure and current code.
2. Write a short implementation plan with:
   - files to read,
   - files to modify or create,
   - expected command(s) to validate the change,
   - risks or assumptions.
3. Implement the smallest coherent change.
4. Run the relevant validation command.
5. Summarize what changed, what passed, what failed, and the next step.

Never make broad rewrites when a focused patch is enough. Never delete or overwrite experiment outputs unless explicitly instructed.

## Reproduction target

Implement GPA as a plug-in module for LT-CIL classifiers.

GPA has three required parts:

1. **Frozen prototype estimation**
   - Before training task `t`, freeze the previous model or feature extractor `phi_{t-1}`.
   - For each new class `c` in the current task, compute the class feature prototype:
     `mu_c = mean(phi_{t-1}(x))` over training samples of class `c`.
   - Normalize prototypes with L2 normalization.
   - Log per-class counts and prototype norms.

2. **Geometric classifier initialization**
   - Expand the classifier for the newly introduced classes.
   - Initialize each new class weight by its normalized prototype:
     `W_c = normalize(mu_c)`.
   - Keep old class weights unchanged.
   - If the classifier uses a bias term, support optional GPA bias initialization:
     `b_c = -log(N_c / (N_ref + eps))`, where `N_ref` should be configurable, defaulting to the maximum class count among the current training classes.
   - Make bias initialization optional via config/CLI because some baselines use bias-free cosine classifiers.

3. **Dynamic anchoring optimization**
   - During current-task training, add a geometric anchoring loss:
     `L_total = L_ce + L_aux + lambda_gpa * L_anchor`.
   - `L_aux` is method-specific and may be zero for the initial finetune baseline.
   - Implement `L_anchor` as MSE or cosine distance between normalized classifier weights and normalized current feature prototypes for classes present in the current batch or cached in the current epoch.
   - Default `lambda_gpa` candidates for CIFAR-100-LT: `0.05, 0.10, 0.12, 0.15, 0.20`; start with `0.12` and `0.15`.
   - Do not backpropagate through stored frozen prototypes. Current feature prototypes may backpropagate if computed from current batch features; keep the implementation numerically stable and document the choice.

## Dataset protocol: CIFAR-100-LT

Implement CIFAR-100-LT before any medical dataset.

Required protocol:
- Dataset: CIFAR-100.
- Training split only is long-tailed; test split remains the standard balanced CIFAR-100 test split.
- Number of classes: 100.
- Main imbalance ratio: `rho = N_min / N_max = 0.01`.
- CIFAR-100 has `N_max = 500` train images per head class, so `rho = 0.01` implies `N_min = 5` images for the rarest class.
- Generate class counts by exponential decay:
  `n_i = N_max * (rho ** (i / (num_classes - 1)))`, where `i` is the rank from head to tail.
- Save the generated class order and class counts to every run directory.

Required class-incremental splits:
- Base classes: 50.
- Remaining classes: 50.
- Implement both 5-task and 10-task incremental settings for the remaining 50 classes.
- Also support local smoke settings with fewer classes and fewer epochs.

Required order protocols:
- `ordered`: classes appear from head to tail.
- `shuffled`: class order is randomized but the same long-tailed class counts are preserved.
- Every shuffled run must save the random seed and exact class order.

## First implementation milestone

Start with a minimal reproducible pipeline:

1. `CIFAR100LT` dataset wrapper.
2. Incremental task splitter.
3. ResNet32 or a repository-native CIFAR backbone.
4. Finetune baseline without GPA.
5. Finetune + GPA.
6. Metrics: average incremental accuracy, final accuracy, forgetting, and many/medium/few accuracy.
7. One local smoke test that runs in minutes.

Do not implement LUCIR, PODNet, prompt methods, or medical datasets until this milestone is complete.

If the repository already contains PyCIL-style code, integrate GPA into the existing trainer rather than creating a parallel framework. If no CIL framework exists, create a minimal, clean training pipeline under `src/` and scripts under `scripts/`.

## Suggested repository structure

Adapt to the existing repository if it already has a structure. Otherwise use:

```text
configs/
  cifar100lt/
    finetune_baseline.yaml
    finetune_gpa.yaml
    smoke.yaml
scripts/
  run_smoke.sh
  run_cifar100lt_finetune.sh
src/
  datasets/
    cifar100lt.py
    incremental_split.py
  models/
    resnet32.py
    classifier.py
  methods/
    gpa.py
    finetune.py
  engine/
    train_cil.py
    eval.py
  utils/
    metrics.py
    seed.py
    logging.py
experiments/
  README.md
```

## Metrics to implement exactly

Let `A_t` be top-1 accuracy on all classes seen up to task `t`, measured after training task `t`.

Report:
- `Acc`: average incremental accuracy, `mean_t A_t`.
- `AccT`: final accuracy after the last task on all seen classes.
- `F`: forgetting. For each earlier task or class group, compute the drop from its best historical accuracy to its final accuracy, then average. Document the exact implementation.
- `Many`: final accuracy for classes with `N_c > 100` training samples.
- `Medium`: final accuracy for classes with `20 <= N_c <= 100` training samples.
- `Few`: final accuracy for classes with `N_c < 20` training samples.

Always save metrics as both human-readable text and machine-readable JSON/CSV.

## Reference targets for sanity checking

Do not expect exact paper-level performance from the local smoke run. For the server run, use the following reported CIFAR-100-LT numbers only as reproduction targets and sanity checks.

Shuffled CIFAR-100-LT, `rho = 0.01`:
- Finetune, 5 tasks: `Acc ≈ 54.39`, `AccT ≈ 40.20`.
- Finetune + GPA, 5 tasks: `Acc ≈ 65.12`, `AccT ≈ 49.88`.
- LUCIR + GPA, 5 tasks: `Acc ≈ 44.68`, `AccT ≈ 37.85`.
- PODNet + GPA, 5 tasks: `Acc ≈ 43.85`, `AccT ≈ 40.62`.

Class-frequency final accuracy on CIFAR-100-LT:
- Finetune: Overall `40.20`, Many `52.00`, Medium `46.80`, Few `34.30`.
- Finetune + GPA: Overall `49.88`, Many `54.30`, Medium `49.90`, Few `46.80`.
- LUCIR: Overall `30.50`, Many `39.40`, Medium `35.50`, Few `26.00`.
- LUCIR + GPA: Overall `37.85`, Many `41.20`, Medium `37.90`, Few `35.40`.

Acceptance criterion for the first serious run:
- GPA should improve final accuracy and especially few-class accuracy over the same baseline.
- If the trend is absent, debug prototypes, classifier normalization, class order, and loss scaling before tuning unrelated hyperparameters.

## Validation commands

Provide or maintain these commands:

```bash
# Fast local sanity check. Must run without a server.
bash scripts/run_smoke.sh

# CIFAR-100-LT finetune baseline.
bash scripts/run_cifar100lt_finetune.sh --method finetune --rho 0.01 --tasks 5 --order shuffled --seed 0

# CIFAR-100-LT finetune + GPA.
bash scripts/run_cifar100lt_finetune.sh --method finetune_gpa --rho 0.01 --tasks 5 --order shuffled --seed 0 --lambda-gpa 0.12
```

If the actual CLI differs, update this section and `experiments/README.md` in the same patch.

## Engineering requirements

- Use PyTorch.
- Use deterministic seeding where practical.
- No absolute hard-coded paths.
- Use `DATA_ROOT` env var or a `--data-root` argument.
- Use `OUTPUT_ROOT` env var or a `--output-root` argument.
- Save every run under a unique directory containing:
  - resolved config,
  - git commit hash if available,
  - class order,
  - class counts,
  - metrics JSON/CSV,
  - stdout/stderr log,
  - checkpoint path(s).
- Keep GPU memory usage modest by default.
- Local smoke tests should work with CPU or a single small GPU.
- Full training can assume a server GPU.

## Debug checklist for GPA

If GPA does not improve the baseline:

1. Verify prototypes are computed with the previous frozen feature extractor before current-task training.
2. Verify labels map correctly between dataset labels, global class IDs, and classifier indices.
3. Verify classifier weights and prototypes have matching dimensions.
4. Verify new class weights are actually overwritten by normalized prototypes.
5. Verify old class weights are not accidentally reinitialized.
6. Verify the training loader contains the expected long-tailed class counts.
7. Verify the test loader uses the balanced CIFAR-100 test set.
8. Check whether the classifier is cosine-normalized; avoid double normalization bugs.
9. Sweep `lambda_gpa` only after the above checks pass.
10. Compare many/medium/few accuracy, not only overall accuracy.

## Medical extension, later

After CIFAR-100-LT is validated, add a generic medical long-tailed classification dataset interface.

Preferred input format:

```csv
path,label,split
relative/or/absolute/image/path.png,class_name,train
relative/or/absolute/image/path.png,class_name,test
```

Medical extension requirements:
- Do not assume balanced labels.
- Log natural class counts before any resampling.
- Support both natural long-tail and controlled downsampling.
- Report macro-F1, balanced accuracy, per-class recall/sensitivity, many/medium/few accuracy, and AUROC/AUPRC when appropriate.
- Keep the CIFAR-100-LT reproduction scripts unchanged.

## Documentation expectations

Update `experiments/README.md` after each meaningful milestone with:
- command used,
- machine used,
- seed,
- dataset protocol,
- result table,
- known deviations from the paper,
- next debugging step.

Do not claim a reproduction is successful unless the exact protocol, class order, counts, and metrics are saved and comparable.
