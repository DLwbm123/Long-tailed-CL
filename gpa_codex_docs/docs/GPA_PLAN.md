# GPA Reproduction Plan for Codex

Use this plan sequentially. Do not skip to later phases before earlier phases pass.

## Phase 0 — Repository inspection

### Goal
Understand the existing codebase before editing.

### Codex actions

1. List the repository tree.
2. Identify the training entry point.
3. Identify dataset loaders.
4. Identify model / classifier definitions.
5. Identify metric and logging utilities.
6. Write a short implementation plan before making changes.

### Do not

- Do not rewrite the whole training framework.
- Do not add medical datasets yet.
- Do not implement LUCIR/PODNet yet.

### Acceptance

Codex can answer:

```text
What command trains a classification model?
Where should CIFAR-100-LT be implemented?
Where should GPA be implemented?
How are classifier heads expanded?
How are metrics logged?
```

---

## Phase 1 — CIFAR-100-LT dataset builder

### Goal
Create a deterministic CIFAR-100-LT dataset with saved class counts and class order.

### Required features

```text
rho = N_min / N_max
ordered class order
shuffled class order
base class count
incremental task count
balanced standard test set
local smoke mode
```

### Deliverables

```text
datasets/cifar100_lt.py or equivalent
continual/task_split.py or equivalent
unit/smoke test or preview command
```

### Validation command

Codex should create or adapt a command like:

```bash
python train.py --dataset cifar100_lt --rho 0.01 --preview_dataset
```

### Acceptance

The preview prints:

```text
100 classes
max count approximately 500
min count approximately 5 for rho=0.01
base classes = 50
remaining classes = 50
incremental tasks = 5 or 10
```

The run directory contains:

```text
class_counts.json
class_order.json
```

---

## Phase 2 — Finetune LT-CIL baseline

### Goal
Implement the minimal sequential training loop without GPA.

### Required behavior

For each task:

1. Expand classifier for new classes.
2. Train only on current task training data.
3. Evaluate on all seen classes.
4. Save metrics.

### Deliverables

```text
methods/finetune.py or equivalent
metrics for all seen classes
accuracy matrix
checkpoint per phase
```

### Local smoke command

```bash
python train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base_classes 10 \
  --incremental_steps 2 \
  --debug_num_classes 20 \
  --method finetune \
  --epochs 1 \
  --seed 0 \
  --output runs/smoke_finetune
```

### Acceptance

```text
all phases complete
no NaN
classifier output dimension grows correctly
metrics.jsonl exists
accuracy matrix exists
```

---

## Phase 3 — GPA plugin

### Goal
Add GPA without changing the Finetune baseline behavior when `--gpa false`.

### Required components

```text
frozen prototype estimation
new-class weight initialization
optional bias initialization
dynamic anchor loss
prototype and anchor-loss logging
```

### Deliverables

```text
methods/gpa.py
CLI/config options:
  --gpa true/false
  --lambda_gpa
  --gpa_init_bias true/false
  --gpa_anchor_mode batch/ema
```

### Local smoke command

```bash
python train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base_classes 10 \
  --incremental_steps 2 \
  --debug_num_classes 20 \
  --method finetune \
  --gpa true \
  --lambda_gpa 0.12 \
  --gpa_init_bias true \
  --epochs 1 \
  --seed 0 \
  --output runs/smoke_finetune_gpa
```

### Acceptance

```text
frozen prototypes are computed before each incremental task
new class weights are initialized from prototypes
anchor loss is non-negative and finite
old class weights are not accidentally overwritten during initialization
run finishes end-to-end
```

---

## Phase 4 — Metrics and result comparison

### Goal
Make results easy to compare with and without GPA.

### Required metrics

```text
final accuracy
average incremental accuracy
forgetting
many / medium / few accuracy
head-tail gap
per-task accuracy matrix
```

### Deliverables

```text
scripts/summarize_results.py or equivalent
runs/<name>/summary.json
runs/<name>/summary.csv
```

### Comparison command

```bash
python scripts/summarize_results.py \
  --runs runs/smoke_finetune runs/smoke_finetune_gpa
```

### Acceptance

A table prints:

```text
method | final_acc | avg_inc_acc | forgetting | many | medium | few | head_tail_gap
```

---

## Phase 5 — Full CIFAR-100-LT server run

### Goal
Run the paper-like setting on the server.

### Required experiments

Minimum set:

```text
Finetune, shuffled, 5 incremental tasks, seed 0/1/2
Finetune+GPA, shuffled, 5 incremental tasks, seed 0/1/2
Finetune, shuffled, 10 incremental tasks, seed 0/1/2
Finetune+GPA, shuffled, 10 incremental tasks, seed 0/1/2
```

Optional but recommended:

```text
ordered 5-task
ordered 10-task
lambda_gpa sweep: 0.05, 0.10, 0.12, 0.15, 0.20
bias initialization on/off
batch-anchor vs EMA-anchor
```

### Acceptance

GPA should show a positive trend over Finetune in at least:

```text
average incremental accuracy
few/tail accuracy
forgetting or head-tail gap
```

Exact paper numbers are not required until optimizer, epochs, augmentation, and baseline details are aligned.

---

## Phase 6 — Baseline expansion

Only after Phase 5 is stable.

Potential next baselines:

```text
LUCIR + GPA
PODNet + GPA
L2P/DualPrompt + GPA if using ViT/foundation model
```

Rules:

1. Add one baseline at a time.
2. Reuse the same dataset, task split, metrics, and logger.
3. Keep GPA as a plugin.
4. Do not mix medical transfer into baseline reproduction.

---

## Phase 7 — Medical long-tailed classification transfer

Only after CIFAR-100-LT behavior is understood.

See:

```text
docs/MEDICAL_LT_TRANSFER.md
```

The medical stage should not be framed as “just another dataset.” It should evaluate rare-class / rare-finding performance, tail recall, macro-F1, AUROC/AUPRC, and calibration.
