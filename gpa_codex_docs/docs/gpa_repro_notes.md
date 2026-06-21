# GPA Reproduction Notes

## 0. Purpose

The goal is to reproduce **Geometric Prototype Alignment (GPA)** for **long-tailed class-incremental learning (LT-CIL)**.

The first target is **CIFAR-100-LT**. Medical long-tailed classification comes later, after the implementation is stable.

Do not start with all baselines. Start with:

```text
CIFAR-100-LT → Finetune baseline → Finetune + GPA → metrics sanity check
```

Only after this path is stable should the repo add LUCIR, PODNet, prompt methods, adapter methods, or medical datasets.

---

## 1. Core GPA idea

GPA is a classifier-level plugin. It should not require rewriting the whole CIL framework.

It has three parts:

1. **Frozen prototype estimation**
2. **Geometric classifier weight initialization**
3. **Dynamic anchoring optimization**

In code, implement GPA as a small module, ideally independent from the baseline method.

Suggested files:

```text
methods/gpa.py
methods/finetune.py
datasets/cifar100_lt.py
continual/task_split.py
utils/metrics.py
utils/logger.py
configs/cifar100lt_finetune_gpa.yaml
```

Adapt the file names to the existing repository if needed, but keep the same separation of concerns.

---

## 2. LT-CIL problem setup

At incremental phase `t`, the model receives only the current training data:

```text
D_t = {(x_i, y_i)}
```

The previous training data `D_1 ... D_{t-1}` should be considered unavailable unless a baseline explicitly uses exemplars.

For the first reproduction milestone, use **exemplar-free Finetune**:

```text
no replay buffer
no exemplar memory
no old training images
```

After training phase `t`, evaluate on the balanced test samples of **all seen classes**:

```text
C_seen = C_0 ∪ C_1 ∪ ... ∪ C_t
```

---

## 3. CIFAR-100-LT construction

Use CIFAR-100.

The training set is made long-tailed. The test set remains the standard balanced CIFAR-100 test set.

Main reproduction setting:

```text
num_classes = 100
max_train_images_per_class = 500
rho = N_min / N_max = 0.01
min_train_images_per_class = 5
```

Recommended class-count formula:

```python
n_i = int(round(N_max * (rho ** (i / (num_classes - 1)))))
n_i = max(1, n_i)
```

where `i` is the class rank from head to tail.

Important: save the exact class counts for each run:

```text
runs/<run_name>/class_counts.json
runs/<run_name>/class_order.json
```

The same class counts and class order must be used for baseline and GPA runs.

---

## 4. Task split protocol

Use this protocol first:

```text
base classes = 50
remaining classes = 50
incremental setting A = 5 incremental tasks, 10 classes per task
incremental setting B = 10 incremental tasks, 5 classes per task
```

Support two order protocols:

### ordered

Classes arrive from head to tail.

```text
base task: the 50 most frequent classes
incremental tasks: progressively rarer classes
```

### shuffled

Classes are randomly permuted while preserving class counts.

```text
class_order = random permutation of 100 classes
```

Always save the seed and the exact order.

For local debugging, also support small smoke settings, for example:

```text
--debug_num_classes 20
--base_classes 10
--incremental_steps 2
--epochs 1
--max_train_per_class 20
```

These smoke settings are not for reporting; they only verify the code path.

---

## 5. GPA implementation details

### 5.1 Frozen prototype estimation

Before training phase `t`, freeze the previous feature extractor `phi_{t-1}`.

For every new class `c` in the current task:

```python
mu_c = mean(phi_prev(x) for x in D_t if y == c)
```

Implementation requirements:

```python
@torch.no_grad()
def compute_class_prototypes(model, loader, class_ids, device):
    model.eval()
    # return dict: class_id -> prototype tensor [feature_dim]
```

Important choices:

- use `model.eval()` during prototype computation;
- no gradient should flow into `phi_{t-1}`;
- compute prototypes only from current-task training samples;
- log class counts and prototype norms;
- if a class has zero samples due to a bug, fail loudly.

### 5.2 Geometric classifier initialization

For each new class:

```python
W_c = normalize(mu_c, p=2)
```

If the classifier uses a bias term, optionally initialize:

```python
b_c = -log(N_c / (N_ref + eps))
```

Recommended default:

```python
N_ref = max class count in current task
```

Make bias initialization configurable:

```text
--gpa_init_bias true/false
```

because some CIL methods use bias-free cosine classifiers.

Implementation requirements:

```python
def init_new_class_weights(classifier, prototypes, class_to_head_index, counts, use_bias=True):
    # expand classifier first if needed
    # initialize only newly introduced class weights
    # leave old class weights unchanged
```

Potential classifier cases:

1. `nn.Linear(feature_dim, num_classes)` with bias.
2. `nn.Linear(feature_dim, num_classes, bias=False)`.
3. Cosine classifier with normalized weights.

Start with case 1. Add case 2 if easy. Do not support every classifier type in the first patch.

### 5.3 Dynamic anchoring optimization

During training phase `t`, maintain current class prototypes from the current model `phi_t`.

Simplest stable implementation:

```python
# For each batch, compute batch-level prototypes for classes present in the batch.
# Compare normalized classifier weights with normalized batch prototypes.
```

Batch-level anchoring:

```python
features = model.extract_features(images)
features_norm = F.normalize(features, dim=1)

for c in labels.unique():
    proto_c = features_norm[labels == c].mean(dim=0)
    proto_c = F.normalize(proto_c, dim=0)
    weight_c = F.normalize(classifier.weight[class_to_head_index[c]], dim=0)
    loss_anchor += mse(weight_c, proto_c)
```

More stable server implementation:

```text
EMA prototype cache per class
```

Recommended EMA update:

```python
ema_proto[c] = m * ema_proto[c] + (1 - m) * batch_proto[c]
ema_proto[c] = normalize(ema_proto[c])
```

Default:

```text
ema_momentum = 0.9
```

Loss:

```python
L_total = L_ce + lambda_gpa * L_anchor + L_aux
```

For Finetune baseline:

```text
L_aux = 0
```

Initial `lambda_gpa` candidates:

```text
0.05, 0.10, 0.12, 0.15, 0.20
```

Start with:

```text
lambda_gpa = 0.12 or 0.15
```

---

## 6. Training procedure for first milestone

### Phase 0: base training

Train the model on the 50 base classes.

For local smoke:

```text
epochs = 1 or 2
batch_size = 64 or 128
```

For server run, tune later.

Save checkpoint:

```text
checkpoints/base_model.pt
```

### Phase t: incremental training

For each incremental task:

1. Load previous model.
2. Expand classifier for new classes.
3. If GPA is enabled:
   - compute frozen prototypes using previous model;
   - initialize new class weights;
   - initialize optional bias;
   - create current prototype cache for anchoring.
4. Train on current task data.
5. Evaluate on all seen classes.
6. Save checkpoint and metrics.

---

## 7. Metrics

Required metrics:

```text
top1_all_seen
average_incremental_accuracy
final_accuracy
forgetting
many_accuracy
medium_accuracy
few_accuracy
tail_accuracy
head_tail_gap
```

### Accuracy after each phase

Let `A_t` be accuracy on all seen classes after phase `t`.

```python
avg_incremental_acc = mean(A_0, A_1, ..., A_T)
final_acc = A_T
```

### Forgetting

Maintain a matrix:

```python
acc_matrix[i][j] = accuracy on task j after training task i
```

For each old task `j < T`:

```python
forgetting_j = max(acc_matrix[i][j] for i in range(j, T)) - acc_matrix[T][j]
```

Average across old tasks.

### Many / Medium / Few

Use training sample counts from the original long-tailed split.

Recommended default thresholds:

```text
many:   n_c > 100
medium: 20 <= n_c <= 100
few:    n_c < 20
```

Make thresholds configurable. Save the class list belonging to each group.

### Tail accuracy

For medical transfer later, tail accuracy or tail recall will be more important than overall accuracy.

For CIFAR-100-LT, also report few-class accuracy.

---

## 8. Logging and reproducibility

Every run directory should contain:

```text
config.yaml
class_counts.json
class_order.json
metrics.jsonl
acc_matrix.npy or acc_matrix.csv
checkpoints/
stdout.log
```

Log these fields every phase:

```json
{
  "phase": 1,
  "seen_classes": 60,
  "new_classes": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
  "top1_all_seen": 0.0,
  "avg_incremental_acc_so_far": 0.0,
  "forgetting_so_far": 0.0,
  "many_acc": 0.0,
  "medium_acc": 0.0,
  "few_acc": 0.0,
  "lambda_gpa": 0.12,
  "gpa_init_bias": true,
  "seed": 0
}
```

Use at least three seeds for server results:

```text
0, 1, 2
```

Use one seed for local smoke.

---

## 9. Local smoke commands

Codex should adapt these commands to the actual repo.

Dataset preview:

```bash
python train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base_classes 10 \
  --incremental_steps 2 \
  --debug_num_classes 20 \
  --preview_dataset
```

Finetune smoke:

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
  --batch_size 64 \
  --seed 0 \
  --output runs/smoke_finetune
```

Finetune + GPA smoke:

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
  --batch_size 64 \
  --seed 0 \
  --output runs/smoke_finetune_gpa
```

Expected local result:

```text
Not NaN.
All phases complete.
Classifier expands correctly.
Prototype logs exist.
Metrics file exists.
Accuracy does not need to match the paper in smoke mode.
```

---

## 10. Server reproduction commands

After local smoke works, run full CIFAR-100-LT.

5-step shuffled:

```bash
python train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base_classes 50 \
  --incremental_steps 5 \
  --method finetune \
  --gpa true \
  --lambda_gpa 0.12 \
  --gpa_init_bias true \
  --seed 0 \
  --output runs/cifar100lt_rho001_shuffled_5step_finetune_gpa_s0
```

10-step shuffled:

```bash
python train.py \
  --dataset cifar100_lt \
  --rho 0.01 \
  --order shuffled \
  --base_classes 50 \
  --incremental_steps 10 \
  --method finetune \
  --gpa true \
  --lambda_gpa 0.12 \
  --gpa_init_bias true \
  --seed 0 \
  --output runs/cifar100lt_rho001_shuffled_10step_finetune_gpa_s0
```

Also run corresponding no-GPA baselines with the same seeds and class orders.

---

## 11. Approximate sanity targets

Do not expect exact paper numbers from the first implementation.

The first server goal is:

```text
GPA improves average incremental accuracy over Finetune.
GPA improves few/tail-class accuracy.
GPA does not catastrophically hurt many/head classes.
```

The paper reports that GPA is plug-and-play and improves average incremental accuracy across methods; however, exact values depend on baseline implementation, optimizer, training epochs, augmentation, and classifier type.

For reproduction reporting, always show:

```text
Finetune vs Finetune+GPA
same seed
same class order
same optimizer
same epochs
same augmentations
same model
```

---

## 12. Common implementation pitfalls

### Pitfall 1: wrong rho direction

The paper uses:

```text
rho = N_min / N_max
```

For CIFAR-100-LT with `rho = 0.01`, the rarest class has about 5 images, not 50,000 images.

### Pitfall 2: class id vs classifier index mismatch

CIFAR class id is not always equal to classifier output index after task reordering.

Always maintain:

```python
class_id_to_head_index
head_index_to_class_id
```

### Pitfall 3: prototype computed with training-mode BatchNorm

Use `model.eval()` for frozen prototype computation.

### Pitfall 4: anchoring over old classes without old data

For the first Finetune+GPA implementation, anchor only current-task classes unless old-class prototype caches are explicitly implemented.

### Pitfall 5: GPA effect hidden by unstable training

Before judging GPA, verify:

```text
baseline training is stable
class counts are correct
evaluation maps logits to the right class ids
same seed/order is used for both runs
```

### Pitfall 6: metric mismatch

Average incremental accuracy, final accuracy, and task-wise forgetting are different metrics. Log all of them.

---

## 13. Minimal GPA module pseudocode

```python
class GPAPlugin:
    def __init__(self, lambda_gpa=0.12, init_bias=True, eps=1e-6, ema_momentum=0.9):
        self.lambda_gpa = lambda_gpa
        self.init_bias = init_bias
        self.eps = eps
        self.ema_momentum = ema_momentum
        self.ema_prototypes = {}

    @torch.no_grad()
    def compute_frozen_prototypes(self, model, loader, class_ids, device):
        model.eval()
        sums = {c: 0 for c in class_ids}
        counts = {c: 0 for c in class_ids}
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feats = model.extract_features(images)
            for c in class_ids:
                mask = labels == c
                if mask.any():
                    sums[c] = sums[c] + feats[mask].sum(dim=0)
                    counts[c] += int(mask.sum().item())
        prototypes = {}
        for c in class_ids:
            assert counts[c] > 0, f"No samples for class {c}"
            prototypes[c] = F.normalize(sums[c] / counts[c], dim=0)
        return prototypes, counts

    @torch.no_grad()
    def init_classifier(self, classifier, prototypes, class_to_idx, counts):
        n_ref = max(counts.values())
        for c, proto in prototypes.items():
            idx = class_to_idx[c]
            classifier.weight[idx].copy_(proto)
            if self.init_bias and classifier.bias is not None:
                classifier.bias[idx].fill_(-math.log(counts[c] / (n_ref + self.eps) + self.eps))

    def anchor_loss(self, classifier, features, labels, class_to_idx):
        losses = []
        for c in labels.unique().tolist():
            mask = labels == c
            if not mask.any():
                continue
            proto = F.normalize(features[mask].mean(dim=0), dim=0)
            idx = class_to_idx[int(c)]
            weight = F.normalize(classifier.weight[idx], dim=0)
            losses.append(F.mse_loss(weight, proto))
        if not losses:
            return features.new_tensor(0.0)
        return torch.stack(losses).mean()
```

This pseudocode is not a drop-in implementation. Codex should adapt it to the repository’s model API.
