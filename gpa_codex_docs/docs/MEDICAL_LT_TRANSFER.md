# Medical Long-Tailed Transfer Notes

This file is for later. Do not implement this before CIFAR-100-LT Finetune+GPA is stable.

## 1. Medical framing

Do not frame the medical experiment as merely:

```text
GPA on another dataset
```

Frame it as:

```text
rare-class / rare-finding continual medical diagnosis under long-tailed streams
```

The key clinical question is whether the model preserves sensitivity to rare but important classes as new categories arrive.

---

## 2. First medical experiment should be conservative

Keep the implementation close to CIFAR-100-LT.

Change only:

```text
dataset adapter
image transforms
number of classes
evaluation metrics
```

Do not add text encoders, adapters, semi-supervision, TTA, or segmentation in the first medical run.

---

## 3. Dataset requirements

For a clean first medical dataset:

```text
classification task
clear class labels
enough total images for train/val/test
natural or constructible long tail
single-label is easier than multi-label
```

Good first choices are small-to-medium medical classification datasets where class imbalance is already present or can be controlled.

For multi-label datasets, do not force them into single-label CIL unless the label semantics are carefully defined. Multi-label generalized CIL is more clinically realistic but more complex.

---

## 4. Medical metrics

In addition to CIL metrics, report:

```text
macro-F1
balanced accuracy
per-class recall / sensitivity
tail recall
tail AUPRC if probabilities are available
AUROC / AUPRC for multi-label datasets
calibration error if useful
```

For medical papers, rare-class recall is often more meaningful than overall accuracy.

---

## 5. Medical long-tail split

If the dataset has natural imbalance:

```text
use natural class counts first
report counts
then optionally create controlled rho splits
```

If constructing controlled long-tail:

```text
use the same exponential formula as CIFAR-100-LT
save selected sample ids
save class counts
keep test set fixed and clinically meaningful
```

Never silently downsample test classes to make the test set long-tailed unless that is explicitly part of the experiment. For diagnosis, balanced and natural test sets answer different questions.

---

## 6. Medical CIL setting

Possible settings:

### Strict class-incremental medical diagnosis

```text
new disease classes arrive over time
old class training images are unavailable
final classifier predicts among all seen diseases
```

This is closest to CIFAR-100-LT and easiest to implement.

### Generalized medical CIL

```text
old and new findings may appear together in later phases
class ratios change dynamically
```

This is more clinically realistic but should be a second-stage experiment.

### Multi-label continual diagnosis

```text
a single image may have multiple findings
new labels are introduced over time
old labels may still be present
```

This is clinically strong but requires different loss, metrics, and label masking.

---

## 7. Medical result table template

```text
Method | Avg Acc | Final Acc | Forgetting | Macro-F1 | Balanced Acc | Tail Recall | Head-Tail Gap | AUPRC
Finetune
Finetune + GPA
```

For each class group:

```text
head classes: common diseases / findings
medium classes
tail classes: rare diseases / findings
```

Report the exact number of images per class.

---

## 8. First medical Codex task

When ready, ask Codex:

```text
Read docs/MEDICAL_LT_TRANSFER.md.
Do not change the GPA implementation.
Add one medical classification dataset adapter and reuse the existing LT-CIL training loop and metrics.
Start with a Plan. The first goal is a local smoke run, not final medical results.
```
