# TaConCM-GPA CIFAR-100-LT Full Seed 0 Results

Date checked: 2026-06-14

This note summarizes the completed server run for analyzing why the final
accuracy is low. It covers one seed only.

## 1. Run Status

- Run type: full six-stage TaConCM/GPA ablation
- Seed: `0`
- Status: completed
- Validation: `tools/summarize_taconcm_ablation.py` reported all six stages as `ok`
- Raw run root:
  `/dev/shm/wangbomin/LongTailedCL/runs/taconcm_ablation_20260613-025914_s0`
- Compact persistent summary backup:
  `/root/ltcl_final_summaries/taconcm_ablation_20260613-025914_s0`
- SwanLab:
  - Project: `LongTailed-CL-TaConCM`
  - Mode: `online`
- Important limitation: only seed `0` was run. Seeds `1` and `2` were not run.

## 2. Experiment Setting

- Dataset: `cifar100_lt`
- Imbalance ratio / rho: `0.01`
- Class order: `shuffled`
- Base classes: `50`
- Incremental steps: `5`
- Number of phases: `6`
- Task sizes: `[50, 10, 10, 10, 10, 10]`
- Epochs: `100` per phase and per stage
- Batch size: `128`
- Optimizer: SGD
- Learning rate: `0.1`
- Momentum: `0.9`
- Weight decay: `5e-4`
- Backbone: `resnet32`
- Device: one V100 GPU via `CUDA_VISIBLE_DEVICES=0`
- Python: `3.10.20`
- PyTorch: `1.13.1+cu117`

Paths:

- Data root: `/dev/shm/wangbomin/LongTailedCL/data`
- Run root: `/dev/shm/wangbomin/LongTailedCL/runs`
- Cache root: `/dev/shm/wangbomin/LongTailedCL/cache`
- Checkpoint root: `/dev/shm/wangbomin/LongTailedCL/checkpoints`

Task split:

```text
task_0: [73, 46, 43, 47, 53, 0, 27, 5, 23, 19, 51, 28, 25, 68, 42, 37, 58, 12, 67, 81, 18, 74, 7, 15, 4, 89, 38, 31, 76, 2, 54, 22, 57, 72, 64, 44, 14, 78, 87, 75, 82, 77, 24, 85, 40, 13, 92, 52, 90, 91]
task_1: [36, 61, 21, 33, 86, 84, 50, 60, 79, 95]
task_2: [88, 63, 3, 62, 96, 30, 34, 98, 20, 9]
task_3: [55, 49, 1, 66, 32, 59, 17, 69, 10, 65]
task_4: [16, 45, 48, 99, 71, 39, 6, 70, 8, 11]
task_5: [93, 26, 97, 35, 83, 56, 94, 80, 41, 29]
```

## 3. Ablation Results

`final_acc` is the final all-seen accuracy after phase 5. `avg_inc_acc` is the
average of all-seen accuracy over phases. `forgetting` is computed from the
task accuracy matrix.

| Stage | final_acc | avg_inc_acc | forgetting | many_acc | medium_acc | few_acc | anchor_target | tail_anchor | T-DSM | match_loss |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `finetune` | 4.31 | 11.464 | 51.54 | 4.40 | 4.971 | 3.433 | `gpa_raw` | false | false | false |
| `finetune_gpa` | 4.87 | 11.698 | 52.60 | 3.971 | 6.829 | 3.633 | `gpa_raw` | false | false | false |
| `taconcm_stage1_calib_init_only` | 4.25 | 11.727 | 53.72 | 4.714 | 5.314 | 2.467 | `gpa_raw` | false | false | false |
| `taconcm_stage2_calib_anchor` | 4.28 | 11.704 | 53.42 | 3.971 | 4.971 | 3.833 | `calibrated` | false | false | false |
| `taconcm_stage3_calib_tailanchor` | 4.47 | 11.702 | 52.52 | 3.943 | 5.229 | 4.200 | `calibrated` | true | false | false |
| `taconcm_stage4_full` | 5.16 | 11.681 | 52.30 | 4.229 | 6.114 | 5.133 | `tdsm` | true | true | true |

Best final all-seen accuracy in this seed:

```text
taconcm_stage4_full: 5.16
```

The improvement over plain finetune is small:

```text
5.16 - 4.31 = +0.85 percentage points
```

## 4. Final Method Configuration

The final stage is `taconcm_stage4_full`.

GPA flags:

```text
lambda_gpa = 0.12
gpa_anchor_start_phase = 1
gpa_init_bias = true
```

TaConCM flags:

```text
use_ltconcm = true
ltconcm_calibrate_prototypes = true
ltconcm_anchor_target = tdsm
ltconcm_tail_anchor = true
ltconcm_anchor_gamma = 0.5
ltconcm_anchor_max_weight = 5.0
ltconcm_memory_topk = 5
ltconcm_alpha_a = 2.0
ltconcm_alpha_b = 1.0
ltconcm_alpha_min = 0.15
ltconcm_alpha_max = 0.95
ltconcm_use_tdsm = true
ltconcm_use_match_loss = true
ltconcm_match_lambda = 0.1
```

Final stage selected metrics:

```text
final_accuracy = 5.16
average_incremental_accuracy = 11.680932539682539
forgetting = 52.3
many_acc = 4.228571428571429
medium_acc = 6.114285714285714
few_acc = 5.133333333333334
balanced_acc = 5.16
macro_f1 = 0.8937184015486914
head_tail_gap = -0.9047619047619051
final_train_loss = 0.6047627608111878
final_anchor_loss = 0.01307410651360179
final_match_loss = 0.24071713411895337
final_extra_loss = 0.02564060659042167
```

## 5. Stage 4 Accuracy Matrix

This is the key evidence for the low final result.

```text
phase,task_0,task_1,task_2,task_3,task_4,task_5
0,34.400000,,,,,
1,0.000000,50.900000,,,,
2,0.000000,0.000000,49.500000,,,
3,0.000000,0.000000,0.000000,64.300000,,
4,0.000000,0.000000,0.000000,0.000000,62.400000,
5,0.000000,0.000000,0.000000,0.000000,0.000000,51.600000
```

Interpretation:

- The model learns the current task reasonably well.
- After each new phase, previous tasks collapse to `0.0`.
- The final `5.16` all-seen accuracy is mostly explained by only task 5
  retaining `51.6%` accuracy while tasks 0 to 4 are `0.0`.
- This is severe catastrophic forgetting, not simply failure to learn the new
  task.

## 6. Main Analysis Takeaways

1. `taconcm_stage4_full` is the best final method for seed 0, but the gain is
   small.
2. GPA and TaConCM components do not prevent old-task collapse in this
   exemplar-free setup.
3. The most suspicious issue is that evaluation over all seen classes is
   dominated by classifier drift toward the latest task.
4. The current method has no rehearsal buffer and no distillation against old
   logits/features, so a final all-seen collapse is plausible.
5. The long-tailed setup makes this harder because many old tail classes have
   very few samples and no future exposure.
6. `average_incremental_accuracy` around `11.7` is much higher than final
   accuracy because early phases briefly perform better before being forgotten.

## 7. What To Ask GPT To Inspect

Useful code areas:

- `src/engine/finetune.py`
  - classifier expansion
  - training loop
  - evaluation mapping from global class id to classifier head index
  - whether old classifier weights/logits are protected after expansion
- `src/methods/gpa.py`
  - GPA prototype initialization
  - anchor loss target and weighting
  - whether anchor loss actually applies to old classes or only current data
- `src/methods/ltconcm.py`
  - prototype calibration
  - T-DSM anchor construction
  - tail anchor weights
- `src/utils/metrics.py`
  - final all-seen evaluation
  - per-task accuracy matrix construction

Specific debugging questions:

1. Is the evaluation protocol correct for class-incremental all-seen accuracy?
2. Does training only on current-task samples make anchor loss insufficient for
   old-task retention?
3. Are old classifier weights preserved mechanically but becoming ineffective
   because features drift?
4. Should old-class distillation, feature regularization, classifier freezing,
   or exemplar replay be added for a meaningful CIL baseline?
5. Is `lambda_gpa=0.12` too weak relative to CE loss and match loss?
6. Should the method log old/new task accuracy separately during training to
   detect when collapse occurs?

## 8. Reproduction Command

The full run was launched with:

```bash
PYTHON_BIN=/opt/miniconda3/envs/torchgpu/bin/python \
DATA_ROOT=/dev/shm/wangbomin/LongTailedCL/data \
OUTPUT_ROOT=/dev/shm/wangbomin/LongTailedCL/runs \
XDG_CACHE_HOME=/dev/shm/wangbomin/LongTailedCL/cache \
TORCH_HOME=/dev/shm/wangbomin/LongTailedCL/cache/torch \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_cifar100lt_taconcm_ablation.sh \
  --full \
  --seed 0 \
  --run-root /dev/shm/wangbomin/LongTailedCL/runs/taconcm_ablation_20260613-025914_s0 \
  --data-root /dev/shm/wangbomin/LongTailedCL/data \
  --cache-dir /dev/shm/wangbomin/LongTailedCL/cache \
  --ckpt-dir /dev/shm/wangbomin/LongTailedCL/checkpoints \
  --use-swanlab \
  --swanlab-project LongTailed-CL-TaConCM \
  --swanlab-mode online
```

Summarizer command:

```bash
/opt/miniconda3/envs/torchgpu/bin/python tools/summarize_taconcm_ablation.py \
  --root /dev/shm/wangbomin/LongTailedCL/runs/taconcm_ablation_20260613-025914_s0
```

## 9. Result Files

Per-stage files exist under:

```text
/dev/shm/wangbomin/LongTailedCL/runs/taconcm_ablation_20260613-025914_s0/<stage>/
```

Each stage contains:

```text
summary.json
metrics.jsonl
config.yaml
acc_matrix.csv
run.log
```

Compact copied summary files are under:

```text
/root/ltcl_final_summaries/taconcm_ablation_20260613-025914_s0
```

