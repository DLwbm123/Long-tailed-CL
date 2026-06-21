# APART Baseline Results

Date: 2026-06-16

Scope: APART official CIFAR100-LT shuffled B50-5 baseline. No TaConCM changes, no GPA changes, and no ConCM-lite implementation in this run.

## Run

Local APART checkout:

- `third_party/APART`
- upstream commit: `f3c5b8da5908b2f78266744a737f7f6bff45a43d`

Remote run:

```bash
cd /dev/shm/wangbomin/APART/code
XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
/opt/miniconda3/envs/torchgpu/bin/python main.py \
  --config ./exps/apart_cifar_shuffle.json \
  --text apart_concm_baseline_check
```

Remote log:

- `/dev/shm/wangbomin/APART/logs/apart_cifar_shuffle_20260616-090022.out`

The official config uses seed `1993`, not seed 0.

## Summary

| method | Acc | AccT | reference Acc | reference AccT | gap Acc | gap AccT |
|---|---:|---:|---:|---:|---:|---:|
| APART official config | 87.1567 | 84.90 | 84.91 | 81.93 | +2.25 | +2.97 |

Finetune could not be run from the official APART checkout without adding code, because `utils/factory.py` only registers `model_name == "apart"` and there is no Finetune learner/config in the repository.

## Phase Curve

| phase | seen classes | all-seen top1 | old | new | many | medium | few |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0-50 | 90.42 | 0.00 | 90.42 | 95.72 | 90.67 | 85.50 |
| 1 | 0-60 | 88.70 | 89.40 | 85.20 | 94.10 | 89.75 | 83.04 |
| 2 | 0-70 | 88.33 | 86.77 | 97.70 | 92.79 | 87.11 | 84.04 |
| 3 | 0-80 | 85.56 | 87.43 | 72.50 | 89.66 | 85.32 | 81.23 |
| 4 | 0-90 | 85.03 | 85.20 | 83.70 | 90.58 | 84.00 | 79.72 |
| 5 | 0-100 | 84.90 | 84.86 | 85.30 | 89.80 | 84.26 | 79.93 |

Final `CNN top1 curve`:

```text
[90.42, 88.70, 88.33, 85.56, 85.03, 84.90]
```

Average Accuracy:

```text
87.15666666666665
```

## Task Accuracy Matrix

Rows are evaluation after each phase. Columns are task class ranges in APART's mapped class order.

| eval phase | 00-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.42 |  |  |  |  |  |
| 1 | 89.40 | 85.20 |  |  |  |  |
| 2 | 88.82 | 76.50 | 97.70 |  |  |  |
| 3 | 87.68 | 76.20 | 97.40 | 72.50 |  |  |
| 4 | 87.50 | 75.90 | 96.60 | 71.60 | 83.70 |  |
| 5 | 87.44 | 75.80 | 96.00 | 71.70 | 83.00 | 85.30 |

## Credibility Check

The baseline is credible for the APART official protocol:

- The official APART config ran end-to-end with no traceback.
- CIFAR100 was loaded from local `/dev/shm` data, not redownloaded into `/`.
- The timm ViT checkpoint was cached in `/dev/shm/wangbomin/APART/cache`.
- Evaluation used APART's own all-seen top-1 `CNN` path with `logits + logits_few`.
- Final Acc/AccT are close to and slightly above the APART reference numbers.

Important boundary:

- This validates APART's own CIFAR100-LT shuffled protocol, not exact GVAlign/Liu fixed-order parity.
- This does not validate a Finetune baseline from the official APART repo, because that entrypoint is absent.

## Decision

APART baseline is strong enough to serve as the control for ConCM-lite Stage 1.

Recommended next step: implement Stage 1 only, as a small prototype-memory / covariance-augmentation ablation that preserves APART inference and keeps APART's existing losses intact.

