# HyperKvasir23 Med-FDM Gate Results

## Scope

Bounded `max_phases=2`, seed0, 5-epoch diagnostic. Phase0 is trained once; phase1 branches are forked from the same phase0 state.

## Exact Command

```bash
/opt/miniconda3/envs/torchgpu/bin/python /dev/shm/wangbomin/LongTailedCL/code/tools/run_hyperkvasir_med_fdm_gate.py --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep --ckpt-dir /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep --cache-dir /dev/shm/wangbomin/LongTailedCL/cache --device cuda --seed 0 --epochs 5 --batch-size 32 --num-workers 4 --lr 0.01 --exemplars-per-class 1 --fdm-lambda 0.01
```

## Exemplar List Summary

- Exemplars per old class: `1`
- Total old exemplars: `13`
- Exemplar file: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`

## Phase0 Anchor Stats

```json
{
  "class_ids": [
    0,
    2,
    3,
    4,
    6,
    8,
    10,
    11,
    18,
    19,
    20,
    21,
    22
  ],
  "finite": true,
  "norm_max": 19.51650619506836,
  "norm_mean": 8.715543746948242,
  "norm_min": 6.117652416229248,
  "num_anchors": 13
}
```

## Freeze Baseline Comparison

| method | mode | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze_baseline | all_seen | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| freeze_baseline | old_only_on_old | 81.1532 | 50.4560 | 45.9362 | 81.1532 |  | 0.0000 |  |
| freeze_baseline | current_only_on_current | 96.7136 | 50.0000 | 49.1647 |  | 96.7136 |  | 0.0000 |
| med_fdm | all_seen | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| med_fdm | old_only_on_old | 81.1532 | 50.4560 | 45.9362 | 81.1532 |  | 0.0000 |  |
| med_fdm | current_only_on_current | 96.7136 | 50.0000 | 49.1647 |  | 96.7136 |  | 0.0000 |

## Calibration Sweep

| method | mode | current_alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze_baseline | scale_current_logits | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| freeze_baseline | scale_current_logits | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| freeze_baseline | scale_current_logits | 0.1000 | 71.1273 | 44.8612 | 38.8068 | 81.1532 | 16.4319 | 0.2582 | 83.5681 |
| freeze_baseline | scale_current_logits | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |
| freeze_baseline | head_norm_current_alpha | 0.8997 | 26.5455 | 12.0202 | 7.8699 | 13.6833 | 96.7136 | 86.3167 | 0.0000 |
| med_fdm | scale_current_logits | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| med_fdm | scale_current_logits | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| med_fdm | scale_current_logits | 0.1000 | 71.1273 | 44.8612 | 38.8068 | 81.1532 | 16.4319 | 0.2582 | 83.5681 |
| med_fdm | scale_current_logits | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |
| med_fdm | head_norm_current_alpha | 0.8997 | 26.5455 | 12.0202 | 7.8699 | 13.6833 | 96.7136 | 86.3167 | 0.0000 |

## Feature Matching Diagnostics

| method | train_ce_loss | fdm_loss_raw | fdm_loss_weighted | fdm_to_ce_ratio | fdm_batches | fdm_requires_grad_batches | fdm_nonzero_value_batches | exemplar_cos_before | exemplar_cos_after | old_eval_cos_before | old_eval_cos_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze_baseline | 0.2382 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.9518 | 0.9518 | 0.9653 | 0.9653 |
| med_fdm | 0.2382 | 0.0414 | 0.0004 | 0.0017 | 135 | 0 | 135 | 0.9518 | 0.9518 | 0.9653 | 0.9653 |

FDM gradient probe:

```json
{
  "fdm_probe_loss": 0.039567071944475174,
  "fdm_probe_nonzero_grad": false,
  "fdm_probe_nonzero_grad_params": 0,
  "fdm_probe_requires_grad": false,
  "fdm_probe_trainable_grad_l1": 0.0
}
```

## Recommendation

Reject this first Med-FDM gate as an active method under the current frozen-backbone protocol: the FDM loss is finite but has no gradient path to the trainable classifier-only branch.

A meaningful feature-space matching method in this ResNet scaffold requires a trainable feature path during phase1, such as unfrozen late blocks, adapters, or a projector. Under strict backbone freezing, feature matching is diagnostic-only.

## Artifact Paths

- Output dir: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep`
- Checkpoint dir: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep`
