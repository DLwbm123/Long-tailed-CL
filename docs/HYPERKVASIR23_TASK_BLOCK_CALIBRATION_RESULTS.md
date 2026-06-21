# HyperKvasir23 Task-Block Calibration Results

## Scope

Evaluation-only non-oracle task-block calibration on the bounded HyperKvasir23 phase1 freeze-backbone+BN checkpoint. No full phases, multi-seed runs, old module matrix, GUIDE, or FDM v2 were run.

## Exact Command

```bash
/opt/miniconda3/envs/torchgpu/bin/python /dev/shm/wangbomin/LongTailedCL/code/tools/run_hyperkvasir_task_block_calibration.py --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 --phase0-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt --phase1-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt --exemplar-csv /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_calibration_20260621_s0 --device cuda --seed 0 --batch-size 64 --num-workers 4
```

## Alpha Selection Rules

- `old_anchor_median_ratio`: median over phase0 old-class anchors of `max_old_logit / (max_current_logit + eps)`, clipped to `[0.05, 1.0]`.
- `old_anchor_mean_stat`: mean-stat matching on phase0 old-class anchors, `mean(max_old_logit) / mean(max_current_logit)`, clipped to `[0.05, 1.0]`.
- `old_exemplar_median_ratio`: median over 1-shot old exemplars of `max_old_logit / (max_current_logit + eps)`, clipped to `[0.05, 1.0]`.
- `old_exemplar_mean_stat`: mean-stat matching on 1-shot old exemplars, clipped to `[0.05, 1.0]`.
- `prior_linear`: `num_current_classes / num_old_classes`.
- `prior_sqrt`: `sqrt(num_current_classes / num_old_classes)`.

## Selection Statistics

```json
{
  "anchor": {
    "current_score_mean": 12.305362701416016,
    "current_score_median": 10.846945762634277,
    "mean_ratio_alpha": 0.4852154218359448,
    "median_ratio_alpha": 0.3686492443084717,
    "old_score_mean": 5.970752239227295,
    "old_score_median": 5.164612770080566
  },
  "exemplar": {
    "current_score_mean": 12.51740837097168,
    "current_score_median": 12.531479835510254,
    "mean_ratio_alpha": 0.47218816239612177,
    "median_ratio_alpha": 0.40304216742515564,
    "old_score_mean": 5.910572528839111,
    "old_score_median": 5.138042449951172
  },
  "num_current_classes": 2,
  "num_old_classes": 13,
  "prior_linear_alpha": 0.15384615384615385,
  "prior_sqrt_alpha": 0.3922322702763681
}
```

## Non-Oracle Calibration

| rule | source | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_anchor_median_ratio | phase0_anchors | 0.3686 | 72.2909 | 39.1097 | 38.5171 | 68.5026 | 92.9577 | 23.4079 | 5.1643 |
| old_anchor_mean_stat | phase0_anchors | 0.4852 | 55.5636 | 29.1986 | 27.9746 | 48.1928 | 95.7746 | 47.6764 | 1.4085 |
| old_exemplar_median_ratio | old_1shot_exemplars | 0.4030 | 64.4364 | 34.5917 | 33.9481 | 58.8640 | 94.8357 | 34.9398 | 2.3474 |
| old_exemplar_mean_stat | old_1shot_exemplars | 0.4722 | 56.4364 | 29.8849 | 28.7059 | 49.3115 | 95.3052 | 46.2134 | 1.8779 |
| prior_linear | class_count_prior | 0.1538 | 75.9273 | 46.5680 | 42.0525 | 80.9811 | 48.3568 | 1.3769 | 51.1737 |
| prior_sqrt | class_count_prior | 0.3922 | 66.4000 | 35.9289 | 35.5983 | 61.3597 | 93.8967 | 32.1859 | 3.2864 |

## Oracle Sweep Reference

Diagnostic only; these alphas are not selected by test performance in the proposed baseline.

| rule | source | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_alpha_0.1 | test_sweep_reference | 0.1000 | 71.1273 | 44.8612 | 38.8068 | 81.1532 | 16.4319 | 0.2582 | 83.5681 |
| oracle_alpha_0.2 | test_sweep_reference | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| oracle_alpha_0.3 | test_sweep_reference | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| oracle_alpha_0.4 | test_sweep_reference | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |
| oracle_alpha_0.5 | test_sweep_reference | 0.5000 | 54.3273 | 28.5096 | 27.4030 | 46.7298 | 95.7746 | 49.2255 | 1.4085 |
| oracle_alpha_0.7 | test_sweep_reference | 0.7000 | 30.1091 | 14.4560 | 11.4066 | 17.9002 | 96.7136 | 81.4114 | 0.0000 |
| oracle_alpha_0.9 | test_sweep_reference | 0.9000 | 26.5455 | 12.0202 | 7.8699 | 13.6833 | 96.7136 | 86.3167 | 0.0000 |
| oracle_alpha_1 | test_sweep_reference | 1.0000 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |

## Recommendation

Revise calibration before promoting it. Best non-oracle rule is `prior_linear` with AccT `75.9273`, which is still not close enough to the oracle-good alpha range.

Per-class recall is saved in `per_class_recall.csv`.

## Artifact Paths

- Output dir: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_calibration_20260621_s0`
- Phase0 checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt`
- Phase1 checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt`
- Exemplar CSV: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`
