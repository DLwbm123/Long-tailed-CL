# HyperKvasir23 Current-Preserving Task-Block Calibration Results

## Scope

Evaluation-only Current-Preserving Task-Block Calibration (CP-TBC). Alpha selection uses phase0 old anchors, 1-shot old exemplars, and phase1 current-task training samples only. Test accuracy is used only after alpha is selected; oracle sweep is diagnostic only. No training, checkpoints, model weights, classifier heads, backbone, BN, replay, FDM, or ConCM modules are modified.

## Exact Command

```bash
cd /dev/shm/wangbomin/LongTailedCL/code && CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/torchgpu/bin/python tools/run_hyperkvasir_current_preserving_calibration.py --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 --phase0-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt --phase1-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt --exemplar-csv /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_current_preserving_calibration_20260621_s0 --report-path /dev/shm/wangbomin/LongTailedCL/code/docs/HYPERKVASIR23_CURRENT_PRESERVING_CALIBRATION_RESULTS.md --device cuda --seed 0 --batch-size 64 --num-workers 4
```

## Calibration Reference

- `current_ref = current_train_acc(alpha=1.0) = 0.9671`
- `rho0.85 threshold = 0.8220`
- `rho0.90 threshold = 0.8704`
- `rho0.95 threshold = 0.9187`

## Calibration-Set Alpha Selection Table

| alpha | old_anchor_retention | old_exemplar_retention | current_train_acc | valid_rho085 | valid_rho090 | valid_rho095 | old_anchor_margin | current_train_margin | cp_margin_balance_abs | hmean_anchor | hmean_exemplar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 1.0000 | 0.4615 | 0.2035 | False | False | False | 4.7402 | -0.6822 | 5.4224 | 0.3382 | 0.2825 |
| 0.1500 | 1.0000 | 0.4615 | 0.5118 | False | False | False | 4.1249 | -0.2355 | 4.3604 | 0.6770 | 0.4854 |
| 0.2000 | 0.9231 | 0.4615 | 0.7235 | False | False | False | 3.5097 | 0.2112 | 3.2984 | 0.8112 | 0.5636 |
| 0.2500 | 0.6923 | 0.4615 | 0.8329 | True | False | False | 2.8944 | 0.6580 | 2.2365 | 0.7561 | 0.5940 |
| 0.3000 | 0.6154 | 0.4615 | 0.8882 | True | True | False | 2.2791 | 1.1047 | 1.1745 | 0.7271 | 0.6074 |
| 0.3500 | 0.6154 | 0.4615 | 0.9188 | True | True | True | 1.6639 | 1.5514 | 0.1125 | 0.7371 | 0.6144 |
| 0.4000 | 0.4615 | 0.3077 | 0.9424 | True | True | True | 1.0486 | 1.9981 | 0.9495 | 0.6196 | 0.4639 |
| 0.4500 | 0.4615 | 0.3077 | 0.9529 | True | True | True | 0.4333 | 2.4448 | 2.0115 | 0.6219 | 0.4652 |
| 0.5000 | 0.3846 | 0.2308 | 0.9576 | True | True | True | -0.1819 | 2.8915 | 3.0735 | 0.5488 | 0.3719 |
| 0.6000 | 0.0769 | 0.0769 | 0.9635 | True | True | True | -1.4125 | 3.7850 | 5.1974 | 0.1425 | 0.1425 |
| 0.7000 | 0.0769 | 0.0769 | 0.9659 | True | True | True | -2.6430 | 4.6784 | 7.3214 | 0.1425 | 0.1425 |
| 0.8000 | 0.0769 | 0.0769 | 0.9659 | True | True | True | -3.8735 | 5.5719 | 9.4454 | 0.1425 | 0.1425 |
| 0.9000 | 0.0769 | 0.0769 | 0.9671 | True | True | True | -5.1041 | 6.4653 | 11.5694 | 0.1425 | 0.1425 |
| 1.0000 | 0.0769 | 0.0769 | 0.9671 | True | True | True | -6.3346 | 7.3587 | 13.6933 | 0.1425 | 0.1425 |

## Selected Alpha By Rule

| rule | alpha | raw_alpha | selection_source | fallback_used | current_train_acc | old_anchor_retention | old_exemplar_retention | cp_margin_balance_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cp_anchor_rho090 | 0.3000 |  | max_old_anchor_retention_with_current_train_acc_ge_0.90_ref | False | 0.8882 | 0.6154 | 0.4615 | 1.1745 |
| cp_exemplar_rho090 | 0.3000 |  | max_old_exemplar_retention_with_current_train_acc_ge_0.90_ref | False | 0.8882 | 0.6154 | 0.4615 | 1.1745 |
| cp_anchor_rho085 | 0.2500 |  | max_old_anchor_retention_with_current_train_acc_ge_0.85_ref | False | 0.8329 | 0.6923 | 0.4615 | 2.2365 |
| cp_anchor_rho095 | 0.3500 |  | max_old_anchor_retention_with_current_train_acc_ge_0.95_ref | False | 0.9188 | 0.6154 | 0.4615 | 0.1125 |
| cp_margin_balance | 0.3500 |  | min_abs_mean_old_anchor_margin_minus_mean_current_train_margin_with_current_train_acc_ge_0.90_ref | False | 0.9188 | 0.6154 | 0.4615 | 0.1125 |
| balanced_hmean_anchor | 0.2000 |  | max_hmean_old_anchor_retention_current_train_acc | False | 0.7235 | 0.9231 | 0.4615 | 3.2984 |
| balanced_hmean_exemplar | 0.3500 |  | max_hmean_old_exemplar_retention_current_train_acc | False | 0.9188 | 0.6154 | 0.4615 | 0.1125 |
| constrained_anchor | 0.2500 |  | max_old_anchor_retention_with_current_train_acc_ge_0.80 | False | 0.8329 | 0.6923 | 0.4615 | 2.2365 |
| constrained_exemplar | 0.2500 |  | max_old_exemplar_retention_with_current_train_acc_ge_0.80 | False | 0.8329 | 0.6923 | 0.4615 | 2.2365 |
| prior_linear | 0.1500 | 0.1538 | nearest_grid_to_num_current_classes_over_num_old_classes | False | 0.5118 | 1.0000 | 0.4615 | 4.3604 |
| prior_sqrt | 0.4000 | 0.3922 | nearest_grid_to_sqrt_num_current_classes_over_num_old_classes | False | 0.9424 | 0.4615 | 0.3077 | 0.9495 |

## Test-Set Evaluation Of Selected Non-Oracle Rules

| rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cp_anchor_rho090 | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| cp_exemplar_rho090 | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| cp_anchor_rho085 | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| cp_anchor_rho095 | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| cp_margin_balance | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| balanced_hmean_anchor | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| balanced_hmean_exemplar | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| constrained_anchor | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_exemplar | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| prior_linear | 0.1500 | 75.7091 | 46.4709 | 41.9089 | 80.9811 | 46.9484 | 1.2909 | 53.0516 |
| prior_sqrt | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |

## Oracle Sweep Reference

Diagnostic only; these alphas are not selected by test performance.

| rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_alpha_0.1 | 0.1000 | 71.1273 | 44.8612 | 38.8068 | 81.1532 | 16.4319 | 0.2582 | 83.5681 |
| oracle_alpha_0.2 | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| oracle_alpha_0.25 | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| oracle_alpha_0.3 | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| oracle_alpha_0.35 | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| oracle_alpha_0.4 | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |
| oracle_alpha_0.5 | 0.5000 | 54.3273 | 28.5096 | 27.4030 | 46.7298 | 95.7746 | 49.2255 | 1.4085 |
| oracle_alpha_1 | 1.0000 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |

## Best Non-Oracle Rule

- Best rule: `cp_anchor_rho090`
- Selected alpha: `0.3000`
- AccT: `79.7818`
- Old/current acc: `79.0017` / `84.0376`

## Comparison With Oracle Alpha 0.3

- Oracle alpha 0.3 AccT: `79.7818`
- Oracle alpha 0.3 old/current acc: `79.0017` / `84.0376`
- Best non-oracle gap vs oracle AccT: `0.0000`

## Recommendation

Use as the stable HyperKvasir23 medical CIL baseline. The best non-oracle rule selects an oracle-good alpha from calibration-side statistics and preserves both old and current accuracy.

Per-class recall is saved in `per_class_recall.csv`.

## Artifact Paths

- Output dir: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_current_preserving_calibration_20260621_s0`
- Final JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_current_preserving_calibration_20260621_s0/final_results.json`
- Report path: `/dev/shm/wangbomin/LongTailedCL/code/docs/HYPERKVASIR23_CURRENT_PRESERVING_CALIBRATION_RESULTS.md`
