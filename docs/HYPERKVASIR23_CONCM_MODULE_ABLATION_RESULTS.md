# HyperKvasir23 ConCM Module Ablation Results

## Purpose

Test ConCM's two separable modules under the same seed0 HyperKvasir23 medical CIL setting: DSM structure matching and MPC-lite visual-memory prototype calibration.

MPC-lite is a medical-safe visual-memory proxy. It uses no external semantic resources and should not be read as faithful original semantic MPC.

## Exact Command

```bash
/opt/miniconda3/envs/torchgpu/bin/python tools/run_hyperkvasir_concm_module_ablation_gate.py --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 --phase0-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt --phase1-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt --exemplar-csv /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv --calibration-json /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json --diagnostics-json /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json --locked-json /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json --dsm-json /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0/final_results.json --dsm-projector /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0/projector.pt --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0 --device cuda --seed 0 --batch-size 64 --num-workers 4 --projector-hidden 2048 --projector-dim 128 --base-projector-epochs 5 --increment-projector-epochs 10 --lr 0.01 --sample-num-old 100 --sample-num-current 50 --cont-weight 1.0 --mpc-lite-alpha 0.6 --mpc-lite-topk 5 --mpc-lite-tau 16.0 --mpc-lite-gamma 0.6 --groups A,B,C,D,E,M,N,O --reuse-existing-dsm --run-mpc-lite --run-mpc-lite-dsm --run-no-cont
```

## Module Branches

- `A`: raw frozen phase1 classifier reference.
- `B`: task-block calibration reference using balanced_hmean_exemplar.
- `C`: locked NC-ConCM reference.
- `D`: reused Pure DSM-ConCM reference.
- `E`: reused or optional Pure DSM without contrastive loss.
- `M`: MPC-lite prototype classifier in frozen feature space.
- `N`: MPC-lite calibrated prototypes/stds feeding DSM projector training.
- `O`: optional MPC-lite DSM without contrastive loss.

## Results

| group | method | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_raw_freeze | raw_freeze | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| B_task_block_only | task_block_only | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| C_locked_NCConCM | locked_NCConCM | 77.5273 | 45.4020 | 43.5763 | 75.5594 | 88.2629 | 14.0275 | 9.8592 |
| D_pure_DSM_ConCM | pure_DSM_ConCM_reused | 76.5818 | 45.7166 | 43.7274 | 74.4406 | 88.2629 | 9.3804 | 9.8592 |
| E_pure_DSM_ConCM_no_cont | pure_DSM_ConCM_no_cont_reused | 66.0364 | 43.1647 | 39.1184 | 71.3425 | 37.0892 | 2.3236 | 38.0282 |
| M_MPC_lite_only | MPC_lite_prototype_classifier | 71.9273 | 48.2049 | 42.9628 | 73.2358 | 64.7887 | 2.9260 | 33.3333 |
| N_MPC_lite_DSM | MPC_lite_DSM | 75.3455 | 45.3386 | 43.6182 | 72.9776 | 88.2629 | 8.3477 | 9.8592 |
| O_MPC_lite_DSM_no_cont | MPC_lite_DSM_no_cont | 67.3455 | 43.3775 | 39.7351 | 72.3752 | 39.9061 | 3.0981 | 33.8028 |

## Acceptance

```json
{
  "N_minus_C": {
    "AccT": -2.181818181818187,
    "balanced_acc": -0.06342781110112838,
    "current_acc": 0.0,
    "current_to_old_rate": 0.0,
    "macro_f1": 0.04194062087086792,
    "old_acc": -2.5817555938037913,
    "old_to_current_rate": -5.67986230636833
  },
  "N_minus_D": {
    "AccT": -1.2363636363636346,
    "balanced_acc": -0.37808238162825347,
    "current_acc": 0.0,
    "current_to_old_rate": 0.0,
    "macro_f1": -0.10915757405326332,
    "old_acc": -1.4629948364888037,
    "old_to_current_rate": -1.032702237521514
  },
  "approaches_locked_C": false,
  "checks": {
    "M_current_to_old_lt_15": false,
    "N_AccT_ge_D": false,
    "N_balanced_ge_D": false,
    "N_current_ge_85": true,
    "N_current_to_old_lt_15": true,
    "N_macro_ge_D": false,
    "N_old_acc_ge_D": false,
    "N_old_to_current_not_worse_than_D_by_gt_2": true
  },
  "status": "failed_vs_D"
}
```

## Interpretation

MPC-lite does not improve over DSM-only in this gate. DSM-only remains the cleaner branch unless a stronger medical prior is added.

## Artifact Paths

- `output_dir`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0`
- `final_results`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/final_results.json`
- `summary`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/summary.csv`
- `delta_vs_C`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/delta_vs_C.csv`
- `delta_vs_D`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/delta_vs_D.csv`
- `mpc_lite_calibration_stats`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/mpc_lite_calibration_stats.csv`
- `mpc_lite_topk_memory`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/mpc_lite_topk_memory.csv`
- `per_class_metrics`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/per_class_metrics.csv`
- `module_ablation_config`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/module_ablation_config.json`
- `run_summary`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/run_summary.md`
- `projector_mpc_lite_dsm`: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_module_ablation_gate_20260621_s0/projector_mpc_lite_dsm.pt`

## Safety Checks

- Backbone parameters were frozen before feature extraction.
- No backbone checkpoint is saved by this runner.
- Only `DSMProjector` parameters are optimized for DSM branches.
- Semantic prior inputs are optional local JSON/CSV files; the runner has no network fetch path.
