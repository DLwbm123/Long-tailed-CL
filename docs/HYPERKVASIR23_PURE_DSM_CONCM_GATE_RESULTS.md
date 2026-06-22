# HyperKvasir23 Pure DSM-ConCM Gate Results

## Purpose

Validate whether original ConCM Dynamic Structure Matching, without MPC or task-block calibration, transfers to the HyperKvasir23 medical CIL/FSCIL phase1 setting.

This is not the previous evaluation-time prototype-relation matching diagnostic. This gate trains a two-layer projector on frozen-backbone feature tensors, builds an SVD/Procrustes-style dynamic geometric structure, and evaluates with a geometric classifier.

## Scope

- Training: only `DSMProjector` is optimized.
- Backbone: loaded from existing checkpoints and frozen with `requires_grad=False`.
- Phase2/full phases: not run.
- Seed sweep: not run; seed0 only.
- FDM v2: not run.
- MPC, WordNet, GloVe, and large-model priors: not implemented or downloaded.
- Existing `train.py`, `src/engine/finetune.py`, task-block calibration, NC-ConCM, and FDM paths are not modified.

## Implementation

- `DSMProjector`: normalize input feature -> Linear -> ReLU -> Linear -> normalize projected feature.
- Gaussian feature augmentation: use class feature mean and diagonal std with an explicit `head_idx < num_old_classes` old/current split.
- Dynamic structure: compute `M = I - 1/N * 11^T`, run `torch.linalg.svd(projected_prototypes.T @ M)`, then build normalized ETF-style geometry vectors.
- `LMatch`: cross entropy over `projected_features @ geometry_vectors.T`.
- `LCont`: anchor-augmented supervised contrastive loss using geometry vectors as class positives.
- Evaluation: freeze checkpoint features -> projector -> geometric classifier; report AccT, balanced accuracy, macro-F1, old acc, current acc, old->current, and current->old.

## Files Changed

- `src/methods/concm_dsm.py`
- `tools/run_hyperkvasir_pure_dsm_concm_gate.py`
- `docs/HYPERKVASIR23_PURE_DSM_CONCM_GATE_RESULTS.md`

## Commands Run

Static compile on server:

```bash
cd /dev/shm/wangbomin/LongTailedCL/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile \
  src/methods/concm_dsm.py \
  tools/run_hyperkvasir_pure_dsm_concm_gate.py
```

Dry run:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
/usr/bin/timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_pure_dsm_concm_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json \
  --output-dir $BASE/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0_dry \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --projector-hidden 2048 --projector-dim 128 \
  --base-projector-epochs 5 --increment-projector-epochs 10 \
  --lr 0.01 --sample-num-old 100 --sample-num-current 50 --cont-weight 1.0 \
  --dry-run
```

Smoke:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
/usr/bin/timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_pure_dsm_concm_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json \
  --output-dir $BASE/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0_smoke \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --projector-hidden 2048 --projector-dim 128 \
  --base-projector-epochs 5 --increment-projector-epochs 2 \
  --lr 0.01 --sample-num-old 100 --sample-num-current 50 --cont-weight 1.0 \
  --max-train-batches 5
```

Bounded full D:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
/usr/bin/timeout 1200 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_pure_dsm_concm_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json \
  --output-dir $BASE/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0 \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --projector-hidden 2048 --projector-dim 128 \
  --base-projector-epochs 5 --increment-projector-epochs 10 \
  --lr 0.01 --sample-num-old 100 --sample-num-current 50 --cont-weight 1.0
```

Optional E without contrastive loss:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
/usr/bin/timeout 1200 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_pure_dsm_concm_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json \
  --output-dir $BASE/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0_no_cont \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --projector-hidden 2048 --projector-dim 128 \
  --base-projector-epochs 5 --increment-projector-epochs 10 \
  --lr 0.01 --sample-num-old 100 --sample-num-current 50 --cont-weight 1.0 \
  --no-cont-loss
```

## A/B/C/D/E Comparison

| group | method | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_raw_freeze | raw_freeze | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| B_task_block_only | task_block_only | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| C_locked_NCConCM | locked_NCConCM | 77.5273 | 45.4020 | 43.5763 | 75.5594 | 88.2629 | 14.0275 | 9.8592 |
| D_pure_DSM_ConCM | pure_DSM_ConCM | 76.5818 | 45.7166 | 43.7274 | 74.4406 | 88.2629 | 9.3804 | 9.8592 |
| E_pure_DSM_ConCM_no_cont | pure_DSM_ConCM_no_cont | 66.0364 | 43.1647 | 39.1184 | 71.3425 | 37.0892 | 2.3236 | 38.0282 |

## Pure DSM vs Locked NC-ConCM

- D AccT gap vs C: `-0.9455`.
- D old acc gap vs C: `-1.1188`.
- D current acc matches C: `88.2629`.
- D old->current is better than C: `9.3804` vs `14.0275`.
- D current->old matches C: `9.8592`.
- E shows that removing LCont is not viable: current acc drops to `37.0892` and current->old rises to `38.0282`.

## Geometry Sanity

D full gate:

```json
{
  "max_abs_offdiag_error": 1.3783574104309082e-06,
  "num_classes": 15,
  "pairwise_diag_mean": 1.0,
  "pairwise_offdiag_mean": -0.071428582072258,
  "pairwise_offdiag_target": -0.07142857142857142,
  "proj_dim": 128
}
```

E no-cont:

```json
{
  "max_abs_offdiag_error": 1.601874828338623e-06,
  "num_classes": 15,
  "pairwise_diag_mean": 1.0,
  "pairwise_offdiag_mean": -0.0714285746216774,
  "pairwise_offdiag_target": -0.07142857142857142,
  "proj_dim": 128
}
```

## Acceptance Gate

D passes the bounded seed0 gate:

```json
{
  "AccT_gap_vs_locked": -0.9454545454545524,
  "checks": {
    "AccT_close_or_above_locked": true,
    "current_acc_ge_85": true,
    "current_to_old_lt_15": true,
    "old_acc_above_raw": true,
    "old_to_current_below_raw": true
  },
  "locked_AccT_reference": 77.52727272727273,
  "passed": true,
  "raw_old_acc_reference": 12.392426850258175,
  "raw_old_to_current_reference": 87.60757314974182
}
```

E does not pass because current accuracy collapses.

## Output Files

D output dir:

`/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0`

E output dir:

`/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0_no_cont`

Each output dir contains:

- `final_results.json`
- `summary.csv`
- `train_trace.csv`
- `dsm_config.json`
- `geometry_stats.json`
- `pairwise_geometry_dot.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `run_summary.md`
- `projector.pt`

`projector.pt` contains only projector state, projector config, geometry vectors, and class/head metadata. It does not save backbone weights.

## Conclusion

Pure DSM-ConCM has a real positive signal on HyperKvasir23 seed0. It is far stronger than raw freeze, passes the current-class and confusion guardrails, and lands within one AccT point of locked NC-ConCM while reducing old->current errors more than C. It should not replace locked NC-ConCM yet because C still has higher AccT and old accuracy, but original ConCM DSM is worth carrying forward as a component.

LCont is important in this medical gate. Without it, DSM overprotects old structure and current-class performance collapses.

## Next Steps

- Module ablation has now been run in `docs/HYPERKVASIR23_CONCM_MODULE_ABLATION_RESULTS.md`; MPC-lite did not improve over DSM-only on seed0.
- Run seed1 bounded DSM only if we want to test stability of this positive seed0 signal.
- Test DSM combined with task-block calibration before adding MPC.
- Keep MPC disabled until DSM-only is stable across at least seed0/seed1.
- Do not replace the locked NC-ConCM skeleton based on this single seed.

## GPU And Process Cleanup

After the runs, both V100s were idle at 1 MiB memory used and 0% utilization. No `run_hyperkvasir_pure_dsm_concm_gate.py` or `torchgpu/bin/python` process remained.
