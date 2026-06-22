# HyperKvasir23 Pure DSM-ConCM Gate Results

## Purpose

Validate whether original ConCM Dynamic Structure Matching, without MPC or task-block calibration, transfers to the HyperKvasir23 medical CIL/FSCIL phase1 setting.

This is not the previous evaluation-time prototype-relation matching diagnostic. This gate trains a two-layer projector on frozen-backbone feature tensors, builds an SVD/Procrustes-style dynamic geometric structure, and evaluates with a geometric classifier.

## Scope

- Training: only `DSMProjector` is optimized.
- Backbone: loaded from existing checkpoints and frozen with `requires_grad=False`.
- Phase2/full phases: not run.
- Seed sweep: not run; seed0 is the default.
- FDM v2: not run.
- MPC, WordNet, GloVe, and large-model priors: not implemented or downloaded.
- Existing `train.py`, `src/engine/finetune.py`, task-block calibration, NC-ConCM, and FDM paths are not modified.

## Implementation

- `DSMProjector`: normalize input feature -> Linear -> ReLU -> Linear -> normalize projected feature.
- Gaussian feature augmentation: use class feature mean and diagonal std with an explicit `head_idx < num_old_classes` old/current split.
- Dynamic structure: compute `M = I - 1/N * 11^T`, run `torch.linalg.svd(projected_prototypes.T @ M)`, then build normalized ETF-style geometry vectors.
- `LMatch`: cross entropy over `projected_features @ geometry_vectors.T`.
- `LCont`: optional anchor-augmented supervised contrastive loss using geometry vectors as class positives.
- Evaluation: freeze checkpoint features -> projector -> geometric classifier; report AccT, balanced accuracy, macro-F1, old acc, current acc, old->current, and current->old.

## Files Changed

- `src/methods/concm_dsm.py`
- `tools/run_hyperkvasir_pure_dsm_concm_gate.py`
- `docs/HYPERKVASIR23_PURE_DSM_CONCM_GATE_RESULTS.md`

## Commands

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
  --output-dir $BASE/runs/hyperkvasir23_pure_dsm_concm_gate_20260621_s0 \
  --device cuda \
  --seed 0 \
  --batch-size 64 \
  --num-workers 4 \
  --projector-hidden 2048 \
  --projector-dim 128 \
  --base-projector-epochs 5 \
  --increment-projector-epochs 10 \
  --lr 0.01 \
  --sample-num-old 100 \
  --sample-num-current 50 \
  --cont-weight 1.0 \
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
  --device cuda \
  --seed 0 \
  --batch-size 64 \
  --num-workers 4 \
  --projector-hidden 2048 \
  --projector-dim 128 \
  --base-projector-epochs 5 \
  --increment-projector-epochs 2 \
  --lr 0.01 \
  --sample-num-old 100 \
  --sample-num-current 50 \
  --cont-weight 1.0 \
  --max-train-batches 5
```

Bounded full gate:

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
  --device cuda \
  --seed 0 \
  --batch-size 64 \
  --num-workers 4 \
  --projector-hidden 2048 \
  --projector-dim 128 \
  --base-projector-epochs 5 \
  --increment-projector-epochs 10 \
  --lr 0.01 \
  --sample-num-old 100 \
  --sample-num-current 50 \
  --cont-weight 1.0
```

## A/B/C/D/E Comparison

Pending remote run.

| group | method | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | raw_freeze | pending | pending | pending | pending | pending | pending | pending |
| B | task_block_only | pending | pending | pending | pending | pending | pending | pending |
| C | locked_NCConCM | pending | pending | pending | pending | pending | pending | pending |
| D | pure_DSM_ConCM | pending | pending | pending | pending | pending | pending | pending |
| E | pure_DSM_ConCM_no_cont | optional | optional | optional | optional | optional | optional | optional |

## Geometry Sanity

Pending remote run. The tool writes:

- `geometry_stats.json`
- `pairwise_geometry_dot.csv`

The report records pairwise diagonal mean, off-diagonal mean, off-diagonal target `-1/(N-1)`, and max absolute off-diagonal error.

## Acceptance Gate

Pure DSM-ConCM is considered useful on seed0 if:

1. old acc is clearly above raw freeze old `12.39`;
2. current acc is at least `85`;
3. current->old is below `15`;
4. old->current is clearly below raw freeze old->current `87.61`;
5. AccT is close to or above locked NC-ConCM C `77.53`, or the result clearly supports DSM transfer value while remaining weaker than locked NC-ConCM.

Current status: pending remote run.

## Output Files

The tool writes these files under `--output-dir`:

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

## Next Steps

- Run seed0 dry-run, smoke, then bounded full gate once the 20035 SSH credential is available.
- If D improves over raw freeze but remains below C, test DSM combined with task-block calibration before adding MPC.
- Add seed1 only if seed0 shows a clear positive DSM signal.
- Keep MPC disabled until DSM-only has stable value.
