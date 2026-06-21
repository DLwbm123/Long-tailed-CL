# HyperKvasir23 Phase1 Collapse Diagnosis

## Scope

This report diagnoses the HyperKvasir23 phase1 old-task collapse observed in the bounded medical CIL gate. It does not expand modules, run full 6-phase experiments, run 3 seeds, or tune new methods.

Current gate:

- Dataset: HyperKvasir23
- Data path: `/dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23`
- Code path: `/dev/shm/wangbomin/LongTailedCL/code`
- Split: 23 classes, `13 base + 5x2 incremental`
- Active diagnostic horizon: `max_phases=2`
- Seed: `0`
- Epochs: `5`

## Commands Run

Static tool check:

```bash
python3 -m py_compile tools/diagnose_hyperkvasir_phase1_collapse.py
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile /dev/shm/wangbomin/LongTailedCL/code/tools/diagnose_hyperkvasir_phase1_collapse.py
```

Eval-only checkpoint diagnosis:

```bash
/opt/miniconda3/envs/torchgpu/bin/python /dev/shm/wangbomin/LongTailedCL/code/tools/diagnose_hyperkvasir_phase1_collapse.py \
  --run-root /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_module_matrix_20260620-111904_s0_5ep \
  --ckpt-root /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_module_matrix_20260620-111904_s0_5ep \
  --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 \
  --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_collapse_diag_20260620 \
  --methods finetune taconcm_stage3_calib_tailanchor taconcm_stage4_full \
  --device cuda \
  --batch-size 64 \
  --num-workers 4
```

Freeze-backbone/BN bounded diagnostic:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
PY=/opt/miniconda3/envs/torchgpu/bin/python
cd "$BASE/code"
RUN="$BASE/runs/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep/finetune_freeze_backbone_bn"
CKPT="$BASE/checkpoints/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep/finetune_freeze_backbone_bn"
timeout 1800 "$PY" train.py \
  --dataset hyper_kvasir23 \
  --data-root "$BASE/data/hyper-kvasir23" \
  --method finetune \
  --base-classes 13 \
  --incremental-steps 5 \
  --max-phases 2 \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4 \
  --lr 0.01 \
  --scheduler cosine \
  --seed 0 \
  --order shuffled \
  --device cuda \
  --output "$RUN" \
  --ckpt-dir "$CKPT" \
  --cache-dir "$BASE/cache" \
  --freeze-backbone-after-base \
  --freeze-bn-after-base \
  --train-new-head-only-after-base \
  --no-download
```

Freeze checkpoint eval-only diagnosis:

```bash
/opt/miniconda3/envs/torchgpu/bin/python /dev/shm/wangbomin/LongTailedCL/code/tools/diagnose_hyperkvasir_phase1_collapse.py \
  --run-root /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep \
  --ckpt-root /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep \
  --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 \
  --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_collapse_diag_20260620/freeze_backbone_bn_eval \
  --methods finetune_freeze_backbone_bn \
  --device cuda \
  --batch-size 64 \
  --num-workers 4
```

## Configs Created Or Modified

- Added `tools/diagnose_hyperkvasir_phase1_collapse.py`.
- Added this report: `docs/HYPERKVASIR23_PHASE1_COLLAPSE_DIAGNOSIS.md`.
- No method module was changed.
- No TaConCM/APART/GPA module expansion was run.

## Class Mapping And Split Audit

Class order:

```text
[18, 4, 20, 10, 11, 2, 21, 6, 19, 3, 22, 8, 0, 12, 16, 13, 7, 5, 17, 14, 9, 1, 15]
```

Base classes:

```text
[18, 4, 20, 10, 11, 2, 21, 6, 19, 3, 22, 8, 0]
```

Phase1 current classes:

```text
[12, 16]
```

Seen classes at phase1:

```text
[18, 4, 20, 10, 11, 2, 21, 6, 19, 3, 22, 8, 0, 12, 16]
```

Checks:

- Old/current overlap: none.
- Seen/future overlap: none.
- Missing classes from task split: none.
- Duplicate task classes: 0.
- Train/test label symmetric difference: none.
- Phase1 evaluation masks logits to seen classes only.

Per-class counts:

| class_id | class_name | split | train | test |
|---:|---|---|---:|---:|
| 0 | barretts | old/base | 32 | 9 |
| 1 | barretts-short-segment | future | 42 | 11 |
| 2 | bbps-0-1 | old/base | 516 | 130 |
| 3 | bbps-2-3 | old/base | 918 | 230 |
| 4 | cecum | old/base | 807 | 202 |
| 5 | dyed-lifted-polyps | future | 801 | 201 |
| 6 | dyed-resection-margins | old/base | 791 | 198 |
| 7 | esophagitis-a | future | 322 | 81 |
| 8 | esophagitis-b-d | old/base | 208 | 52 |
| 9 | hemorrhoids | future | 4 | 2 |
| 10 | ileum | old/base | 7 | 2 |
| 11 | impacted-stool | old/base | 104 | 27 |
| 12 | polyps | current/phase1 | 822 | 206 |
| 13 | pylorus | future | 799 | 200 |
| 14 | retroflex-rectum | future | 312 | 79 |
| 15 | retroflex-stomach | future | 611 | 153 |
| 16 | ulcerative-colitis-grade-0-1 | current/phase1 | 28 | 7 |
| 17 | ulcerative-colitis-grade-1 | future | 160 | 41 |
| 18 | ulcerative-colitis-grade-1-2 | old/base | 8 | 3 |
| 19 | ulcerative-colitis-grade-2 | old/base | 354 | 89 |
| 20 | ulcerative-colitis-grade-2-3 | old/base | 22 | 6 |
| 21 | ulcerative-colitis-grade-3 | old/base | 106 | 27 |
| 22 | z-line | old/base | 745 | 187 |

## Checkpoint Availability

Existing phase checkpoints were available, so the module matrix was not rerun:

- `finetune`: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_module_matrix_20260620-111904_s0_5ep/finetune/model_phase_1.pt`
- `taconcm_stage3_calib_tailanchor`: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_module_matrix_20260620-111904_s0_5ep/taconcm_stage3_calib_tailanchor/model_phase_1.pt`
- `taconcm_stage4_full`: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_module_matrix_20260620-111904_s0_5ep/taconcm_stage4_full/model_phase_1.pt`
- `finetune_freeze_backbone_bn`: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep/finetune_freeze_backbone_bn/model_phase_1.pt`

## Oracle-Mask Evaluation

| method | mode | AccT | bal_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| finetune | all_seen | 14.9818 | 6.6667 | 2.4902 | 0.0000 | 96.7136 | 100.0000 | 0.0000 |
| finetune | old_only_on_old | 12.5645 | 10.1291 | 5.4006 | 12.5645 |  | 0.0000 |  |
| finetune | current_only_on_current | 96.7136 | 50.0000 | 49.1647 |  | 96.7136 |  | 0.0000 |
| taconcm_stage3_calib_tailanchor | all_seen | 15.0545 | 7.6190 | 2.9003 | 0.0000 | 97.1831 | 87.7797 | 0.0000 |
| taconcm_stage3_calib_tailanchor | old_only_on_old | 21.9449 | 17.0704 | 11.5933 | 21.9449 |  | 0.0000 |  |
| taconcm_stage3_calib_tailanchor | current_only_on_current | 97.1831 | 57.1429 | 61.7823 |  | 97.1831 |  | 0.0000 |
| taconcm_stage4_full | all_seen | 14.9818 | 6.6667 | 1.8484 | 0.0000 | 96.7136 | 97.5043 | 0.0000 |
| taconcm_stage4_full | old_only_on_old | 29.4320 | 16.0139 | 12.6641 | 29.4320 |  | 0.0000 |  |
| taconcm_stage4_full | current_only_on_current | 96.7136 | 50.0000 | 49.1647 |  | 96.7136 |  | 0.0000 |
| finetune_freeze_backbone_bn | all_seen | 26.6182 | 13.0606 | 8.7031 | 13.6833 | 97.1831 | 86.3167 | 0.0000 |
| finetune_freeze_backbone_bn | old_only_on_old | 81.1532 | 51.7274 | 47.3205 | 81.1532 |  | 0.0000 |  |
| finetune_freeze_backbone_bn | current_only_on_current | 97.1831 | 57.1429 | 61.7823 |  | 97.1831 |  | 0.0000 |

Interpretation:

- Class mapping and seen-mask are not the root cause.
- Current phase classes are learned: current-only accuracy is about 96.7-97.2%.
- Original phase1 training severely damages old-class separability: old-only is only 12.6-29.4%.
- Freezing backbone and BN preserves old representation: old-only rises to 81.15%.
- Even with preserved features, all-seen old accuracy is still low because current-task logits dominate old samples.

## Logit Block Diagnostics

| method | sample | max_old | max_current | current_minus_old | current_gt_old | old_gt_current | old_norm | current_norm | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| finetune | old | 9.0834 | 26.9126 | 17.8293 | 1.0000 |  | 2.9035 | 2.9492 | 1.0157 |
| finetune | current | 0.6894 | 12.2369 | 11.5475 |  | 0.0000 | 2.9035 | 2.9492 | 1.0157 |
| taconcm_stage3_calib_tailanchor | old | 17.5369 | 20.3061 | 2.7692 | 0.8778 |  | 2.9060 | 1.5878 | 0.5464 |
| taconcm_stage3_calib_tailanchor | current | 2.1665 | 12.0414 | 9.8748 |  | 0.0000 | 2.9060 | 1.5878 | 0.5464 |
| taconcm_stage4_full | old | 6.5410 | 12.7292 | 6.1882 | 0.9750 |  | 2.9078 | 1.5715 | 0.5405 |
| taconcm_stage4_full | current | 2.7876 | 14.3235 | 11.5358 |  | 0.0000 | 2.9078 | 1.5715 | 0.5405 |
| finetune_freeze_backbone_bn | old | 8.4680 | 11.5832 | 3.1152 | 0.8632 |  | 2.9036 | 3.0619 | 1.0545 |
| finetune_freeze_backbone_bn | current | 1.7664 | 9.8215 | 8.0551 |  | 0.0000 | 2.9036 | 3.0619 | 1.0545 |

Interpretation:

- The original finetune checkpoint is pure current-logit dominance: every old sample has current max logit above old max logit.
- TaConCM stages reduce the current-over-old margin but do not recover old accuracy.
- Freeze-backbone/BN reduces representation drift, but the classifier still assigns most old samples to current classes under all-seen logits.

## Calibration Sweep

Best rows from current-task logit scaling:

| method | alpha | AccT | bal_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| finetune | 0.1000 | 22.1091 | 12.1294 | 6.6133 | 10.6713 | 84.5070 | 10.6713 | 13.6150 |
| finetune | 0.2000 | 23.2727 | 12.6459 | 6.1798 | 10.4991 | 92.9577 | 17.6420 | 4.2254 |
| taconcm_stage3_calib_tailanchor | 0.2000 | 27.3455 | 16.6057 | 11.5480 | 18.9329 | 73.2394 | 8.6919 | 24.8826 |
| taconcm_stage4_full | 0.3000 | 33.0182 | 15.9948 | 10.1530 | 22.3752 | 91.0798 | 19.4492 | 6.5728 |
| finetune_freeze_backbone_bn | 0.3000 | 80.8000 | 46.9388 | 44.4272 | 79.3460 | 88.7324 | 8.3477 | 9.8592 |

Best rows from old-logit scaling:

| method | beta | AccT | bal_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| finetune | 3.0000 | 22.1818 | 11.7625 | 4.8131 | 8.6059 | 96.2441 | 38.7263 | 0.9390 |
| taconcm_stage3_calib_tailanchor | 3.0000 | 27.7818 | 15.6643 | 9.5928 | 15.4905 | 94.8357 | 18.5026 | 2.3474 |
| taconcm_stage4_full | 3.0000 | 32.2182 | 15.6227 | 9.4254 | 21.2565 | 92.0188 | 23.2358 | 5.6338 |
| finetune_freeze_backbone_bn | 3.0000 | 80.3636 | 46.2972 | 44.6734 | 78.0551 | 92.9577 | 10.9294 | 5.1643 |

Interpretation:

- Calibration alone cannot rescue the original no-freeze checkpoints because old-only accuracy is already low.
- Calibration becomes highly effective once backbone/BN drift is removed: current-logit alpha `0.3` gives AccT `80.80`, old acc `79.35`, current acc `88.73`.
- Old-logit beta `3.0` is similarly effective after freezing: AccT `80.36`, old acc `78.06`, current acc `92.96`.

## Freeze / BN Diagnostic

Freeze configuration:

- `--freeze-backbone-after-base`
- `--freeze-bn-after-base`
- `--train-new-head-only-after-base`

Result:

- Base test accuracy remains `85.9725`.
- Phase1 all-seen AccT improves from `14.9818` to `26.6182`.
- Phase1 old acc improves from `0.0` to `13.6833`.
- Phase1 old-only accuracy improves to `81.1532`.
- Current-only accuracy remains high at `97.1831`.
- Backbone parameter max delta after phase1: `0.0`.
- BN running mean/var max delta after phase1: `0.0`.
- Old eval feature drift mean cosine: `1.0`.

Interpretation:

- Freezing backbone/BN prevents representation forgetting.
- The remaining collapse is a classifier/logit block bias problem created by phase1 classifier training without replay or a calibrated task-block objective.

## Tiny Exemplar Diagnostic

Not run.

Reason: Step 7 condition was not met. Once backbone/BN was frozen, old-only accuracy recovered to `81.15`, and evaluation-only block calibration recovered AccT to about `80.8`. This is enough to identify the main failure path without launching a replay diagnostic.

## Final Diagnosis

Root-cause ranking:

1. Evaluation/class mapping bug: unlikely. Split and masks are consistent; no unseen future class is used in phase1 evaluation.
2. Seen-class mask bug: unlikely. The diagnostic evaluator explicitly masks to phase1 seen classifier rows.
3. Latest-task logit dominance: confirmed. Old samples are overwhelmingly assigned to phase1 logits under all-seen evaluation.
4. Representation forgetting / BN-backbone drift: confirmed for the original no-freeze checkpoints. Old-only accuracy is low before freezing and high after freezing.
5. Replay-free setting too hard: likely. Without old samples or a stronger retention/calibration mechanism, phase1 CE on current classes overwrites the usable old decision surface.

Practical conclusion:

The HyperKvasir23 CIL gate failure is not evidence that NC-ConCM/TaConCM modules are invalid. The current gate first needs a stable medical-CIL baseline. The minimum viable direction is:

1. Freeze backbone and BN after base, or use a pretrained/frozen medical backbone.
2. Add a non-oracle task-block calibration rule for all-seen inference.
3. Then test a tiny exemplar replay upper bound only if freeze+calibration is still insufficient.
4. Re-evaluate methods against this stabilized baseline before running more modules or full 6-phase experiments.

## Artifact Paths

- Main diagnosis root: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_collapse_diag_20260620`
- Freeze diagnosis root: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_collapse_diag_20260620/freeze_backbone_bn_eval`
- Freeze training run: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep/finetune_freeze_backbone_bn`
- Freeze checkpoint root: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep/finetune_freeze_backbone_bn`

## Cleanup Status

- No `diagnose_hyperkvasir` or HyperKvasir `train.py` process remained after diagnostics.
- GPU status after completion: GPU0 `1 MiB, 0%`; GPU1 `1 MiB, 0%`.
