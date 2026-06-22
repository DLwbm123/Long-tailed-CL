# HyperKvasir23 NC-ConCM Rule-Lock Results

## 2026-06-21 Full-ConCM Diagnostic Follow-Up

Follow-up results are in `docs/HYPERKVASIR23_FULL_CONCM_DIAGNOSTIC_RESULTS.md`.

That run opened reliability gating, matching diagnostics/correction, constrained diagnostic rules, and a quality-gated selector under evaluation-only constraints. The full-module rows did not satisfy the replacement rule: F is worse than locked C on seed0, and G fails the seed0 current/current->old guardrail. Keep locked C as the current main method skeleton; treat reliability, matching, constrained rules, and selector logic as diagnostics/future work.

## Scope

Purpose: lock the current HyperKvasir23 medical CIL main-method skeleton cleanly:

Freeze + task-block calibration + NC-ConCM class-wise correction.

This run is evaluation-only. It did not train, did not run phase2, did not run full phases, did not run a seed sweep, did not run FDM v2, did not save large weights, and did not change training logic. It only reused existing seed0/seed1 phase0 anchors, phase1 freeze checkpoints, old exemplar CSVs, calibration JSONs, and diagnostic JSONs.

Locked configuration:

- Alpha rule: `balanced_hmean_exemplar`.
- Alpha source: existing non-oracle task-block calibration JSON.
- Rung: B task-block only and C task-block + NC-ConCM only.
- Lambda: fixed `0.30`.
- Lambda grid search: disabled.
- Reliability gating: disabled.
- Matching diagnostic: disabled.
- Matching correction: disabled.
- Rule search: disabled.
- Test-oracle parameter selection: disabled.

## Tool Change

`tools/run_hyperkvasir_concm_min_gate.py` was minimally extended with rule-lock controls:

- `--fixed-lambda`
- `--disable-lambda-grid`
- `--groups`
- `--disable-reliability-gate`
- `--disable-matching-diagnostic`
- `--disable-matching-correction`

The tool remains evaluation-only. Static checks passed:

- `py_compile` passed locally and remotely.
- No `torch.save`.
- No `loss.backward` or `.backward(`.
- No `optimizer.step`.
- No `model.train(`.

## Artifacts

Seed0 inputs:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt`
- Phase1 freeze: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt`
- Exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`
- Calibration JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json`
- Diagnostics JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json`

Seed1 inputs:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt`
- Phase1 freeze: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt`
- Exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv`
- Calibration JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json`
- Diagnostics JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json`

Outputs:

- Seed0: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0`
- Seed1: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s1`

Each output directory contains only the B/C rule-lock outputs and shared confusion summaries. No prototype relation or top-drift matching files were written.

## Commands

Seed0:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_ncconcm_rule_lock_20260621_s0
timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar \
  --fixed-lambda 0.30 \
  --groups B,C \
  --disable-lambda-grid \
  --disable-reliability-gate \
  --disable-matching-diagnostic \
  --disable-matching-correction
```

Seed1:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_ncconcm_rule_lock_20260621_s1
timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 1 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar \
  --fixed-lambda 0.30 \
  --groups B,C \
  --disable-lambda-grid \
  --disable-reliability-gate \
  --disable-matching-diagnostic \
  --disable-matching-correction
```

## Locked B/C Results

| seed | group | alpha | lambda | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | B_task_block_only | 0.35 | 0.00 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| seed0 | C_task_block_nc_concm | 0.35 | 0.30 | 77.5273 | 45.4020 | 43.5763 | 75.5594 | 88.2629 | 14.0275 | 9.8592 |
| seed1 | B_task_block_only | 0.20 | 0.00 | 53.5407 | 32.8560 | 24.1794 | 40.3430 | 89.8263 | 36.0108 | 3.4739 |
| seed1 | C_task_block_nc_concm | 0.20 | 0.30 | 54.2687 | 33.6778 | 26.1992 | 41.6968 | 88.8337 | 32.6715 | 4.2184 |

## Delta

Delta is C minus B.

| seed | Delta AccT | Delta old | Delta current | Delta old_to_current | Delta current_to_old |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed0 | +1.0182 | +1.7212 | -2.8169 | -2.7539 | +2.8169 |
| seed1 | +0.7280 | +1.3538 | -0.9926 | -3.3394 | +0.7444 |

## Consistency Check

The locked B/C rows exactly reproduce the corresponding rows from the previous ConCM-min gate at full JSON precision:

| seed | group | max_abs_diff_vs_previous |
| --- | --- | ---: |
| seed0 | B_task_block_only | 0.00000000 |
| seed0 | C_task_block_nc_concm | 0.00000000 |
| seed1 | B_task_block_only | 0.00000000 |
| seed1 | C_task_block_nc_concm | 0.00000000 |

## Oracle Leakage Check

- Alpha comes from `balanced_hmean_exemplar`, selected by the existing non-oracle calibration path using phase0 anchors, old exemplars, and current-task training samples.
- Lambda is fixed at `0.30` for this locked gate.
- This run did not sweep lambda.
- This run did not search alpha rules.
- This run did not use test results to choose rule, alpha, lambda, reliability gating, or matching.
- Previous oracle/grid rows remain diagnostic history only and are not method rows here.

## Guardrail

Acceptance criteria:

1. Both seeds satisfy current accuracy `>= 85`.
2. Both seeds satisfy current->old `< 15`.
3. Both seeds improve AccT or old accuracy over B.
4. Seed0 old->current drops.
5. Seed1 old->current does not worsen and preferably drops.
6. Results match the previous ConCM-min gate.
7. No oracle selection, grid search, reliability gate, or matching module is active.

Result: pass.

- Seed0 C current `88.2629`, current->old `9.8592`.
- Seed1 C current `88.8337`, current->old `4.2184`.
- Both seeds improve AccT and old accuracy over B.
- Old->current drops on both seeds.
- Matching disabled in JSON: `enabled=false`, `correction_enabled=false`, top drift rows `0`.
- Summary groups are exactly `B_task_block_only` and `C_task_block_nc_concm`.

## Conclusion

Locked NC-ConCM provides a clean evaluation-only positive gate over task-block calibration under fixed non-oracle alpha selection and fixed `lambda=0.30`. The effect is bounded but consistent, supporting Freeze + Task-block Calibration + NC-ConCM as the current main method skeleton.

This is not a full stable validation. Seed1 remains artifact-weak, and the gains are modest. Do not promote reliability gating or prototype relation matching into the main method from this gate.

## Next Step

Freeze this rule-lock configuration as the current method skeleton before any larger validation. The next useful step is to define a non-oracle rationale for `lambda=0.30` or run one narrowly bounded artifact-quality check. Do not move to phase2, full phases, seed sweep, or FDM v2 solely from this result.

## Cleanup

After the locked gate, no HyperKvasir training/evaluation process was running. `nvidia-smi` showed both V100 GPUs at 1 MiB memory, 0% utilization, and no active GPU process.
