# HyperKvasir23 Full-ConCM Diagnostic Results

## Scope

Purpose: test whether turning on the full evaluation-time ConCM module set improves over the locked main skeleton:

Freeze + Task-block Calibration + NC-ConCM.

This run is diagnostic-only and evaluation-only. It did not train, did not run phase2, did not run full phases, did not run a seed sweep, did not run FDM v2, did not modify `train.py`, did not change training logic, and did not save large weights.

Current main method skeleton remains C locked NC-ConCM:

- alpha rule: `balanced_hmean_exemplar`
- lambda: fixed `0.30`
- reliability gating: disabled
- matching: disabled
- no grid search
- no oracle selection

F/G rows below are full-module diagnostics only. They are not promoted method rows unless they clearly improve over locked C on both seeds while satisfying the current-class guardrail.

## Tool Change

`tools/run_hyperkvasir_concm_min_gate.py` was extended only as an evaluation/diagnostic tool:

- `--locked-json` loads the prior rule-lock skeleton row.
- `--enable-quality-gated-selector` enables a non-oracle rule selector.
- `--groups A,B,C,D,E,F,G` emits explicit diagnostic groups.
- `per_class_relation_drift.csv`, `selected_per_class_metrics.csv`, `selected_old_to_current_confusion.csv`, and `selected_current_to_old_confusion.csv` are written for matching diagnostics.

Static checks passed locally and remotely:

- `py_compile` passed.
- No `torch.save`.
- No `loss.backward` or `.backward(`.
- No `optimizer.step`.
- No `model.train(`.

## Artifacts

Seed0:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt`
- Phase1 freeze: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt`
- Exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`
- Calibration JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json`
- Diagnostics JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json`
- Locked JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json`

Seed1:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt`
- Phase1 freeze: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt`
- Exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv`
- Calibration JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json`
- Diagnostics JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json`
- Locked JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s1/final_results.json`

Outputs:

- Seed0: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_full_concm_diag_20260621_s0`
- Seed1: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_full_concm_diag_20260621_s1`

Each output directory contains `final_results.json`, `summary.csv`, `alpha_lambda_landscape.csv`, `selector_stats.csv`, `selector_decisions.csv`, reliability CSVs, prototype relation CSVs, selected per-class metrics, selected block confusion CSVs, and logs.

## Commands

Seed0:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_full_concm_diag_20260621_s0
timeout 900 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0/final_results.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar,constrained_anchor,constrained_exemplar \
  --lambda-grid 0.05,0.10,0.15,0.20,0.30 \
  --matching-grid 0.05,0.10,0.15 \
  --groups A,B,C,D,E,F,G \
  --enable-reliability-gate \
  --enable-matching-diagnostic \
  --enable-matching-correction \
  --enable-quality-gated-selector
```

Seed1:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_full_concm_diag_20260621_s1
timeout 900 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json \
  --locked-json $BASE/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s1/final_results.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 1 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar,constrained_anchor,constrained_exemplar \
  --lambda-grid 0.05,0.10,0.15,0.20,0.30 \
  --matching-grid 0.05,0.10,0.15 \
  --groups A,B,C,D,E,F,G \
  --enable-reliability-gate \
  --enable-matching-diagnostic \
  --enable-matching-correction \
  --enable-quality-gated-selector
```

## Seed0 Results

| group | alpha_rule | alpha | lambda | match | reliability | selector | AccT | bal acc | macro-F1 | old | current | old->current | current->old |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A raw_freeze | none | 1.00 | 0.00 | 0.00 | off | none | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| B task_block_only | balanced_hmean_exemplar | 0.35 | 0.00 | 0.00 | off | none | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| B task_block_only | constrained_anchor | 0.25 | 0.00 | 0.00 | off | none | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| B task_block_only | constrained_exemplar | 0.25 | 0.00 | 0.00 | off | none | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| C locked_NCConCM | balanced_hmean_exemplar | 0.35 | 0.30 | 0.00 | off | fixed_balanced | 77.5273 | 45.4020 | 43.5763 | 75.5594 | 88.2629 | 14.0275 | 9.8592 |
| D + reliability | balanced_hmean_exemplar | 0.35 | 0.30 | 0.00 | on | fixed_balanced | 77.0182 | 43.4368 | 42.1105 | 74.6988 | 89.6714 | 15.6627 | 8.4507 |
| E + matching | balanced_hmean_exemplar | 0.35 | 0.30 | 0.05 | off | fixed_balanced | 77.6000 | 45.4350 | 43.6023 | 75.6454 | 88.2629 | 13.9415 | 9.8592 |
| F full_all_fixed_balanced | balanced_hmean_exemplar | 0.35 | 0.30 | 0.15 | on | fixed_balanced | 77.0909 | 43.4698 | 42.1361 | 74.7849 | 89.6714 | 15.5766 | 8.4507 |
| G full_all_selector | constrained_anchor | 0.25 | 0.05 | 0.05 | on | quality_gated | 79.8545 | 47.2416 | 43.9857 | 80.3787 | 76.9953 | 5.1635 | 21.5962 |

## Seed1 Results

| group | alpha_rule | alpha | lambda | match | reliability | selector | AccT | bal acc | macro-F1 | old | current | old->current | current->old |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A raw_freeze | none | 1.00 | 0.00 | 0.00 | off | none | 24.6856 | 12.3413 | 5.1979 | 0.0000 | 92.5558 | 100.0000 | 0.0000 |
| B task_block_only | balanced_hmean_exemplar | 0.20 | 0.00 | 0.00 | off | none | 53.5407 | 32.8560 | 24.1794 | 40.3430 | 89.8263 | 36.0108 | 3.4739 |
| B task_block_only | constrained_anchor | 0.15 | 0.00 | 0.00 | off | none | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| B task_block_only | constrained_exemplar | 0.15 | 0.00 | 0.00 | off | none | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| C locked_NCConCM | balanced_hmean_exemplar | 0.20 | 0.30 | 0.00 | off | fixed_balanced | 54.2687 | 33.6778 | 26.1992 | 41.6968 | 88.8337 | 32.6715 | 4.2184 |
| D + reliability | balanced_hmean_exemplar | 0.20 | 0.05 | 0.00 | on | fixed_balanced | 53.6731 | 32.9673 | 24.3736 | 40.5235 | 89.8263 | 35.8303 | 3.4739 |
| E + matching | balanced_hmean_exemplar | 0.20 | 0.10 | 0.15 | off | fixed_balanced | 55.4600 | 34.0969 | 26.3978 | 43.2310 | 89.0819 | 31.5884 | 4.2184 |
| F full_all_fixed_balanced | balanced_hmean_exemplar | 0.20 | 0.15 | 0.15 | on | fixed_balanced | 54.5334 | 33.3919 | 25.0444 | 41.9675 | 89.0819 | 33.4838 | 4.2184 |
| G full_all_selector | constrained_anchor | 0.15 | 0.05 | 0.15 | on | quality_gated | 59.6294 | 36.1076 | 28.3994 | 50.3610 | 85.1117 | 20.2166 | 8.9330 |

## Delta Vs Locked C

| seed | group | Delta AccT | Delta old | Delta current | Delta old->current | Delta current->old |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seed0 | D + reliability | -0.5091 | -0.8606 | +1.4085 | +1.6351 | -1.4085 |
| seed0 | E + matching | +0.0727 | +0.0861 | +0.0000 | -0.0861 | +0.0000 |
| seed0 | F full_all_fixed_balanced | -0.4364 | -0.7745 | +1.4085 | +1.5491 | -1.4085 |
| seed0 | G full_all_selector | +2.3273 | +4.8193 | -11.2676 | -8.8640 | +11.7371 |
| seed1 | D + reliability | -0.5956 | -1.1733 | +0.9926 | +3.1588 | -0.7444 |
| seed1 | E + matching | +1.1913 | +1.5343 | +0.2481 | -1.0830 | +0.0000 |
| seed1 | F full_all_fixed_balanced | +0.2647 | +0.2708 | +0.2481 | +0.8123 | +0.0000 |
| seed1 | G full_all_selector | +5.3607 | +8.6643 | -3.7221 | -12.4549 | +4.7146 |

## Module Contribution

Reliability gating is not a stable improvement. It improves current accuracy and reduces current->old on both seeds, but it lowers AccT and old accuracy on both seeds and worsens old->current. It should remain a robustness diagnostic.

Matching correction is mixed. E gives a tiny seed0 gain (`+0.0727` AccT, `+0.0861` old) and a larger seed1 diagnostic gain (`+1.1913` AccT, `+1.5343` old), while preserving current guardrails. The effect is not consistent enough across seeds to enter the main method.

Full modules with fixed balanced rule do not beat locked C consistently. F is worse than C on seed0 and only slightly better on seed1, so F should not replace locked C.

The quality-gated selector is not ready. It selects `constrained_anchor` on both seeds based on non-test proxies. On seed1 this is strong and passes the guardrail. On seed0 it fails the current guardrail badly: current `76.9953 < 85` and current->old `21.5962 > 15`. Therefore G cannot replace locked C.

Constrained rules should stay competing diagnostics. They are useful for showing the accuracy/old-class ceiling, especially on seed1, but seed0 violates current-class preservation.

## Matching Diagnostic

Top prototype relation drifts:

| seed | pair | drift_abs | phase0_cosine | phase1_cosine |
| --- | --- | ---: | ---: | ---: |
| seed0 | 20-19 | 0.2239 | 0.9949 | 0.7709 |
| seed0 | 20-2 | 0.2196 | 0.7324 | 0.5128 |
| seed0 | 20-10 | 0.2134 | 0.9925 | 0.7791 |
| seed1 | 16-22 | 0.3773 | 0.4852 | 0.8625 |
| seed1 | 16-7 | 0.3627 | 0.5123 | 0.8750 |
| seed1 | 15-17 | 0.2818 | 0.6865 | 0.4047 |

Selected high-drift class recalls:

| seed | class | drift_norm | C recall | E recall | F recall | G recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | 19 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| seed0 | 20 | 0.9916 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| seed0 | 21 | 0.5951 | 29.6296 | 29.6296 | 11.1111 | 33.3333 |
| seed0 | 4 | 0.5099 | 89.6040 | 90.0990 | 90.0990 | 96.0396 |
| seed1 | 15 | 1.0000 | 0.6536 | 0.6536 | 0.0000 | 0.6536 |
| seed1 | 16 | 0.9505 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| seed1 | 22 | 0.8255 | 80.2139 | 88.7701 | 91.9786 | 95.1872 |
| seed1 | 7 | 0.7785 | 13.5802 | 8.6420 | 6.1728 | 2.4691 |

Relation drift is real and often overlaps weak old-class regions, but matching correction does not reliably repair those classes. Seed0's highest drift classes 19 and 20 remain at zero recall under C/E/F/G. Seed1 class 22 improves, but class 7 degrades. Matching remains diagnostic-only.

## Oracle Leakage Check

- The G selector chooses the alpha rule from non-test proxy statistics only: reliability, old-to-current proxy, current retention proxy, and rule availability.
- G does not use test AccT, test old/current accuracy, or test old/current confusion to select the rule.
- Lambda and matching grids are diagnostic. The selected D/E/F/G rows are chosen from evaluation grids for analysis, so they are not clean non-oracle method rows.
- No F/G row should be used to replace locked C unless a future non-test selector for lambda and matching strength is defined and validated.

## Decision

Full modules do not satisfy the replacement rule:

1. F/G do not both improve over C on both seeds.
2. G fails seed0 current accuracy and current->old guardrails.
3. F is worse than C on seed0.
4. Matching/reliability benefits are not stable across seeds.
5. Lambda/matching strength selection remains diagnostic, not a locked non-oracle method policy.

Keep the current main method skeleton:

Freeze + Task-block Calibration + NC-ConCM.

Position full modules as follows:

- Reliability gating: robustness diagnostic.
- Matching: diagnostic-only / motivation.
- Constrained rules: competing diagnostic.
- Quality-gated selector: future improvement, not current main method.

## Next Step

Do not move to phase2/full phases/seed sweep/FDM v2 based on full-module diagnostics. The next clean step is to keep locked C as the method skeleton and, if needed, design a truly non-oracle selector for lambda/matching strength before rerunning a bounded diagnostic.

## Cleanup

After both runs, no HyperKvasir training/evaluation process was running. `nvidia-smi` showed both V100 GPUs at 1 MiB memory, 0% utilization, and no active GPU process.
