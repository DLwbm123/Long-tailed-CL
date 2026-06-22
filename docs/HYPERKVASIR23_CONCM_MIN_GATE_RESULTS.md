# HyperKvasir23 ConCM-min Gate Results

## 2026-06-21 NC-ConCM Rule-Lock Follow-Up

Follow-up results are in `docs/HYPERKVASIR23_NCCONCM_RULE_LOCK_RESULTS.md`.

That run locks the current main-method skeleton to `balanced_hmean_exemplar` + NC-ConCM C with fixed `lambda=0.30`. It is evaluation-only, runs only B/C, disables lambda grid search, disables reliability gating, disables matching diagnostic/correction, and does not use test-oracle selection. The locked B/C rows exactly reproduce the corresponding rows from this ConCM-min gate for seed0 and seed1.

## Scope

This is a bounded full-method gate for HyperKvasir23 medical CIL under the freeze + task-block calibration setting. It is truly evaluation-only: no training was started, no phase2 or full phases were run, no multi-seed sweep was run, no FDM v2 was run, no `train.py` or training logic was changed, and no large weights were saved.

The only new tool used here is `tools/run_hyperkvasir_concm_min_gate.py`. It loads existing phase0/phase1 checkpoints, old exemplars, calibration JSON, and diagnostic JSON, then writes CSV/JSON diagnostics and evaluation-time logit-correction results.

## Inputs

Seed0:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt`
- Phase1 freeze checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt`
- Old exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`
- Task-block calibration: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json`
- Task-block diagnostics: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json`

Seed1:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt`
- Phase1 freeze checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt`
- Old exemplars: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv`
- Task-block calibration: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json`
- Task-block diagnostics: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json`

## Modules

- M1 confusion statistics: writes `confusion_matrix.csv`, `old_to_current_confusion.csv`, `current_to_old_confusion.csv`, and `per_class_confusion_strength.csv`.
- M2 task-block baseline: reuses `balanced_hmean_exemplar` and `constrained_anchor` from existing calibration JSON.
- M3 NC-ConCM: applies conservative evaluation-time class-wise logit correction with lambda grid `0.00,0.05,0.10,0.15,0.20,0.30`.
- M4 reliability gating: computes per-class reliability and optionally gates correction strength.
- M5 prototype relation matching: writes phase0/phase1 relation matrices, drift matrices, top drift pairs, and an optional conservative matching correction with grid `0.00,0.05,0.10,0.15`.

C/D/E rows below are diagnostic grid selections using the reported guardrail-first rule from the evaluation output: select rows satisfying `current_acc >= 85` and `current_to_old_rate < 15`, then choose higher old accuracy / AccT. This is diagnostic evidence, not a finalized non-oracle method-selection policy.

## Commands

Seed0:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_concm_min_gate_20260621_s0
timeout 900 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s0/diagnostics.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 0 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar,constrained_anchor,constrained_exemplar \
  --enable-reliability-gate \
  --enable-matching-diagnostic \
  --enable-matching-correction
```

Seed1:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
OUT=hyperkvasir23_concm_min_gate_20260621_s1
timeout 900 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_concm_min_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv \
  --calibration-json $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json \
  --diagnostics-json $BASE/runs/hyperkvasir23_task_block_diag_20260621_s1/diagnostics.json \
  --output-dir $BASE/runs/$OUT \
  --device cuda --seed 1 --batch-size 64 --num-workers 4 \
  --alpha-rule balanced_hmean_exemplar,constrained_anchor,constrained_exemplar \
  --enable-reliability-gate \
  --enable-matching-diagnostic \
  --enable-matching-correction
```

## Outputs

- Seed0 output: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_min_gate_20260621_s0`
- Seed1 output: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_concm_min_gate_20260621_s1`

Each output directory contains `final_results.json`, `summary.csv`, `selected_rows.csv`, `alpha_lambda_landscape.csv`, confusion CSVs, reliability CSV, prototype relation CSVs, `top_drift_pairs.csv`, `per_class_metrics.csv`, and `run_summary.md`.

## Seed0 Results

| rule | group | alpha | lambda | matching | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | A_raw_freeze | 1.00 | 0.00 | 0.00 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| balanced_hmean_exemplar | B_task_block_only | 0.35 | 0.00 | 0.00 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| balanced_hmean_exemplar | C_task_block_nc_concm | 0.35 | 0.30 | 0.00 | 77.5273 | 45.4020 | 43.5763 | 75.5594 | 88.2629 | 14.0275 | 9.8592 |
| balanced_hmean_exemplar | D_task_block_nc_concm_reliability | 0.35 | 0.30 | 0.00 | 77.0182 | 43.4368 | 42.1105 | 74.6988 | 89.6714 | 15.6627 | 8.4507 |
| balanced_hmean_exemplar | E_task_block_nc_concm_reliability_matching | 0.35 | 0.30 | 0.15 | 77.0909 | 43.4698 | 42.1361 | 74.7849 | 89.6714 | 15.5766 | 8.4507 |
| constrained_anchor | B_task_block_only | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_anchor | C_task_block_nc_concm | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_anchor | D_task_block_nc_concm_reliability | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_anchor | E_task_block_nc_concm_reliability_matching | 0.25 | 0.00 | 0.10 | 79.8545 | 47.2416 | 43.9849 | 80.3787 | 76.9953 | 5.0775 | 21.5962 |
| constrained_exemplar | B_task_block_only | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_exemplar | C_task_block_nc_concm | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_exemplar | D_task_block_nc_concm_reliability | 0.25 | 0.00 | 0.00 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_exemplar | E_task_block_nc_concm_reliability_matching | 0.25 | 0.00 | 0.10 | 79.8545 | 47.2416 | 43.9849 | 80.3787 | 76.9953 | 5.0775 | 21.5962 |

Seed0 balanced gate interpretation:

- B already satisfies current guardrail: current `91.0798`, current->old `7.0423`.
- C improves B by `+1.0182` AccT, `+3.0337` balanced accuracy, `+2.4988` macro-F1, `+1.7212` old accuracy, and reduces old->current from `16.7814` to `14.0275`, while current remains above 85 and current->old remains below 15.
- D reduces current->old compared with C, but sacrifices old recovery and AccT. It is useful as a safety diagnostic, not the best seed0 row.
- E adds only a very small gain over D and remains below C on AccT, balanced accuracy, macro-F1, and old accuracy.

## Seed1 Results

| rule | group | alpha | lambda | matching | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current | current_to_old |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | A_raw_freeze | 1.00 | 0.00 | 0.00 | 24.6856 | 12.3413 | 5.1979 | 0.0000 | 92.5558 | 100.0000 | 0.0000 |
| balanced_hmean_exemplar | B_task_block_only | 0.20 | 0.00 | 0.00 | 53.5407 | 32.8560 | 24.1794 | 40.3430 | 89.8263 | 36.0108 | 3.4739 |
| balanced_hmean_exemplar | C_task_block_nc_concm | 0.20 | 0.30 | 0.00 | 54.2687 | 33.6778 | 26.1992 | 41.6968 | 88.8337 | 32.6715 | 4.2184 |
| balanced_hmean_exemplar | D_task_block_nc_concm_reliability | 0.20 | 0.05 | 0.00 | 53.6731 | 32.9673 | 24.3736 | 40.5235 | 89.8263 | 35.8303 | 3.4739 |
| balanced_hmean_exemplar | E_task_block_nc_concm_reliability_matching | 0.20 | 0.15 | 0.15 | 54.5334 | 33.3919 | 25.0444 | 41.9675 | 89.0819 | 33.4838 | 4.2184 |
| constrained_anchor | B_task_block_only | 0.15 | 0.00 | 0.00 | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| constrained_anchor | C_task_block_nc_concm | 0.15 | 0.15 | 0.00 | 59.5632 | 36.1214 | 28.6042 | 50.0903 | 85.6079 | 20.1264 | 8.4367 |
| constrained_anchor | D_task_block_nc_concm_reliability | 0.15 | 0.05 | 0.00 | 59.6294 | 36.1438 | 28.5713 | 50.0000 | 86.1042 | 20.2166 | 7.6923 |
| constrained_anchor | E_task_block_nc_concm_reliability_matching | 0.15 | 0.00 | 0.15 | 59.6294 | 36.1076 | 28.3994 | 50.3610 | 85.1117 | 20.2166 | 8.9330 |
| constrained_exemplar | B_task_block_only | 0.15 | 0.00 | 0.00 | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| constrained_exemplar | C_task_block_nc_concm | 0.15 | 0.15 | 0.00 | 59.5632 | 36.1214 | 28.6042 | 50.0903 | 85.6079 | 20.1264 | 8.4367 |
| constrained_exemplar | D_task_block_nc_concm_reliability | 0.15 | 0.05 | 0.00 | 59.6294 | 36.1438 | 28.5713 | 50.0000 | 86.1042 | 20.2166 | 7.6923 |
| constrained_exemplar | E_task_block_nc_concm_reliability_matching | 0.15 | 0.00 | 0.15 | 59.6294 | 36.1076 | 28.3994 | 50.3610 | 85.1117 | 20.2166 | 8.9330 |

Seed1 artifact-quality-gated interpretation:

- Seed1 remains weak at the artifact level: raw-freeze old accuracy is `0.0000` and old->current is `100.0000`.
- For `balanced_hmean_exemplar`, C improves B by `+0.7280` AccT, `+0.8217` balanced accuracy, `+2.0197` macro-F1, `+1.3538` old accuracy, and reduces old->current from `36.0108` to `32.6715`. Current remains `88.8337` and current->old remains `4.2184`, so the guardrail holds.
- Reliability gating is conservative on seed1: D preserves current and current->old but nearly removes the C gain.
- Matching correction gives the highest balanced-rule AccT, but the gain is small and old->current is worse than C. Treat it as diagnostic evidence only.

## Prototype Relation Matching Diagnostic

Top drift pairs:

| seed | pair | drift_abs | phase0_cosine | phase1_cosine |
| --- | --- | ---: | ---: | ---: |
| seed0 | 20-19 | 0.2239 | 0.9949 | 0.7709 |
| seed0 | 20-2 | 0.2196 | 0.7324 | 0.5128 |
| seed0 | 20-10 | 0.2134 | 0.9925 | 0.7791 |
| seed1 | 16-22 | 0.3773 | 0.4852 | 0.8625 |
| seed1 | 16-7 | 0.3627 | 0.5123 | 0.8750 |
| seed1 | 15-17 | 0.2818 | 0.6865 | 0.4047 |

The diagnostic finds substantial old-class prototype relation drift in both seeds, so matching is a plausible signal. However, the conservative matching correction is not yet compelling as a method component: it adds little on seed0 and gives only a small diagnostic gain on seed1 while not consistently improving old->current over C.

## Acceptance Gate

Guardrail used here:

- current accuracy must stay at least `85`.
- current->old must stay below `15`.
- old-class recovery should improve over B without creating a current-class leak.

Balanced-rule outcome:

- Seed0 C passes: current `88.2629`, current->old `9.8592`, AccT and old accuracy improve over B.
- Seed1 C passes: current `88.8337`, current->old `4.2184`, AccT and old accuracy improve over B.
- Reliability gating is useful as a safety diagnostic but not the best-performing selected rung.
- Matching correction remains diagnostic-only and should not be promoted yet.

Constrained-rule outcome:

- Seed0 constrained_anchor and constrained_exemplar are identical and remain accuracy-heavy but biased toward old classes: current `76.9953` and current->old `21.5962` fail the guardrail.
- Seed1 constrained_anchor and constrained_exemplar are identical and pass the guardrail, but gains from C/D/E are negligible. They remain accuracy-heavy diagnostics, not the first-choice baseline unless future runs show consistent current-guardrail compliance.

## Conclusion

`balanced_hmean_exemplar` remains the preferred freeze + task-block calibration baseline candidate because it keeps current accuracy high and current->old low on both seed0 and seed1. It should still be described as a baseline candidate, not a fully stable baseline, because seed1 old accuracy remains weak under the 5-epoch artifact.

ConCM-min is effective as a bounded diagnostic on top of that baseline: NC-ConCM C improves old recovery and AccT on both seed0 and seed1 while satisfying the current-class guardrail. The effect is modest but real enough to justify keeping NC-ConCM as the next evaluation-only rung.

Reliability gating should remain a safety option. Prototype relation matching should remain diagnostic-only; it is not ready to enter the main method or training loss.

Next minimal step: freeze a non-oracle lambda-selection proxy before any larger method claim. Do not move to phase2/full phases/FDM v2 solely from this gate.

## Cleanup

After the seed0/seed1 evaluation-only runs, no HyperKvasir training/evaluation process was left running. `nvidia-smi` showed both V100 GPUs at 1 MiB memory, 0% utilization, and no active GPU process.
