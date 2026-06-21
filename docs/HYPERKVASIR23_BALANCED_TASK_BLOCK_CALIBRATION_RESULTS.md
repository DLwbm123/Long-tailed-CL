# HyperKvasir23 Balanced Task-Block Calibration Results

## Scope

Evaluation-only balanced non-oracle calibration. Alpha selection uses phase0 old anchors, 1-shot old exemplars, and phase1 current-task training samples only. Test accuracy is used only after alpha is selected; oracle sweep is diagnostic only.

## 2026-06-21 Seed1 Minimal Bounded Validation

Goal: answer whether the seed0 `balanced_hmean_exemplar` result is clearly not a seed0 artifact.

This run only generated the minimal seed1 artifacts needed for evaluation-only calibration: phase0 anchors, a freeze-baseline phase1 checkpoint, and `old_exemplars.csv`. It reused the seed0 5-epoch fair gate settings, changed only `--seed 1` and output paths, and used `--branches freeze_baseline` to avoid running the unnecessary Med-FDM branch. No seed sweep, full phases, phase2/phase3, FDM v2, test-oracle alpha selection change, or hyperparameter search was run. No training logic was changed; the tool only received a narrow branch-selection switch so this bounded run would not save unnecessary branch weights.

### Seed1 Minimal Artifact Generation

Training command:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
RUN_TAG=hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep
timeout 1200 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_med_fdm_gate.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --output-dir $BASE/runs/$RUN_TAG \
  --ckpt-dir $BASE/checkpoints/$RUN_TAG \
  --cache-dir $BASE/cache \
  --device cuda \
  --seed 1 \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4 \
  --lr 0.01 \
  --exemplars-per-class 1 \
  --fdm-lambda 0.01 \
  --branches freeze_baseline
```

Generated seed1 artifacts:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/phase0_with_anchors.pt`
- Phase1 freeze checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/freeze_baseline_phase1.pt`
- Exemplar CSV: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/old_exemplars.csv`
- Training log: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep/train_seed1_minimal.log`

Artifact checks:

- `phase0_with_anchors.pt`, `freeze_baseline_phase1.pt`, and `old_exemplars.csv` exist.
- `med_fdm_phase1.pt` was not generated.
- No phase2 checkpoint was generated.
- Seed1 phase1 branch metadata has `use_fdm=false` and `fdm_batches=0`.
- Logs had no traceback/runtime error.

### Seed1 Evaluation-Only Calibration

Calibration command:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
TRAIN_TAG=hyperkvasir23_med_fdm_gate_fair_20260621_s1_5ep
OUT_TAG=hyperkvasir23_balanced_task_block_calibration_20260621_s1
timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_balanced_task_block_calibration.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt $BASE/checkpoints/$TRAIN_TAG/phase0_with_anchors.pt \
  --phase1-ckpt $BASE/checkpoints/$TRAIN_TAG/freeze_baseline_phase1.pt \
  --exemplar-csv $BASE/runs/$TRAIN_TAG/old_exemplars.csv \
  --output-dir $BASE/runs/$OUT_TAG \
  --device cuda \
  --seed 1 \
  --batch-size 64 \
  --num-workers 4
```

Seed1 output:

- Output dir: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1`
- Final JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/final_results.json`
- Calibration log: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s1/calibration_seed1.log`

### Seed1 Result Table

| rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_hmean_anchor | 0.1000 | 55.2614 | 34.4556 | 27.3971 | 50.5415 | 68.2382 | 15.6137 | 27.0471 |
| balanced_hmean_exemplar | 0.2000 | 53.5407 | 32.8560 | 24.1794 | 40.3430 | 89.8263 | 36.0108 | 3.4739 |
| constrained_anchor | 0.1500 | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| constrained_exemplar | 0.1500 | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| margin_balance | 0.1500 | 59.5632 | 36.1148 | 28.5382 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |

### Seed0 vs Seed1

| seed | rule | alpha | AccT | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | balanced_hmean_exemplar | 0.3500 | 76.5091 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| seed1 | balanced_hmean_exemplar | 0.2000 | 53.5407 | 40.3430 | 89.8263 | 36.0108 | 3.4739 |
| seed0 | constrained_anchor | 0.2500 | 79.7818 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| seed1 | constrained_anchor | 0.1500 | 59.5632 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |
| seed0 | margin_balance | 1.0000 | 25.4545 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |
| seed1 | margin_balance | 0.1500 | 59.5632 | 49.9097 | 86.1042 | 20.3069 | 7.6923 |

Guardrail check:

- `balanced_hmean_exemplar` seed1 current accuracy passes: `89.8263 >= 85`.
- `balanced_hmean_exemplar` seed1 current->old passes: `3.4739 < 15`.
- `balanced_hmean_exemplar` seed1 alpha is `0.2000`, below the expected `0.25-0.40` seed0-like range.
- Old accuracy is recovered from the uncalibrated all-seen collapse, but the recovered level is weak (`40.3430`) and far below seed0 (`73.8382`).
- The seed1 result does not look like a pure current-class sacrifice, because current accuracy remains high, but it leaves substantial old->current confusion (`36.0108`).

Conclusion for `balanced_hmean_exemplar`: **partially supported but still weak**. Seed1 confirms that the rule can preserve current accuracy without large current->old leakage, so seed0 was not purely a current-class accident. However, seed1 old-class recovery is much weaker and the selected alpha shifts to `0.2000`; stability is not strong enough to call it a stable baseline candidate yet.

`constrained_anchor` / `constrained_exemplar` remain accuracy-heavy diagnostics. On seed1 they have the best non-oracle AccT (`59.5632`) and higher old accuracy than `balanced_hmean_exemplar`, while still passing current accuracy (`86.1042`) and current->old (`7.6923`) guardrails. They also reduce current accuracy relative to `balanced_hmean_exemplar` and still leave high old->current confusion (`20.3069`), so they should stay diagnostic rather than become the preferred baseline without a clearer stability argument.

`margin_balance` is not stable as an independent rule. It failed on seed0 and collapsed to the same alpha/result as constrained rules on seed1, so it should remain a failed/unstable route.

Cleanup status after seed1 validation: no HyperKvasir training/evaluation process was left running. `nvidia-smi` showed GPU0 and GPU1 each using 1 MiB with 0% utilization and no active GPU process.

## 2026-06-21 Bounded Validation Follow-Up

Goal: check whether `balanced_hmean_exemplar` is more than a seed0 artifact without training, without full phases, without a multi-seed sweep, and without FDM v2.

Outcome at that point: no new evaluation run was launched. The remote `/dev/shm/wangbomin/LongTailedCL` inventory only contained HyperKvasir23 seed0 phase0/phase1 artifacts for this calibration setup. No seed1/seed2 phase0+phase1 freeze checkpoint pair with exemplar CSV was found, and no HyperKvasir23 phase2 checkpoint was found. Therefore that follow-up took the bounded path C: document the missing inputs and the next minimal command instead of starting training. The seed1 minimal bounded validation above was run afterward once minimal seed1 generation was explicitly allowed.

This follow-up was read-only on remote artifacts. It did not start training, did not run full HyperKvasir phases, did not run a seed sweep, did not run FDM v2, did not modify training logic, and did not save large weights.

### Remote Candidate Inventory

Usable seed0 calibration inputs:

- Phase0 anchors: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt`
- Phase1 freeze checkpoint: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt`
- Exemplar CSV: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv`
- Existing seed0 balanced calibration output: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json`

Other HyperKvasir23 artifacts found:

- Non-fair duplicate seed0 gate: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_20260621_s0_5ep`
- Seed0 module matrix checkpoints: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_module_matrix_20260620-111904_s0_5ep`
- Seed0 freeze diagnosis checkpoints: `/dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_phase1_freeze_backbone_bn_diag_s0_5ep`

Missing bounded-validation inputs at that time:

- No HyperKvasir23 `s1`, `s2`, `seed1`, `seed2`, `1994`, or `1995` directory/file was found under `/dev/shm/wangbomin/LongTailedCL`.
- No HyperKvasir23 phase2 checkpoint was found under `/dev/shm/wangbomin/LongTailedCL`.
- No new-seed `old_exemplars.csv` or equivalent exemplar CSV was found.

Selection logic:

- Path A was preferred but unavailable because no second seed had phase0 anchors, phase1 freeze checkpoint, and exemplar CSV.
- Path B was unavailable because no HyperKvasir23 phase2 checkpoint was present. The current script is phase1-oriented; it was not extended because there was no phase2 input to evaluate.
- Path C was selected: no training and no new evaluation run; keep seed0 result as the only completed evaluation-only calibration and record the missing files.

Next minimal command after a second seed checkpoint triplet exists:

```bash
BASE=/dev/shm/wangbomin/LongTailedCL
timeout 600 /opt/miniconda3/envs/torchgpu/bin/python \
  $BASE/code/tools/run_hyperkvasir_balanced_task_block_calibration.py \
  --data-root $BASE/data/hyper-kvasir23 \
  --phase0-ckpt <FOUND_SEED1_OR_SEED2_PHASE0_WITH_ANCHORS_PT> \
  --phase1-ckpt <FOUND_SEED1_OR_SEED2_FREEZE_BASELINE_PHASE1_PT> \
  --exemplar-csv <FOUND_SEED1_OR_SEED2_OLD_EXEMPLARS_CSV> \
  --output-dir $BASE/runs/hyperkvasir23_balanced_task_block_calibration_20260621_<seed_tag> \
  --device cuda \
  --seed <seed> \
  --batch-size 64 \
  --num-workers 4
```

### Follow-Up Result Table

No new row was produced in this follow-up because the required non-seed0 or phase2 inputs were absent. The existing seed0 evaluation-only table remains the only completed bounded result:

| validation | rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | balanced_hmean_anchor | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| seed0 | balanced_hmean_exemplar | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| seed0 | constrained_anchor | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| seed0 | constrained_exemplar | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| seed0 | margin_balance | 1.0000 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |

Guardrail check on the existing seed0 row:

- `balanced_hmean_exemplar` satisfies current accuracy >= 85 (`91.0798`) and current->old < 15 (`7.0423`).
- `balanced_hmean_exemplar` recovers old accuracy to `73.8382`, but this has not yet been replicated beyond seed0.
- `constrained_anchor` / `constrained_exemplar` remain accuracy-heavy diagnostics: higher AccT and old accuracy than `balanced_hmean_exemplar`, but current accuracy is below the current-preservation guardrail (`76.9953`) and current->old is high (`21.5962`).
- `margin_balance` remains failed/unstable under all-seen evaluation: AccT `25.4545`, old accuracy `12.3924`, and old->current `87.6076`.

Follow-up conclusion: `balanced_hmean_exemplar` remains the best seed0 baseline candidate for freeze+task-block calibration, but stability is inconclusive because no non-seed0 or phase2 evaluation-only artifact exists yet. It should not be promoted as a stable multi-seed baseline until one additional bounded evaluation-only run is available.

Cleanup status after the follow-up inventory: no HyperKvasir training/evaluation process was left running. `nvidia-smi` showed GPU0 and GPU1 each using 1 MiB with 0% utilization and no active GPU process.

## Exact Command

```bash
/opt/miniconda3/envs/torchgpu/bin/python /dev/shm/wangbomin/LongTailedCL/code/tools/run_hyperkvasir_balanced_task_block_calibration.py --data-root /dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23 --phase0-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/phase0_with_anchors.pt --phase1-ckpt /dev/shm/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/freeze_baseline_phase1.pt --exemplar-csv /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_med_fdm_gate_fair_20260621_s0_5ep/old_exemplars.csv --output-dir /dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0 --device cuda --seed 0 --batch-size 64 --num-workers 4
```

## Calibration-Set Alpha Selection Table

| alpha | old_anchor_retention | old_exemplar_retention | current_train_acc | old_anchor_margin | old_exemplar_old_vs_current_margin | current_train_current_vs_old_margin | hmean_anchor | hmean_exemplar | margin_balance_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 1.0000 | 0.4615 | 0.2035 | 4.7402 | 4.6588 | -0.6822 | 0.3382 | 0.2825 | 4.0580 |
| 0.1500 | 1.0000 | 0.4615 | 0.5118 | 4.1249 | 4.0330 | -0.2355 | 0.6770 | 0.4854 | 3.8895 |
| 0.2000 | 0.9231 | 0.4615 | 0.7235 | 3.5097 | 3.4071 | 0.2112 | 0.8112 | 0.5636 | 3.7209 |
| 0.2500 | 0.6923 | 0.4615 | 0.8329 | 2.8944 | 2.7812 | 0.6580 | 0.7561 | 0.5940 | 3.5524 |
| 0.3000 | 0.6154 | 0.4615 | 0.8882 | 2.2791 | 2.1554 | 1.1047 | 0.7271 | 0.6074 | 3.3838 |
| 0.3500 | 0.6154 | 0.4615 | 0.9188 | 1.6639 | 1.5295 | 1.5514 | 0.7371 | 0.6144 | 3.2153 |
| 0.4000 | 0.4615 | 0.3077 | 0.9424 | 1.0486 | 0.9036 | 1.9981 | 0.6196 | 0.4639 | 3.0467 |
| 0.4500 | 0.4615 | 0.3077 | 0.9529 | 0.4333 | 0.2777 | 2.4448 | 0.6219 | 0.4652 | 2.8782 |
| 0.5000 | 0.3846 | 0.2308 | 0.9576 | -0.1819 | -0.3481 | 2.8915 | 0.5488 | 0.3719 | 2.7096 |
| 0.6000 | 0.0769 | 0.0769 | 0.9635 | -1.4125 | -1.5999 | 3.7850 | 0.1425 | 0.1425 | 2.3725 |
| 0.7000 | 0.0769 | 0.0769 | 0.9659 | -2.6430 | -2.8516 | 4.6784 | 0.1425 | 0.1425 | 2.0354 |
| 0.8000 | 0.0769 | 0.0769 | 0.9659 | -3.8735 | -4.1034 | 5.5719 | 0.1425 | 0.1425 | 1.6983 |
| 0.9000 | 0.0769 | 0.0769 | 0.9671 | -5.1041 | -5.3551 | 6.4653 | 0.1425 | 0.1425 | 1.3612 |
| 1.0000 | 0.0769 | 0.0769 | 0.9671 | -6.3346 | -6.6068 | 7.3587 | 0.1425 | 0.1425 | 1.0241 |

## Test-Set Evaluation Of Selected Non-Oracle Rules

| rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_hmean_anchor | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| balanced_hmean_exemplar | 0.3500 | 76.5091 | 42.3683 | 41.0775 | 73.8382 | 91.0798 | 16.7814 | 7.0423 |
| constrained_anchor | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| constrained_exemplar | 0.2500 | 79.7818 | 47.2086 | 43.9581 | 80.2926 | 76.9953 | 5.3356 | 21.5962 |
| margin_balance | 1.0000 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |

## Oracle Sweep Reference

Diagnostic only; these alphas are not selected by test performance.

| rule | alpha | AccT | balanced_acc | macro_f1 | old_acc | current_acc | old_to_current_rate | current_to_old_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_alpha_0.1 | 0.1000 | 71.1273 | 44.8612 | 38.8068 | 81.1532 | 16.4319 | 0.2582 | 83.5681 |
| oracle_alpha_0.2 | 0.2000 | 78.4000 | 47.0247 | 43.2754 | 80.7229 | 65.7277 | 2.7539 | 32.8638 |
| oracle_alpha_0.3 | 0.3000 | 79.7818 | 46.5301 | 43.9963 | 79.0017 | 84.0376 | 8.5198 | 14.5540 |
| oracle_alpha_0.4 | 0.4000 | 64.7273 | 34.7083 | 34.0972 | 59.2943 | 94.3662 | 34.3373 | 2.8169 |
| oracle_alpha_0.5 | 0.5000 | 54.3273 | 28.5096 | 27.4030 | 46.7298 | 95.7746 | 49.2255 | 1.4085 |
| oracle_alpha_0.7 | 0.7000 | 30.1091 | 14.4560 | 11.4066 | 17.9002 | 96.7136 | 81.4114 | 0.0000 |
| oracle_alpha_0.9 | 0.9000 | 26.5455 | 12.0202 | 7.8699 | 13.6833 | 96.7136 | 86.3167 | 0.0000 |
| oracle_alpha_1 | 1.0000 | 25.4545 | 11.5152 | 7.5254 | 12.3924 | 96.7136 | 87.6076 | 0.0000 |

## Best Non-Oracle Rule

- Best rule: `constrained_anchor`
- Selected alpha: `0.2500`
- AccT: `79.7818`
- Old/current acc: `80.2926` / `76.9953`

## Comparison With Oracle Alpha 0.3

- Oracle alpha 0.3 AccT: `79.7818`
- Oracle alpha 0.3 old/current acc: `79.0017` / `84.0376`
- Gap best non-oracle vs oracle AccT: `0.0000`

## Recommendation

Use `balanced_hmean_exemplar` as a partially supported but still weak medical-CIL calibration baseline candidate. Seed0 keeps current-task accuracy high (`91.0798`) while recovering old accuracy (`73.8382`), and seed1 still preserves current accuracy (`89.8263`) with low current->old leakage (`3.4739`). The weakness is seed1 old accuracy (`40.3430`) and alpha shift (`0.2000`), so the stability claim is not strong enough to call it a stable baseline candidate yet.

Keep `constrained_anchor` as an accuracy-heavy diagnostic: it matches oracle alpha `0.3` on AccT (`79.7818`) and has slightly higher old accuracy (`80.2926`), but current accuracy falls below the preferred `85%` guardrail (`76.9953`).

Do not proceed to adapter-FDM v2 yet. First diagnose whether the seed1 weakness is seed sensitivity, weaker phase1 checkpoint quality, or exemplar-selection sensitivity.

Per-class recall is saved in `per_class_recall.csv`.

## Artifact Paths

- Output dir: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0`
- Final JSON: `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_balanced_task_block_calibration_20260621_s0/final_results.json`
