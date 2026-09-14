# HyperKvasir23 Locked NC-DSM Auxiliary Results

## Scope

This document reports the current `my-gpu` reconstructed HyperKvasir23 phase1 checkpoints for `locked_nc_dsm_aux` / `dsm_assisted_nc_concm`.

- Main branch: locked task-block + NC-ConCM correction.
- Auxiliary branch: DSM projector trained on frozen feature tensors.
- Trainable loss: `L_match_dsm + lambda_cont * L_cont_dsm`.
- `lambda_dsm` controls the normalized DSM residual at evaluation time.
- MPC-lite / Medical-MPC: not used.
- Phase2 / full phases: not run.

Because the repository does not expose a trainable locked `L_nc` optimizer path, `L_nc` is represented by the frozen locked NC-ConCM branch. Classifier and old NC anchors are not updated by this tool.

## Current my-gpu Reconstructed Checkpoint Conclusion

Under the current `my-gpu:/remote-home/wangbomin/LongTailedCL` 20260709 reconstructed checkpoints, `locked_nc_dsm_aux` is consistently better than Locked NC-ConCM.

Across seeds 1/2/3:

- `lambda_dsm=0.1` improves AccT by `+7.3700+/-0.8398` and HM by `+8.1345+/-1.2608` over Locked NC-ConCM.
- `lambda_dsm=0.2` improves AccT by `+8.6630+/-0.5897` and HM by `+9.6450+/-1.6978` over Locked NC-ConCM.
- `lambda_dsm=0.2` is the better multi-seed setting. Seed1 alone preferred `0.1`, but seeds 2/3 prefer `0.2`.
- Gains are not only from current recovery: `lambda_dsm=0.2` also improves old accuracy by `+6.4543+/-2.4171` on average.
- Pure DSM has the highest mean AccT, but it is not stable: seed3 has current collapse (`current_acc=36.3057`, `current_to_old=57.9618`). For a guardrailed medical old/current tradeoff, `locked_nc_dsm_aux lambda=0.2` is the safer recommendation.

## Per-Seed Table

| seed | method | lambda_dsm | AccT | old_acc | current_acc | HM_old_current | old_to_current | current_to_old | output_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Locked NC-ConCM | 0.0000 | 53.7392 | 47.2022 | 71.7122 | 56.9312 | 17.3285 | 24.8139 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s1/final_results.json` |
| 1 | Pure DSM-ConCM |  | 72.8657 | 67.2383 | 88.3375 | 76.3571 | 8.2130 | 10.6700 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s1/final_results.json` |
| 1 | locked_nc_dsm_aux | 0.1000 | 62.0781 | 52.7978 | 87.5931 | 65.8835 | 20.5776 | 11.4144 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s1/final_results.json` |
| 1 | locked_nc_dsm_aux | 0.2000 | 61.8134 | 51.2635 | 90.8189 | 65.5352 | 22.4729 | 8.1886 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s1/final_results.json` |
| 2 | Locked NC-ConCM | 0.0000 | 49.0669 | 44.8468 | 70.8134 | 54.9152 | 47.9109 | 24.8804 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s2/final_results.json` |
| 2 | Pure DSM-ConCM |  | 61.8974 | 55.3389 | 95.6938 | 70.1251 | 28.1337 | 0.4785 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s2/final_results.json` |
| 2 | locked_nc_dsm_aux | 0.1000 | 55.9876 | 49.5822 | 88.9952 | 63.6839 | 43.7326 | 8.1340 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s2/final_results.json` |
| 2 | locked_nc_dsm_aux | 0.2000 | 58.3204 | 51.2535 | 94.7368 | 66.5194 | 40.2971 | 2.3923 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s2/final_results.json` |
| 3 | Locked NC-ConCM | 0.0000 | 54.3307 | 50.4942 | 81.5287 | 62.3638 | 35.8491 | 0.0000 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s3/final_results.json` |
| 3 | Pure DSM-ConCM |  | 75.9055 | 81.4915 | 36.3057 | 50.2322 | 2.3360 | 57.9618 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_pure_dsm_concm_gate_20260709_s3/final_results.json` |
| 3 | locked_nc_dsm_aux | 0.1000 | 61.1811 | 57.6819 | 85.9873 | 69.0463 | 28.3917 | 0.0000 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s3/final_results.json` |
| 3 | locked_nc_dsm_aux | 0.2000 | 62.9921 | 59.3890 | 88.5350 | 71.0907 | 27.7628 | 0.0000 | `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_20260709_s3/final_results.json` |

Full per-seed CSV: `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_per_seed.csv`.

## Mean/Std Table

| method | lambda_dsm | n | AccT | old_acc | current_acc | HM_old_current | old_to_current | current_to_old |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Locked NC-ConCM | 0.0000 | 3 | 52.3789+/-2.8835 | 47.5144+/-2.8366 | 74.6847+/-5.9440 | 58.0701+/-3.8527 | 33.6961+/-15.4044 | 16.5648+/-14.3455 |
| Pure DSM-ConCM |  | 3 | 70.2228+/-7.3685 | 68.0229+/-13.0939 | 73.4457+/-32.3737 | 65.5715+/-13.6447 | 12.8942+/-13.5209 | 23.0367+/-30.6722 |
| locked_nc_dsm_aux | 0.0500 | 3 | 56.7885+/-3.4929 | 50.9302+/-3.6832 | 82.5994+/-1.3812 | 62.9784+/-3.2206 | 32.5512+/-13.8338 | 10.3087+/-8.9335 |
| locked_nc_dsm_aux | 0.1000 | 3 | 59.7489+/-3.2882 | 53.3540+/-4.0784 | 87.5252+/-1.5051 | 66.2046+/-2.6956 | 30.9006+/-11.7796 | 6.5161+/-5.8767 |
| locked_nc_dsm_aux | 0.2000 | 3 | 61.0420+/-2.4295 | 53.9687+/-4.6942 | 91.3636+/-3.1366 | 67.7151+/-2.9645 | 30.1776+/-9.1542 | 3.5270+/-4.2106 |

Full mean/std CSV: `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_mean_std.csv`.

## Delta Table

| seed | comparison | delta_AccT | delta_old_acc | delta_current_acc | delta_HM_old_current | delta_old_to_current | delta_current_to_old |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | locked_nc_dsm_aux lambda=0.1 vs Locked NC-ConCM | 8.3388 | 5.5957 | 15.8809 | 8.9523 | 3.2491 | -13.3995 |
| 1 | locked_nc_dsm_aux lambda=0.2 vs Locked NC-ConCM | 8.0741 | 4.0614 | 19.1067 | 8.6039 | 5.1444 | -16.6253 |
| 1 | Pure DSM-ConCM vs locked_nc_dsm_aux lambda=0.1 | 10.7876 | 14.4404 | 0.7444 | 10.4736 | -12.3646 | -0.7444 |
| 1 | Pure DSM-ConCM vs Locked NC-ConCM | 19.1264 | 20.0361 | 16.6253 | 19.4259 | -9.1155 | -14.1439 |
| 2 | locked_nc_dsm_aux lambda=0.1 vs Locked NC-ConCM | 6.9207 | 4.7354 | 18.1818 | 8.7687 | -4.1783 | -16.7464 |
| 2 | locked_nc_dsm_aux lambda=0.2 vs Locked NC-ConCM | 9.2535 | 6.4067 | 23.9234 | 11.6041 | -7.6137 | -22.4880 |
| 2 | Pure DSM-ConCM vs locked_nc_dsm_aux lambda=0.1 | 5.9098 | 5.7567 | 6.6986 | 6.4411 | -15.5989 | -7.6555 |
| 2 | Pure DSM-ConCM vs Locked NC-ConCM | 12.8305 | 10.4921 | 24.8804 | 15.2098 | -19.7772 | -24.4019 |
| 3 | locked_nc_dsm_aux lambda=0.1 vs Locked NC-ConCM | 6.8504 | 7.1878 | 4.4586 | 6.6825 | -7.4573 | 0.0000 |
| 3 | locked_nc_dsm_aux lambda=0.2 vs Locked NC-ConCM | 8.6614 | 8.8949 | 7.0064 | 8.7269 | -8.0863 | 0.0000 |
| 3 | Pure DSM-ConCM vs locked_nc_dsm_aux lambda=0.1 | 14.7244 | 23.8095 | -49.6815 | -18.8141 | -26.0557 | 57.9618 |
| 3 | Pure DSM-ConCM vs Locked NC-ConCM | 21.5748 | 30.9973 | -45.2229 | -12.1316 | -33.5130 | 57.9618 |

## Delta Mean/Std Table

| comparison | n | delta_AccT | delta_old_acc | delta_current_acc | delta_HM_old_current | delta_old_to_current | delta_current_to_old |
| --- | --- | --- | --- | --- | --- | --- | --- |
| locked_nc_dsm_aux lambda=0.1 vs Locked NC-ConCM | 3 | 7.3700+/-0.8398 | 5.8396+/-1.2443 | 12.8404+/-7.3495 | 8.1345+/-1.2608 | -2.7955+/-5.4855 | -10.0486+/-8.8618 |
| locked_nc_dsm_aux lambda=0.2 vs Locked NC-ConCM | 3 | 8.6630+/-0.5897 | 6.4543+/-2.4171 | 16.6788+/-8.7159 | 9.6450+/-1.6978 | -3.5185+/-7.5060 | -13.0378+/-11.6654 |
| Pure DSM-ConCM vs locked_nc_dsm_aux lambda=0.1 | 3 | 10.4739+/-4.4157 | 14.6689+/-9.0286 | -14.0795+/-30.9756 | -0.6331+/-15.8737 | -18.0064+/-7.1560 | 16.5206+/-36.0551 |
| Pure DSM-ConCM vs Locked NC-ConCM | 3 | 17.8439+/-4.5110 | 20.5085+/-10.2608 | -1.2391+/-38.3141 | 7.5014+/-17.1328 | -20.8019+/-12.2310 | 6.4720+/-44.8855 |

## Lambda 0 Sanity

For every new seed run, `lambda_dsm=0` exactly reproduced the locked NC-ConCM reference row saved in the corresponding locked JSON. The new residual path did not change the locked baseline.

## Executed Command Pattern

For each seed `S in {2,3}`, the following sequence was run on `my-gpu` with `CUDA_VISIBLE_DEVICES=2`:

```bash
BASE=/remote-home/wangbomin/LongTailedCL
TRAIN=hyperkvasir23_med_fdm_gate_fair_20260709_s${S}_5ep
CAL=hyperkvasir23_balanced_task_block_calibration_20260709_s${S}
DIAG=hyperkvasir23_task_block_diag_20260709_s${S}
LOCK=hyperkvasir23_ncconcm_rule_lock_20260709_s${S}
RUN=hyperkvasir23_locked_nc_dsm_aux_20260709_s${S}
PURE=hyperkvasir23_pure_dsm_concm_gate_20260709_s${S}

/root/anaconda3/bin/python tools/run_hyperkvasir_med_fdm_gate.py \
  --data-root "$BASE/data/hyper-kvasir23" \
  --output-dir "$BASE/runs/$TRAIN" \
  --ckpt-dir "$BASE/checkpoints/$TRAIN" \
  --device cuda --seed "$S" --epochs 5 --batch-size 32 --num-workers 4 --lr 0.01 \
  --exemplars-per-class 1 --fdm-lambda 0.01 --branches freeze_baseline

/root/anaconda3/bin/python tools/run_hyperkvasir_balanced_task_block_calibration.py ...
/root/anaconda3/bin/python tools/diagnose_hyperkvasir_task_block_calibration.py ...
/root/anaconda3/bin/python tools/run_hyperkvasir_concm_min_gate.py ...
/root/anaconda3/bin/python tools/run_hyperkvasir_locked_nc_dsm_aux_gate.py \
  ... --lambda-dsm-grid 0.0,0.05,0.1,0.2
/root/anaconda3/bin/python tools/run_hyperkvasir_pure_dsm_concm_gate.py ...
```

Exact full commands are saved in each run's `final_results.json`.

## Baseline Mismatch Diagnosis

### What was compared

The apparent `53.74` vs `77.53` mismatch is not a direct same-seed/same-checkpoint mismatch.

- Old `77.5273` row is documented as 20260621 `seed0`:
  `/dev/shm/wangbomin/LongTailedCL/runs/hyperkvasir23_ncconcm_rule_lock_20260621_s0`.
- Old 20260621 `seed1` row is `54.2687`, not `77.5273`.
- Current 20260709 `my-gpu` reconstructed `seed1` row is `53.7392`.
- Therefore current seed1 is close to old seed1 (`-0.5295 AccT`), while the large `77.53 -> 53.74` gap comes from comparing old seed0 to current seed1.

### Current checkpoint evidence

All current reconstructed checkpoints use the same 5-epoch freeze-baseline recipe, but different shuffled class orders:

| seed | phase0 ckpt bytes | phase1 ckpt bytes | epochs | old classes | current classes | locked AccT |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 1954228 | 1952150 | 5 | `[1,12,21,16,7,20,22,15,17,2,11,10,3]` | `[4,5]` | 53.7392 |
| 2 | 1954228 | 1952150 | 5 | `[21,19,18,14,16,7,6,22,12,17,2,11,10]` | `[13,0]` | 49.0669 |
| 3 | 1954228 | 1952150 | 5 | `[9,17,12,18,0,15,3,20,8,13,6,1,10]` | `[2,11]` | 54.3307 |

The current paths are:

- Seed1 phase0: `/remote-home/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260709_s1_5ep/phase0_with_anchors.pt`
- Seed1 phase1: `/remote-home/wangbomin/LongTailedCL/checkpoints/hyperkvasir23_med_fdm_gate_fair_20260709_s1_5ep/freeze_baseline_phase1.pt`
- Seed2 phase0/phase1: same pattern with `s2_5ep`
- Seed3 phase0/phase1: same pattern with `s3_5ep`

### Config and data/eval checks

Verified common settings:

- Dataset: `hyper_kvasir23`
- `base_classes=13`, `incremental_steps=5`, `max_phases=2`
- Phase1 branch: `freeze_baseline`
- Epochs: `5`
- Batch size: training `32`, eval `64`
- LR: `0.01`
- Exemplars: `1` per old class
- Alpha rule for locked baseline: `balanced_hmean_exemplar`
- Locked NC lambda: `0.30`
- Eval scripts: current runs use the same local code path family as the old documented command: `run_hyperkvasir_concm_min_gate.py`, `run_hyperkvasir_pure_dsm_concm_gate.py`, and the new `run_hyperkvasir_locked_nc_dsm_aux_gate.py`.
- Test totals differ by seed because old/current partitions differ: seed1 total `1511`, seed2 total `1286`, seed3 total `1270`.
- Exemplar file checks passed for seeds 1/2/3; each old class has one readable exemplar.

Important calibration evidence:

- Current seed2 `balanced_hmean_exemplar` selected alpha `0.45`, while diagnostics show oracle-best alpha `0.15`; this costs `8.6314 AccT`.
- Current seed3 `balanced_hmean_exemplar` selected alpha `0.60`, while diagnostics show best alpha `0.25`; this costs `6.6142 AccT`.
- This explains why current locked baselines are weak under the fixed balanced_hmean_exemplar rule for seeds 2/3.

### Environment checks

- Current runs executed on `my-gpu` at `/remote-home/wangbomin/LongTailedCL`.
- Interpreter: `/root/anaconda3/bin/python`.
- Old documented runs used `/dev/shm/wangbomin/LongTailedCL` and `/opt/miniconda3/envs/torchgpu/bin/python`.
- The old `/dev/shm/wangbomin/LongTailedCL` tree is not present on current `my-gpu`, so old checkpoint file sizes/mtimes cannot be directly rechecked now.

### Diagnosis

Evidence-supported judgment:

1. The headline `77.53` is old seed0, not old seed1.
2. Current `my-gpu` seed1 locked AccT `53.74` is close to old documented seed1 locked AccT `54.27`; seed1 is not clearly abnormal.
3. Current seed2/3 locked baselines also fall in the `49-54` range, so the current non-seed0 shuffled partitions are systematically weaker than old seed0 under `balanced_hmean_exemplar`.
4. The old `77.53` likely comes from a much easier/different seed0 class partition and alpha selection (`alpha=0.35`, current AccT `88.26`, old AccT `75.56`), not from a same-seed checkpoint regression.
5. Direct binary checkpoint comparison against old `/dev/shm` artifacts is not possible with current evidence because the old tree is absent.

Therefore all new conclusions are scoped to the current `my-gpu` reconstructed checkpoints.

## Recommendation

Use `locked_nc_dsm_aux lambda_dsm=0.2` as the best current guardrailed auxiliary residual setting.

- It is stable across seeds 1/2/3.
- It improves Locked NC-ConCM on AccT, old_acc, current_acc, HM, old_to_current, and current_to_old on average.
- It avoids the seed3 Pure DSM current-class collapse.

Keep Pure DSM-ConCM as a strong ablation branch, not the main method, until its current-class instability is solved.

## Source Artifacts

Remote summary output:

- `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_multiseed_summary.md`
- `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_per_seed.csv`
- `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_mean_std.csv`
- `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_deltas.csv`
- `/remote-home/wangbomin/LongTailedCL/runs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_delta_mean_std.csv`

Local source-copy summary output:

- `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_multiseed_summary.md`
- `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_per_seed.csv`
- `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_mean_std.csv`
- `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_deltas.csv`
- `docs/hyperkvasir23_locked_nc_dsm_aux_multiseed_20260709/locked_nc_dsm_aux_delta_mean_std.csv`
