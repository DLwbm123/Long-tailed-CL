# APART ConCM-lite Stage 1 Schedule Gates

Date: 2026-06-17

Scope: bounded `max_tasks=2` APART CIFAR100-LT shuffled B50-5 gates. This does not run full 6-task training, does not run 3 seeds, does not touch TaConCM or GPA, and does not change APART inference.

## Files Changed

- `third_party/APART/models/apart.py`
  - Added ConCM Stage1 schedule controls:
    - `concm_stage1_start_epoch`
    - `concm_stage1_ramp_epochs`
    - `concm_stage1_max_synth_total`
  - Added final eval diagnostics:
    - `new_eval_pred_old_rate`
    - `old_eval_pred_new_rate`
    - old/new block prediction rates
    - old/new classifier head and head_few weight norms
  - Expanded Stage1 training logs with 6-decimal raw loss, weighted loss, effective weight, CE loss, weighted-to-CE ratio, raw-to-CE ratio, and nonzero-gradient batch count.
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_late_phase1_gpu0.json`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_ramp_phase1_gpu1.json`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_phase1_gpu0.json`

## Baseline Reuse

The existing 2-task APART baseline result was reused, per instruction:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc |
|---|---|---:|---:|---:|---:|
| baseline | `[90.42, 88.70]` | 89.560 | 88.70 | 89.40 | 85.20 |

The current Stage1 result from the previous gate was also reused:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc |
|---|---|---:|---:|---:|---:|
| current Stage1 | `[90.42, 88.55]` | 89.485 | 88.55 | 90.94 | 76.60 |

Those previous logs were produced before the new prediction-bias diagnostics were added, so `new_eval_pred_old_rate` and classifier norm diagnostics are unavailable for those two rows.

## Commands Run

Late schedule:

```bash
cd /dev/shm/wangbomin/APART/code
LOG_LATE=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_late_phase1_gate_20260617-021846.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_late_phase1_gpu0.json \
    --text apart_concm_stage1_late_phase1_gate \
  > "$LOG_LATE" 2>&1 &
```

Ramp schedule:

```bash
cd /dev/shm/wangbomin/APART/code
LOG_RAMP=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_ramp_phase1_gate_20260617-021846.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_ramp_phase1_gpu1.json \
    --text apart_concm_stage1_ramp_phase1_gate \
  > "$LOG_RAMP" 2>&1 &
```

Capped synthetic-old schedule:

```bash
cd /dev/shm/wangbomin/APART/code
LOG_CAPPED=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_capped_phase1_gate_20260617-025651.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_phase1_gpu0.json \
    --text apart_concm_stage1_capped_phase1_gate \
  > "$LOG_CAPPED" 2>&1 &
```

## Result Table

Decision rule:

- Promising: old acc improves while new acc drop is less than 1%.
- Inconclusive: old acc improves but new acc drops 1-3%.
- Negative: new acc drops more than 3%, even if old acc improves.

| run | schedule | curve | Avg Acc | task1 AccT | old acc | new acc | old delta | new delta | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | Stage1 disabled | `[90.42, 88.70]` | 89.560 | 88.70 | 89.40 | 85.20 | 0.00 | 0.00 | reference |
| current Stage1 | weight 0.05 from epoch1, all old synth | `[90.42, 88.55]` | 89.485 | 88.55 | 90.94 | 76.60 | +1.54 | -8.60 | negative |
| Stage1-late | weight 0.05 from epoch6 | `[90.42, 87.45]` | 88.935 | 87.45 | 90.94 | 70.00 | +1.54 | -15.20 | negative |
| Stage1-ramp | linear 0 to 0.05 over task1 | `[90.42, 88.03]` | 89.225 | 88.03 | 90.86 | 73.90 | +1.46 | -11.30 | negative |
| Stage1-capped | weight 0.05, max 48 synthetic old samples/iter | `[90.42, 88.58]` | 89.500 | 88.58 | 90.86 | 77.20 | +1.46 | -8.00 | negative |

## Prediction Bias Diagnostics

| run | new_eval_pred_old_rate | old_eval_pred_new_rate | old->old | new->new | head norm old/new | head_few norm old/new |
|---|---:|---:|---:|---:|---|---|
| baseline | N/A | N/A | N/A | N/A | N/A | N/A |
| current Stage1 | N/A | N/A | N/A | N/A | N/A | N/A |
| Stage1-late | 29.30 | 0.24 | 99.76 | 70.70 | 1.349961 / 0.968585 | 1.373310 / 0.996807 |
| Stage1-ramp | 25.40 | 0.50 | 99.50 | 74.60 | 1.262268 / 0.968586 | 1.283829 / 0.996807 |
| Stage1-capped | 22.20 | 0.76 | 99.24 | 77.80 | 1.288610 / 0.968578 | 1.308362 / 0.996805 |

Interpretation:

- The failure mode is old-class prediction bias on new-class test samples.
- Stage1-late makes the bias worst among the new schedules: `29.30%` of new-task test samples are predicted as old classes.
- Stage1-ramp reduces that to `25.40%`.
- Stage1-capped reduces it further to `22.20%`, but this is still large and still leaves new acc `8.00` points below baseline.
- Old eval samples almost never get predicted as new classes (`0.24-0.76%`), so the bias is asymmetric: the classifier becomes conservative toward old classes at the expense of new-task learning.
- Old classifier row norms are consistently higher than new classifier row norms in the final model. The norm gap aligns with the observed new-to-old prediction drift.

## Loss Diagnostics

| run | key behavior | final task1 loss diagnostics |
|---|---|---|
| Stage1-late | epochs 1-5 effective weight `0.000000`; epochs 6-10 `0.050000` | epoch10 raw `0.000001`, weighted `0.000000`, CE `0.212378`, weighted/CE `0.000001`, nonzero grad `16/16` |
| Stage1-ramp | effective weight ramps `0.000000 -> 0.050000` | epoch10 raw `0.000000`, weighted `0.000000`, CE `0.212379`, weighted/CE `0.000000`, nonzero grad `16/16` |
| Stage1-capped | effective weight `0.050000`, synthetic-old cap `48` | epoch10 raw `0.000007`, weighted `0.000000`, CE `0.212393`, weighted/CE `0.000003`, nonzero grad `16/16` |

All three runs completed without `Traceback`, `RuntimeError`, CUDA OOM, or logged `Error`.

## Recommendation

None of the tested Stage1 schedules deserves 3-seed or full 6-task testing.

Reason: every schedule improves old-task accuracy by about `+1.46` to `+1.54`, but the new-task accuracy drop is far beyond the 3% negative threshold. The safest tested variant is Stage1-capped, but it still drops new acc from `85.20` to `77.20`.

The next useful design step is not increasing `loss_weight`. The diagnostic points to old-class logit/norm bias. A safer follow-up would need an explicit anti-bias constraint, balanced old/new synthetic sampling with new-task preservation, or a normalization/calibration mechanism validated again by the same 2-task gate before any full run.
