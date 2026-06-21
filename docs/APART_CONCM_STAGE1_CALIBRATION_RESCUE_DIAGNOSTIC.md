# APART ConCM-lite Stage 1 Calibration Rescue Diagnostic

Date: 2026-06-17

Scope: no-training checkpoint audit for Stage1-capped calibration rescue. No training, no 3-seed run, and no full 6-task run was launched.

## Checkpoint Availability

Result: no usable APART checkpoint is available for evaluation-only calibration.

Remote checks run:

```bash
find /dev/shm/wangbomin/APART/code -maxdepth 6 -type f \
  \( -name "*.pth" -o -name "*.pt" -o -name "*.pkl" -o -name "*.ckpt" -o -name "*checkpoint*" \)

find /dev/shm/wangbomin/APART -maxdepth 6 -type f \
  \( -name "*.pth" -o -name "*.pt" -o -name "*.pkl" -o -name "*.ckpt" -o -name "*checkpoint*" \)
```

Both returned no model checkpoint files. The only checkpoint-like hit in logs was the pretrained ViT download under the Torch cache, not a trained APART task checkpoint.

Code audit:

- `third_party/APART/models/base.py` defines `BaseLearner.save_checkpoint()`.
- The APART trainer does not call this function.
- `rg "torch.save|save_checkpoint"` shows no active training-time save path for this APART run.

Therefore the completed Stage1-capped model cannot be reloaded for evaluation-only logit scaling. The final trained weights existed only in the finished Python process memory.

## Available Diagnostics From Logs

Baseline result reused from the existing 2-task run:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc |
|---|---|---:|---:|---:|---:|
| baseline | `[90.42, 88.70]` | 89.560 | 88.70 | 89.40 | 85.20 |

Baseline prediction-bias diagnostics are unavailable because the baseline run was completed before those logging fields were added, and no checkpoint exists for no-training re-evaluation.

Stage1-capped uncalibrated result from existing logs:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage1-capped | `[90.42, 88.58]` | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |

Classifier norm diagnostics:

| run | head old/new norm | head_few old/new norm |
|---|---|---|
| Stage1-capped | 1.288610 / 0.968578 | 1.308362 / 0.996805 |

Interpretation from available diagnostics:

- Stage1-capped improves old acc by `+1.46` over baseline.
- It drops new acc by `-8.00`.
- `22.20%` of new-task test samples are predicted as old classes.
- Old classifier row norms are substantially larger than new classifier row norms.
- This is consistent with classifier/logit old-class bias, but representation damage cannot be ruled out without checkpointed features/logits or model weights.

## Calibration Sweep Status

Requested eval-only alpha sweep:

```text
alpha in {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4}
old-class logits *= alpha
new-class logits unchanged
```

Status: not run.

Reason: no trained Stage1-capped checkpoint exists. Running this sweep would require either:

1. a saved task1 Stage1-capped checkpoint, or
2. retraining Stage1-capped to recreate the model state.

Option 2 violates the no-training constraint for this task.

Head-norm equalization eval is blocked for the same reason.

## Decision

No calibration rescue conclusion can be made from evaluation-only tests because the necessary checkpoint is absent.

Current evidence still points to old-class logit/classifier bias, but it is not enough to claim Stage1 can be rescued by calibration. The correct next step, if calibration rescue remains important, is to add task-boundary checkpoint saving for future runs before launching any further training. Do not rerun training solely for this diagnostic unless explicitly approved.
