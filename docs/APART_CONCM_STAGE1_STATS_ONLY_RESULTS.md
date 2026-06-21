# APART ConCM-lite Stage 1 Stats-Only Safety Results

Date: 2026-06-16

Scope: safety validation for ConCM-lite Stage 1 prototype statistics only. This run did not enable prototype augmentation loss, did not change APART inference, and did not touch TaConCM or GPA.

## Run

Remote code:

- `/dev/shm/wangbomin/APART/code`

Command:

```bash
cd /dev/shm/wangbomin/APART/code
XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
/opt/miniconda3/envs/torchgpu/bin/python main.py \
  --config ./exps/apart_cifar_shuffle_concm_stage1_stats.json \
  --text apart_concm_stage1_stats_rngfix
```

Remote log:

- `/dev/shm/wangbomin/APART/logs/apart_concm_stage1_stats_rngfix_20260616-105813.out`

Stats-only config:

- `concm_stage1=true`
- `concm_stage1_loss_weight=0.0`
- `concm_stage1_synth_per_class=0`

This means Stage 1 memory statistics were collected after each phase, but no `ConCMStage1_loss` was added to training.

## Earlier Safety Failure

The first stats-only attempt was stopped after task 1 started:

- PID: `37136`
- log: `/dev/shm/wangbomin/APART/logs/apart_concm_stage1_stats_20260616-102503.out`

Task 0 matched the APART baseline, but task 1 epoch 1 diverged from baseline:

| run | task1 epoch1 loss | task1 epoch1 train acc |
|---|---:|---:|
| APART baseline | 43.621 | 76.26 |
| first stats-only attempt | 42.520 | 79.31 |

Root cause: phase-end statistics extraction used a DataLoader over the train dataset and consumed global RNG state.

Fix applied:

- Capture and restore Python, NumPy, Torch CPU, and CUDA RNG state around `_update_concm_stage1_memory()`.
- Use an isolated `torch.Generator` for the stats extraction DataLoader.

## Safety Result

The rerun completed all 6 phases. It exactly matches the reproduced APART baseline curve and final metrics.

| run | Acc | AccT | curve |
|---|---:|---:|---|
| APART baseline | 87.1567 | 84.90 | `[90.42, 88.70, 88.33, 85.56, 85.03, 84.90]` |
| Stage1 stats-only | 87.1567 | 84.90 | `[90.42, 88.70, 88.33, 85.56, 85.03, 84.90]` |

Final many/medium/few also matches baseline:

| run | many | medium | few |
|---|---:|---:|---:|
| APART baseline | 89.80 | 84.26 | 79.93 |
| Stage1 stats-only | 89.80 | 84.26 | 79.93 |

No training loss injection was observed:

- `grep -c "ConCMStage1_loss" = 0`
- `grep -Ec "Traceback|RuntimeError|Error" = 0`

## Memory Updates

| task | updated classes | total memory classes | synth per class | loss weight |
|---:|---|---:|---:|---:|
| 0 | 0-49 | 50 | 0 | 0.0 |
| 1 | 50-59 | 60 | 0 | 0.0 |
| 2 | 60-69 | 70 | 0 | 0.0 |
| 3 | 70-79 | 80 | 0 | 0.0 |
| 4 | 80-89 | 90 | 0 | 0.0 |
| 5 | 90-99 | 100 | 0 | 0.0 |

## Verdict

ConCM-lite Stage 1 statistics collection is safe after the RNG fix:

- It does not perturb APART training or evaluation when loss weight is zero.
- It preserves APART baseline Acc, AccT, phase curve, and final many/medium/few metrics.
- It leaves inference unchanged.

Next valid step: run a small Stage 1 loss ablation with `concm_stage1_loss_weight=0.05` only after this stats-only result is accepted. Do not implement Stage 2 or Stage 3 until Stage 1 loss has a measured benefit or at least no material degradation.
