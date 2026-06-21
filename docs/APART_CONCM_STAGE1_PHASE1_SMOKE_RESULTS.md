# APART ConCM-lite Stage 1 Phase1 Smoke Results

Date: 2026-06-16

Scope: bounded smoke test for ConCM-lite Stage 1 prototype augmentation loss. This is not a full ablation. It does not change APART inference, does not touch TaConCM, does not continue GPA, and does not implement MPC or DSM.

## Baseline max_tasks Gate

Config:

- `third_party/APART/exps/apart_cifar_shuffle_phase1.json`
- `max_tasks=2`

Remote log:

- `/dev/shm/wangbomin/APART/logs/apart_phase1_baseline_20260616-124030.out`

The gate was stopped early after code-path validation, per instruction.

Evidence:

- `max_tasks: 2` was parsed.
- Trainer logged `Limiting run to max_tasks=2 of 6 total tasks.`
- Task 0 early epochs exactly matched the known APART baseline:
  - epoch 1: loss `24.846`, train acc `74.04`
  - epoch 2: loss `6.206`, train acc `85.29`
  - epoch 3: loss `2.822`, train acc `85.41`
  - epoch 4: loss `2.197`, train acc `86.48`
  - epoch 5: loss `1.795`, train acc `87.78`, pool_all `89.46`
  - epoch 6: loss `2.918`, train acc `88.60`
- No traceback or runtime error was observed.
- GPU execution was normal.

Verdict: baseline `max_tasks` gate passed for code-path validation.

## Stage1 Loss Smoke

Config:

- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_smoke.json`
- `max_tasks=2`
- `tuned_epoch=2`
- `concm_stage1=true`
- `concm_stage1_loss_weight=0.05`
- `concm_stage1_synth_per_class=4`

Remote command:

```bash
cd /dev/shm/wangbomin/APART/code
timeout 900s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_smoke.json \
    --text apart_concm_stage1_loss_smoke
```

Remote log:

- `/dev/shm/wangbomin/APART/logs/apart_concm_stage1_smoke_20260616-130100.out`

The run was bounded by `timeout 900s`. It completed the 2-epoch, 2-task smoke before manual stopping was needed.

## Evidence

Task 0 created prototype memory:

```text
ConCM-lite Stage1 memory updated: task=0, classes=[0, ..., 49], total_memory_classes=50, synth_per_class=4, loss_weight=0.05
```

Stage1 loss activated in task 1:

| phase | epoch | train loss | train acc | ConCMStage1_loss |
|---:|---:|---:|---:|---:|
| 1 | 1 | 35.824 | 77.98 | 0.005 |
| 1 | 2 | 20.776 | 91.64 | 0.000 |

The first Stage1-loss logging point is finite and nonzero. The second is finite and rounded to `0.000` by the existing three-decimal log formatting.

Task 1 also updated memory:

```text
ConCM-lite Stage1 memory updated: task=1, classes=[50, ..., 59], total_memory_classes=60, synth_per_class=4, loss_weight=0.05
```

Sanity counters:

- `ConCMStage1_loss` log hits: `3` including tqdm progress duplication.
- `Traceback|RuntimeError|CUDA out of memory|Error`: `0`
- GPUs were idle after completion: GPU0 `1 MiB`, GPU1 `1 MiB`.

## Verdict

The Stage1 prototype augmentation loss smoke passed:

- Prototype memory is available before phase 1 training.
- `ConCMStage1_loss` is activated in phase 1.
- The first logged Stage1 loss is finite and nonzero.
- Backward and optimizer steps completed without crash.
- No traceback, runtime error, CUDA OOM, or visible NaN/Inf failure was observed.
- Stats-only mode had already been verified baseline-identical in `docs/APART_CONCM_STAGE1_STATS_ONLY_RESULTS.md`.

This smoke only validates the execution path. It does not establish accuracy benefit. The next experimental step, if requested, should be a short phase1 ablation comparing APART phase1 vs Stage1 phase1 under the same `max_tasks=2` setting.
