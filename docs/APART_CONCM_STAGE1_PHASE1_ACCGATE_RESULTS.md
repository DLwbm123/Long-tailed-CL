# APART ConCM-lite Stage 1 Phase1 Accuracy Gate

Date: 2026-06-16

Scope: paired 2-task effectiveness gate for APART CIFAR100-LT shuffled B50-5. This does not run the full 6-task experiment, does not touch TaConCM or GPA, does not implement ConCM Stage 2/3, and does not change APART inference.

## Checkpoint Reuse

Task0 was not reused from a checkpoint.

Reason: APART has a `BaseLearner.save_checkpoint()` helper, but the current APART trainer does not call a task-boundary save path for this model flow, and there is no safe task-boundary resume/fork path that restores the model, task counters, data manager/class order, RNG state, and ConCM Stage1 memory. Adding this just for the gate would be more invasive than the test itself.

Fallback used: two independent `max_tasks=2` runs with the normal `tuned_epoch=10` budget and `timeout 3600s`.

## Commands

Baseline, Stage1 disabled:

```bash
cd /dev/shm/wangbomin/APART/code
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_phase1.json \
    --text apart_phase1_baseline_accgate \
  > /dev/shm/wangbomin/APART/logs/apart_phase1_baseline_accgate_20260616-132300.out 2>&1 &
```

ConCM-lite Stage1 enabled:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_phase1_accgate_20260616-140037.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_phase1.json \
    --text apart_concm_stage1_phase1_accgate \
  > "$LOG" 2>&1 &
```

## Configs

| run | config | max_tasks | tuned_epoch | Stage1 weight | synth/class |
|---|---|---:|---:|---:|---:|
| baseline | `third_party/APART/exps/apart_cifar_shuffle_phase1.json` | 2 | 10 | 0.0 | 0 |
| Stage1 | `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_phase1.json` | 2 | 10 | 0.05 | 4 |

Both runs use the APART CIFAR shuffle protocol: `cifar224`, `shuffle=true`, `init_cls=50`, `increment=10`, `longtail=0.01`, ViT-B/16 adapter pool, seed `1993`.

## Task0 Alignment

The Stage1 run task0 matched the baseline exactly before Stage1 becomes active:

| task0 metric | baseline | Stage1 |
|---|---:|---:|
| epoch1 loss / train acc | 24.846 / 74.04 | 24.846 / 74.04 |
| epoch5 pool_all | 89.46 | 89.46 |
| epoch10 pool_all | 90.42 | 90.42 |
| task0 final CNN | 90.42 | 90.42 |

This keeps the accuracy gate focused on task1.

## Accuracy Gate

| run | CNN top1 curve | Average Accuracy | task1 AccT | old-task acc after task1 | new-task acc after task1 | many | medium | few |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `[90.42, 88.70]` | 89.560 | 88.70 | 89.40 | 85.20 | 94.10 | 89.75 | 83.04 |
| Stage1 | `[90.42, 88.55]` | 89.485 | 88.55 | 90.94 | 76.60 | 92.29 | 88.81 | 84.96 |
| delta | `[+0.00, -0.15]` | -0.075 | -0.15 | +1.54 | -8.60 | -1.81 | -0.94 | +1.91 |

Interpretation: Stage1 improves old-task retention by `+1.54`, but it sharply hurts new-task learning by `-8.60`. Average Accuracy and task1 AccT both decrease.

## Stage1 Loss Diagnostics

Stage1 loss is active, finite, and contributes gradients, but its weighted scale is tiny relative to CE after the first epoch.

| epoch | raw Stage1 loss | weighted Stage1 loss | CE loss | Stage1/CE ratio | nonzero grad batches |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000788 | 0.000039 | 0.774860 | 0.001007 | 16/16 |
| 2 | 0.000039 | 0.000002 | 0.256743 | 0.000203 | 16/16 |
| 3 | 0.000012 | 0.000001 | 0.299740 | 0.000053 | 16/16 |
| 4 | 0.000009 | 0.000000 | 0.312813 | 0.000045 | 16/16 |
| 5 | 0.000011 | 0.000001 | 0.226470 | 0.000181 | 16/16 |
| 6 | 0.000012 | 0.000001 | 0.196094 | 0.000085 | 16/16 |
| 7 | 0.000007 | 0.000000 | 0.170316 | 0.000454 | 16/16 |
| 8 | 0.000009 | 0.000000 | 0.195450 | 0.000129 | 16/16 |
| 9 | 0.000006 | 0.000000 | 0.236295 | 0.000061 | 16/16 |
| 10 | 0.000005 | 0.000000 | 0.212324 | 0.000056 | 16/16 |

Runtime checks:

- `Traceback|RuntimeError|CUDA out of memory|Error`: 0
- exact-token `nan|inf`: no hits
- final Stage1 run completed before the `timeout 3600s` limit

## Decision

Decision: negative for this Stage1 prototype augmentation setting.

It is not justified to start a 3-seed or full 6-task run with the current Stage1 loss (`loss_weight=0.05`, `synth_per_class=4`) because the paired phase1 gate reduces Average Accuracy and task1 AccT, and the old-task gain comes with a large new-task accuracy drop.

Recommended next step: do not scale this exact Stage1 loss to full runs. If continuing ConCM-lite, first redesign the Stage1 loss so it does not suppress new-task learning, or make it routing/class-balanced and validate again with the same 2-task gate.
