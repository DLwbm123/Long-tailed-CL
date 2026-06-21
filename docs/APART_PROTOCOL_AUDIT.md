# APART Protocol Audit

Date: 2026-06-16

Local checkout: `third_party/APART`

Upstream: `https://github.com/vita-qzh/APART`

Commit: `f3c5b8da5908b2f78266744a737f7f6bff45a43d`

## Executive Summary

The APART repository can be cloned and imported after installing the README dependencies `timm==0.6.12` and `easydict`. The CIFAR100-LT shuffled B50-5 APART entrypoint is present, but the repository does not include a runnable Finetune learner or config. Therefore APART baseline can be attempted from the official code, while Finetune requires either an external baseline implementation or a new minimal learner and cannot be reproduced "without code changes" from this repository alone.

The code is exemplar-free for the provided APART configs (`memory_size=0`, `memory_per_class=0`) and uses a frozen timm ViT-B/16 backbone with trainable adapter pools and classifier heads. Evaluation is all-seen top-1 over the summed logits from the main and auxiliary heads.

## Required Questions

| question | audit answer |
|---|---|
| CIFAR100-LT B50-5 shuffled command | From the APART repo root: `python main.py --config ./exps/apart_cifar_shuffle.json`. |
| Class order / long-tail sampling vs Liu et al. 2022 | The code builds CIFAR100-LT with exponential counts from 500 to about 5 samples per class (`longtail=0.01`). For shuffled CIFAR, `DataManager._setup_data()` creates a seeded random class permutation, and `order=false` separately shuffles the long-tail count list. The repository does not contain the fixed Liu et al. / GVAlign class-order list, so exact parity with Liu et al. 2022 is not established from code alone. |
| Exemplar-free? | Yes for official APART configs: `memory_size=0`, `memory_per_class=0`, and `models/apart.py` trains only the current task dataset without appending rehearsal memory. |
| Backbone | Code calls `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)` inside `backbone/vision_transformer_adapter_pool_a.py`. With `timm==0.6.12`, the resolved checkpoint download is an AugReg ViT-B/16 checkpoint whose filename includes i21k pretraining and ImageNet2012 finetuning. The commented `vit_base_patch16_224_in21k` path is not used by the CIFAR config. |
| Adapter pool A | Two pools are constructed in `VisionTransformer`: `self.pool` and `self.pool_few` in `backbone/vision_transformer_adapter_pool_a.py`. Runtime naming is non-obvious: the main `logits` path uses `head` on features routed through `pool_few`, while the auxiliary `logits_few` path uses `head_few` on features routed through `pool`. |
| Auxiliary adapter pool A_aux | Same code location: `self.pool` / `self.pool_few`. The auxiliary output path is `x_few = self.head_few(x_few)` in `VisionTransformer.forward()`. |
| Adaptive routing `w(x,y)` | `AdapterPool.forward()` computes normalized similarity between the original ViT `cls_features` and learned `prompt_key`, then selects top-k adapter ids. `PoolAssigner.forward()` maps `cls_features` and class-frequency weight to a sigmoid `pool_id`, which is used by the APART training losses. |
| Classifier `g` and `g_aux` | `VisionTransformer` defines `self.head` and `self.head_few`. `head` produces `logits`; `head_few` produces `logits_few`. |
| Inference logits | Yes, inference uses `outputs = res["logits"][:, :total_classes]` followed by `outputs += res["logits_few"][:, :total_classes]` in `models/apart.py`. |
| Evaluation Acc / AccT | `trainer.py` appends `cnn_accy["top1"]` after each task and logs `Average Accuracy (CNN)` as the mean over the top-1 curve. The last top-1 value is AccT. `utils/toolkit.py::accuracy()` computes all-seen top-1 over the currently seen classes. |
| Many/medium/few metrics | Yes. `utils/toolkit.py::accuracy()` returns `h-m-f` using thresholds `head > 100`, `tail < 20`, and medium otherwise, based on the current long-tail count list. |
| Training epochs, batch size, LR, optimizer, scheduler | For `apart_cifar_shuffle.json`: `tuned_epoch=10`, `batch_size=48`, `init_lr=0.003`, `optimizer=adamw`, `weight_decay=0`, `min_lr=1e-5`. The JSON says `scheduler=constant`, but `models/apart.py::get_scheduler()` overrides B50-5 to cosine because `init_cls != increment`. Adapter-pool parameters get `0.1 * init_lr`; other trainable params get `init_lr`. |

## CIFAR100-LT Protocol Details

The official shuffled config is `third_party/APART/exps/apart_cifar_shuffle.json`:

- dataset: `cifar224`
- class split: `init_cls=50`, `increment=10`, giving 6 phases: B50 + 5 x 10.
- shuffle: `true`
- long-tail: `imbalance=true`, `longtail=0.01`
- seed: `[1993]`
- memory: `0`
- device: `["1"]`

For CIFAR, `get_img_num_per_cls(500, 100, "exp", 0.01)` creates class counts with a head around 500 images and a tail around 5 images. The code maps labels to the shuffled class-order rank before sampling. With `order=false`, the long-tail count list is shuffled independently, so head/tail status is not simply tied to original CIFAR class id.

This is APART's own CIFAR100-LT shuffled protocol. It is not proven identical to the fixed GVAlign / Liu et al. 2022 protocol because no fixed Liu order file is present in APART.

## Model And Training Path

Primary entrypoints:

- `main.py`: loads a JSON config and calls `trainer.train()`.
- `trainer.py`: constructs `DataManager`, creates the learner through `utils/factory.py`, runs all tasks, and logs the top-1 curve and average.
- `utils/factory.py`: only registers `model_name == "apart"`.
- `models/apart.py`: APART learner, current-task training loop, optimizer, scheduler, and APART losses.
- `utils/inc_net.py`: builds `AdapterVitNet` and the original frozen ViT feature extractor.
- `backbone/vision_transformer_adapter_pool_a.py`: ViT-B/16 adapter-pool implementation, dual pools, routing, and dual heads.

The training loop is current-task only. It slices logits to the current task during training, while evaluation uses all seen classes. The loss is the average of current-task CE on `logits`, CE on `logits + logits_few`, and a few/auxiliary loss, plus the pool matching loss and pull-constraint terms.

## Reproduction Attempt Status

Remote server path:

- code: `/dev/shm/wangbomin/APART/code`
- data link: `/dev/shm/wangbomin/APART/code/data -> /dev/shm/wangbomin/GVAlign/data/cifar100`
- cache: `/dev/shm/wangbomin/APART/cache`
- logs: `/dev/shm/wangbomin/APART/logs`

Environment observations:

- `/` is nearly full, so code/cache/logs/data were kept under `/dev/shm`.
- GPUs: 2 x Tesla V100-PCIE-16GB.
- Python: `/opt/miniconda3/envs/torchgpu/bin/python`.
- Initial missing dependencies: `timm`, `easydict`.
- Installed for the APART run: `timm==0.6.12`, `easydict==1.13`.

Syntax/import gate passed for the APART entrypoint and core files.

APART baseline command launched:

```bash
cd /dev/shm/wangbomin/APART/code
XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
/opt/miniconda3/envs/torchgpu/bin/python main.py \
  --config ./exps/apart_cifar_shuffle.json \
  --text apart_concm_baseline_check
```

Run status:

- PID: `57049`
- log: `/dev/shm/wangbomin/APART/logs/apart_cifar_shuffle_20260616-090022.out`
- The run successfully loaded CIFAR100, printed the shuffled class order and long-tail counts, downloaded the timm pretrained ViT checkpoint into `/dev/shm/wangbomin/APART/cache/torch/hub/checkpoints`, initialized APART, and entered task 0 training.
- At task 0 epoch 5/10, the first base-phase test sanity was `Test_accy(pool1 86.28, pool2 87.98, pool_all 89.46)`, with `Train_accy 87.78`.
- The run completed all 6 phases. Final `CNN top1 curve` was `[90.42, 88.70, 88.33, 85.56, 85.03, 84.90]`.
- Final APART Acc was `87.1567`; final AccT was `84.90`.
- After completion, PID `57049` had exited and both GPUs were idle.

Finetune baseline status:

- Official APART repository has no `finetune` model registered in `utils/factory.py`.
- No `models/finetune.py` or Finetune JSON config exists in the checkout.
- Therefore Finetune cannot be run from APART official code without adding a learner/config.
