# APART-ConCM Integration Plan

Date: 2026-06-16

Scope: plan only. Do not copy the existing TaConCM/ConCM implementation into APART. Do not mix FSCIL-specific ConCM metrics with APART LT-CIL Acc / AccT.

## Starting Point

APART is an exemplar-free LT-CIL method built around a frozen pretrained ViT, two adapter pools, adaptive routing, and two classifier heads. Its official CIFAR100-LT shuffled B50-5 config trains only current-task data and evaluates all seen classes using `logits + logits_few`.

ConCM-style components should therefore be integrated as lightweight prototype/statistics modules first, not as a wholesale method transplant. The safe path is to preserve APART inference and add diagnostics/regularizers in small ablations.

## Stage 1: APART + Prototype Memory / Covariance Augmentation

| item | plan |
|---|---|
| Insert code location | Add feature/stat collection around `models/apart.py::_init_train()` and after each task in `after_task()`. Feature extraction should use `self._network.eval()` and the APART feature path already returned by `VisionTransformer.forward()`. |
| New memory state | Per mapped class id: prototype mean, diagonal covariance or compact low-rank covariance, sample count, task id, many/medium/few bin, optional separate stats for `logits` path and `logits_few` path. Store numeric tensors only; no image exemplars. |
| Loss function | Prototype augmentation CE: sample compact synthetic features from class mean/covariance and pass through `head` / `head_few`; add CE or balanced CE over seen classes. Start with training-only loss and no inference change. |
| Inference impact | None in the first ablation. Evaluation remains APART's `logits + logits_few`. |
| Expected benefit | Helps tail and old-class classifier calibration without violating APART's exemplar-free setting. Gives a controlled bridge from APART to ConCM-style prototype augmentation. |
| Risks | Synthetic feature distribution may be miscalibrated because APART routing changes feature geometry by task and pool. Full covariance can be too large; use diagonal or low-rank summaries first. |

Suggested ablation:

| run | prototype memory | covariance | synthetic per class | loss weight | inference | expected signal |
|---|---:|---|---:|---:|---|---|
| APART baseline | no | none | 0 | 0 | raw APART | reference |
| Stage1-mean | yes | none | 0 | 0 | raw APART | diagnostics only |
| Stage1-diag-aug | yes | diagonal | 16 | 0.05 | raw APART | old/tail retention |
| Stage1-diag-aug-balanced | yes | diagonal | 16 | 0.1 | raw APART | many/medium/few tradeoff |

## Stage 2: APART + Routing-Aware Prototype Calibration

| item | plan |
|---|---|
| Insert code location | Use `backbone/vision_transformer_adapter_pool_a.py::AdapterPool.forward()` and `VisionTransformer.forward_features()` to expose selected adapter ids and route similarity. Consume them in `models/apart.py::_init_train()` for calibration losses. |
| New memory state | Per class and per route: routed prototype, route frequency, route-conditioned logit bias, old/new route margin statistics, and separate stats for main and auxiliary paths. |
| Loss function | Routing-aware prototype consistency: for samples routed to pool `r`, pull features toward the class prototype conditioned on `r`; add a small calibration regularizer on old-vs-new logit margins per route. Keep APART's existing CE and pool matching losses intact. |
| Inference impact | First ablation: no inference change. Second ablation: optional route-conditioned bias correction before `logits + logits_few`, but only after diagnostics show raw-logit route bias. |
| Expected benefit | APART's routing is adaptive and class-frequency-aware; route-specific prototypes should be less noisy than a single global class prototype, especially for tail classes and old-task drift. |
| Risks | Route ids are learned and can shift during training, so stale routed prototypes may hurt. Bias correction at inference can overfit phase1 and should be gated by held-out diagnostics. |

Suggested ablation:

| run | routed prototypes | route bias | consistency loss | inference calibration | expected signal |
|---|---:|---:|---:|---:|---|
| APART baseline | no | no | 0 | no | reference |
| Stage2-route-stats | yes | no | 0 | no | route drift diagnostics |
| Stage2-route-consistency | yes | no | 0.05 | no | feature retention |
| Stage2-route-bias-train | yes | yes | 0.05 | no | margin stabilization |
| Stage2-route-bias-eval | yes | yes | 0.05 | yes | logit calibration effect |

## Stage 3: APART + DSM Projector And LMatch/LCont

| item | plan |
|---|---|
| Insert code location | Add a small projector module near `AdapterVitNet` or in `models/apart.py` that consumes APART features. Do not alter the frozen ViT blocks or adapter-pool internals initially. |
| New memory state | Frozen previous-phase projector copy, per-class projected prototypes, pairwise prototype similarity matrix, route-aware similarity matrix, and compact class-frequency metadata. |
| Loss function | `LMatch`: match current projected prototype structure to stored calibrated old structure. `LCont`: supervised contrastive or prototype contrastive loss over current-task features plus generated prototype features. Keep loss weights small and report Acc/AccT plus many/medium/few. |
| Inference impact | Initially none: projector is training-only. A later ablation can use the projector for prototype-based auxiliary logits if Stage 3 improves retention without hurting APART. |
| Expected benefit | Dynamic structure matching is closer to ConCM's core contribution and may complement APART by preserving inter-class geometry across routed adapter updates. |
| Risks | Highest complexity and highest overfitting risk. If Stage 1/2 do not show stable gains, Stage 3 is premature. Projector losses can conflict with APART's existing pool matching and pull-constraint terms. |

Suggested ablation:

| run | projector | LMatch | LCont | prototype aug | inference | gate |
|---|---:|---:|---:|---:|---|---|
| APART baseline | no | 0 | 0 | no | APART | reference |
| Stage3-projector-stats | yes | 0 | 0 | no | APART | diagnostics only |
| Stage3-LMatch | yes | 0.05 | 0 | no | APART | old retention improves |
| Stage3-LMatch-LCont | yes | 0.05 | 0.05 | no | APART | old/tail improves |
| Stage3-full-lite | yes | 0.05 | 0.05 | Stage1 diag | APART | Acc/AccT improves |

## Recommendation

Implement Stage 1 first after APART baseline is reproducible. It is the smallest change, preserves exemplar-free behavior, avoids inference changes, and gives the statistics needed to decide whether route-aware calibration is justified.

Stage 2 should follow only if route histograms show systematic old/new or many/medium/few routing bias. Stage 3 should wait until Stage 1/2 produce a measurable gain over APART baseline or at least clear diagnostics showing prototype geometry drift.

