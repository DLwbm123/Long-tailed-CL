# GPA Paper Implementation Spec

This document defines the single-head GPA reproduction scaffold now copied into this Long-tailed CL workspace. GPA has no available public training implementation to audit, so the paper is treated as the method specification and GVAlign / Long-Tailed-CIL is treated as the data and protocol reference.

## Component Decisions

| component | paper source | implementation decision | ambiguity |
|---|---|---|---|
| classifier type | paper uses `h_t: R^d -> R^{|C_1:t|}` | single shared classifier | paper code unavailable |
| prototype estimation | Eq. 3 | `prev_model.eval()`, current task samples only, features extracted before current-phase training | none |
| weight init | Eq. 4 | `W_c = normalize(mu_c)` for new class rows | old rows unchanged unless variant |
| bias init | Eq. 4 | `b_c = -log(N_c / N_ref + eps)` | `N_ref` definition not fully specified |
| old weight freezing | Algorithm 1 | freeze old rows by zeroing old classifier gradients and optionally restoring old rows after `optimizer.step()` | replay setting ambiguity |
| dynamic anchor | Eq. 5 | current-task moving-average centroid initialized from frozen prototypes | paper says moving-average centroid, exact update timing is not code-specified |
| anchor loss | Eq. 5 | L2 squared sum over current-task class centroids | exact reduction not code-specified |
| lambda | Fig. 5 / text | `0.12` for CIFAR100-LT | paper hyperparameter only |
| evaluation | paper Acc/AccT | all-seen top-1 single-head accuracy | none |
| protocol | GVAlign / LT-CIL | CIFAR100-LT fixed shuffled order, 50 + 5x10 classes, 20 exemplars/class, herding | paper code unavailable |

## Assumptions

1. The classifier is monolithic: at phase `t`, logits are `num_seen_classes` wide and all seen classes compete in one softmax.
2. Labels remain GVAlign's global incremental labels: base classes are `0..49`, phase1 classes are `50..59`, and so on.
3. Prototype estimation is done with a frozen copy of the model from the end of phase `t-1`, before expanding or training on phase `t`.
4. Dynamic anchor is implemented as a moving-average target per current-task class. The first target is the frozen prototype.
5. Because the public GPA training code is unavailable, `N_ref` is exposed as a CLI switch. The default scaffold uses `max_current`.
6. Exemplar replay follows GVAlign: growing memory with `num_exemplars_per_class=20` and herding selection.
7. This scaffold is not GVAlign multi-head GPA and does not use TaConCM or ConCM.

## Non-Goals

- No TaConCM or ConCM code.
- No task-masked accuracy as the main metric.
- No no-anchor variant counted as faithful GPA.
- No official implementation audit, because no complete official training implementation is available.
- No full seed0 launch until phase1 single-head sanity passes Finetuning.
