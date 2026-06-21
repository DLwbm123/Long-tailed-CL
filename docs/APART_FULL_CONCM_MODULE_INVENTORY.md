# APART Full-ConCM Module Inventory

## Scope

This is an inventory and integration plan only. No training was launched.

Current validated reference:

```text
NC-ConCM = APART + ConCM Stage1 capped prototype augmentation + head_norm_effective_sum old-logit calibration
```

Validated NC-ConCM full 3-seed result:

```text
Avg Acc: 86.624 -> 87.584, delta +0.961
AccT:    84.03  -> 85.68,  delta +1.65
few:     75.91  -> 78.67,  delta +2.76
```

Any future module must be compared primarily against NC-ConCM, not only against APART baseline.

## Branch / Config Isolation

The local `Long-tailed CL` checkout is not a git repository, so a git branch cannot be created in this workspace. The isolated config path for future full-ConCM work is:

```text
third_party/APART/exps/full_concm_ladder/
```

Do not overwrite the validated NC-ConCM configs, logs, reports, or checkpoints.

## What NC-ConCM Already Includes

| component | current code path | included behavior |
|---|---|---|
| Prototype memory | `third_party/APART/models/apart.py::_update_concm_stage1_memory()` | Stores per-class means and diagonal variances for `pre_logits` and `pre_logits_few`. |
| Prototype augmentation | `third_party/APART/models/apart.py::_concm_stage1_sample_memory()` and `_concm_stage1_loss()` | Samples old-class synthetic features and applies CE through `head + head_few`. |
| Capped synthetic budget | `concm_stage1_max_synth_total` | Caps total old synthetic features per iteration. |
| Head-norm calibration | `_eval_calibration_info()` and `_eval_cnn()` | Computes `effective_weight = head.weight + head_few.weight`; scales old logits by `alpha = new_norm / old_norm` at evaluation. |
| Old/new diagnostics | `_evaluate()` | Logs old acc, new acc, new->old, old->new, head norms, effective norms, calibrated and uncalibrated metrics. |

## Full Modules Not Currently Included

| module | source / available code | acts on | updates backbone / classifier / head_few | required state | interaction with APART effective head | old-class bias risk | minimal switches |
|---|---|---|---|---|---|---|---|
| Reliability-aware prototype calibration | `src/methods/ltconcm.py::ReliabilityAwarePrototypeCalibrator` | Prototype memory and synthetic feature means | No direct update if used to rewrite stored prototypes; indirect head/head_few gradients through synthetic CE | class prototypes, counts, uncertainty/variance, memory top-k | Should calibrate both `mean_main` and `mean_few`, or a shared normalized prototype projected back carefully. Do not calibrate only `head`. | Medium. Mixing new prototypes toward old memory may strengthen old-class regions and increase new->old unless head-norm calibration remains active. | `concm_proto_calibration`, `concm_proto_topk`, `concm_alpha_a`, `concm_alpha_b`, `concm_alpha_min`, `concm_alpha_max` |
| Calibrated prototype anchors | `src/methods/gpa.py::_target_anchor_loss()` uses `anchor_targets` | Classifier/effective-head weight geometry | Would update APART `head` and `head_few` if applied to effective rows | anchor target per class, class-to-row mapping, anchor weights | Must anchor `head.weight + head_few.weight`, not just `head.weight`. Need split gradients back to both heads. | High. Anchoring old effective rows can preserve old classes but suppress current-task plasticity. | `concm_effective_head_anchor`, `concm_anchor_lambda`, `concm_anchor_target=calibrated` |
| Tail-balanced anchor weights | `src/methods/ltconcm.py::tail_anchor_weights()` | Loss weighting / prototype or effective-head anchor | No update alone; affects whichever anchor/match loss uses it | class counts, gamma, max weight | Can weight effective-head anchor or feature-anchor loss by class frequency. | High if tail/old weights dominate current-task CE; previously old-logit bias was the main Stage1 failure mode. | `concm_tail_anchor`, `concm_anchor_gamma`, `concm_anchor_max_weight` |
| T-DSM anchor construction | `src/methods/ltconcm.py::build_tail_aware_structure_anchors()` | Prototype/anchor geometry | No direct update; produces anchors used by later losses | calibrated prototypes, class counts, old anchors, old/new mask | Anchors should live in APART feature space. If used for logits, compare to effective head rows. | Medium-high. Margin constraints can move current anchors away from true current-task features and increase new->old. | `concm_use_tdsm`, `concm_anchor_target=tdsm`, `concm_tdsm_steps`, `concm_tdsm_lr`, `concm_tdsm_margin_*` |
| Feature-structure match loss | `src/methods/gpa.py::_match_loss()` | Features / adapter outputs | Updates trainable APART adapters and routing; not heads unless combined with CE | anchor target per class, anchor weights, feature tensor, labels | Use APART `pre_logits` and possibly `pre_logits_few` separately; avoid only one path. | Medium. It can preserve old geometry but may reduce new-task feature adaptation. | `concm_use_match_loss`, `concm_match_lambda`, `concm_match_paths={main,few,both}` |
| GPA-style classifier imprinting | `src/methods/gpa.py::init_new_class_weights()` | Classifier rows / logits | Updates classifier rows directly before phase training | new class prototypes, counts, class-to-row map | Not directly appropriate: APART has fixed 100-way `head` and `head_few`, and training slices current-task logits. Any imprinting must handle both heads and preserve official APART protocol. | High. Prior GPA paper-bias and active-anchor routes collapsed in this project. Treat as diagnostic only, not main full-ConCM. | `concm_imprint_effective_head=false` by default |
| Route-aware prototype calibration | APART-specific; planned in `docs/APART_CONCM_INTEGRATION_PLAN.md`, not implemented in TaConCM code | Routing, memory, prototype statistics, optional loss | Could update adapter pools through consistency loss | per-class/per-route prototypes, route ids, route frequencies, old/new route margins | Must use APART selected adapter ids and route stats; effective-head calibration remains separate. | Medium. Route ids can drift across tasks, stale route prototypes can hurt current classes. | `concm_route_stats`, `concm_route_consistency`, `concm_route_loss_weight` |
| DSM projector with LMatch/LCont | Planned only; not present as reusable code in current TaConCM implementation | Projected features / prototype structure / contrastive loss | Updates projector and possibly adapters, depending on detach policy | projector, frozen previous projector, projected class prototypes, pairwise similarity matrix | Initially training-only. Do not use projector logits for APART inference until a gate passes. | High. Adds the most moving parts and can conflict with APART routing and pull constraints. | `concm_projector`, `concm_lmatch_weight`, `concm_lcont_weight` |

## Important Interaction Constraints

1. APART inference is `logits + logits_few`. Any classifier/logit module must use:

   ```python
   effective_weight = head.weight + head_few.weight
   ```

2. APART training is current-task sliced, while evaluation is all-seen. A module that strengthens old logits can look good on old retention but fail by raising `new_eval_pred_old_rate`.

3. NC-ConCM already fixed the first Stage1 failure through non-oracle head-norm calibration. Full modules must improve over calibrated NC-ConCM, not rediscover the same calibration effect.

4. Direct GPA-style bias/imprinting is not recommended as a primary APART full-ConCM module because this project already found GPA active-anchor routes unstable.

## Incremental Integration Ladder

| rung | method | modules enabled | first gate | keep only if |
|---:|---|---|---|---|
| 0 | APART baseline | none | already done | reference only |
| 1 | NC-ConCM reference | Stage1-capped augmentation + head-norm calibration | already passed 3-seed full | primary baseline |
| 2 | NC-ConCM + A | reliability-aware prototype calibration for Stage1 memory/synthetic means | max_tasks=2 | improves over NC-ConCM task1 Avg/AccT or old retention without new-class drop |
| 3 | NC-ConCM + B | T-DSM anchors + feature-structure match, no prototype calibration | max_tasks=2 | improves over NC-ConCM and does not increase new->old |
| 4 | NC-ConCM + A + B | prototype calibration + T-DSM/match | max_tasks=2 | better than both A-only and B-only, or clear complementary mechanism |
| 5 | full ConCM + headnorm calibration | A + tail-balanced anchor weights + T-DSM + match loss + head-norm calibration | seed1993 full only after 2-task pass | beats NC-ConCM full seed1993 and avoids late-task degradation |

Recommended concrete naming:

| rung | config prefix |
|---:|---|
| 1 | `nc_concm_reference` |
| 2 | `fullconcm_a_proto_calib` |
| 3 | `fullconcm_b_tdsm_match` |
| 4 | `fullconcm_ab_proto_tdsm_match` |
| 5 | `fullconcm_headnorm` |

## Bounded Gates

Every candidate must follow this sequence:

1. `max_tasks=2` only.
2. No full 6-task run until the 2-task gate beats NC-ConCM or shows a clear mechanism benefit without plasticity loss.
3. No 3-seed full until one full seed passes.
4. No hyperparameter search unless diagnostics identify a specific failure mode.

Primary comparison:

```text
candidate > NC-ConCM
```

Not:

```text
candidate > APART baseline but candidate < NC-ConCM
```

## Required Diagnostics

For every 2-task or full run:

- Avg Acc
- AccT
- old acc
- new/current acc
- `new_eval_pred_old_rate`
- `old_eval_pred_new_rate`
- alpha_t
- old effective norm
- new effective norm
- uncalibrated vs calibrated metrics where feasible
- many / medium / few for full runs

Additional full-ConCM diagnostics:

- prototype calibration alpha mean by many/medium/few
- prototype raw-to-calibrated cosine by many/medium/few
- T-DSM align/margin/old losses
- feature-match loss and match-to-CE ratio
- route histogram if route-aware modules are enabled

## Decision Rule

A module is useful only if it improves over NC-ConCM or provides a mechanism benefit without hurting plasticity.

Reject or demote to diagnostic-only if:

- it only beats APART baseline but underperforms NC-ConCM;
- it increases `new_eval_pred_old_rate` without a compensating AccT/Avg gain;
- it improves old acc by sacrificing current-task/new acc;
- alpha becomes extreme or unstable;
- few-class accuracy drops meaningfully in full runs.

## Recommendation

Start with rung 2:

```text
NC-ConCM + reliability-aware prototype calibration
```

Reason: it is closest to ConCM's prototype calibration contribution, can be implemented without touching APART inference or adding projector complexity, and uses states already present in NC-ConCM memory. T-DSM/match should wait until prototype calibration is gated, because anchor/match losses have higher risk of recreating old-class logit bias.
