# Full-ConCM Ladder Config Path

This directory is reserved for APART full-ConCM ladder configs.

Do not overwrite validated NC-ConCM configs:

- `../apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json`
- `../apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1994_gpu1.json`
- `../apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1995_gpu1.json`

Current rung 2 configs:

- `rung2_proto_calib_disabled_identity_phase1_seed1993_gpu0.json`
- `rung2_proto_calib_smoke_seed1993_gpu0.json`
- `rung2_proto_calib_phase1_seed1993_gpu0.json`

Current rung 3 configs:

- `rung3_match_disabled_identity_phase1_seed1993_gpu0.json`
- `rung3_match_smoke_seed1993_gpu0.json`
- `rung3_match_phase1_seed1993_gpu0.json`

Current rung 4 configs:

- `rung4_bias_controller_disabled_identity_phase1_seed1993_gpu0.json`
- `rung4_bias_controller_smoke_seed1993_gpu0.json`
- `rung4_bias_controller_phase1_seed1993_gpu0.json`

Current GUIDE-style uncertainty diagnostic configs:

- `guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json`
- `guide_uncertainty_diag_phase1_seed1993_gpu0.json`

Current rung 5b configs:

- `rung5b_ugr_disabled_identity_phase1_seed1993_gpu0.json`
- `rung5b_ugr_smoke_seed1993_gpu0.json`
- `rung5b_ugr_phase1_seed1993_gpu0.json`

Current USFM configs:

- `usfm_disabled_identity_phase1_seed1993_gpu0.json`
- `usfm_smoke_seed1993_gpu0.json`
- `usfm_phase1_seed1993_gpu0.json`

These configs test only:

```text
NC-ConCM + reliability-aware prototype calibration
```

The prototype calibration is optional and acts only on ConCM Stage1 memory / synthetic feature statistics. It does not modify APART classifier weights, `head_few`, routing, inference heads, T-DSM, match loss, tail anchors, projectors, or GPA-style imprinting.

Rung 3 tests only:

```text
NC-ConCM + dual-path feature-structure matching
```

The match loss operates in feature space on `pre_logits` and `pre_logits_few`. It does not anchor classifier weights or optimize `head.weight` / `head_few.weight` directly. Rung 3 keeps `concm_proto_calibration=false` because rung 2 was demoted to diagnostic-only.

Rung 4 tests only:

```text
NC-ConCM + bias-adaptive replay controller
```

The controller estimates current-task old-class prediction bias from training batches, then only reduces Stage1 prototype replay loss weight and capped synthetic sample count. It keeps `concm_proto_calibration=false` and `concm_use_match_loss=false`, keeps head-norm evaluation calibration active, and does not modify classifier heads, anchors, projectors, routing, or inference logic.

The GUIDE-style uncertainty diagnostic tests only:

```text
NC-ConCM + APART main/few branch uncertainty logging
```

It reuses the GUIDE uncertainty decomposition idea, but not GUIDE's multi-expert architecture, controllers, gates, validation meta-update, or DERM residual refinement. The diagnostic computes entropy-based aleatoric and epistemic statistics from APART `head(pre_logits)` and `head_few(pre_logits_few)` over the all-seen class set. It is detached from gradients and must not affect training or inference decisions.

Rung 5b tests only:

```text
NC-ConCM + uncertainty-gated Stage1 replay reduction
```

The gate reuses the APART main/few uncertainty diagnostic to estimate current-task batch aleatoric and epistemic uncertainty. It only reduces Stage1 synthetic replay loss weight and capped synthetic sample count, clipped to `[0.5, 1.0]`. It does not change CE loss, classifier heads, head-norm calibration, APART routing, prototype memory, feature matching, or any GUIDE/DERM controller.

USFM tests only:

```text
NC-ConCM + uncertainty-selective current-task feature matching
```

USFM reuses APART main/few branch uncertainty over all-seen logits. It builds detached EMA anchors for current-task real samples only, then applies a small feature-space cosine loss only when high epistemic and low aleatoric uncertainty produce a non-degenerate detached gate. It does not touch old synthetic samples, classifier heads, prototype calibration, replay controllers, tail anchors, route-aware modules, projectors, or GUIDE/DERM controllers.

Future planned config names:

- `fullconcm_a_proto_calib_phase1_seed1993_gpu0.json`
- `fullconcm_b_tdsm_match_phase1_seed1993_gpu0.json`
- `fullconcm_ab_proto_tdsm_match_phase1_seed1993_gpu0.json`
- `fullconcm_headnorm_full_seed1993_gpu0.json`

No full 6-task or 3-seed full-ConCM config should be launched from this directory until the bounded rung gates pass.
