# GPA Single-Head Implementation Notes

This update fixes the most important issue in the previous scaffold: the dynamic anchoring term was implemented as a feature-centroid-to-frozen-prototype penalty. GPA Eq. 5 instead regularizes the classifier row against the normalized moving-average class centroid:

```text
L_anchor = sum_c || W_c - normalize(mu_c^(t)) ||_2^2
```

## What changed

- `train_one_phase()` now updates a raw EMA feature centroid for each current-task class and computes anchor loss on classifier weights, not on features.
- The default anchor reduction is `sum`, matching the paper's summation form. `mean` remains available for diagnostics.
- GPA bias is exposed as `--gpa-bias-mode paper|zero|old_mean|none`.
- Evaluation calibration is exposed as `--eval-mode raw|no_bias|cosine_eval|norm_weight_eval` and `--eval-calibration` for final-phase diagnostics.
- Phase summaries now include before-expansion, after-initialization, and after-training diagnostics, including old/new logit decomposition and classifier norm/bias statistics.

## Current faithful default

```text
--method finetune_gpa
--gpa-bias-mode paper
--lambda-gpa 0.12
--gpa-anchor-reduction sum
--freeze-old-rows
--restore-old-rows
--eval-mode raw
```

## Why this matters

The previous feature-anchor implementation could directly pull feature centroids toward stale frozen prototypes and harm both old and new accuracy. The corrected implementation constrains classifier geometry, which is the mechanism described by GPA.

## Recommended first server check

Run phase0+phase1 only:

```bash
bash scripts/run_cifar100lt_singlehead_phase1.sh \
  --gvalign-root /path/to/GVAlign \
  --data-root /path/to/GVAlign/data \
  --output-root /tmp/gpa_singlehead_weight_anchor_phase1 \
  --method finetune_gpa \
  --epochs 500 \
  --max-task 2 \
  --gpa-bias-mode paper \
  --lambda-gpa 0.12 \
  --gpa-anchor-reduction sum \
  --eval-calibration
```

Gate before full seed0:

- GPA after-init should not catastrophically reduce old/base accuracy.
- GPA after-training old accuracy and phase1 all-seen accuracy should be at least comparable to single-head Finetune.
- `new_weight_cos_to_frozen_proto_mean` should be close to 1.0.
- `old_head_weight_max_delta_after_phase` and `old_head_bias_max_delta_after_phase` should be 0.0 when freeze/restore is enabled.
