# Decision-entry veto and equal-mean fusion

Frozen B, original scales and lambda=0.5. Equal mean only when both prior fusions leave B. No parameter scan, model/image processing, scale refit or E/test access.

| Dataset | Query | Entry BA % | Delta B all/old/new pp | Delta consensus all pp | Rescue/harm vs B | Rescue/harm vs consensus | Harm bound |
|---|---|---:|---|---:|---|---|---:|
| ISIC | Qfit | 70.0826 | +1.4966 / +1.6497 / +1.1905 | +0.0536 | 19/4 | 2/1 | 4 |
| ISIC | Qgate | 55.3716 | +0.6527 / +0.5435 / +0.8711 | +0.0000 | 14/2 | 0/0 | 2 |
| HK | Qfit | 61.5670 | +0.4048 / +0.4670 / +0.0000 | +0.0000 | 2/0 | 0/0 | 0 |
| HK | Qgate | 70.2406 | +0.2381 / +0.2747 / +0.0000 | +0.0000 | 1/0 | 0/0 | 0 |

## Retention of consensus rescues

| Dataset | Query | Retained/previous rescues | Lost | Additional |
|---|---|---|---:|---:|
| ISIC | Qfit | 17/18 | 1 | 2 |
| ISIC | Qgate | 14/14 | 0 | 0 |
| HK | Qfit | 2/2 | 0 | 0 |
| HK | Qgate | 1/1 | 0 | 0 |

## Target audit and interpretation

Both preidentified ISIC Qfit targets recover: mapped class3 and mapped class4. However, a different mapped class4 sample loses a previous rescue. Only three predictions change across all four conditions: two wrong-to-correct and one correct-to-wrong versus consensus, all on ISIC Qfit. All three were wrong under B. Thus B-relative rescue increases from18 to19 while B-relative harm remains4. The target class4 gain is canceled by the other class4 loss; the net BA gain is exactly one extra class3 correct image, 100/(311*6)=0.05359 pp. Old BA rises0.08039 pp, new BA is unchanged. No other condition changes predictions, and the HK target recoveries persist because all HK predictions remain identical to consensus.

The score evidence explains both benefits and cost. For the recovered class3 sample, wrong class4's advantage over true class3 is +0.14096 under density and -0.15156 under distance; averaging selects true class3 by only0.00530. For the recovered class4 target, wrong class0's advantage is -0.09600/+0.01561; averaging selects class4 by0.04020. For the newly lost class4 rescue, wrong class2's advantage is -0.05479/+0.08012; averaging instead selects class2 by0.01266. These are small compromise margins, not evidence that equal averaging identifies the reliable branch. Per-row identities remain private.

The label-free gate invariant and harm subset pass on every row and old/new subgroup. Observed B harms are ISIC4/2 and HK0/0, matching the fixed upper bounds. This guarantee controls damage relative to B, not loss of rescues relative to another fusion; the newly lost rescue does not violate it. Current zero HK harm is dataset-specific, not a universal generalization guarantee.

Both datasets still improve overall BA versus B on both queries without old/new degradation. However, this change barely increases retained rescue coverage: ISIC Qfit keeps17/18 consensus rescues and adds2; all other conditions retain exactly the same rescues. Versus density/distance it remains a conservative tradeoff, not a broadly stronger method. No lambda, threshold, class-specific protection, seed expansion or follow-up experiment was performed. Reused development queries and post-hoc design preclude independent performance claims.

## Gates

- ISIC versus B: PASS (strict overall improvement on both queries and old/new loss at most1 pp).
- ISIC versus density_fusion: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).
- ISIC versus distance_fusion: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).
- ISIC versus consensus: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).
- HK versus B: PASS (strict overall improvement on both queries and old/new loss at most1 pp).
- HK versus density_fusion: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).
- HK versus distance_fusion: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).
- HK versus consensus: FAIL (strict overall improvement on both queries and old/new loss at most1 pp).

## Validation and artifacts

Synthetic checks cover either-branch veto, averaging across all classes, permitted return to B and the harm subset. All three reference runs reproduce full metric dictionaries, labels, B and scales. Public files include per-class comparisons, all/old/new metrics, confusion matrices, rescue retention and harm bounds. Private changes retain row indices, labels, predictions and score gaps. Hybrid-score CE is descriptive and can change without prediction changes. CPU runtime in status.json excludes all prior cache construction. Previous consensus delivery:61ea835df655feda64f40206f4326e44ce93f21f.
