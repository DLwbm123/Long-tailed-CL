# Full ConCM Multi-Seed Results - GPU1

## Five Incremental Sessions

Mean/std is computed over seed-level averages of sessions 1-5; base session 0 is excluded.

| method | AccT | old | current | HM | current->old | collapse sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Locked NC | 28.75 +/- 5.58 | 32.70 | 1.13 | 1.82 +/- 1.78 | 98.19 | 15 |
| full_dynamic | 44.77 +/- 8.69 | 39.70 | 74.51 | 43.45 +/- 16.97 | 12.95 | 3 |
| nc_anchored | 31.32 +/- 0.97 | 33.16 | 13.19 | 10.48 +/- 8.71 | 84.26 | 15 |

## Final Session

| method | AccT | old | current | HM | current->old |
| --- | ---: | ---: | ---: | ---: | ---: |
| Locked NC | 33.75 +/- 4.39 | 36.39 | 0.57 | 1.10 +/- 1.91 | 99.43 |
| full_dynamic | 45.22 +/- 9.46 | 47.57 | 33.70 | 30.64 +/- 27.65 | 34.72 |
| nc_anchored | 31.05 +/- 2.47 | 33.51 | 0.00 | 0.00 +/- 0.00 | 98.94 |

At final session, full_dynamic improves over Locked NC by mean `+11.46` AccT and `+29.54` HM. nc_anchored changes AccT by `-2.71` and HM by `-1.10`.

## Stability

The collapse criterion is `current_to_old >= 30%` or `current accuracy < 50%`.

- Locked NC: 15/15 incremental sessions collapse.
- nc_anchored: 15/15 collapse.
- full_dynamic: 3/15 collapse.

full_dynamic is clearly the most stable of the three, but it is not stable enough for promotion:

- seed1 session2: current 18.03, current-to-old 45.90;
- seed1 session5: current 0.00, current-to-old 100.00;
- seed3 session5: current 43.97, current-to-old 2.59.

Seed2 full_dynamic is stable across all five sessions and finishes with AccT/HM `50.91/53.74`. Seed1 final-session collapse drives the large cross-seed variance.

All summary values come from the current GPU1 full-session runs. Old seed0 and bounded 20260709 session1 values are excluded.


## Later decision and provenance

Copied from the completed GPU1 output bundle on 2026-09-14. These aggregate metrics are retained as historical evidence. The later [branch closure](../HYPERKVASIR23_FULL_DYNAMIC_BRANCH_CLOSED.md) supersedes the July 10 candidate recommendation: Full Dynamic is closed and is not promoted. The reconstructed Locked NC control here is different from the earlier task-block calibrated Locked NC-ConCM session1 method.
