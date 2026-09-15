# Completed frozen representation comparison

A is reused unchanged. B uses the originally specified revision and SHA256, now recovered through local download and transfer. No new model or hyperparameter was introduced. Final eight-class numbers represent one deterministic fit, not three independent repetitions.

| Encoder | Classifier | Final validation BA |
|---|---|---:|
| A | NCM | 43.5199 |
| A | CBRidge | 55.4973 |
| B | NCM | 34.1276 |
| B | CBRidge | 54.1929 |

Both use final normalized CLS and per-sample L2, RGB 224-square bicubic/antialias, encoder-specific normalization. CBRidge uses lambda=0.001, no bias, float64 class-balanced sufficient statistics. Stage access, final order invariance and batch/stream equivalence passed. Detailed per-class and lesion-equal metrics are supplied.

New test predictions: 0.
