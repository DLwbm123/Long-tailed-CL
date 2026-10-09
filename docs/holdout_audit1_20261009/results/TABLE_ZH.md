# 完整任务边界对照

所有差值为 native − static_pc；BA 单位为百分点，平方风险越低越好。各状态共用冻结编码器，但 fit_only/all 的 bank 不同。

| 数据集 | 阶段 | bank | meta Δ风险 | 当前开发 Δ风险 | meta ΔBA | 当前开发 ΔBA | 全部开发 ΔBA | fit_only meta选择 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ISIC | 1 | all | +0.012463 | -0.009871 | -1.8187 | -0.1318 | -0.1318 | 不适用 |
| ISIC | 1 | fit_only | -0.004655 | -0.010181 | +0.2284 | -0.6757 | -0.6757 | native |
| ISIC | 2 | all | +0.012750 | -0.000727 | -2.7730 | +3.0100 | +2.0164 | 不适用 |
| ISIC | 2 | fit_only | +0.003066 | -0.001669 | -2.4263 | +3.0100 | +0.1024 | static_pc |
| ISIC | 3 | all | +0.011141 | +0.013405 | -1.0623 | -2.3864 | +0.7549 | 不适用 |
| ISIC | 3 | fit_only | +0.008836 | +0.013212 | -1.4966 | -1.2500 | +0.4295 | static_pc |
| HK | 1 | all | +0.008237 | -0.009694 | -0.5843 | -0.9732 | -0.9732 | 不适用 |
| HK | 1 | fit_only | -0.005706 | -0.011422 | +0.2950 | +0.0768 | +0.0768 | native |
| HK | 2 | all | +0.017356 | +0.007487 | +0.0000 | +0.0000 | -0.9173 | 不适用 |
| HK | 2 | fit_only | +0.003459 | +0.007240 | +0.0000 | +0.0000 | -0.9727 | static_pc |
| HK | 3 | all | +0.027482 | +0.005677 | +0.0000 | +0.0000 | -1.5444 | 不适用 |
| HK | 3 | fit_only | -0.014408 | -0.002022 | +0.0000 | +0.0000 | -1.2996 | native |
| HK | 4 | all | +0.012530 | +0.008195 | -0.3937 | -3.0303 | -1.2412 | 不适用 |
| HK | 4 | fit_only | +0.006691 | +0.007563 | +0.0000 | -3.0303 | -0.8311 | static_pc |
| HK | 5 | all | +0.017555 | -0.002588 | +0.0000 | +0.0000 | -1.3301 | 不适用 |
| HK | 5 | fit_only | -0.005846 | -0.004075 | +0.7576 | +12.5000 | -0.2982 | native |
| HK | 6 | all | +0.011813 | +0.011592 | -0.3876 | -1.5924 | -1.1297 | 不适用 |
| HK | 6 | fit_only | +0.010975 | +0.011514 | -0.3876 | -1.5924 | -1.3261 | static_pc |

## 逐类方向计数

同向包含双零变化；另列双非零条件下同向，避免将不变预测作为动作成功。

| 数据集 | 阶段 | bank | 指标 | 同向/类数 | 双非零同向/双非零 | 双零 |
|---|---:|---|---|---:|---:|---:|
| ISIC | 1 | all | balanced_accuracy | 3/4 | 2/3 | 1 |
| ISIC | 1 | all | macro_square_risk | 0/4 | 0/4 | 0 |
| ISIC | 1 | fit_only | balanced_accuracy | 3/4 | 1/1 | 2 |
| ISIC | 1 | fit_only | macro_square_risk | 3/4 | 3/4 | 0 |
| ISIC | 2 | all | balanced_accuracy | 0/2 | 0/1 | 0 |
| ISIC | 2 | all | macro_square_risk | 1/2 | 1/2 | 0 |
| ISIC | 2 | fit_only | balanced_accuracy | 0/2 | 0/1 | 0 |
| ISIC | 2 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
| ISIC | 3 | all | balanced_accuracy | 2/2 | 2/2 | 0 |
| ISIC | 3 | all | macro_square_risk | 2/2 | 2/2 | 0 |
| ISIC | 3 | fit_only | balanced_accuracy | 1/2 | 1/1 | 0 |
| ISIC | 3 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 1 | all | balanced_accuracy | 12/13 | 4/4 | 8 |
| HK | 1 | all | macro_square_risk | 8/13 | 8/13 | 0 |
| HK | 1 | fit_only | balanced_accuracy | 10/13 | 3/3 | 7 |
| HK | 1 | fit_only | macro_square_risk | 12/13 | 12/13 | 0 |
| HK | 2 | all | balanced_accuracy | 2/2 | 0/0 | 2 |
| HK | 2 | all | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 2 | fit_only | balanced_accuracy | 2/2 | 0/0 | 2 |
| HK | 2 | fit_only | macro_square_risk | 1/2 | 1/2 | 0 |
| HK | 3 | all | balanced_accuracy | 2/2 | 0/0 | 2 |
| HK | 3 | all | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 3 | fit_only | balanced_accuracy | 2/2 | 0/0 | 2 |
| HK | 3 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 4 | all | balanced_accuracy | 0/2 | 0/0 | 0 |
| HK | 4 | all | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 4 | fit_only | balanced_accuracy | 1/2 | 0/0 | 1 |
| HK | 4 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 5 | all | balanced_accuracy | 2/2 | 0/0 | 2 |
| HK | 5 | all | macro_square_risk | 1/2 | 1/2 | 0 |
| HK | 5 | fit_only | balanced_accuracy | 0/2 | 0/0 | 0 |
| HK | 5 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 6 | all | balanced_accuracy | 2/2 | 1/1 | 1 |
| HK | 6 | all | macro_square_risk | 2/2 | 2/2 | 0 |
| HK | 6 | fit_only | balanced_accuracy | 2/2 | 1/1 | 1 |
| HK | 6 | fit_only | macro_square_risk | 2/2 | 2/2 | 0 |
