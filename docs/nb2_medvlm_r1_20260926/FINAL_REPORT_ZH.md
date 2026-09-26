# NB2-MedVLM-R1 最终报告

G/B 主矩阵完成；BiomedCLIP 对 Generic 主效用门槛：FAIL。

本轮是既有开发数据上的固定医学 VLM 对照；不是独立确认或临床验证。新增神经训练与 optimizer steps 均为 0，test/reserved 访问为 0。
本轮 G.J 使用 λ=5e-4，原 R1 A2 对应 G.J1（λ=1e-3）；旧 R1 保持原样。
BiomedCLIP 使用官方视觉编码器、PubMedBERT 文本编码器、context=256 tokenizer 与预处理。预训练直接样本暴露 UNKNOWN。

## 各方法三顺序均值

| 数据集 | 方法 | Final BA (%) | Macro-F1 (%) | AvgBA_inc (%) |
|---|---|---:|---:|---:|
| ISIC | P | 55.497 | 55.560 | 61.703 |
| ISIC | F | 58.276 | 58.721 | 63.970 |
| ISIC | F2 | 58.298 | 58.294 | 63.328 |
| ISIC | G.V | 52.937 | 51.853 | 60.177 |
| ISIC | G.J | 59.055 | 59.845 | 64.413 |
| ISIC | G.J1 | 58.944 | 59.380 | 64.647 |
| ISIC | G.Z | 17.113 | 13.315 | 22.031 |
| ISIC | G.L | 59.232 | 59.889 | 64.325 |
| ISIC | G.R | 59.055 | 59.845 | 64.529 |
| ISIC | G.R-no-gate | 59.146 | 59.887 | 64.912 |
| ISIC | G.R-identity | 59.055 | 59.845 | 64.529 |
| ISIC | G.R-zero-prior | 59.055 | 59.845 | 64.529 |
| ISIC | G.L-permute | 59.068 | 59.843 | 64.616 |
| ISIC | G.template-0 | 16.248 | 11.855 | 21.034 |
| ISIC | G.template-1 | 18.609 | 16.033 | 23.748 |
| ISIC | B.V | 51.084 | 49.460 | 55.914 |
| ISIC | B.J | 61.001 | 62.024 | 64.091 |
| ISIC | B.J1 | 60.112 | 60.920 | 64.202 |
| ISIC | B.Z | 23.239 | 22.128 | 31.949 |
| ISIC | B.L | 61.274 | 62.217 | 64.184 |
| ISIC | B.R | 60.888 | 61.910 | 64.053 |
| ISIC | B.R-no-gate | 61.110 | 62.092 | 64.052 |
| ISIC | B.R-identity | 60.888 | 61.910 | 64.053 |
| ISIC | B.R-zero-prior | 61.001 | 62.024 | 64.091 |
| ISIC | B.L-permute | 60.332 | 61.323 | 64.060 |
| ISIC | B.template-0 | 29.219 | 24.471 | 34.505 |
| ISIC | B.template-1 | 18.933 | 16.649 | 26.825 |
| HK | P | 61.128 | 55.525 | 73.643 |
| HK | F | 61.213 | 55.646 | 73.572 |
| HK | F2 | 60.216 | 54.292 | 72.966 |
| HK | G.V | 58.211 | 52.033 | 71.771 |
| HK | G.J | 62.756 | 57.628 | 74.697 |
| HK | G.J1 | 61.865 | 56.217 | 74.661 |
| HK | G.Z | 6.100 | 2.146 | 10.695 |
| HK | G.L | 62.802 | 57.592 | 74.909 |
| HK | G.R | 62.765 | 57.640 | 74.698 |
| HK | G.R-no-gate | 63.159 | 57.833 | 74.839 |
| HK | G.R-identity | 62.765 | 57.640 | 74.698 |
| HK | G.R-zero-prior | 62.765 | 57.640 | 74.698 |
| HK | G.L-permute | 63.001 | 57.918 | 74.781 |
| HK | G.template-0 | 5.845 | 2.220 | 11.466 |
| HK | G.template-1 | 6.911 | 2.353 | 12.550 |
| HK | B.V | 59.112 | 47.599 | 68.306 |
| HK | B.J | 61.185 | 55.836 | 73.118 |
| HK | B.J1 | 59.847 | 54.068 | 72.814 |
| HK | B.Z | 27.477 | 23.636 | 45.163 |
| HK | B.L | 60.724 | 55.937 | 73.047 |
| HK | B.R | 61.161 | 55.801 | 73.108 |
| HK | B.R-no-gate | 60.317 | 55.480 | 72.987 |
| HK | B.R-identity | 61.162 | 55.791 | 73.110 |
| HK | B.R-zero-prior | 61.161 | 55.800 | 73.105 |
| HK | B.L-permute | 60.011 | 54.725 | 71.957 |
| HK | B.template-0 | 28.049 | 24.100 | 45.688 |
| HK | B.template-1 | 25.900 | 21.439 | 44.021 |

## 固定门槛

- ISIC B.J − G.J：+1.946 pp，FAIL；未通过项：avg_ba_inc_nonnegative。
- ISIC B.J − F：+2.725 pp，FAIL；未通过项：final_tail_nonnegative。
- HK B.J − G.J：-1.571 pp，FAIL；未通过项：final_ba_mean_gain_ge_1pp, at_least_two_orders_improve, avg_ba_inc_nonnegative, final_tail_nonnegative, no_new_zero_recall。
- HK B.J − F：-0.028 pp，FAIL；未通过项：final_ba_mean_gain_ge_1pp, avg_ba_inc_nonnegative, final_tail_nonnegative, no_new_zero_recall。

## 机制比较与区间

| 数据集 | 比较 | Final BA 差值 (pp) | 固定模型条件区间 |
|---|---|---:|---|
| ISIC | B.J − G.J | +1.946 | [-1.163, +5.208] |
| ISIC | B.J − F | +2.725 | [+0.002, +5.591] |
| ISIC | B.V − G.V | -1.853 | [-7.532, +4.111] |
| ISIC | B.Z − G.Z | +6.126 | [+0.450, +11.737] |
| ISIC | G.L − G.J | +0.176 | [-1.053, +1.635] |
| ISIC | G.R − G.J | +0.000 | [+0.000, +0.000] |
| ISIC | G.J − F | +0.780 | [-2.095, +3.375] |
| ISIC | G.J1 − F2 | +0.646 | [-1.735, +2.933] |
| ISIC | G.R-no-gate − G.R | +0.091 | [-0.324, +0.523] |
| ISIC | G.R-identity − G.R | +0.000 | [+0.000, +0.000] |
| ISIC | G.R-zero-prior − G.R | +0.000 | [+0.000, +0.000] |
| ISIC | G.L − G.L-permute | +0.164 | [-1.409, +1.973] |
| ISIC | G.template-0 − G.template-1 | -2.360 | [-5.296, +0.582] |
| ISIC | B.L − B.J | +0.274 | [-1.068, +1.565] |
| ISIC | B.R − B.J | -0.113 | [-0.347, +0.000] |
| ISIC | B.J1 − F2 | +1.814 | [-0.777, +4.504] |
| ISIC | B.R-no-gate − B.R | +0.221 | [-0.225, +0.756] |
| ISIC | B.R-identity − B.R | +0.000 | [+0.000, +0.000] |
| ISIC | B.R-zero-prior − B.R | +0.113 | [+0.000, +0.347] |
| ISIC | B.L − B.L-permute | +0.943 | [-0.861, +2.712] |
| ISIC | B.template-0 − B.template-1 | +10.286 | [+4.600, +15.899] |
| HK | B.J − G.J | -1.571 | [-4.021, +0.913] |
| HK | B.J − F | -0.028 | [-2.076, +1.928] |
| HK | B.V − G.V | +0.901 | [-3.814, +5.647] |
| HK | B.Z − G.Z | +21.377 | [+17.692, +25.055] |
| HK | G.L − G.J | +0.046 | [-0.557, +0.599] |
| HK | G.R − G.J | +0.010 | [+0.000, +0.029] |
| HK | G.J − F | +1.543 | [-0.604, +3.832] |
| HK | G.J1 − F2 | +1.650 | [-0.490, +3.786] |
| HK | G.R-no-gate − G.R | +0.393 | [-0.160, +1.297] |
| HK | G.R-identity − G.R | +0.000 | [+0.000, +0.000] |
| HK | G.R-zero-prior − G.R | +0.000 | [+0.000, +0.000] |
| HK | G.L − G.L-permute | -0.199 | [-0.810, +0.417] |
| HK | G.template-0 − G.template-1 | -1.066 | [-2.522, +0.872] |
| HK | B.L − B.J | -0.461 | [-2.568, +1.083] |
| HK | B.R − B.J | -0.024 | [-0.071, +0.000] |
| HK | B.J1 − F2 | -0.369 | [-2.292, +1.326] |
| HK | B.R-no-gate − B.R | -0.845 | [-2.163, +0.270] |
| HK | B.R-identity − B.R | +0.001 | [-0.132, +0.106] |
| HK | B.R-zero-prior − B.R | +0.000 | [+0.000, +0.000] |
| HK | B.L − B.L-permute | +0.713 | [-1.714, +3.009] |
| HK | B.template-0 − B.template-1 | +2.149 | [-0.698, +4.939] |

条件区间仅重采样已有验证身份组件，不覆盖训练随机性或未见领域；跨零表示证据较弱。HK 极小尾类的实际分母见 class_metrics.csv。
完整阶段、类别、遗忘与纠错/破坏分解均已公开；私有 logits、图像 ID、父状态、统计 bank/W 留在规定服务器存储。
DermLIP：BLOCKED_ACCESS_LICENSE。当前官方访问需帐号授权，模型卡许可证字段与正文不一致，未满足探索项执行条件。
独立备份状态见 backup_report.json；NEXT_DECISION=STOP。
