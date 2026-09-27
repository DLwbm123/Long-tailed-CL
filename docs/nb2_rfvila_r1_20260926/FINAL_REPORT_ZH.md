# NB2-RFVILA-R1 最终报告

本轮是 VILA 风格医学适配，不是论文原始复现。正常两类 Task1 后全部冻结，新增神经训练和 optimizer steps 为 0。
主投影 67101；67102 独立报告，不挑赢家、不平均 logits。所有比较均使用相同 Task1 九点正则选择预算；CV 表征已学习 Task1，不是端到端独立验证。

已准入矩阵完成：32 方法，1440 阶段行，14688 逐类行。

| 数据集 | 方法 | Final BA % | Macro-F1 % | AvgBA_inc % |
|---|---|---:|---:|---:|
| ISIC | F.LIN | 57.459 | 57.194 | 63.046 |
| ISIC | G.LIN | 57.044 | 56.829 | 63.501 |
| ISIC | B.LIN | 60.562 | 61.468 | 64.678 |
| ISIC | F.RF1 | 55.048 | 54.566 | 62.019 |
| ISIC | G.RF1 | 56.163 | 55.920 | 62.617 |
| ISIC | B.RF1 | 59.978 | 60.850 | 64.006 |
| ISIC | G.LIN.CSE1 | 56.864 | 57.024 | 63.776 |
| ISIC | G.LIN.CSE025 | 57.162 | 57.254 | 63.623 |
| ISIC | G.RF1.CSE1 | 55.840 | 55.927 | 62.377 |
| ISIC | G.RF1.CSE025 | 56.217 | 56.146 | 62.641 |
| ISIC | B.LIN.CSE1 | 60.910 | 61.623 | 65.338 |
| ISIC | B.LIN.CSE025 | 60.474 | 61.352 | 64.895 |
| ISIC | B.RF1.CSE1 | 59.626 | 60.514 | 64.057 |
| ISIC | B.RF1.CSE025 | 59.277 | 60.111 | 63.808 |
| ISIC | F.LOCK1e3 | 58.276 | 58.721 | 63.970 |
| ISIC | G.LOCK5e4 | 59.055 | 59.845 | 64.413 |
| ISIC | G.LOCK1e3 | 58.944 | 59.380 | 64.647 |
| ISIC | B.LOCK5e4 | 61.001 | 62.024 | 64.091 |
| ISIC | B.LOCK1e3 | 60.112 | 60.920 | 64.202 |
| ISIC | F.RF2 | 57.713 | 58.470 | 63.125 |
| ISIC | G.RF2 | 57.822 | 58.832 | 63.662 |
| ISIC | B.RF2 | 58.585 | 59.414 | 62.863 |
| ISIC | G.RF2.CSE1 | 58.168 | 59.150 | 63.625 |
| ISIC | G.RF2.CSE025 | 57.806 | 58.776 | 63.690 |
| ISIC | B.RF2.CSE1 | 58.539 | 59.267 | 63.180 |
| ISIC | B.RF2.CSE025 | 58.159 | 59.009 | 62.614 |
| ISIC | G.SPLIT1 | 56.489 | 56.310 | 62.466 |
| ISIC | B.SPLIT1 | 58.239 | 59.225 | 62.756 |
| ISIC | G.RPLIN1 | 56.996 | 56.840 | 63.598 |
| ISIC | B.RPLIN1 | 60.646 | 61.615 | 64.724 |
| ISIC | G.RF1.CSEperm1 | 55.271 | 54.622 | 62.238 |
| ISIC | B.RF1.CSEperm1 | 59.882 | 60.687 | 64.214 |
| HK | F.LIN | 62.418 | 57.373 | 74.061 |
| HK | G.LIN | 62.892 | 57.927 | 75.092 |
| HK | B.LIN | 61.849 | 56.639 | 73.399 |
| HK | F.RF1 | 63.437 | 58.211 | 74.480 |
| HK | G.RF1 | 62.138 | 57.476 | 74.372 |
| HK | B.RF1 | 62.279 | 57.507 | 73.593 |
| HK | G.LIN.CSE1 | 63.633 | 58.292 | 75.284 |
| HK | G.LIN.CSE025 | 62.873 | 57.881 | 75.083 |
| HK | G.RF1.CSE1 | 62.783 | 57.698 | 74.546 |
| HK | G.RF1.CSE025 | 62.154 | 57.395 | 74.430 |
| HK | B.LIN.CSE1 | 61.339 | 56.399 | 72.998 |
| HK | B.LIN.CSE025 | 61.657 | 56.594 | 73.413 |
| HK | B.RF1.CSE1 | 62.388 | 57.556 | 73.798 |
| HK | B.RF1.CSE025 | 62.114 | 57.388 | 73.580 |
| HK | F.LOCK1e3 | 61.213 | 55.646 | 73.572 |
| HK | G.LOCK5e4 | 62.756 | 57.628 | 74.697 |
| HK | G.LOCK1e3 | 61.865 | 56.217 | 74.661 |
| HK | B.LOCK5e4 | 61.185 | 55.836 | 73.118 |
| HK | B.LOCK1e3 | 59.847 | 54.068 | 72.814 |
| HK | F.RF2 | 62.775 | 57.773 | 73.998 |
| HK | G.RF2 | 63.395 | 58.520 | 74.783 |
| HK | B.RF2 | 60.978 | 58.174 | 70.786 |
| HK | G.RF2.CSE1 | 63.330 | 58.550 | 74.904 |
| HK | G.RF2.CSE025 | 63.335 | 58.451 | 74.840 |
| HK | B.RF2.CSE1 | 60.594 | 57.837 | 70.397 |
| HK | B.RF2.CSE025 | 60.998 | 58.208 | 70.812 |
| HK | G.SPLIT1 | 64.613 | 59.884 | 75.150 |
| HK | B.SPLIT1 | 62.279 | 58.131 | 73.592 |
| HK | G.RPLIN1 | 63.256 | 57.961 | 74.673 |
| HK | B.RPLIN1 | 61.408 | 56.265 | 73.450 |
| HK | G.RF1.CSEperm1 | 63.032 | 58.093 | 74.543 |
| HK | B.RF1.CSEperm1 | 60.926 | 55.967 | 72.362 |

## RF 对公平 LIN 的固定效用门槛

- ISIC G.RF1 − G.LIN: -0.881 pp，FAIL；未通过：final_ba_mean_gain_ge_1pp, at_least_two_orders_improve, avg_ba_inc_nonnegative。
- ISIC B.RF1 − B.LIN: -0.584 pp，FAIL；未通过：final_ba_mean_gain_ge_1pp, at_least_two_orders_improve, avg_ba_inc_nonnegative, final_tail_nonnegative。
- HK G.RF1 − G.LIN: -0.754 pp，FAIL；未通过：final_ba_mean_gain_ge_1pp, at_least_two_orders_improve, avg_ba_inc_nonnegative, final_tail_nonnegative。
- HK B.RF1 − B.LIN: +0.430 pp，FAIL；未通过：final_ba_mean_gain_ge_1pp, at_least_two_orders_improve, final_tail_nonnegative。

## 固定机制比较

| 数据集 | 比较 | Final BA 差 pp | 条件区间 |
|---|---|---:|---|
| ISIC | F.RF1 - F.LIN | -2.412 | [-4.067, -0.899] |
| ISIC | G.RF1 - G.LIN | -0.881 | [-2.850, +1.079] |
| ISIC | B.RF1 - B.LIN | -0.584 | [-2.202, +1.098] |
| ISIC | G.RF1 - F.RF1 | +1.115 | [-1.081, +3.382] |
| ISIC | B.RF1 - F.RF1 | +4.930 | [+2.466, +7.484] |
| ISIC | G.RF1 - F.LIN | -1.296 | [-3.462, +0.828] |
| ISIC | B.RF1 - F.LIN | +2.518 | [-0.366, +5.298] |
| ISIC | B.RF1 - G.RF1 | +3.815 | [+0.204, +7.192] |
| ISIC | B.LIN - G.LIN | +3.517 | [+0.352, +6.658] |
| ISIC | G.RF1 - G.SPLIT1 | -0.326 | [-1.790, +1.139] |
| ISIC | G.RF1 - G.RPLIN1 | -0.833 | [-2.451, +0.864] |
| ISIC | G.RF2 - G.LIN | +0.778 | [-0.972, +2.505] |
| ISIC | G.LIN.CSE1 - G.LIN | -0.180 | [-1.575, +1.222] |
| ISIC | G.LIN.CSE025 - G.LIN | +0.118 | [-0.379, +0.639] |
| ISIC | G.RF1.CSE1 - G.RF1 | -0.323 | [-1.831, +1.218] |
| ISIC | G.RF1.CSE025 - G.RF1 | +0.054 | [-0.605, +0.690] |
| ISIC | G.RF2.CSE1 - G.RF2 | +0.346 | [-0.797, +1.583] |
| ISIC | G.RF2.CSE025 - G.RF2 | -0.016 | [-0.514, +0.469] |
| ISIC | G.RF1.CSEperm1 - G.RF1 | -0.892 | [-2.348, +0.699] |
| ISIC | G.RF1.CSE1 - G.RF1.CSEperm1 | +0.568 | [-0.926, +1.935] |
| ISIC | B.RF1 - B.SPLIT1 | +1.738 | [+0.051, +3.537] |
| ISIC | B.RF1 - B.RPLIN1 | -0.669 | [-2.577, +1.088] |
| ISIC | B.RF2 - B.LIN | -1.977 | [-3.933, -0.036] |
| ISIC | B.LIN.CSE1 - B.LIN | +0.348 | [-1.229, +1.968] |
| ISIC | B.LIN.CSE025 - B.LIN | -0.087 | [-0.637, +0.547] |
| ISIC | B.RF1.CSE1 - B.RF1 | -0.352 | [-2.004, +1.084] |
| ISIC | B.RF1.CSE025 - B.RF1 | -0.701 | [-1.373, -0.100] |
| ISIC | B.RF2.CSE1 - B.RF2 | -0.045 | [-1.554, +1.512] |
| ISIC | B.RF2.CSE025 - B.RF2 | -0.426 | [-1.240, +0.324] |
| ISIC | B.RF1.CSEperm1 - B.RF1 | -0.096 | [-1.419, +1.210] |
| ISIC | B.RF1.CSE1 - B.RF1.CSEperm1 | -0.256 | [-1.961, +1.324] |
| ISIC | F.meanRP - F.LIN | -1.079 | [-2.680, +0.382] |
| ISIC | G.meanRP - G.LIN | -0.052 | [-1.580, +1.497] |
| ISIC | B.meanRP - B.LIN | -1.280 | [-2.821, +0.240] |
| ISIC | G.RF1 - F.RF1 - (G.LIN - F.LIN) | +1.530 | [-0.748, +3.930] |
| ISIC | G.meanRP - F.meanRP | +0.612 | [-1.450, +2.597] |
| ISIC | B.RF1 - F.RF1 - (B.LIN - F.LIN) | +1.828 | [-0.169, +3.856] |
| ISIC | B.meanRP - F.meanRP | +2.901 | [+0.577, +5.127] |
| ISIC | B.meanRP - G.meanRP | +2.289 | [-0.969, +5.254] |
| HK | F.RF1 - F.LIN | +1.019 | [-0.422, +2.718] |
| HK | G.RF1 - G.LIN | -0.754 | [-1.410, -0.144] |
| HK | B.RF1 - B.LIN | +0.430 | [-0.673, +1.589] |
| HK | G.RF1 - F.RF1 | -1.299 | [-3.464, +0.614] |
| HK | B.RF1 - F.RF1 | -1.158 | [-2.882, +0.330] |
| HK | G.RF1 - F.LIN | -0.280 | [-1.957, +1.409] |
| HK | B.RF1 - F.LIN | -0.139 | [-1.752, +1.549] |
| HK | B.RF1 - G.RF1 | +0.142 | [-2.218, +2.568] |
| HK | B.LIN - G.LIN | -1.043 | [-3.289, +1.375] |
| HK | G.RF1 - G.SPLIT1 | -2.475 | [-4.458, -0.757] |
| HK | G.RF1 - G.RPLIN1 | -1.118 | [-2.009, -0.338] |
| HK | G.RF2 - G.LIN | +0.503 | [-0.833, +1.791] |
| HK | G.LIN.CSE1 - G.LIN | +0.741 | [+0.024, +1.495] |
| HK | G.LIN.CSE025 - G.LIN | -0.019 | [-0.175, +0.153] |
| HK | G.RF1.CSE1 - G.RF1 | +0.645 | [+0.001, +1.455] |
| HK | G.RF1.CSE025 - G.RF1 | +0.016 | [-0.134, +0.176] |
| HK | G.RF2.CSE1 - G.RF2 | -0.064 | [-0.677, +0.482] |
| HK | G.RF2.CSE025 - G.RF2 | -0.060 | [-0.333, +0.183] |
| HK | G.RF1.CSEperm1 - G.RF1 | +0.894 | [-0.165, +2.252] |
| HK | G.RF1.CSE1 - G.RF1.CSEperm1 | -0.249 | [-1.098, +0.528] |
| HK | B.RF1 - B.SPLIT1 | -0.000 | [-0.985, +1.094] |
| HK | B.RF1 - B.RPLIN1 | +0.871 | [-0.068, +1.802] |
| HK | B.RF2 - B.LIN | -0.871 | [-2.896, +0.957] |
| HK | B.LIN.CSE1 - B.LIN | -0.510 | [-1.369, +0.241] |
| HK | B.LIN.CSE025 - B.LIN | -0.192 | [-0.821, +0.249] |
| HK | B.RF1.CSE1 - B.RF1 | +0.108 | [-1.636, +1.775] |
| HK | B.RF1.CSE025 - B.RF1 | -0.165 | [-0.785, +0.283] |
| HK | B.RF2.CSE1 - B.RF2 | -0.385 | [-1.575, +0.382] |
| HK | B.RF2.CSE025 - B.RF2 | +0.020 | [-0.100, +0.133] |
| HK | B.RF1.CSEperm1 - B.RF1 | -1.353 | [-2.373, -0.352] |
| HK | B.RF1.CSE1 - B.RF1.CSEperm1 | +1.462 | [-0.179, +3.187] |
| HK | F.meanRP - F.LIN | +0.688 | [-0.278, +1.719] |
| HK | G.meanRP - G.LIN | -0.126 | [-0.990, +0.733] |
| HK | B.meanRP - B.LIN | -0.220 | [-1.553, +1.048] |
| HK | G.RF1 - F.RF1 - (G.LIN - F.LIN) | -1.773 | [-3.819, -0.078] |
| HK | G.meanRP - F.meanRP | -0.340 | [-2.263, +1.377] |
| HK | B.RF1 - F.RF1 - (B.LIN - F.LIN) | -0.589 | [-2.876, +1.388] |
| HK | B.meanRP - F.meanRP | -1.477 | [-2.970, -0.086] |
| HK | B.meanRP - G.meanRP | -1.137 | [-3.195, +0.990] |

RF−LIN 检验非线性读出；RF−F.RF 检验同容量下 VLM 的附加信息；B−G 检验固定系统的差异，不能单因果归于医学语料。
CSE 的净贡献与真实类别 Top-K、纠错/破坏见 candidate_coverage.csv；两个 RP 的方向一致性见 projection_robustness.csv。
区间按 identity component 成对重采样 2000 次；固定父状态与投影，不覆盖训练随机性或未见领域。多重开发、极小尾类和 UNKNOWN 预训练暴露限制保留。
未准入项见 METHOD_MATRIX.csv；所有私有图像、身份/组件映射、逐样本特征/分数和 S/Q/W/R 只留规定存储。关键资产目标端 SHA 已核验，大图像归档未声明全量逐字节一致。
NEXT_DECISION=STOP。

本轮曾因本机 SSH 备份中继中断而到达原时限。用户随后授权取消时间预算，从已封存的 10 阶段续跑；原始 T0 和失败记录未重置。服务器端备份中继替代本机中继，模型、方法、数据顺序及超参数保持不变。
