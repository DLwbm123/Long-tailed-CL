# FEEDBACK-INFORMATION1 完整汇总

BA 与增量均以百分点表示。A/B 仅覆盖身份足够的类别；full 包含全部已见类别。所有 oracle 都是事后诊断，不是可部署策略。

## ISIC_T2_cached

A/B 排除类别：[]

| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_00 | 60.8787 | 56.4519 | 69.7324 | 58.4009 | 0.0000 | 2.0765 | 2.0765 |
| linear_ridge | 61.4758 | 56.3860 | 71.6555 | 59.2966 | -0.5971 | 2.0765 | 1.4794 |
| prototype_ridge | 61.4758 | 56.3860 | 71.6555 | 59.2966 | -0.5971 | 2.0765 | 1.4794 |

### full

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_13 | 2.0765 | 3.1147 | 0.0000 | 3.1147 | 5 | 0 |
| empirical_A_norm | norm_05 | 1.8605 | 1.8293 | 1.9231 | 2.7908 | 4 | 0 |
| gaussian_A_edge | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_13 | 2.0765 | 0 |
| edge | group | edge_13 | 2.0765 | 0 |
| edge | per_class | edge_13 | 2.0765 | 0 |
| edge | per_sample | edge_13 | 2.0765 | 0 |
| norm | unconstrained | norm_05 | 1.8605 | 0 |
| norm | group | norm_05 | 1.8605 | 0 |
| norm | per_class | norm_05 | 1.8605 | 0 |
| norm | per_sample | norm_05 | 1.8605 | 0 |

### A

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_13 | 3.3772 | 5.0658 | 0.0000 | 5.0658 | 4 | 0 |
| empirical_A_norm | norm_05 | 2.9487 | 2.5000 | 3.8462 | 4.4231 | 3 | 0 |
| gaussian_A_edge | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_13 | 3.3772 | 0 |
| edge | group | edge_13 | 3.3772 | 0 |
| edge | per_class | edge_13 | 3.3772 | 0 |
| edge | per_sample | edge_13 | 3.3772 | 0 |
| norm | unconstrained | norm_05 | 2.9487 | 0 |
| norm | group | norm_05 | 2.9487 | 0 |
| norm | per_class | norm_05 | 2.9487 | 0 |
| norm | per_sample | norm_05 | 2.9487 | 0 |

### B

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_13 | 0.7937 | 1.1905 | 0.0000 | 1.1905 | 1 | 0 |
| empirical_A_norm | norm_05 | 0.7937 | 1.1905 | 0.0000 | 1.1905 | 1 | 0 |
| gaussian_A_edge | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_01 | 0.7937 | 0 |
| edge | group | edge_01 | 0.7937 | 0 |
| edge | per_class | edge_01 | 0.7937 | 0 |
| edge | per_sample | edge_01 | 0.7937 | 0 |
| norm | unconstrained | norm_15 | 2.5132 | 1 |
| norm | group | norm_15 | 2.5132 | 1 |
| norm | per_class | norm_04 | 0.7937 | 0 |
| norm | per_sample | norm_04 | 0.7937 | 0 |

| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |
|---|---|---:|---:|---:|
| gaussian_A_vs_empirical_A | edge | 6.4294 | 1.0910 | 0.1809806914982567 |
| gaussian_A_vs_empirical_A | norm | 5.5525 | 4.2639 | -0.04588043047057028 |
| empirical_A_vs_B | edge | 9.9496 | 0.7987 | 0.4458570174826006 |
| empirical_A_vs_B | norm | 9.3555 | 3.4172 | 0.07795077513677329 |
| gaussian_A_vs_B | edge | 6.5520 | 0.7751 | 0.0 |
| gaussian_A_vs_B | norm | 8.0843 | 2.4665 | -0.9783188862212846 |
## ISIC_T3

A/B 排除类别：[]

| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_00 | 55.8387 | 50.7773 | 71.0227 | 50.8282 | 0.0000 | 3.2271 | 3.2271 |
| linear_ridge | 56.6060 | 52.2171 | 69.7727 | 53.3990 | -0.7673 | 3.2271 | 2.4599 |
| prototype_ridge | 56.2681 | 51.7666 | 69.7727 | 53.3990 | -0.4295 | 3.2271 | 2.7977 |

### full

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.2717 | -0.3623 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_07 | -0.2717 | -0.3623 | 0.0000 | 0.0000 | 0 | 1 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 1.5574 | 2.0765 | 0.0000 | 3.1147 | 5 | 0 |
| empirical_A_norm | norm_07 | 2.1461 | 2.4827 | 1.1364 | 1.8293 | 8 | 1 |
| gaussian_A_edge | edge_13 | 3.2271 | 3.8483 | 1.3636 | 8.0786 | 14 | 5 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_13 | 3.2271 | 5 |
| edge | group | edge_13 | 3.2271 | 5 |
| edge | per_class | edge_01 | 1.5574 | 0 |
| edge | per_sample | edge_01 | 1.5574 | 0 |
| norm | unconstrained | norm_07 | 2.1461 | 1 |
| norm | group | norm_07 | 2.1461 | 1 |
| norm | per_class | norm_07 | 2.1461 | 1 |
| norm | per_sample | norm_03 | 1.2393 | 0 |

### A

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.5208 | -0.6944 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_07 | -0.5208 | -0.6944 | 0.0000 | 0.0000 | 0 | 1 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 0.6250 | 0.8333 | 0.0000 | 1.2500 | 1 | 0 |
| empirical_A_norm | norm_07 | 1.8037 | 1.5716 | 2.5000 | 0.0000 | 3 | 0 |
| gaussian_A_edge | edge_13 | 0.2323 | -0.5994 | 2.7273 | 2.5000 | 4 | 4 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_03 | 0.6963 | 2 |
| edge | group | edge_01 | 0.6250 | 0 |
| edge | per_class | edge_01 | 0.6250 | 0 |
| edge | per_sample | edge_01 | 0.6250 | 0 |
| norm | unconstrained | norm_07 | 1.8037 | 0 |
| norm | group | norm_07 | 1.8037 | 0 |
| norm | per_class | norm_07 | 1.8037 | 0 |
| norm | per_sample | norm_07 | 1.8037 | 0 |

### B

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_decision | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 2.4802 | 3.3069 | 0.0000 | 4.9603 | 4 | 0 |
| empirical_A_norm | norm_07 | 2.4233 | 3.3069 | -0.2273 | 3.5714 | 5 | 1 |
| gaussian_A_edge | edge_13 | 6.3267 | 8.4355 | 0.0000 | 13.7897 | 10 | 1 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_13 | 6.3267 | 1 |
| edge | group | edge_13 | 6.3267 | 1 |
| edge | per_class | edge_01 | 2.4802 | 0 |
| edge | per_sample | edge_01 | 2.4802 | 0 |
| norm | unconstrained | norm_12 | 2.4928 | 3 |
| norm | group | norm_03 | 1.8579 | 0 |
| norm | per_class | norm_03 | 1.8579 | 0 |
| norm | per_sample | norm_03 | 1.8579 | 0 |

| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |
|---|---|---:|---:|---:|
| gaussian_A_vs_empirical_A | edge | 5.0086 | 1.1818 | -0.28122049591070253 |
| gaussian_A_vs_empirical_A | norm | 4.7776 | 2.0479 | 0.6777676605061005 |
| empirical_A_vs_B | edge | 11.4523 | 1.6760 | 0.12095699073350799 |
| empirical_A_vs_B | norm | 11.2270 | 3.8991 | -0.11905519106126237 |
| gaussian_A_vs_B | edge | 12.8690 | 1.4663 | 0.6944776605898034 |
| gaussian_A_vs_B | norm | 12.5073 | 3.4247 | -0.5670584359333394 |
## HK_T2_cached

A/B 排除类别：[5, 6, 19]

| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_00 | 66.1108 | 62.8201 | 87.5000 | 53.5431 | 0.0000 | 0.3743 | 0.3743 |
| linear_ridge | 65.1747 | 61.7400 | 87.5000 | 53.6397 | 0.9361 | 0.3743 | 1.3104 |
| prototype_ridge | 65.1747 | 61.7400 | 87.5000 | 53.6397 | 0.9361 | 0.3743 | 1.3104 |

### full

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.1741 | -0.2009 | 0.0000 | 0.0000 | 1 | 2 |
| original_decision | edge_04 | 0.1177 | 0.1358 | 0.0000 | 0.0000 | 3 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 0.1177 | 0.1358 | 0.0000 | 0.0000 | 3 | 0 |
| empirical_A_norm | norm_05 | -0.5944 | 0.2757 | -6.2500 | -1.5625 | 4 | 4 |
| gaussian_A_edge | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.3743 | 1 |
| edge | group | edge_14 | 0.3743 | 1 |
| edge | per_class | edge_14 | 0.3743 | 1 |
| edge | per_sample | edge_01 | 0.1177 | 0 |
| norm | unconstrained | norm_03 | 0.1093 | 1 |
| norm | group | norm_03 | 0.1093 | 1 |
| norm | per_class | norm_03 | 0.1093 | 1 |
| norm | per_sample | norm_01 | 0.0444 | 0 |

### A

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.2688 | -0.3226 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_04 | 0.2943 | 0.3531 | 0.0000 | 0.0000 | 3 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 0.2943 | 0.3531 | 0.0000 | 0.0000 | 3 | 0 |
| empirical_A_norm | norm_05 | 0.5376 | 0.6452 | 0.0000 | 0.0000 | 3 | 1 |
| gaussian_A_edge | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_01 | 0.2943 | 0 |
| edge | group | edge_01 | 0.2943 | 0 |
| edge | per_class | edge_01 | 0.2943 | 0 |
| edge | per_sample | edge_01 | 0.2943 | 0 |
| norm | unconstrained | norm_05 | 0.5376 | 1 |
| norm | group | norm_05 | 0.5376 | 1 |
| norm | per_class | norm_05 | 0.5376 | 1 |
| norm | per_sample | norm_01 | 0.1111 | 0 |

### B

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.1667 | -0.2000 | 0.0000 | 0.0000 | 1 | 1 |
| original_decision | edge_04 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_norm | norm_05 | -2.0278 | 0.0667 | -12.5000 | -5.0000 | 1 | 3 |
| gaussian_A_edge | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.6667 | 0 |
| edge | group | edge_14 | 0.6667 | 0 |
| edge | per_class | edge_14 | 0.6667 | 0 |
| edge | per_sample | edge_14 | 0.6667 | 0 |
| norm | unconstrained | norm_03 | 0.1667 | 1 |
| norm | group | norm_03 | 0.1667 | 1 |
| norm | per_class | edge_00 | 0.0000 | 0 |
| norm | per_sample | edge_00 | 0.0000 | 0 |

| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |
|---|---|---:|---:|---:|
| gaussian_A_vs_empirical_A | edge | 3.3461 | 0.4187 | 0.9558985062876292 |
| gaussian_A_vs_empirical_A | norm | 3.8278 | 1.6078 | 0.4525363440699853 |
| empirical_A_vs_B | edge | 10.2403 | 0.3259 | 0.88182473671466 |
| empirical_A_vs_B | norm | 10.5410 | 1.7079 | 0.45432024277781374 |
| gaussian_A_vs_B | edge | 9.2973 | 0.3529 | 0.9096055254216285 |
| gaussian_A_vs_B | norm | 9.8129 | 1.8267 | 0.3520805289275468 |
## HK_T4

A/B 排除类别：[5, 6, 19]

| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_00 | 62.7924 | 61.5171 | 73.6327 | 51.6654 | 0.0000 | 0.6101 | 0.6101 |
| linear_ridge | 61.9613 | 60.9447 | 70.6024 | 51.6878 | 0.8311 | 0.6101 | 1.4412 |
| prototype_ridge | 61.9613 | 60.9447 | 70.6024 | 51.6878 | 0.8311 | 0.6101 | 1.4412 |

### full

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.0797 | -0.0891 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_05 | 0.0065 | 0.1856 | -1.5152 | -0.2755 | 2 | 1 |
| empirical_A_norm | norm_11 | 0.1601 | -0.3200 | 4.2406 | -1.2948 | 19 | 12 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.6101 | 1 |
| edge | group | edge_14 | 0.6101 | 1 |
| edge | per_class | edge_14 | 0.6101 | 1 |
| edge | per_sample | edge_15 | 0.3544 | 0 |
| norm | unconstrained | norm_07 | 0.9306 | 5 |
| norm | group | norm_07 | 0.9306 | 5 |
| norm | per_class | edge_00 | 0.0000 | 0 |
| norm | per_sample | edge_00 | 0.0000 | 0 |

### A

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.1894 | -0.2165 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_05 | 0.1894 | 0.2165 | 0.0000 | 0.0000 | 1 | 0 |
| empirical_A_norm | norm_11 | 1.1215 | 0.4414 | 5.8824 | 0.6373 | 8 | 5 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_05 | 0.1894 | 0 |
| edge | group | edge_05 | 0.1894 | 0 |
| edge | per_class | edge_05 | 0.1894 | 0 |
| edge | per_sample | edge_05 | 0.1894 | 0 |
| norm | unconstrained | norm_11 | 1.1215 | 5 |
| norm | group | norm_11 | 1.1215 | 5 |
| norm | per_class | norm_05 | 0.9603 | 1 |
| norm | per_sample | edge_00 | 0.0000 | 0 |

### B

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_decision | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_05 | -0.1823 | 0.2381 | -3.1250 | -0.7812 | 1 | 1 |
| empirical_A_norm | norm_11 | -1.3730 | -1.9284 | 2.5152 | -5.4688 | 11 | 7 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 1.3159 | 0 |
| edge | group | edge_14 | 1.3159 | 0 |
| edge | per_class | edge_14 | 1.3159 | 0 |
| edge | per_sample | edge_14 | 1.3159 | 0 |
| norm | unconstrained | norm_05 | 1.1232 | 3 |
| norm | group | norm_05 | 1.1232 | 3 |
| norm | per_class | norm_01 | 0.7765 | 0 |
| norm | per_sample | norm_01 | 0.7765 | 0 |

| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |
|---|---|---:|---:|---:|
| gaussian_A_vs_empirical_A | edge | 4.5023 | 0.2887 | 0.6072046434206483 |
| gaussian_A_vs_empirical_A | norm | 5.0299 | 2.7162 | 0.09208118745741979 |
| empirical_A_vs_B | edge | 11.6334 | 0.4902 | 0.3590662493587659 |
| empirical_A_vs_B | norm | 13.4864 | 3.2506 | -0.027093616605009585 |
| gaussian_A_vs_B | edge | 11.6839 | 0.4981 | 0.41604554005867544 |
| gaussian_A_vs_B | norm | 11.8348 | 3.3479 | 0.9108832195435612 |
## HK_T6

A/B 排除类别：[5, 6, 19]

| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_00 | 62.5525 | 59.7778 | 91.6864 | 47.1664 | 0.0000 | 0.3049 | 0.3049 |
| linear_ridge | 61.2264 | 58.4771 | 90.0941 | 46.1954 | 1.3261 | 0.3049 | 1.6310 |
| prototype_ridge | 61.2264 | 58.4771 | 90.0941 | 46.1954 | 1.3261 | 0.3049 | 1.6310 |

### full

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.2030 | -0.2224 | 0.0000 | -0.2525 | 0 | 2 |
| original_decision | edge_15 | -0.1689 | -0.1546 | -0.3185 | -0.3968 | 1 | 2 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_09 | 0.0369 | 0.0404 | 0.0000 | 0.0000 | 1 | 1 |
| empirical_A_norm | norm_03 | 0.4078 | 0.4761 | -0.3101 | -0.4864 | 14 | 7 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.3049 | 3 |
| edge | group | edge_09 | 0.0369 | 1 |
| edge | per_class | edge_00 | 0.0000 | 0 |
| edge | per_sample | edge_00 | 0.0000 | 0 |
| norm | unconstrained | norm_05 | 0.6992 | 11 |
| norm | group | edge_00 | 0.0000 | 0 |
| norm | per_class | edge_00 | 0.0000 | 0 |
| norm | per_sample | edge_00 | 0.0000 | 0 |

### A

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.2941 | -0.3268 | 0.0000 | -0.6536 | 0 | 1 |
| original_decision | edge_15 | 0.1515 | 0.1684 | 0.0000 | 0.0000 | 1 | 0 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_09 | 0.1515 | 0.1684 | 0.0000 | 0.0000 | 1 | 0 |
| empirical_A_norm | norm_03 | 1.0659 | 1.1843 | 0.0000 | 0.2832 | 7 | 1 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.4837 | 1 |
| edge | group | edge_09 | 0.1515 | 0 |
| edge | per_class | edge_09 | 0.1515 | 0 |
| edge | per_sample | edge_09 | 0.1515 | 0 |
| norm | unconstrained | norm_08 | 1.7262 | 3 |
| norm | group | norm_03 | 1.0659 | 1 |
| norm | per_class | norm_02 | 0.9100 | 0 |
| norm | per_sample | norm_02 | 0.9100 | 0 |

### B

| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---|---:|---:|---:|---:|---:|---:|
| original_zero | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| original_original_reward | edge_07 | -0.1667 | -0.1852 | 0.0000 | 0.0000 | 0 | 1 |
| original_decision | edge_15 | -0.5641 | -0.5556 | -0.6410 | -1.1111 | 0 | 2 |
| original_norm_decision | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| empirical_A_edge | edge_09 | -0.0667 | -0.0741 | 0.0000 | 0.0000 | 0 | 1 |
| empirical_A_norm | norm_03 | -0.1666 | -0.1158 | -0.6242 | -1.6667 | 7 | 6 |
| gaussian_A_edge | edge_11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| gaussian_A_norm | edge_00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |
|---|---|---|---:|---:|
| edge | unconstrained | edge_14 | 0.1964 | 2 |
| edge | group | edge_00 | 0.0000 | 0 |
| edge | per_class | edge_00 | 0.0000 | 0 |
| edge | per_sample | edge_00 | 0.0000 | 0 |
| norm | unconstrained | norm_04 | 0.1516 | 6 |
| norm | group | edge_00 | 0.0000 | 0 |
| norm | per_class | edge_00 | 0.0000 | 0 |
| norm | per_sample | edge_00 | 0.0000 | 0 |

| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |
|---|---|---:|---:|---:|
| gaussian_A_vs_empirical_A | edge | 3.6397 | 0.2588 | 0.708666311790598 |
| gaussian_A_vs_empirical_A | norm | 3.9061 | 1.6616 | 0.7096390180051818 |
| empirical_A_vs_B | edge | 10.6513 | 0.3883 | 0.3760250342568424 |
| empirical_A_vs_B | norm | 12.8400 | 4.0684 | -0.267813469907433 |
| gaussian_A_vs_B | edge | 10.2547 | 0.3709 | 0.14790159000488431 |
| gaussian_A_vs_B | norm | 11.4871 | 3.8855 | -0.4414470480258694 |
