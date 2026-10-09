# SCARCITY-FACTOR1 全部固定头与选择结果

所有增量单位为百分点；A/B为固定覆盖类别，full包括全部已见类。

## ISIC_T2_cached

A/B排除类别：[]

### full

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 60.8787 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 62.9552 | 2.0765 | 3.1147 | 0.0000 | 3.1147 | 5 | 0 |
| full_minus | 61.5594 | 0.6807 | -0.0659 | 2.1739 | -0.0659 | 3 | 1 |
| row_plus | 62.9552 | 2.0765 | 3.1147 | 0.0000 | 3.1147 | 5 | 0 |
| row_minus | 61.6034 | 0.7246 | 0.0000 | 2.1739 | 0.0000 | 2 | 0 |
| within_plus | 62.5487 | 1.6700 | 2.5049 | 0.0000 | 2.5049 | 4 | 0 |
| within_minus | 60.8787 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### A

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 58.7781 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 62.1553 | 3.3772 | 5.0658 | 0.0000 | 5.0658 | 4 | 0 |
| full_minus | 60.1670 | 1.3889 | 0.0000 | 4.1667 | 0.0000 | 2 | 0 |
| row_plus | 62.1553 | 3.3772 | 5.0658 | 0.0000 | 5.0658 | 4 | 0 |
| row_minus | 60.1670 | 1.3889 | 0.0000 | 4.1667 | 0.0000 | 2 | 0 |
| within_plus | 62.1553 | 3.3772 | 5.0658 | 0.0000 | 5.0658 | 4 | 0 |
| within_minus | 58.7781 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### B

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 62.9361 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 63.7298 | 0.7937 | 1.1905 | 0.0000 | 1.1905 | 1 | 0 |
| full_minus | 62.8039 | -0.1323 | -0.1984 | 0.0000 | -0.1984 | 1 | 1 |
| row_plus | 63.7298 | 0.7937 | 1.1905 | 0.0000 | 1.1905 | 1 | 0 |
| row_minus | 62.9361 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| within_plus | 62.9361 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| within_minus | 62.9361 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

删除稳定性（仅当前meta身份）：
{"full": {"identity_deletions": 241, "same_winner": 241, "unsupported_deletions": 0, "winner_counts": {"zero": 241, "full_plus": 0, "full_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "row": {"identity_deletions": 241, "same_winner": 241, "unsupported_deletions": 0, "winner_counts": {"zero": 241, "row_plus": 0, "row_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "within": {"identity_deletions": 241, "same_winner": 241, "unsupported_deletions": 0, "winner_counts": {"zero": 241, "within_plus": 0, "within_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}}

## ISIC_T3

A/B排除类别：[]

### full

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 55.8387 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 59.0658 | 3.2271 | 3.8483 | 1.3636 | 8.0786 | 14 | 5 |
| full_minus | 56.4145 | 0.5758 | -0.4065 | 3.5227 | -1.2854 | 4 | 2 |
| row_plus | 57.4063 | 1.5677 | 2.0902 | 0.0000 | 4.8980 | 7 | 3 |
| row_minus | 56.4145 | 0.5758 | -0.4065 | 3.5227 | -1.2854 | 4 | 2 |
| within_plus | 57.1573 | 1.3186 | 1.7581 | 0.0000 | 3.1806 | 5 | 1 |
| within_minus | 55.5085 | -0.3302 | -0.8570 | 1.2500 | -1.2854 | 1 | 2 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | within_plus | 1.3186 | 1.7581 | 0.0000 | 3.1806 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### A

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 60.3453 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 60.5775 | 0.2323 | -0.5994 | 2.7273 | 2.5000 | 4 | 4 |
| full_minus | 60.3453 | 0.0000 | -0.8333 | 2.5000 | -1.2500 | 1 | 1 |
| row_plus | 59.2707 | -1.0746 | -1.4327 | 0.0000 | 1.2500 | 1 | 3 |
| row_minus | 60.3453 | 0.0000 | -0.8333 | 2.5000 | -1.2500 | 1 | 1 |
| within_plus | 60.4494 | 0.1042 | 0.1389 | 0.0000 | 1.2500 | 1 | 1 |
| within_minus | 60.3453 | 0.0000 | -0.8333 | 2.5000 | -1.2500 | 1 | 1 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | within_plus | 0.1042 | 0.1389 | 0.0000 | 1.2500 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### B

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 51.2100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 57.5367 | 6.3267 | 8.4355 | 0.0000 | 13.7897 | 10 | 1 |
| full_minus | 52.3464 | 1.1364 | -0.0000 | 4.5455 | -1.3889 | 3 | 1 |
| row_plus | 55.5255 | 4.3155 | 5.7540 | 0.0000 | 8.6310 | 6 | 0 |
| row_minus | 52.3464 | 1.1364 | -0.0000 | 4.5455 | -1.3889 | 3 | 1 |
| within_plus | 53.7894 | 2.5794 | 3.4392 | 0.0000 | 5.1587 | 4 | 0 |
| within_minus | 50.5156 | -0.6944 | -0.9259 | 0.0000 | -1.3889 | 0 | 1 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | within_plus | 2.5794 | 3.4392 | 0.0000 | 5.1587 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

删除稳定性（仅当前meta身份）：
{"full": {"identity_deletions": 1470, "same_winner": 1470, "unsupported_deletions": 0, "winner_counts": {"zero": 1470, "full_plus": 0, "full_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "row": {"identity_deletions": 1470, "same_winner": 1470, "unsupported_deletions": 0, "winner_counts": {"zero": 1470, "row_plus": 0, "row_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "within": {"identity_deletions": 1470, "same_winner": 1467, "unsupported_deletions": 0, "winner_counts": {"zero": 3, "within_plus": 1467, "within_minus": 0}, "primary_point": "within_plus", "stable_choice": "zero", "population_confidence_guarantee": false}}

## HK_T2_cached

A/B排除类别：[5, 6, 19]

### full

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 66.1108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 65.5679 | -0.5429 | -0.6264 | 0.0000 | -0.3049 | 0 | 7 |
| full_minus | 66.4851 | 0.3743 | 0.4319 | 0.0000 | 0.2083 | 7 | 1 |
| row_plus | 65.5679 | -0.5429 | -0.6264 | 0.0000 | -0.3049 | 0 | 7 |
| row_minus | 66.3758 | 0.2650 | 0.3057 | 0.0000 | 0.2083 | 6 | 1 |
| within_plus | 65.7944 | -0.3163 | -0.3650 | 0.0000 | -0.3049 | 0 | 3 |
| within_minus | 66.0904 | -0.0204 | -0.0235 | 0.0000 | 0.0000 | 2 | 1 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### A

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 84.3957 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 83.7463 | -0.6494 | -0.7792 | 0.0000 | -0.9524 | 0 | 2 |
| full_minus | 84.6736 | 0.2780 | 0.3336 | 0.0000 | 0.0000 | 4 | 1 |
| row_plus | 83.7463 | -0.6494 | -0.7792 | 0.0000 | -0.9524 | 0 | 2 |
| row_minus | 84.6736 | 0.2780 | 0.3336 | 0.0000 | 0.0000 | 4 | 1 |
| within_plus | 83.9988 | -0.3968 | -0.4762 | 0.0000 | -0.9524 | 0 | 1 |
| within_minus | 84.2380 | -0.1577 | -0.1892 | 0.0000 | 0.0000 | 1 | 1 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### B

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 80.8084 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 80.1058 | -0.7027 | -0.8432 | 0.0000 | 0.0000 | 0 | 5 |
| full_minus | 81.4751 | 0.6667 | 0.8000 | 0.0000 | 0.6667 | 3 | 0 |
| row_plus | 80.1058 | -0.7027 | -0.8432 | 0.0000 | 0.0000 | 0 | 5 |
| row_minus | 81.1973 | 0.3889 | 0.4667 | 0.0000 | 0.6667 | 2 | 0 |
| within_plus | 80.4196 | -0.3889 | -0.4667 | 0.0000 | 0.0000 | 0 | 2 |
| within_minus | 80.9196 | 0.1111 | 0.1333 | 0.0000 | 0.0000 | 1 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

删除稳定性（仅当前meta身份）：
{"full": {"identity_deletions": 24, "same_winner": 24, "unsupported_deletions": 0, "winner_counts": {"zero": 24, "full_plus": 0, "full_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "row": {"identity_deletions": 24, "same_winner": 24, "unsupported_deletions": 0, "winner_counts": {"zero": 24, "row_plus": 0, "row_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "within": {"identity_deletions": 24, "same_winner": 24, "unsupported_deletions": 0, "winner_counts": {"zero": 24, "within_plus": 0, "within_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}}

## HK_T4

A/B排除类别：[5, 6, 19]

### full

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 62.7924 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 62.3491 | -0.4433 | -0.4955 | 0.0000 | 0.5031 | 2 | 17 |
| full_minus | 63.4025 | 0.6101 | 0.6819 | 0.0000 | 0.0000 | 14 | 1 |
| row_plus | 62.3259 | -0.4665 | -0.5214 | 0.0000 | 0.2919 | 2 | 14 |
| row_minus | 63.1947 | 0.4023 | 0.4497 | 0.0000 | 0.0000 | 9 | 1 |
| within_plus | 62.5723 | -0.2201 | -0.2460 | 0.0000 | 0.0000 | 0 | 5 |
| within_minus | 62.9873 | 0.1949 | 0.2179 | 0.0000 | 0.0000 | 3 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### A

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 78.3491 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 77.6516 | -0.6975 | -0.7971 | 0.0000 | 0.0000 | 0 | 6 |
| full_minus | 78.4889 | 0.1398 | 0.1598 | 0.0000 | 0.0000 | 3 | 1 |
| row_plus | 77.9493 | -0.3999 | -0.4570 | 0.0000 | 0.5952 | 1 | 6 |
| row_minus | 78.2308 | -0.1183 | -0.1352 | 0.0000 | 0.0000 | 1 | 1 |
| within_plus | 77.9931 | -0.3561 | -0.4069 | 0.0000 | 0.0000 | 0 | 3 |
| within_minus | 78.3491 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### B

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 69.9008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 69.5759 | -0.3249 | -0.3713 | 0.0000 | 1.4583 | 2 | 11 |
| full_minus | 71.2167 | 1.3159 | 1.5039 | 0.0000 | 0.0000 | 11 | 0 |
| row_plus | 69.1863 | -0.7145 | -0.8166 | 0.0000 | 0.2083 | 1 | 8 |
| row_minus | 70.9813 | 1.0805 | 1.2349 | 0.0000 | 0.0000 | 8 | 0 |
| within_plus | 69.7341 | -0.1667 | -0.1905 | 0.0000 | 0.0000 | 0 | 2 |
| within_minus | 70.3672 | 0.4664 | 0.5330 | 0.0000 | 0.0000 | 3 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

删除稳定性（仅当前meta身份）：
{"full": {"identity_deletions": 152, "same_winner": 152, "unsupported_deletions": 0, "winner_counts": {"zero": 152, "full_plus": 0, "full_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "row": {"identity_deletions": 152, "same_winner": 152, "unsupported_deletions": 0, "winner_counts": {"zero": 152, "row_plus": 0, "row_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "within": {"identity_deletions": 152, "same_winner": 151, "unsupported_deletions": 0, "winner_counts": {"zero": 151, "within_plus": 0, "within_minus": 1}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}}

## HK_T6

A/B排除类别：[5, 6, 19]

### full

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 62.5525 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 62.1593 | -0.3932 | -0.4913 | 0.6369 | 0.0000 | 4 | 14 |
| full_minus | 62.8574 | 0.3049 | 0.3331 | 0.0083 | -0.5357 | 15 | 3 |
| row_plus | 62.2758 | -0.2767 | -0.3030 | 0.0000 | 0.2033 | 3 | 13 |
| row_minus | 62.6727 | 0.1202 | 0.1309 | 0.0083 | -0.5357 | 11 | 3 |
| within_plus | 62.2625 | -0.2900 | -0.3176 | 0.0000 | -0.2525 | 0 | 5 |
| within_minus | 62.3836 | -0.1689 | -0.1546 | -0.3185 | -0.3968 | 1 | 2 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### A

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 75.2708 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 75.0825 | -0.1883 | -0.2092 | 0.0000 | 0.0000 | 2 | 5 |
| full_minus | 75.7545 | 0.4837 | 0.5375 | 0.0000 | -0.3704 | 6 | 1 |
| row_plus | 75.3206 | 0.0498 | 0.0554 | 0.0000 | 0.5291 | 3 | 5 |
| row_minus | 75.5363 | 0.2655 | 0.2950 | 0.0000 | -0.3704 | 4 | 1 |
| within_plus | 74.9100 | -0.3608 | -0.4009 | 0.0000 | -0.6536 | 0 | 2 |
| within_minus | 75.4223 | 0.1515 | 0.1684 | 0.0000 | 0.0000 | 1 | 0 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
### B

| 固定规则 | BA | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 67.9019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full_plus | 67.1838 | -0.7180 | -0.9403 | 1.2821 | 0.0000 | 2 | 9 |
| full_minus | 68.0983 | 0.1964 | 0.2164 | 0.0169 | -1.1111 | 9 | 2 |
| row_plus | 67.2072 | -0.6947 | -0.7719 | 0.0000 | 0.0000 | 0 | 8 |
| row_minus | 67.8919 | -0.0100 | -0.0130 | 0.0169 | -1.1111 | 7 | 2 |
| within_plus | 67.6019 | -0.3000 | -0.3333 | 0.0000 | 0.0000 | 0 | 3 |
| within_minus | 67.3378 | -0.5641 | -0.5556 | -0.6410 | -1.1111 | 0 | 2 |

| 训练侧选择器 | 已选头 | ΔBA | Δold | Δcurrent | Δtail |
|---|---|---:|---:|---:|---:|
| full_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| row_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_point | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| within_stable | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

删除稳定性（仅当前meta身份）：
{"full": {"identity_deletions": 260, "same_winner": 260, "unsupported_deletions": 0, "winner_counts": {"zero": 260, "full_plus": 0, "full_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "row": {"identity_deletions": 260, "same_winner": 260, "unsupported_deletions": 0, "winner_counts": {"zero": 260, "row_plus": 0, "row_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}, "within": {"identity_deletions": 260, "same_winner": 260, "unsupported_deletions": 0, "winner_counts": {"zero": 260, "within_plus": 0, "within_minus": 0}, "primary_point": "zero", "stable_choice": "zero", "population_confidence_guarantee": false}}

