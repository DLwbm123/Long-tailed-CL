# R111结果一览

A1–A4归档分析已完成。新工作仅为输入可行性检查，**正式P1b未启动**；真实分类器/验证器/RL更新=0，VAL/TEST读取=0。

| 流 | seed74003有效状态 | seed74004有效状态 | 输入门 |
|---|---:|---:|---|
| Ordered | 4/16 | 4/16 | 通过 |
| order0 | 0/16 | 0/16 | 未通过 |

联合门未过。order0缺合法旧尾S/FIT/AUDIT角色，不能用缺失罚项0称安全。Ordered末阶段也出现合法池小于256的no-op；容量未降低，种子未替换。下一步需要单独修订任务构造，暂不能进行有效的双流P1b确认。

A分析的固定门后规则L1的事后mean R为Ordered 0.019164081、order0 0.000264423；所有非ZERO执行都在确无真实旧尾的根，不能当尾类保护证据。原R110结论仍为Ordered -0.010501924、order0 0.048658126、联合门未过。

详见[完整报告](REPORT_ZH.md)、[固定规则比较](results/A4_STREAM_COMPARISON.csv)、[准入分层](READINESS_STAGE_COUNTS.csv)、[冻结协议](READINESS_PROTOCOL.md)和[成本](COST_SUMMARY.json)。
