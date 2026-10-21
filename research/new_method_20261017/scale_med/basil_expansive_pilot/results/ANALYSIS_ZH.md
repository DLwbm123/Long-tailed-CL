# basil_expansive_pilot 验证报告

结论：STOP_BASIL_EXPANSIVE_PILOT；method_success=false。

原19.06 MB完整模型上限按用户明确指令取消；完整资产与实际分配仍如实计量，独立512MiB新增预算、600秒时钟和窗口限制保持。

这是现有冻结特征上的机制迁移，具体与论文差异见PROTOCOL；不是完整论文复现。历史query已暴露，无盲测或完整未来类CL结论。

| 数据 / 臂 | BA % | correct | counts |
|---|---:|---|---|
| ISIC/F | 64.713626 | [237, 13, 64, 1626] | [307, 20, 170, 2058] |
| ISIC/P | 56.280891 | [209, 11, 67, 1289] | [307, 20, 170, 2058] |
| ISIC/R | 48.982015 | [187, 5, 39, 1792] | [307, 20, 170, 2058] |
| ISIC/T | 56.329482 | [209, 11, 67, 1293] | [307, 20, 170, 2058] |
| ISIC/M | 56.208769 | [197, 9, 51, 1763] | [307, 20, 170, 2058] |
| E/F | 62.956731 | [165, 0, 0, 17, 28, 103, 182, 123, 58, 34, 125, 25, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/P | 62.710417 | [160, 0, 0, 15, 43, 100, 182, 119, 57, 29, 110, 35, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/R | 62.068921 | [164, 0, 0, 9, 52, 112, 182, 122, 57, 28, 120, 27, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/T | 62.898034 | [160, 0, 0, 15, 43, 100, 182, 119, 57, 30, 110, 35, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/M | 62.424358 | [163, 0, 0, 11, 47, 112, 182, 122, 57, 28, 120, 29, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| Qgate/F | 71.257239 | [62, 0, 0, 7, 12, 33, 73, 48, 24, 14, 54, 10, 1] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/P | 64.095005 | [61, 0, 0, 6, 18, 33, 73, 48, 25, 13, 46, 14, 0] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/R | 63.301861 | [62, 0, 0, 4, 24, 38, 73, 48, 25, 12, 51, 8, 0] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/T | 64.390863 | [61, 0, 0, 6, 18, 33, 73, 48, 25, 13, 46, 15, 0] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/M | 65.358719 | [62, 0, 0, 5, 22, 38, 73, 48, 25, 12, 50, 14, 0] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |

ISIC冻结门：`{"BA_GAIN": false, "CLASS1_GAIN": false, "CLASS1_NO_HARM": false, "CELL_BA": false, "CELL_CLASS1": false, "ALLCLASS_COST": false, "CELL_ALLCLASS_COST": false, "VS_PROTOTYPES": false, "pass_all": false, "missing": false}`
HK冻结门：`{"BA_GAIN": false, "CLASS1_GAIN": false, "CLASS1_NO_HARM": true, "CELL_BA": false, "CELL_CLASS1": false, "ALLCLASS_COST": false, "CELL_ALLCLASS_COST": false, "VS_PROTOTYPES": true, "CELL_VS_PROTOTYPES": false, "pass_all": false, "missing": false}`

全部fold/cohort、逐类混淆和救回/损害计数见aggregate.json；所有臂均公开，不挑最佳控制替代主方法。点门未全过则保持STOP，不扫参数救门。无bootstrap，CI=NA。

BASIL：F原模型，P均值，R真实支持最大相似度，T切向参考，M完整BASIL迁移。真实支持与随机切向参考均未改变原F；全支持局部最大值的类别样本量效应不能解释为语义信息增益。此固定迁移失败不否证整篇论文。

先HEAD/MODEL_SEAL再query，F原score/计数与A/B残差、finite、回读、覆盖检查由entry断言执行；未执行到的检查不得视为通过。

成本：`{"activity_wall_seconds": 21.161725759506226, "known_worker_wait4_CPU_seconds": 30.422857, "coordinator_CPU_seconds": 2.724046, "peak_RSS_bytes": 1083506688, "round_remote_bytes": 248269955, "window_remote_bytes": 14108059681, "unmetered": "reading/preparation/network/report CPU unknown; worker body not added again"}`
累计：`{"activity_wall_seconds": 32275.858914674114, "known_worker_wait4_CPU_seconds": 40617.926879999985, "coordinator_CPU_seconds": 1264.9692579999999}`

源码/协议/聚合/失败与NA公开；身份、逐例、特征、模型、Gram、原日志和第三方PDF/代码包保持私有。父子已退出并退休，不再监测或重跑。
