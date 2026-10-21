# cafe_closed_loop_pilot 验证报告

结论：STOP_CAFE_CLOSED_LOOP_PILOT；method_success=false。

原19.06 MB完整模型上限按用户明确指令取消；完整资产与实际分配仍如实计量，独立512MiB新增预算、600秒时钟和窗口限制保持。

这是现有冻结特征上的机制迁移，具体与论文差异见PROTOCOL；不是完整论文复现。历史query已暴露，无盲测或完整未来类CL结论。

| 数据 / 臂 | BA % | correct | counts |
|---|---:|---|---|
| ISIC/F | 64.713626 | [237, 13, 64, 1626] | [307, 20, 170, 2058] |
| ISIC/P | 56.280891 | [209, 11, 67, 1289] | [307, 20, 170, 2058] |
| ISIC/R | 64.713626 | [237, 13, 64, 1626] | [307, 20, 170, 2058] |
| ISIC/T | 64.713626 | [237, 13, 64, 1626] | [307, 20, 170, 2058] |
| ISIC/M | 64.713626 | [237, 13, 64, 1626] | [307, 20, 170, 2058] |
| E/F | 62.956731 | [165, 0, 0, 17, 28, 103, 182, 123, 58, 34, 125, 25, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/P | 62.710417 | [160, 0, 0, 15, 43, 100, 182, 119, 57, 29, 110, 35, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/R | 62.956731 | [165, 0, 0, 17, 28, 103, 182, 123, 58, 34, 125, 25, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/T | 62.840181 | [165, 0, 0, 17, 27, 103, 182, 123, 58, 34, 125, 25, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| E/M | 62.956731 | [165, 0, 0, 17, 28, 103, 182, 123, 58, 34, 125, 25, 0] | [165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2] |
| Qgate/F | 71.257239 | [62, 0, 0, 7, 12, 33, 73, 48, 24, 14, 54, 10, 1] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/P | 64.095005 | [61, 0, 0, 6, 18, 33, 73, 48, 25, 13, 46, 14, 0] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/R | 71.257239 | [62, 0, 0, 7, 12, 33, 73, 48, 24, 14, 54, 10, 1] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/T | 71.257239 | [62, 0, 0, 7, 12, 33, 73, 48, 24, 14, 54, 10, 1] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |
| Qgate/M | 71.257239 | [62, 0, 0, 7, 12, 33, 73, 48, 24, 14, 54, 10, 1] | [62, 1, 1, 8, 28, 40, 73, 48, 25, 16, 59, 26, 1] |

ISIC冻结门：`{"BA_GAIN": false, "CLASS1_GAIN": false, "CLASS1_NO_HARM": true, "CELL_BA": true, "CELL_CLASS1": true, "ALLCLASS_COST": true, "CELL_ALLCLASS_COST": true, "VS_PROTOTYPES": false, "pass_all": false, "missing": false}`
HK冻结门：`{"BA_GAIN": false, "CLASS1_GAIN": false, "CLASS1_NO_HARM": true, "CELL_BA": false, "CELL_CLASS1": false, "ALLCLASS_COST": true, "CELL_ALLCLASS_COST": true, "VS_PROTOTYPES": false, "CELL_VS_PROTOTYPES": false, "pass_all": false, "missing": false}`

全部fold/cohort、逐类混淆和救回/损害计数见aggregate.json；所有臂均公开，不挑最佳控制替代主方法。点门未全过则保持STOP，不扫参数救门。无bootstrap，CI=NA。

CAFE：R真实样本CE，T同提议池不经虚拟风险准入，M经虚拟风险筛选的闭环。P仅均值参考。训练支持风险只用于机制内选择，不是独立验证。详见各TRAIN统计；即使支持风险改善也不等于query少数类安全。

先HEAD/MODEL_SEAL再query，F原score/计数与A/B残差、finite、回读、覆盖检查由entry断言执行；未执行到的检查不得视为通过。

成本：`{"activity_wall_seconds": 26.154610872268677, "known_worker_wait4_CPU_seconds": 46.752533, "coordinator_CPU_seconds": 3.4481200000000003, "peak_RSS_bytes": 993644544, "round_remote_bytes": 60746769, "window_remote_bytes": 14169133317, "unmetered": "reading/preparation/network/report CPU unknown; worker body not added again"}`
累计：`{"activity_wall_seconds": 32302.013525546383, "known_worker_wait4_CPU_seconds": 40664.67941299998, "coordinator_CPU_seconds": 1268.4173779999999}`

源码/协议/聚合/失败与NA公开；身份、逐例、特征、模型、Gram、原日志和第三方PDF/代码包保持私有。父子已退出并退休，不再监测或重跑。
