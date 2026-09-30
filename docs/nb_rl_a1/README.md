# NB-RL-A1 已完成实验

本分支发布已执行的代码、协议、聚合结果与审计摘要。最终结果以
[四卡正式报告](parallel_run/FINAL_REPORT_ZH.md) 和 `parallel_run/` 为准。
`evidence/` 保存先前单卡预算准入未通过的工程记录，不是正式实验结果。

四张 A100 40GB；3 epoch；S/H/K/G/R/E 六条件、三个顺序、四任务；
F_S/F_R 使用本轮对应正常 Task1 的冻结模型。共 21,150 次正式更新，
96 个逻辑阶段、480 条逐类结果。北京时间 2026-09-30 22:16 完成，
从原始 T0 起墙钟约 4 小时 10 分钟，累计 GPU 驻留约 13.23 小时。
独立备份最终 ACK 已通过。

R 的最终平均 BA 为 42.283%，S 为 42.972%；主效用、安全和强控制净效用门槛未通过。
保持组合与动作采样的附加收益均未建立。未因结果调整训练、奖励或评价口径。
完整比较、条件 bootstrap 区间和解释边界见正式报告。

## 公开内容与复现入口

- `tools/run_nb_rl_a1.py`：训练、冻结参照与封存评价；`nb_rl_a1_core.py`：目标与数学自检。
- `tools/nb_rl_a1_parallel.py`：四卡轨迹调度与共享资源账本。
- `tools/nb_rl_a1_benchmark.py`、`nb_rl_a1_admission.py`、`nb_rl_a1_parallel_admission.py`：工程计时与预算准入。
- `tools/report_nb_rl_a1.py`：从私有封存预测生成聚合结果；`nb_rl_a1_transfer.py`：独立备份传输。
- `parallel_run/`：最终结果表、奖励/梯度诊断、协议锁、环境与完成/备份回执。

代码复用本分支已有 APART、CT1/CT3P 和 holdout 报告工具；外部预训练权重与数据版本
由 `parallel_run/SOURCE_AND_ASSET_LOCK.json` 标识。环境见 `ENVIRONMENT_LOCK.json`。
运行入口从 `N78_ROOT/private/INPUT.json` 读取本地私有配置，所需键包括
`legacy`、`deps`、`weight`、`dataset.manifests`、`dataset.images` 和 `wall_T0_unix`；
阶段预算与初始资源账本须先按协议准入，不能直接绕过工程门槛。
`N78_ROLE` 指定工程/训练/冻结/评价角色，四卡主管通过 `N78_MODE` 调度。
启动所需中性入口、Python 环境与 SSH 备份别名由部署环境准备，发布不自动启动新实验。

不公开原始图像、身份/样本清单、逐样本预测、模型权重、checkpoint、原始进程日志和私有配置。
因此本仓库可审阅源码与聚合结果；重新评价封存预测或训练需要合法取得上述私有资产。
最终 ACK 中的完整性校验属于既有传输流程，不表示 GitHub 包含模型备份。

原协议和运行期回执中的“无自动 push”/`AWAITING_AUTHORIZATION` 保留历史语义。
用户在实验完成后明确要求直接发布，授权记录见 [PUBLICATION_STATUS.json](PUBLICATION_STATUS.json)。
历史主报告未改写；该授权只改变发布状态，不改变 `NEXT_DECISION=STOP` 的科学决策。
