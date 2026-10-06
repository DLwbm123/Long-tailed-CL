# Long-tailed Class-Incremental Learning

长尾类别增量学习研究代码与实验报告，包含 CIFAR-100-LT 上的 GPA/TaConCM 诊断、APART + ConCM-lite，以及 HyperKvasir23 上的校准与动态几何实验。

- [单标签原型解析：24 小时监测与自动改进计划](docs/prototype_analytic_longrun_20261006/PROGRAM_ZH.md)
- [PA2 完整单标签候选：多原型与可靠性图传播](docs/prototype_analytic_pa2/PROTOCOL_ZH.md)
- [PA1 原型与解析分类头：single-label 开发协议](docs/prototype_analytic/PROTOCOL_ZH.md)
- [项目现状：数据集、方法与结果](docs/PROJECT_STATUS_20260914.md)
- [NB-RL-A1 四卡实验：完整结果与审计](docs/nb_rl_a1/README.md)
- [NB-RL-A2：保持强度与精确奖励开发实验](docs/nb_rl_a2/README.md)
- [基础运行与 smoke 命令](experiments/README.md)
- [医学数据集接口](docs/FOPRO_MEDICAL_DATASETS.md)
- [APART + ConCM-lite 完整三种子结果](docs/APART_CONCM_STAGE1_HEADNORM_3SEED_FULL_RESULTS.md)
- [HyperKvasir Full Dynamic 分支关闭结论](docs/HYPERKVASIR23_FULL_DYNAMIC_BRANCH_CLOSED.md)

这是研究工作区的源代码快照；历史诊断分支也保留在仓库中，不代表全部方法复现成功。不同 backbone、回放设置、类别顺序和阶段数的结果不可直接排名。依赖和运行参数请按对应入口及报告配置。

数据集图像、模型权重、checkpoint、运行日志、缓存、论文 PDF 和压缩包不随仓库发布。第三方代码保留原有来源和许可；本仓库不重新授权第三方材料。
