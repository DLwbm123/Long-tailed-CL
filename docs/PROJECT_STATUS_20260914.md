# 项目现状（2026-09-14 核对）

本摘要依据当前工作区代码、已保存的实验报告与小型指标文件整理。本次没有重新训练，也没有登录历史服务器验证旧 checkpoint 或进程；报告中的完成状态属于历史实验记录。最新分支结论见 7 月底的 Full Dynamic 关闭报告。

## 数据集与当前准备状态

| 数据集 | 项目中的协议 | 当前证据与状态 |
|---|---|---|
| CIFAR-100-LT | 100 类，主要使用 rho=0.01（约 100:1）、shuffled、50+5×10；早期 ResNet32 与 APART 预训练 ViT 分属不同路线 | 已有完整实验报告。本次本地 `data/cifar-100-python` 只有 `test`、`meta` 和空备份文件，缺少 `train`，不能声称本地已完整就绪 |
| HyperKvasir23 | 23 类、10,662 张；自然长尾，确定性分层 5 折；fold1 训练 8,519、测试 2,143，训练类样本数 4–918；后续 CIL 为 13+5×2 | 当前 DataP 数据根目录存在；样本规模来自已保存的数据预览记录，本次未重新全量扫描。多数早期 gate 仅运行 base+session1，不能当作完整 23 类结果 |
| ISIC2019-LT | 8 类，支持 100:1 / 200:1 / 500:1，5 套 split；验证每类 50、测试每类 100 | 已实现接口和 split。本次确认 DataP 的 `ISIC2019_FoProKD/ISIC_2019_Training_Input` 有 25,331 个 JPEG，抽取一张检查了 JPEG 文件头；未核对全部 split 覆盖，未发现本项目 ISIC 效果报告 |

医学数据集接口参考 [FOPRO_MEDICAL_DATASETS.md](FOPRO_MEDICAL_DATASETS.md)。其中 6 月 20 日“未找到 ISIC 图像”的描述已被本次目录核对更新。接口接入不等于 FoPro-KD 论文复现；原始医学接口沿用 ResNet32，后续实验必须以对应脚本配置为准。

## 已尝试方法与效果

数值单位为百分比。Avg Acc 为阶段平均 all-seen accuracy；AccT 为对应协议最后一个评估阶段的 all-seen accuracy。不同表中的阶段数、模型和训练设置不同。± 沿用各报告的标准差口径；HM 为旧类/当前类准确率的调和平均。

### CIFAR：早期无回放 ResNet32 路线

完整 seed0、50+5×10、每阶段 100 epoch：

| 方法 | Avg Acc | 最终 AccT | 结论 |
|---|---:|---:|---|
| Finetune | 11.464 | 4.31 | 严重遗忘 |
| Finetune + GPA | 11.698 | 4.87 | 未解决旧任务崩塌 |
| TaConCM-GPA 完整组合 | 11.681 | 5.16 | 比 Finetune 高 0.85 个百分点，但旧任务仍降到 0% |

尝试过原型校准初始化、校准 anchor、tail anchor、T-DSM 和 matching loss。随后又检查了 GVAlign 多头插件与单头 GPA scaffold；单头 phase1 Finetune AccT 为 31.67，paper-bias + mean-anchor GPA 为 15.35，sum-anchor 为 13.30。GPA 忠实复现路线已停止，不能把这些结果称为论文成功复现。

证据：[六阶段消融](TACONCM_FULL_SEED0_RESULTS.md)、[GPA 负结果报告](GPA_NEGATIVE_REPRODUCTION_REPORT.md)。

### CIFAR：APART + ConCM-lite 路线

完整六阶段配对实验，seeds=1993/1994/1995，APART 预训练 ViT 协议：

| 方法 | Avg Acc | 最终 AccT | few 类准确率 |
|---|---:|---:|---:|
| APART | 86.624 ± 0.377 | 84.03 ± 0.67 | 75.91 ± 3.06 |
| APART + ConCM-lite Stage1 capped + head-norm effective-sum 校准 | 87.584 ± 0.542 | 85.68 ± 0.70 | 78.67 ± 3.88 |
| 配对提升 | +0.961 | +1.65 | +2.76 |

三个种子的 Avg Acc 与 AccT 均提升。这是当前最完整、最清楚的正向证据，但不是与上述无回放 ResNet32 的公平横向比较。

后续还试过原型校准、relation matching、bias controller、不确定性门控 replay 和 USFM。这些主要是两阶段有界 gate，未展示继续升级所需的收益；例如 replay gate 的 AccT 从 89.62 降至 89.52，USFM 为 89.60，不应计为新的完整三种子成功方法。

证据：[完整三种子结果](APART_CONCM_STAGE1_HEADNORM_3SEED_FULL_RESULTS.md)、[门控 replay](APART_FULL_CONCM_RUNG5B_UNCERTAINTY_GATED_REPLAY_RESULTS.md)、[USFM](APART_FULL_CONCM_USFM_RESULTS.md)。

### HyperKvasir：有界 session1 校准与 DSM

尝试过冻结骨干/旧头、Med-FDM、task-block calibration、固定 NC 几何、可靠性加权、prototype matching、Pure DSM、去 LCont、MPC-lite，以及 NC 主分支叠加 DSM residual。

7 月 9 日重建 checkpoint、seeds=1/2/3、仅 base+session1：

| 方法 | session1 AccT | 当前类准确率 | HM |
|---|---:|---:|---:|
| Locked NC-ConCM | 52.38 ± 2.88 | 74.68 ± 5.94 | 58.07 ± 3.85 |
| + DSM auxiliary，lambda=0.2 | 61.04 ± 2.43 | 91.36 ± 3.14 | 67.72 ± 2.96 |
| Pure DSM-ConCM | 70.22 ± 7.37 | 73.45 ± 32.37 | 65.57 ± 13.64 |

DSM auxiliary 相对 Locked NC 平均提升 8.66 个百分点，且三种子均提升；但仍是有界阶段的诊断候选。Pure DSM 的总准确率更高，却在 seed3 把当前类准确率压到 36.31%，不能只按总准确率判断稳定性。

历史 seed0 的 Locked NC 77.53 属于不同类别划分和历史 checkpoint，不能和这里 seed1 的 53.74 当作同设置退化对比。

证据：[辅助 DSM 三种子报告](HYPERKVASIR23_LOCKED_NC_DSM_AUX_RESULTS.md)。

### HyperKvasir：完整增量与后续关闭

7 月 10 日 GPU1 记录包含 3 方法 × 3 种子 × 6 阶段，共 54 个阶段。这里重建的 Locked NC 控制采用固定 simplex/raw prototypes，**不是上一张表的 task-block calibrated Locked NC-ConCM**。

| 方法 | 5 个增量阶段平均 AccT | 最后阶段 AccT | 崩塌阶段 / 15 |
|---|---:|---:|---:|
| 重建 Locked NC 控制 | 28.75 ± 5.58 | 33.75 ± 4.39 | 15/15 |
| Full Dynamic | 44.77 ± 8.69 | 45.22 ± 9.46 | 3/15 |
| NC-anchored | 31.32 ± 0.97 | 31.05 ± 2.47 | 15/15 |

Full Dynamic 改善平均结果，但 seed1/session5 当前类准确率为 0%。后续 raw-prototype fallback、验证 checkpoint 诊断及 NC-safe dynamic 修复没有通过门槛；最新关闭报告判定旧 anchor 漂移为主要原因，`FULL_DYNAMIC=CLOSED`、`NO_FULL_PROMOTION`。

最新报告建议回到固定几何的 Locked NC-ConCM 主线；这是一项方法路线决策，**不代表其已获得稳定的完整 23 类多种子验证**。当前不能声称医学长尾 CIL 问题已经解决。

证据：[本次从完成输出中保存的完整指标](hyperkvasir23_full_concm_20260710/FULL_CONCM_MULTI_SEED_RESULTS.md)、[54 阶段 CSV](hyperkvasir23_full_concm_20260710/all_session_metrics.csv)、[最新关闭报告](HYPERKVASIR23_FULL_DYNAMIC_BRANCH_CLOSED.md)。

另有 7 月 29 日的独立 OLCD（old-NC/current-dynamic）实现和 sequence-dev 收尾工具。收尾工具描述了 seed1 gate 失败、旧类退化、禁止进入 confirmatory evaluation 的处理；但当前本地没有对应的运行报告/指标包。因此这里只列为已有实现及失败收尾逻辑，不能从代码模板推算或声称实际完成的数值结果。

## 证据边界与发布范围

- 7 月 10 日非 GPU1 的旧输出目录记录“未启动”，完成结果以 GPU1 文件为准。
- 本地 7 月 26 日 validation-checkpoint 输出记录 `NOT_EVALUABLE/NOT_RUN`，而后来的关闭报告记载了进一步诊断与修复。因此后续关闭结论目前有报告依据，但本地这批旧 JSON 不能用来独立复算后续修复数值。
- 本次发布当前源代码、配置、测试、脚本、历史报告、本摘要以及筛选的聚合 CSV/JSON；原始数据、逐样本材料、checkpoint、大型运行输出和参考论文不上传。
- 本次不以旧日志中的“无进程”或旧服务器路径推断当前运行状态，也没有新增实验或持续监测。
