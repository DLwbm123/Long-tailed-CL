# CT13-ISIC 已核验聚合结果

本文件只记录服务器租期结束前成功核验的公开聚合证据，不包含私有图像、逐样本预测、特征、权重或 checkpoint。

## 执行范围

- 数据集：ISIC2019-LT，三个既有 seed/order（1993/1994/1995）
- 阶段：2+2+2+2，共 12 个阶段
- 固定读出：10 个，主读出 A6
- APART 与 OpenCLIP 双预处理，联合特征维度 1536+512=2048
- 新增神经训练 epoch：0；optimizer step：0
- test/reserved 访问：0

## 最后一次成功核验

服务器租期结束前，以下文件均已存在且进程已正常退出：

- `COMPLETE.json`
- `REPORT.json`
- `stage_metrics.csv`（120 行阶段指标）
- `class_metrics.csv`（600 行逐类指标）
- `PREDICTIONS_LOCK.json`
- 12 个 `STATE_W_LOCK.json`

汇总账本记录：

| 项目 | 数值 |
|---|---:|
| 阶段指标行 | 120 |
| 逐类指标行 | 600 |
| train 图像读取 | 56,154 |
| val 图像读取 | 885 |
| historical fit rereads | 0 |
| within-run old fit reads | 0 |
| test 访问 | 0 |
| reserved 访问 | 0 |

报告状态为 `PENDING_UTILITY_GATE`：A0 历史 F1 只记录为 `LOCKED_REFERENCE_PROVIDED`，效用门槛尚未在该执行器中完成判定。这不是训练失败，也不应解读为正向性能结论。

## 谱系与可复核边界

- 实现基点：`674942161b3bed60a3264a2e21a75c3785c81660`
- CT13 执行代码最终本地提交：`ab5fff2`
- 固定 OpenCLIP 权重 SHA256：`5dd0b33400e35f5ddb982b37d5925ab98a999079855dd5465cc9728aee224271`
- 服务器原始逐阶段/逐类 CSV 未在租期结束前同步到本地；本公开分支不声称包含这些缺失明细。

