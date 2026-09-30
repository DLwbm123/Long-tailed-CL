# M1 公开交付说明（2026-09-30）

用户于 2026-09-30 明确授权提交并推送本轮源码和脱敏结果。本目录对应 DFD-T4-P-v1-M1；基线为 80f61322129ecc851411c72195a0d87d0466411b，发布目标为 exp/dfd-t4-p-v1 分支。

训练与统一评价已于北京时间 2026-09-30 03:06 完成：12 条轨迹、5500 正式更新（12 工程更新另计）、120 epoch、120 梯度探针、24 行阶段指标、192 行逐类结果及 6 行 FZ1 缺失登记。所有训练单元、预测和最终报告备份回执均为 PASS，最后 completion_receipt 请求与 ACK 匹配。GPU 累计驻留 24078.665 秒；持久化含独立备份 2852479539 字节，均在修订预算内。

科学结果见 [最终报告](FINAL_REPORT_ZH.md) 和 [判定表](decision_gates.json)：ISIC 与 HK 的 M/U1/U2 均为 FAIL，S 为 PASS；FZ1 后置，完整验收仍为 PENDING_FZ1，原五条件计划未完成。NEXT_DECISION=STOP；本轮小时监测已暂停。

## 公开范围和历史记录

本次仅交付三个 M1 Python 源文件、聚合指标、诊断和脱敏审计。图像、逐样本 ID/component、访问原始日志、Q/W/特征、预测、checkpoint 和私有交接均未纳入。

RUN_STATUS、PROCESS_RECEIPT、早期预算审计和失败记录是各自时间点的历史快照，不改写为最终状态。DELIVERY_AUDIT 中 AWAITING_SEPARATE_AUTHORIZATION 是运行结束时的历史发布状态；本说明记录随后获得的明确发布授权，Git 提交及远端访问验证用于确认公开交付。

FAILURE_1790684657666509650.json 的 traceback 中一处私有绝对路径替换为 <PRIVATE_PATH>；未改动异常类型、行号、代码和科学结果。原件保存在私有主归档、独立备份及本地私有副本。已有回执中的 SHA 指向原始产物，不声称脱敏副本与其字节相同。RESOURCE_LEDGER 与 BACKUP_RECEIPTS 已同步 worker 结束后的最终账和回执；旧备份快照仍按其原时间解释。
