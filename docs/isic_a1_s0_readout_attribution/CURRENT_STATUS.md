# A1 当前状态

2026-09-17（北京时间）：**COMPLETE_A1_ATTRIBUTION，NEXT_DECISION=STOP**。正式worker退出码0，完整8变体×3固定父模型/顺序配对×3阶段完成，主指标72行、逐类432行（核心36/216），正式特征57,039行。保存分数复核通过，神经训练/optimizer step/test访问均0。

主比较S-J-CB的Final BA为59.706%，A-CB为55.497%，差值+4.208 pp，条件性95%区间[+1.704,+6.876]。tail差值+1.496 pp，区间[−2.667,+5.797]；标签6平均召回下降，不能称尾类全面改善或独立确认。M/F仅作固定诊断，不替换J主比较。

执行源码`4c74c4b1e04b396f24c79bbd8819a59bf662d408`。首轮18个A拟合原位复用，新S拟合54个。累计GPU驻留观测约21.4分钟、保守上界22.02分钟；新增文件约350 MiB、最低空闲1.767 GiB，预算通过，无需删除历史资产。本次限时监测随完整交付提前关闭，不自动扩展实验。

- [最终中文报告](repair_01/FINAL_REPORT_ZH.md)
- [来源与局限](repair_01/SOURCE_AND_LIMITATIONS.md)
- [交付核验](repair_01/DELIVERY_AUDIT.json)
- [主指标72行](repair_01/val_metrics.csv) / [逐类432行](repair_01/val_per_class_metrics.csv)
- [配对区间](repair_01/bootstrap_intervals.csv)
- [资源计账](repair_01/RESOURCE_REPORT.json)

- [工程修复证据](oversight_20260916/REPAIR_REPORT_ZH.md)
- [正式启动记录](repair_01/FORMAL_LAUNCH.json)
- [正式工程与预算](repair_01/ENGINEERING_A1.json)
- [首轮阻塞报告（历史，保留不覆盖）](FINAL_REPORT_ZH.md)
