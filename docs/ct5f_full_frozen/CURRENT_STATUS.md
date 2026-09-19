# CT5-F CURRENT_STATUS

RUNNING — 2026-09-19 启动已验证；不是实验完成。

- 独立分支：analysis/ct5f-full-frozen-reference。
- 科学源码：c466ff68625502fbb5fb9096c44df6785b2c7058；PROTOCOL_LOCK已在运行前写入。
- CT3-P/CT4-F均已完成关闭，未重启。
- 服务器当前端口30128，中性根/tmp/p24root，驱动PID11441，GPU worker PID11442。使用实时退出码与回执判断，PID仅供定位。
- 已通过实际数据盘写读探针、六父可用性、90个历史控制分数锁、18个CT4前缀W锁核验及CPU边界/指标测试。
- 启动后日志已到HK1993 Task8解析拟合，前3阶段W数值重现通过。
- 首个新增Task4：667图像/5.716秒；峰值CUDA allocated 1.9765GB。盘剩余约3.30GB，未清理历史资产。
- 45个F1阶段全部W锁定后再进行新增27阶段val；18阶段复用原预测。新增神经epoch/step=0，test/reserved=0。
- 顺序重建Task2/3统计是明确披露的前缀重建，不重新生成其val预测。完整输出预计15–30分钟；实际资源以进程回执为准。

远程证据：output/inference.log、inference.exit、report.log、report.exit、public/PROCESS_RECEIPTS.json、FIT_AUDIT.json、STATE_W_LOCK.json、PREDICTIONS_LOCK.json、RESOURCES.json、INFERENCE_COMPLETE.json、COMPLETE.json；失败保留FAILURE/DRIVER_FAILURE。

按小时监测。成功后复制公开聚合交付、完善科学结论、推送本分支并匿名验证。无论结果正负，本固定研究NEXT_DECISION=STOP。
