# CT4-F状态

PREPARING_PARENT_TRANSFER。独立控制，正常CT1-U Task1末冻结全部参数，Task2/3仅合法current-fit解析统计累积；18/72结果，无新增训练/test。固定计划见PLAN.md。CPU合成非恒等类别映射/配对BA/bootstrap检查PASS。父状态正在通过tar stream复制到hb01 port30128 `/tmp/p23root/output/private/parents`；全部完成并严格校验后才启动。源依赖只读复用`/tmp/p22root/code`，无CT3-P重训；后续读PROTOCOL_LOCK、RESOURCES、FIT_AUDIT、STATE_W_LOCK、PREDICTIONS_LOCK、COMPLETE或FAILURE和进程退出码。
