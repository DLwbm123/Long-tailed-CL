# CT6-F CURRENT_STATUS

RUNNING — 2026-09-19，六父严格验收后实际启动，未完成。

- 独立分支exp/ct6f-full-fd-continuation；科学编排源码2d4aa7b12ddbe6a4d15633aaf0bc071197bb810c；驱动源码00736901141ff0736a62f0d455b4b11a5d71dafa，原CT3依赖SHA保持不变。
- hb01 SSH30128，中性根/tmp/p25root；GPU主入口p25.py PID14034、驱动p25driver.py PID14033，仅定位用，判断状态必须读实时日志/退出码/锁/资源。
- 严格普通restore_new验证全部六父模型/原配置/原代码/manifest与三种RNG；A/T计数与6类W重现，Task4教师等于Task3自身且参数不共享。工程0训练步，GPU资格检查耗时以QUALIFY_RECEIPT为准。
- 数据盘创建/读写成功，启动前约2.2GiB空闲；父副本逐条用后删除，原my-gpu归档保留。新27checkpoint归档服务在my-gpu /tmp/p25archive（实际NFS /remote-home/wangbomin/LongTailedCL/archives/ct6f_q8m25），pid2399902，日志service.log。不要新建重复服务。
- 固定270task-epochs/5570正式steps/27新checkpoint；六轨迹只Task4起，HK至11、ISIC至4。C2/C3两读出共享轨迹。全部W/checkpoint锁定后新增54阶段val预测，原36前缀预测复用；test/reserved/oracle/旧fit访问0。
- 预计1.5–3小时，仅实际吞吐估计，无GPU小时上限；CPU/空间限制保留。负结果不取消后续配对。

远程output/train.log、evaluate.log、report.log及对应.exit；public/ENGINEERING_GATE、QUALIFY_RECEIPT、RESOURCE_LEDGER、TRAIN_EPOCH_METRICS.jsonl、STATE_W_LOCK、PREDICTIONS_LOCK、PROCESS_RECEIPTS、COMPLETE/FAILURE。初次启动已看到原10epoch循环开始，四loader worker命令行均为中性p25.py。

新工作没有恢复CT3/4/5已关闭任务。完成后按PLAN比较并公开聚合交付，NEXT_DECISION=STOP。技术中断不得盲重启：保留失败、累计资源，明确同锁恢复路径和源码绑定；不得重训前缀或已完成任务。
