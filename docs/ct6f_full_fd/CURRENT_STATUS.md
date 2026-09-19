# CT6-F CURRENT_STATUS

TRAINING_COMPLETE_EVALUATING — 2026-09-19 15:38 UTC heartbeat实测。train_r1.exit=0；270/270 task-epochs、5570/5570正式steps、27/27末checkpoint全部完成。STATE_W_LOCK=LOCKED，54个W与27个checkpoint已锁定。禁止重启训练。驱动29287已转入统一val，评价PID39847（仅定位）；待evaluate_r1/report_r1.exit、PREDICTIONS_LOCK、COMPLETE与公开报告验收，当前尚未完整交付。累计GPU-process residence最新约13587秒，包含历史失败及恢复，不归零；test/old-fit/future-fit/oracle访问仍0，空闲约3.24GB。

以下为历史恢复及启动记录：

RUNNING_RECOVERED_R1 — 2026-09-19 13:41 UTC，归档超时已修复且实际续训，未完成。

原训练在180epochs/2350steps后归档第18个checkpoint超时退出1，见RECOVERY_R1.md与R0证据。严格普通restore、末次A/T统计重建及既有W核对全部PASS；同字节归档重传105.219秒成功，没有重复训练。已经从HK1995 Task6续训并完成两个epoch。恢复启动时剩余90epochs/3220steps/9个新checkpoint。

当前驱动/tmp/p25driver_r1.py PID29287、GPU入口/tmp/p25resume.py PID29288；归档my-gpu /tmp/p25s_r1.py PID2413513，使用新/tmp/q25-transfer-r1 master。旧进程已退出，不得启动原driver/train。优先读train_r1/evaluate_r1/report_r1日志和.exit、PROCESS_RECEIPTS_R1；原train.exit=1及R0为保留历史，不代表再次失败。原科学源不变，恢复源码4608f49ccfd5e742755f1a1f2936898edcbc9bbf及RECOVERY_LOCK_R1单独绑定。传输600秒/ACK660秒为有界技术修复，模型/loss/seed/epoch/data不变。

以下为原启动审计（旧PID仅历史）：

- 独立分支exp/ct6f-full-fd-continuation；科学编排源码2d4aa7b12ddbe6a4d15633aaf0bc071197bb810c；驱动源码00736901141ff0736a62f0d455b4b11a5d71dafa，原CT3依赖SHA保持不变。
- hb01 SSH30128，中性根/tmp/p25root；GPU主入口p25.py PID14034、驱动p25driver.py PID14033，仅定位用，判断状态必须读实时日志/退出码/锁/资源。
- 严格普通restore_new验证全部六父模型/原配置/原代码/manifest与三种RNG；A/T计数与6类W重现，Task4教师等于Task3自身且参数不共享。工程0训练步，GPU资格检查耗时以QUALIFY_RECEIPT为准。
- 数据盘创建/读写成功，启动前约2.2GiB空闲；父副本逐条用后删除，原my-gpu归档保留。新27checkpoint归档服务在my-gpu /tmp/p25archive（实际NFS /remote-home/wangbomin/LongTailedCL/archives/ct6f_q8m25），pid2399902，日志service.log。不要新建重复服务。
- 固定270task-epochs/5570正式steps/27新checkpoint；六轨迹只Task4起，HK至11、ISIC至4。C2/C3两读出共享轨迹。全部W/checkpoint锁定后新增54阶段val预测，原36前缀预测复用；test/reserved/oracle/旧fit访问0。
- 首两个正式epoch各约25秒/14steps，更新预计2.5–3.5小时，仅实际吞吐估计，无GPU小时上限；CPU/空间限制保留。负结果不取消后续配对。

远程output/train.log、evaluate.log、report.log及对应.exit；public/ENGINEERING_GATE、QUALIFY_RECEIPT、RESOURCE_LEDGER、TRAIN_EPOCH_METRICS.jsonl、STATE_W_LOCK、PREDICTIONS_LOCK、PROCESS_RECEIPTS、COMPLETE/FAILURE。初次启动已确认HK1993 Task4完成2epoch/28steps，loss有限、FD_mean为正、teacher_unchanged=true、adapter梯度检查通过。峰值CUDA allocated 18,599,083,008字节，空闲盘2,148,687,872字节。四loader worker命令行均为中性p25.py。

新工作没有恢复CT3/4/5已关闭任务。完成后按PLAN比较并公开聚合交付，NEXT_DECISION=STOP。技术中断不得盲重启：保留失败、累计资源，明确同锁恢复路径和源码绑定；不得重训前缀或已完成任务。
