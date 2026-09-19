# CT9-P CURRENT_STATUS

RUNNING — 六个原Task3父状态严格恢复及工程验收全部PASS，qualify.exit=0，工程optimizer steps=0。2026-09-19 19:43 UTC轮次实测正式训练已完成至少42/60 epochs、1670/2750 steps；HK三seed及ISIC1993四条轨迹完成并归档，当前ISIC1994。训练记录loss有限、teacher固定、adapter梯度通过；没有新FAILURE。尚未释放新增val，test/旧fit/未来fit/oracle访问均0。

科学源e7f9b9afa36139d6ceffe818d0840fbee3fd7082不变，独立分支exp/ct9p-fixed-fd-perturbation。驱动/tmp/p28driver.py PID41272，正式GPU入口/tmp/p28.py PID41383；my-gpu归档服务/tmp/p28s.py PID2442833。PID仅供定位，以实时回执与日志为准。当前累计GPU process residence约3397秒（含工程），峰值分配18.61GB，活动输出约278MB、空闲约2.96GB；快照不是最终资源账本。

固定beta1仅Task4、HK/ISIC×三原seed，60epochs/2750steps/6checkpoint；B1A/B1T共享轨迹，与CT6 Task4 C2/C3及CT5 F1比较，HK仅8类前缀。原CT3–8均STOP，不重启。

工程已通过，driver按既定train→evaluate→report继续，不重复启动。状态以output/*.exit、PROCESS_RECEIPTS、ENGINEERING_GATE、TRAINING_STARTED、STATE_W_LOCK、PREDICTIONS_LOCK、COMPLETE/FAILURE、RESOURCE_LEDGER为准。全部checkpoint/W锁后才val，test/旧fit/未来fit/oracle0。

活动盘≤2GiB、空闲≥1GiB；GPU无小时截止，仅固定矩阵。新归档/tmp/p28archive在my-gpu真实NFS，已挂载/写读探针通过；逐父临时副本验证后可删，六原父和所有新末状态归档保留。传输600秒/ACK660秒，原master/tmp/q25-transfer-r1；不得重复service或worker。若技术失败，保留失败/资源，明确同锁恢复并绑定新源码，不盲重跑已更新步骤。
