# CT9-P CURRENT_STATUS

QUALIFYING — 单worker已实际启动，六个原Task3父状态正在逐个拉取并做严格恢复验收，尚未确认正式训练启动。科学源e7f9b9afa36139d6ceffe818d0840fbee3fd7082，独立分支exp/ct9p-fixed-fd-perturbation。驱动/tmp/p28driver.py初始PID41272，工程GPU入口/tmp/p28.py初始PID41273；my-gpu归档服务/tmp/p28s.py PID2442833，中性命令行已检查。

固定beta1仅Task4、HK/ISIC×三原seed，60epochs/2750steps/6checkpoint；B1A/B1T共享轨迹，与CT6 Task4 C2/C3及CT5 F1比较，HK仅8类前缀。原CT3–8均STOP，不重启。

工程通过后driver自动train→evaluate→report，不重复启动。状态以output/*.exit、PROCESS_RECEIPTS、ENGINEERING_GATE、TRAINING_STARTED、STATE_W_LOCK、PREDICTIONS_LOCK、COMPLETE/FAILURE、RESOURCE_LEDGER为准。全部checkpoint/W锁后才val，test/旧fit/未来fit/oracle0。

活动盘≤2GiB、空闲≥1GiB；GPU无小时截止，仅固定矩阵。新归档/tmp/p28archive在my-gpu真实NFS，已挂载/写读探针通过；逐父临时副本验证后可删，六原父和所有新末状态归档保留。传输600秒/ACK660秒，原master/tmp/q25-transfer-r1；不得重复service或worker。若技术失败，保留失败/资源，明确同锁恢复并绑定新源码，不盲重跑已更新步骤。
