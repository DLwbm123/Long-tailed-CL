# CT11-H CURRENT_STATUS

QUALIFYING — 已实际启动单worker工程验收，首个CT6 Task4状态严格恢复PASS，正在逐个恢复12个模型。尚未确认新增val预测；工程全部通过后driver自动evaluate→report，不新增训练。驱动/tmp/p30driver.py PID49637，GPU入口/tmp/p30.py PID49638，my-gpu GET-only服务/tmp/p30s.py PID2474551。以实时exit/日志/锁/账本确认，PID仅作定位。

独立分支analysis/ct11h-fixed-head-readout。诊断科学源码4f70240a317d18f3cd2745cf830fbf3ef11b6f1f，锁交付44d9895eba280534a7bf4e71de225ed59c9ec0c1（公开已匿名核验）。CT9 Student恢复仍绑定原e7f9及原PROTOCOL_LOCK字节，新的DIAGNOSTIC_LOCK独立约束本次推理，未绕过原普通恢复。

固定CT6 beta10/CT9 beta1 × HK/ISIC × 三seed Task4共12状态；新增N10/N1原神经main+few预测12单元，复用原24 ridge，共36/288行。全部pointwise probe、batch48同布局，仅原val；0新训练/steps/checkpoint/analytic fit/test/旧fit/未来fit/oracle。统一评价须复现每状态原A/T ridge分数与argmax，不一致即BLOCKED，不改容忍或路由。HK仅八类前缀；不选择更优head替换正式成绩。

服务器hb30128，中性根/tmp/p30root实为数据盘ct11h。启动GPU空闲24081MiB、数据盘空闲3.23GB。新盘≤1GiB、空闲≥1GiB，GPU无小时截止，CPU报告≤600秒；只此固定矩阵。归档服务日志在my-gpu/tmp/p30archive真实NFS，源/tmp/p25archive与/tmp/p28archive只读。逐个临时checkpoint删除仅发生在严格恢复后，源归档不删；服务仅GET、600秒传输/660秒ACK，不自动重试。完成或失败driver写STOP_TRANSFER；不重启历史p22–p29。

完成后交付全部正负结果及条件性区间，NEXT_DECISION=STOP。本轮不扩任务、不训练、不搜beta或校准。失败保留已做访问和资源，不盲重跑已完成单元。
