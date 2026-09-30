# DFD-T4-P-v1-M1：真实工程准入与资源阻塞报告

**execution_status=BLOCKED_RESOURCE_TIME；D/R 正式 optimizer steps=0；未启动 P1，未产生新 D/R val。**

基线为 `80f61322129ecc851411c72195a0d87d0466411b`。原 BLOCKED_ASSET 和 BLOCKED_UPSTREAM_STATE 报告保持原样。M1 已落实 FZ1 后置，本次阻塞来自 GPU 驻留预算，不是 FZ1 或资产缺失。

## 已完成的实际检查

- 六个原父、共享权重、137 份原绑定源码、原 manifest 和封存 Q 的 SHA/来源检查通过；六份 Q 的独立备份复读 SHA 通过。
- 12 份恢复控制的真实新 SHA、样本/标签/component/列顺序与当前 val 布局一致；控制没有重复训练或评价。
- 六个 case 均经过原完整 real loss 入口，各执行 2 个隔离 AdamW 更新，共 12 步，低于上限 16。初始 FD=0；更新后 D/R 的真实 FD 梯度均到达 main/few adapter。
- 教师独立、冻结、不变、无梯度；pointwise flags、路由对象、RNG 恢复；相同 loader/RNG 的首 batch 索引和增强逐张量一致。正式 D/R 全轨迹逐 batch 对齐检查已实现，尚未运行。
- 原 real loss 数值回归和空 Q/等系数旧 FD 回归通过。新 schema 在独立实例中恢复网络、optimizer、scheduler、全 RNG；错误 method/Q/coefficients/worker/protocol 被拒绝。
- 真实压缩保存、传输、独立端 SHA 复读通过。正式训练、rolling checkpoint、统一评价、配对 bootstrap、报告和逐单元备份入口已实现；其完整科学轨迹尚未执行。

|数据|seed|工程更新|非首 batch 训练秒/步|压缩保存 MiB|保存秒|备份秒|
|---|---:|---:|---:|---:|---:|---:|
|ISIC|1993|2|2.469|76.60|8.02|14.36|
|ISIC|1994|2|3.366|93.95|9.34|14.42|
|ISIC|1995|2|2.578|78.70|8.06|12.36|
|HK|1993|2|2.670|108.70|10.47|34.58|
|HK|1994|2|1.645|114.53|10.80|16.49|
|HK|1995|2|4.671|102.53|9.88|14.43|

## 为什么未启动正式训练

当前 prompt 第九节明确保留单卡 GPU 驻留上限 6 小时，包含工程、正式训练、评价与失败尝试；资源不足必须 BLOCKED。
固定 5500 步按各 case 真实非首 batch 耗时外推约 4.27 小时。已实做的 6098 张当前 fit 完整提取耗时 123.401 秒、128 batch；将该较低单位耗时用于全矩阵 6172 个提取/评价 batch，约 1.65 小时。加上本轮已消耗 16.51 分钟，低开销外推已为 6.19 小时，尚未加额外梯度探针、模型恢复、rolling 保存和独立备份。
自动准入的保守外推为 10.20 小时。该上界将小批提取中的固定 loader/同步开销也按 batch 放大，偏保守，不能称为预计实际完成时间；即便改用上述已完成的大批提取测量，6 小时也不足以准入完整矩阵。
存储准入外推 2.958 GiB，含主端、独立副本、15% 余量和活跃/其他预留，低于 3 GiB 但余量较小，运行时仍须执行硬上限。没有通过缩减 epoch、batch、精度、方法或删除历史资产来满足预算。

## 失败与修复均保留

第一次在备份操作器的 tar 迭代/版本切换处失败，SHA 不匹配被正确拒绝；没有训练更新。第二次检查脚本对 adapt_list 内的模块做深拷贝，再比较对象身份，误报上下文未恢复；修正为检查原对象身份，原科学依赖未修改，该失败同样发生在 optimizer 更新前。后续六 case 的检查全部通过。
截至阻塞，本轮累计 GPU 进程计时 990.686 秒（含保留失败账），工程更新 12，正式更新 0；实际 fit 读取 13702，新 val 读取 0；旧 fit、未来 fit、test/reserved、oracle 均为 0。P0/P0R 旧账另列，不清零，不重新复制已验证父备份。

## 证据边界与下一步

Task3 W：REBUILT_FROM_VERIFIED_PARENT，独立历史 W 匹配 NOT_AVAILABLE。CT6/CT9：REBUILT_FIXED_CONTROL，历史逐样本一致性 UNKNOWN。FZ1 六项保留 DEFERRED_MISSING_TASK4_READOUT、数值为空。
M/U1/U2/S=NOT_EVALUATED；U3=NOT_EVALUABLE_MISSING_FZ1；完整方法验收=PENDING_FZ1；original_five_condition_plan_complete=false；frozen_superiority=NOT_ESTABLISHED；full_method_promotion=false。
需要新的 GPU 时限授权才能准入 P1；可只把 6 小时放宽至 12 小时，科学矩阵、数据边界和存储上限保持原协议。现有真实工程结果可以复用，不需要重跑工程更新。未创建周期监测，未自动推送 GitHub。
NEXT_DECISION=STOP_PENDING_RESOURCE_AUTHORIZATION。
