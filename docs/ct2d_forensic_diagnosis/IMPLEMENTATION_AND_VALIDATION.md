# CT2-D 实现与执行边界

- 以 CT1 交付 345bd7714662a8b7404c4609770df4bba4808451 建立独立 worktree / 分支。旧源、报告和私有运行目录只读；没有实例化原 Run 或执行原 formal/main。
- 90 个归档仅在 CPU 上逐份读状态与 SHA，核对全部非共享参数名字/组哈希及连续变化，不将它称为重放 90 个 encoder。GPU 正式诊断恢复 12 个最终父模型及指定 4 个前任务模型；工程恢复额外计数。
- 独立 metric/confusion 实现复算全部 270/2754，并核对正确数、分母、宏平均、old/current 和 restricted 候选。保留原最小 original_label tie，bootstrap 仅复用固定 component 抽样工具。
- 沿用原 native 训练算子做零更新梯度探针；提取时原 FP32 Linear 调度与 pointwise 路由在独立模型启用。每次梯度探针前后比较完整模型指纹，正式提取结束再检查状态未变。AdamW/SGD/Optimizer.step 在诊断进程中被禁止。
- 54 个正式离线读出：48 个 2×2 矩反事实、6 个 U/Q11-Risk；另计 12 个独立加权批量校验解、P0 合成校验解。所有新拟合均如实记账。
- 当前类 mu/v/raw moment 的初始容差为 atol/rtol=1e-5；Q00 分数与原逐样本预测必须复现；Q11 逐样本加权解使用 1e-9 检查。明显非 PSD 阻塞；只对称化，不截断/加 jitter。混合矩 trace 改变不调整 lambda。
- P2 四个转移只用 current fit 估计共同对角映射；旧 fit/val 仅验证误差。输出 a 分布、边界比例、几何、类间距离和两个池的切换。梯度 B/2B 检查保留原 SUM/MEAN，不改 loss。
- 原 45 个 Risk 日志保留；6 个最终 U 的原 Risk 与刷新 Risk 补充 B/W 变化、无 slack margin、xi 和翻转；不新增其他 Risk 求解。
- 输入图像以固定 train/val 清单建立白名单；成功 Dataset 读取用共享计数器记账，含 loader worker。明确计入事后旧 train 读取；原 CT1 账本保持不变。所有 test/reserved 资产拒绝。
- 单 GPU worker，torch/BLAS 4 线程，loader 4 worker。内存映射仅用于当前模型 raw 暂存，完成该模型诊断后删除已登记的本轮 tmp 文件。没有历史资产清理。
- GPU 预算 3 小时，CPU 额外预算 2 小时；新持久文件 512 MiB，活跃文件 1 GiB，磁盘空闲至少 1 GiB。后台固定流水线结束后 STOP，不建监测。

工程证据：独立 CT1 全指标复算、合成矩公式检查、严格父状态恢复和 pointwise 同伴一致性、真实零更新梯度 B/2B 检查、54/837 合成报告覆盖与恒等配对 bootstrap 已分别执行。正式结果以运行完成凭证为准。
