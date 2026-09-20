# 固定 HK1-M1 方法卡

计划矩阵：三个新 S0，每个 10 epoch，任务 13+2+2+2+2+2；A-CB、S-J-CB、S-M-CB，J−A 唯一主比较。实际完成状态以 COMPLETION_AUDIT 为准。

沿用原生 APART S0 三项分类项、原 reduction、/3、pool assignment、pull 与 theta。原核心冻结，12726×16 频次 embedding 以真实 fit 数索引，未来十行无 CE 梯度但可有 AdamW weight decay。原始预训练核心共享引用，最终全部非共享张量保留，滚动恢复仅保留当前 run。

S0 后冻结两分支；推理按 A1 固定关闭两池 batchwise majority，并沿用 FP32 三维 Linear 逐图调度，native B1 不变；该补丁不进入训练。A 是原始 AugReg CLS 的 float32 L2；J 是 raw main/few 拼接后的整体 float64 L2；M 是 raw main 的 float64 L2。

全部读出为 mean-of-class-mean 平方误差 + 0.001||W||²，float64，无 bias。逐类到达不读取未来类统计，所有 W 锁定后统一评价 val。S1–S5 神经更新为零。保存统计与缓存是科研资产，不是隐私安全或零辅助存储。

固定 2000 次、seed 44001 的类分层精确内容组件配对 bootstrap，共享所有方法/阶段/顺序的抽样；父模型固定不重采样。未知患者关联、历史开发暴露和极少 val 组数都限制解释。
