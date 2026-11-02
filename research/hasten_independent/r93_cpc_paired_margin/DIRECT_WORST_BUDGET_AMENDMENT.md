# R93 0正式science工程计时附注

2026-11-02。在attempt1预算拒绝、正式optimizer/encoder/VAL均0的边界记录。原FIRST/deadline、科学模型/lambda/全部6144updates/角色/门不改，原工程收据和成本保留。不是partial optimizer恢复。

首预算使用C13数据时间乘C平方与查询行数的组合因子5.4547。该因子保守但不是实际最坏C23操作的测量；模块主要开销同时含固定小MLP和线性分数计算，C平方相乘可能重复放大。唯一新工程问题：直接执行与原训练相同代码的最坏C23/B1713完整pair forward/backward，而不再尺寸外推。

一次合成profile固定seed，不读取F/VAL：S按全部原TRAIN count生成合成单位向量，总6821行大于实际S3405，各类count不超过原TRAIN；B1713、y循环23类。对五个cap1/2/4/8/FULL各执行两pass、均计成本；只采用预定第二pass，各cap取最大（不是挑最快），训练时每pairzero_grad，因此计时包含zero_grad。另一次合成AdamW step计engineering1，不进入正式计数，不沿用该参数或optimizer，另计完整eval-context最坏C23/F1703、4模型×2side×3档的指标与前向。

新投影=最大第二pair时间×6144×4×安全4 + AdamW单步时间×6144×安全4 + 实测eval-context时间×120×安全4 + 3600秒I/O/保存/分析余量。全部既有准备/排队/重试成本消耗原clock，只有新投影<剩余−180s才准入；安全4不变，不改科学更新/模型/采样/门。若仍不通过，不再重复profile/挑profile或减安全系数。本预算方法替代外推不是二者择小，原拒绝完整披露。

工程profile受同原clock控制器、CPU4/内存/磁盘lock守卫；profilesynthetic optimizer1单列非science。首次fit前复用原软件与真实TRAIN梯度/restore PASS，不重复其测试，重新从原seed初始化所有模型/优化器；绝无中途优化器状态恢复。启动前核原attempt退出、所有实际计数、无正式端点与无其他launcherreceipt，原位保存attempt2工程与后续主attempt成本。
