# D43 原适配器求和的幅度可行性（运行前冻结）
目的：区分“新增adapter改变旧表示方向”（D36已诊断）与“按任务求和使表示幅度系统增长”这两个命题。只计算封存参数的仿射输出范数，不复算分类、距离、logits或既有科学指标，不训练/拟合/编码。

R28linear64三域全部36增量边界×3seed，固定旧TRAIN类原始均值（旧state fc.weight）与旧类单位prompt bank（严格从checkpoint取），两模态共990类×边界行。使用同一固定输入，比前一模型旧模块和、当前模型相同旧前缀和（须exact）、当前全部模块和。计算norm_ratio=||newsum||/||oldsum||，原绝对对数范数变化，以及反事实“每task模块平均”两边分别除以K、K-1后的同量。不测试其他缩放，不拟合系数。

事前可行性：NIH在Image和Text两模态各至少2/3seed，原sum ratio中位数>=1.05，且固定mean反事实平均绝对log范数变化相对减少>=10%；ISIC/HK两模态各自跨seed平均相对减少不为负。若不支持，停止固定mean路线，不扫1/sqrt(K)、指数、模态单侧、class/stage缩放或阈值。不改数据、loss、树、projection或fusion。即使可行性通过也只支持独立完整训练，不证明hyper径向保持、性能或因果；未通过不否定所有ensemble方案。

0优化/0分类评分/0图像/0VAL清单元信息特征响应/0TEST/0新RNG。训练配置保留VAL路径元字段但不使用，不宣称配置完全无元信息。checkpoint/缓存都是封存TRAIN依赖。全部旧参数不变，完整类/seed正负结果保留；CPU4、独立600s/300MB含16MiB余量、CPU2400s/RSS8GiB、file64MiB，预计<5MB。pro/data ext4容量写读probe，CPU-only不占GPU但先核存储；中性完整主子argv、独立guard、原钟含所有修复。预算不重置，不删除共享历史。结束安全结果/源码/报告交付。
