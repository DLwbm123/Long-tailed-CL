# 来源与实现边界

用户 M1 修订优先于原 HK1 的数值 ID 相等及输入镜像限制；其他数据、训练、方法、精度、统计和预算规则不变。原 BLOCKED 保留在上一级目录。

复用 V2 的原生 Learner._init_train、真实预训练加载 mapping、频次 embedding 扩展、AdamW 和 cosine scheduler。训练工程将当前 S0 与提交 6d8bd3e700924c864e229c47efe5f108ce728c78 的原函数做同输入、同随机状态的一步 loss/gradient/update 验收。独立入口避免 generic incremental_train 构造 test_loader。

复用 A1 已接受的逐图 Linear 推理实现；复用 G1 Increment 的类均衡累计、直接 solve、数值检查和重复嵌入控制。R1 bootstrap 仅增加 labels 参数，默认仍为旧 8 类，本轮显式 range(23)。未改历史 R1 分析或扩大本轮矩阵。

新增部分限于 HK1 manifest/标签适配、紧凑 checkpoint 与恢复、资源记账、六阶段指标和报告。工程状态、真实源码摘要及失败证据以 JSON 锁为准；本文不代替验收。
