# R93 用户明确授权的实现优化（原时钟，0正式science边界）

2026-11-02。用户最新明确要求先优化实现，不改方法、步数、对照，通过24h预算即启动；这覆盖此前工程结案中不再profile的执行限制，不重置FIRST/deadline。先核原attempt1/2均exit0、正式参数端点0；保留旧预算拒绝与全部费用。

唯一性能改动：每个load_stage内对不可变S角色和标签一次分组并排序，缓存类→有序ID列表；原实现每pair、每类扫描全部S并逐项读取Torch标量标签。配对a/b offset、m、成员排序和索引、state计算、网络、loss、梯度、lambda、随机seed、held隔离、全部6144steps及对照/评估门均不变。cache随stage数据释放，不缓存网络输出或梯度，不读F/VAL，内存只小型ID元数据。预算与独立硬守卫沿用，无GPU需要。

等价检查：所有六stage合法S池与原定义列表逐项相同；完整1024训练pair及20评估pair每stage的选择IDs/overlap/m与原公式一致，held取同池的相同子类。两端stage、五档、训练和评估索引的原/新双侧state/p0/coverage相等。新网络非零末层权重的lambda0/1、cap4/FULL损失及完整参数梯度逐元素比较，容差1e−12且报告实际误差，0正式optimizer/encoder/VAL。测试和profile全计工程费用。

等价通过后一次复用原五cap、每cap双pass直接C23/B1713最坏合成profile，固定第二pass、跨cap取最大、安全4及3600s余量、eval120和AdamW计时不变。profile第一pass包括缓存构建，均计成本；预定投影只用第二pass，原全plan余量包含六stage一次缓存构建。投影须小于原剩余−180s且资源/锁准入才正式fit；从原seed精确新建全部模型和optim，不加载任何已训练状态、不partialresume。原预算拒绝不删除，新的工程计数单列。
