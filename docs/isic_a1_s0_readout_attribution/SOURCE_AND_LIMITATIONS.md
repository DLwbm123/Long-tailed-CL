# A1 来源与限制

A1的联合输入、逐样本路由、解析目标、矩阵及资源界限来自本轮用户批准的设计，不是GSR、ConCM或APART原论文规则。固定依据G1 8507c2d、R1 2ab2c269、V3 cda1b371、V2交付a18b25a8及训练源6d8bd3e。

恢复直接使用历史AdapterVitNet结构，完整strict load网络state_dict；先验证checkpoint文件、训练配置/顺序/来源/协议元数据，再匹配R1所记录的完整网络tensor哈希。采用mmap读取checkpoint容器，立即丢弃未使用的optimizer/scheduler等条目，不创建训练learner、不实例化或恢复optimizer、不把其tensor搬到GPU。父训练checkpoint未修改。

与原方法的明确差异只有探针两项batchwise_prompt旗标，以及A1既定归一化和新解析读出。保持原生逐样本top-1及所有既有adapter；main/few按实际输出键读取。绝不将结果包装成“对K只换分类器”。不同基础类对应不同父模型，每个父只配其历史顺序，不全交叉。

重用G1的float64统计/直接solve/状态恢复，以及R1的指标、分层component bootstrap。R1指标函数增加一个显式可选的最小original_label tie规则，仅A1启用；旧R1/G1默认行为不变并运行原selfcheck。重复嵌入控制不构成新增候选；原始A不二次归一化。没有新增方法、正则搜索、训练或test授权。

raw主/辅特征仅各提取一次，按当前到达类写入float32 memmap；派生视图即时在float64构造。只保存一个可恢复累计状态，完整旧类Q不归档。完成全部阶段后做同父模型、类内逆序/分块的隔离重聚合检查；它不回流前期模型。可选train重代入评分未执行，保留必需train几何诊断。

这些结果只能说明固定探针的线性可用性；cosine、margin及condition不证明不可逆信息损坏或临床有效性。validation反复用于开发，既有test反馈、未知患者关联、临床标签语义和预训练暴露限制仍然存在。区间不代表独立确认或训练总体，原始A最终是单一确定性参照，S的三个父模型/顺序配对不是独立患者划分。

历史资产、现有source-only目录及五个历史CSV工作区修改保持原状，独立分支不合并main。原始数据、raw缓存、患者/病灶关联、精确路径、W及逐样本分数私有。协议、hash与必要聚合公开；不把统计记忆称为天然匿名或隐私安全。无论方向如何，P3之后停止。
