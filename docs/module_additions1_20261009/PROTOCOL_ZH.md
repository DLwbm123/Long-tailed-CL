# 单模块增量实验：冻结协议

用户授权在已有方法上逐一尝试四个推荐模块，使用GPU0/1/2并每小时监测。此新周期不是复活RL，也不重置历史成本。

主体为最近 static_pc：预训练ViT、既有adapter、类别/组件充分统计、原型先验、竞争二次目标、FD10与解析求解均保留。ISIC和HK各五臂，共10条正式轨迹。种子74002、原shuffled顺序、任务大小、identity fit/meta划分、两epoch、batch64、AdamW .0003及原cosine schedule保持；pace臂只按历史秩缩放整个schedule。ISIC468、HK176更新/臂，共3220正式更新。3个ISIC零更新预检加CPU自检。新周期墙钟上限24小时，不添加科学配置或自动重跑失败正式轨迹。

## 四个单模块

1. **层级先验（HASTEN思想）**：依据公开训练标签名称预定义粗粒度研究family，只有当前已见同family至少两类时，将其类均值等权聚合、L2归一化为父锚点。父锚点与原P拼接，仍用R=(I+PPᵀ)⁻¹及原解析求解。ISIC：melanocytic={NV,MEL}、keratinocytic={BCC,BKL,AK,SCC}，VASC/DF单独；HK：Barretts、BBPS、polyp/procedure、oesophagitis、UC分级，同名类族之外各类单独。这是研究设计，未声称临床本体已验证。无CLIP文本编码、超曲空间或未来样本。
2. **局部证据（COMPOSE思想）**：同一ViT两分支最终patch token，在四固定象限内按与CLS余弦的softmax聚合，局部向量减每图四区域均值后归一化；类别到达时统计每类四个局部均值，每区域以1/2权重加入P。原全局表示及线性推理保留，原mu/Q和竞争规则保留。无DINO换骨干、额外slot/router、旧样本回放、Chamfer或推理支持集。历史局部均值沿用共同平移假设；不能把这种简化看作完整COMPOSE复现。
3. **步幅节奏（PaLoRA思想）**：历史每任务adapter参数增量向量的Gram特征值，累计能量95%有效秩R；当前完整原LR曲线乘1/√max(1,R)。参数历史秩不同于原LoRA更新有效秩；不声称原理论适用，未实现原方向投影/补偿损失。
4. **任务末融合（DAF思想）**：上一global adapter、历史优化adapter算术平均与当前优化adapter，按原Eq3/14逐坐标融合；alpha1.25、beta截断[.001,.499]。仅增加任务末模块，不重置当前起点。当前fit额外一次canonical CE梯度通道估计batch Fisher/梯度，近零位移采用曲率项回退；这是batch近似，不是逐样本Fisher。融合后重新提取当前特征再按原方式运输bank，无旧影像/生成特征回放。

## 评价与停止

按base→hierarchy→local→paced→fusion顺序向3个GPU派发，两数据集独立匹配，不组合模块。10条全部TRAINED且更新计数一致后才打开固定开发评价。base逐阶段逐类召回须复现既有static_pc；不通过则标工程/来源异常，不能放宽。方法效果门：ΔBA≥1pp、Δtail≥−0.5pp、Δforgetting≤1pp、Δnew≥−1pp、Δold≥−1pp。完整报告所有正负、逐类分母、激活、附加计算及失败。旧开发集已反复查看，单seed单order不是独立确认，通过只称开发信号。无test/CIFAR读取、调参、额外seed或自动组合。

后台进程使用中性命令行。每小时监测同一任务，健康无实质变化保持安静；实际完成后整理匿名报告、源码和聚合结果，代理推送公开GitHub并核实访问，再暂停同一监测。私有配置、影像、身份、权重、特征、原型向量和原日志不公开。
