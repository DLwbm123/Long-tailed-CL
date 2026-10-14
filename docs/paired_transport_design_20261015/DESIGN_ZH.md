# 配对仿射统计运输：CPU设计与调用链核对

本次仅合成CPU、无影像/医学checkpoint/val/test/optimizer/GPU。不更改Core、LT或旧轮科学源；两个新增helper不是生产worker。估计当前fit上teacher/current配对关系，并把同一估计仿射变换推到旧类充分统计；跨类泛化是未来需检验的假设，不是实际旧类漂移真值。

## 现有调用链

- `ct3p_core.features`：两分支CLS拼接后逐样本L2归一化，1536维。
- `prototype_coherent.append`：每类质量为1，保存加权均值μ、原始二阶矩Q、实际计数n；components保存center/mass/count及标量radius。Q不是中心化协方差，也不是未归一化的样本和。
- `run_prototype_coherent.common_shift`：先按当前类别求同图差均值，再对类别等权。
- `run_module_additions`：task起点提取teacher canonical特征作为before；epoch后同一canonical顺序提取current。每个epoch从未改变的task起点bank出发，fit-only估计start→current的shift；不把上epoch candidate累加进旧bank。最终boundary使用全部current train估计shift，并加入meta的原生新类统计。当前顺序检查只比较标签；未来配对映射需额外用私有行索引检查，不能只靠同类标签顺序。
- `prototype_coherent.translate`：同时更新μ、完整Q和全局component中心；平移下radius不变。
- `module_additions.local_parts`：CLS条件的四区域池化，经区域中心化与逐区域归一化。它不是global CLS的四份复制。
- `local_memory`：旧区域中心使用各区域的class-balanced共同位移，再追加当前类实际区域均值。`metric`只将区域中心组成4K列先验；head和competition不消费旧component的radius。

因此不可把global矩阵直接套到四区域，也不可在仿射运输后再加原common_shift。任何epoch候选都必须从相同teacher frame的task-start bank重建；boundary只封存一次。

## 估计与可识别范围

行向量约定`y=xM+b`，`M=I+UV`。权重`w_i=1/(当前类别数×该类fit n)`；按该质量中心化before，配对位移减去class-balanced共同shift。对`Z=diag(sqrt(w))(X−μx)`作SVD，秩阈值仅为浮点数值阈值`eps·max(N,d)·smax`，不按val选秩。

沿数值可识别方向估计残差映射，ridge为`.001·sum(s²)/rank`，.001前瞻复用原ridge数量级，不搜索。`b=class_balanced_shift−μx UV`。该正则带来偏差，不声称无噪声时精确恢复真实矩阵。

未观察输入增量h若满足hU=0，则`f(x+h)−f(x)=h`；这是**线性残差在未观察输入方向保持identity**，不是断言该方向真实无漂移，也不是说公共平移b不作用于旧类。合成中完全看不见的旧方向旋转仍有√2误差，必须保留这一负结果。

实际特征是归一化后的空间，映射也在此空间拟合。对旧统计不做额外逐样本归一化：旧样本不在，均值和二阶矩不足以恢复非线性归一化后的新矩。原生prior对aggregate中心的normalize保持原封，不等同于重新归一化历史样本。

## 原始矩与区域处理

`μ'=μM+b`，`Q'=MᵀQM+(μM)ᵀb+bᵀ(μM)+bᵀb`。

counts、class identity、component mass不变，component center做相同仿射变换。当前新类使用current实际特征的统计，绝不再套一次旧类映射。实现用低秩更新，不为每类另做完整稠密三次方矩阵乘。

标量radius仅给出了component协方差的trace，不能精确得到任意非等方变换后的trace。因此非平移时明确置`radius=None`并保存有效上下界；不伪造精确半径。原生head/competition/edge描述只用class μ/Q、component center/mass，已核对不消费旧radius；当前membership仍只使用当前真实seed半径。未来生产必须版本化这一schema并对完整调用链回归，不允许把None流入其他radius消费者。

区域每个域单独使用对应before/current特征估计自己的map，带`region_0...3`标签；禁止global map混用。四区域frame必须一致。原生LT先验仍是原公式，不再追加一份shift，也不重复加中心。是否值得让区域也升级仿射，必须有贡献对照，而不是因维度相同就默认共享。

## 当前状态与下一步

合成CPU设计通过，已验证1536×23统计/解析头与虚拟样本一致、零漂移精确、平移退化、坐标frame防重复、配对索引防错位及checkpoint roundtrip。没有生产loader、完整训练调用链、真实医学数据、真实GPU预检或模型效果。

下一步先完成隔离的生产CPU接线、原控制disabled精确回归、boundary fit/meta时序、最高旧类数的内存路径和版本metadata；前瞻贡献协议见CONTRIBUTION_PROTOCOL_ZH。只有完整生产CPU通过、正式科学源码和协议公开冻结后，才可以安排全矩阵真实零更新预检；不能把这个设计PASS当GPU-ready。
