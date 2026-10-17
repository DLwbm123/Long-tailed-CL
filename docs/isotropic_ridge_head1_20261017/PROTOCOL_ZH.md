# 保留类内散布总量、移除方向结构的固定开发消融

## 证据与假设

原型岭删除全部类内散布后，相对完整岭四切片总体BA分别下降9.5020、3.6762、3.5064、8.9398pp，且总正确数均下降，数值求解正常。该结果未区分散布的总量与方向结构。本轮检验：只保留类内散布trace是否足以保留完整岭表现。仅在既有ISIC/HK分类特征与现有支持范围内运行，不扩展种子、数据或标注。

## 唯一主要改动

令K为seen类数，d=1536，原始类均值μ不额外单位化；G=(1/K)Σ_c mean_i z_ci z_ciᵀ，G_between=μᵀμ/K，G_within=G−G_between。固定τ=trace(G_within)/d，由本轮支持特征确定。

G_iso=G_between+τI；W=(G_iso+0.001I)^−1(μᵀ/K)；评分固定8zW，无bias、无列归一化。只将类内散布替换为同trace的各向同性矩阵，保留类均值外积和右端。τ不按查询选择，无强度、秩、温度或正则网格。等价表达是原型岭的λ_eff=0.001+τ，因此不宣称新方法，也不称为原逐样本最小二乘的等价解。保留trace不保留谱或方向几何。

复用既有analytic.ridge；以新dict中的S=μᵀμ+trace(S−μᵀμ)/d·I求解，原bank不改。run_raw_ridge_head.py新增isotropic_within_second_moment=true，与prototype_only_second_moment及diagonal_second_moment互斥；旧分支默认保留。所有调用仍为既有中性worker包装器，无新增库或额外服务。

## 冻结和访问顺序

固定seed74002、原类别顺序/划分、raw单位features；旧类H为ISIC64、HK169，新S为2290、95。类内均匀、每类总质量1/K。先读取H、新S，构造并保存头、记录seal，再读取Qfit/Qgate及历史聚合。旧S、E、原图、新标签禁止；teacher仅读取零残差模型形状，不使用教师适配器作表示。禁止GPU。

保留1536维、非空、finite、均匀权重、标签范围、头1536×6/15、query封存前访问守卫。类内trace必须有限非负，G_iso trace与原G一致；求解残差<=1e−7。失败即记录，不调λ、不裁剪负trace、不回退求解或自动重跑。

每数据集3个二维工程解：无类内散布时解析解、复制单类样本类平衡解不变、具有非零各向异性散布的已知同trace各向同性闭式解（同时检查原矩阵不改、不是完整方程解）。共6个工程矩阵解，2个正式矩阵解，0逐元素正式解，0梯度（包括工程）。

## 固定参考与判定

主要raw_ridge；raw_prototype_ridge用于总量/方向机制比较，raw_centroid为必要简单基准。全部11参考：raw_centroid、MM、HH、identity_ETF、teacher_centroid、teacher_ETF、coordinate_median、centered_centroid、raw_ridge、raw_diagonal_ridge、raw_prototype_ridge。历史只复用已交付聚合，不重跑。

每个查询报告全部seen逐类recall/CE/正确与数量，all/old/new宏BA及完整seen交叉熵，所有参考正负差。描述性规则仍为每数据集双查询均总体BA严格正、旧/新BA各>=−1pp；不据查询选模型、不改主要指标。主门槛不通过则不声称完整岭被简化替代。对原型岭的改善也不自动等于保留完整岭。报告τ及λ_eff、原类内trace/Frobenius比、头范数、残差和微计数；不将稀有类一例变化解释为普遍安全。

## 有限成本与结束条件

两个CPU任务，各900秒、总3600秒，两线程；0GPU/梯度/适配器教师编码器记忆更新。2618支持+2155查询=4773特征前向。两拟合头32256个float64元素258048字节（252KiB），0bias。仍构建完整G用于trace及诊断，不声称节省Gram成本。历史成本不重计，客户端准备null；保存全部失败/缺失/超时成本。

任务退役后收集PROGRAM、两结果和日志，公开全部11参考、逐类组、诊断与成本，再作下一步决策。现有查询已参与设计，患者分组未知，仅开发机制证据；不能当独立医学、跨种子或全阶段持续学习验证。不访问保留测试集，不自动扩大预算。
