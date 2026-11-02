# D92 固定候选信用分解：冻结诊断协议

新独立诊断，非R91续跑或新方法效益实验。原R91负门不变。只读取原六stage、原五档offset0、原四类0/1/2/3的冻结builder及TRAIN特征。0optimizer/新encoder/VALTEST，不新增候选、样本、支持重复或部署选择。全部30情境保留，不择优。

A：同一held builder四格动作：全NOOP、仅held全修正、仅其余类全修正、全修正。对相同held查询的宏/尾召回报告self、competitor、interaction，验三项和等于R91整体held差。不能把竞争类效应叫自身去噪。

B：原81动作(B/F)全部同域，报告B选动作相对uniform的宏/尾/D/U；F最优相对uniform的H、F最优减B选的regret，验证H−regret恒等式。固定块全0/.5/1作为同域强控制；全类Nfull只单独报告。不按F选择部署动作，oracle仅诊断，不称独立验证。

C：对原81动作U，计算四类各3动作的均匀条件q与logit梯度g=(q−mean_k q)/3；报告B/F范数、内积、余弦，零范数余弦null；分解macro/tail/−D分量。不是共享actor参数梯度，不声称GRPO/优化有效。

精度：R91已有main/held/context指标重建最大差<=1e−8pp；新分解恒等式<=1e−10（pp单位<=1e−8）；固定81合成枚举中心/条件均值恒等式检验。任何不符失败封存，不自动改门重算。结果描述性，无新的正负准入门、无显著性、无后验部署选择。

预算：原FIRST起1800s包括本诊断准备/代码/准入/运行/分析，不重置（严于24h）；CPU4/总400000s，RAM64GiB准入，worker28GB/group29GB每2s，总30GB；输出cap1GB+512MiB余量，每server<50GB。CPU-only，CUDA不可见，不加载encoder，不需要GPU分配但复用独占共享memorylock。remote-home存科学输出；写前NFS小probe及空间检查；不复制基础模型，不动历史/他人。保守最多60builder前向、120候选score-bank调用，另reference的TRAIN读出按实际行数计；utility评估按调用与action-classifier行计。端点/分数/类别级q私存，仅聚合机制/费用安全交付。预算前置按R91已测profile保守允许<=600s科学段，另1200s准备分析；若剩余不足不启动。必须identity/noreceipt防重复，不partialresume。复用现有守卫与中性control.py/campaign.py启动，独立deadline。软件最小新恒等式只一次，不重复R90/R91旧toy。
