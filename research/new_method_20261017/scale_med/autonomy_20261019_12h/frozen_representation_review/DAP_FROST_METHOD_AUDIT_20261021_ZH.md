# DAP / FROST 正文核查与复用决定

两篇均已通过官方OpenReview下载正文。阅读范围见JSON：已读核心方法、实现配置、控制/超参附录；不是官方代码复现或每张结果表逐行认证。投稿不代表录用。

## DAP

DAP明确是基础类均值/协方差按余弦softmax迁移到novel类，合成Gaussian特征，再进入固定RBF锚点空间，最后类平衡递推ridge。仅递推闭式求解不是我们缺少的机制。原流程还包含基础阶段监督适配，不是完全不训练。其校准分布不同于旧FeCAM距离头，但不能因此认定已有正效用。当前initial四类没有不依赖暴露query的rich-base/novel划分。正文给出不同数据集的经验alpha/beta/K/gamma/tau，没有可直接用于当前医疗数据的唯一配置。 inspected实现没有找到Nsyn及lambda的具体数值。

K=3000/5000时，float64 packed Gram单独36,012,000/100,020,000bytes，超过19,058,800预算；锚点、类协方差、B/W还需另计。不能缩K/改成对角/切pseudo-base来改名复活旧核/协方差实验。文中范数集中和Cauchy-Schwarz不足单独保证不同类可分，更不是类1风险保证。

## FROST

FROST原版有基础adapter训练、2000维随机ReLU、未做class mass normalization的充分统计ridge、20%base holdout选lambda，以及原生feature归一化类均值余弦头，最终固定0.5融合。它不是全流程免训练、免选择。按当前1536原生维，FP32随机投影12,288,000加FP32 packed Gram8,004,000，已经20,292,000bytes，尚未计adapter/B/W等，默认整套不满足当前cap。

只复用论文Eq10的固定双读出估计器。原F每类总质量已均衡，因此不宣称纠正base频次偏置。可证伪问题是全局inverse-moment判别与原生类均值方向是否互补，尤其HK单样本类的均值方向仍有定义、不会像旧within-group罚一样严格为零。这不是独立语义信息或原创函数族；在适当条件下仍可归入同一仿射函数族，不妨碍检验不同support估计器。

有界旧控制核对：旧covariance_reference是mapper后的Euclidean NCM和协方差距离，无本次固定融合；core.py是Gaussian patch/global证据相加；group_out_ridge2_audit的A3也是mapped Euclidean NCM。未认证全历史未测。旧T末端换头/alpha、I权重、核、FeCAM、组一致性停止均保留。

## 选择与严格控制

选择FROST-inspired已知强基线，不声称复现FROST。冻结原F/mapper/alpha.01/逐图质量；原生类均值按原class-group-image权拟合。唯一主臂M=.5F+.5cosine-mean；独立P余弦均值与N原生Euclidean均值控制。无新编码、mapperfit、ridgefit/solve或优化器，仅3次prototypefit。全部artifact先于query封存。

ISIC原两fold与HK原S_old/E/Qgate均完成，维持类1优先、全类每fold/cohort成本门；HK两个cohort不冒充两训练fold。所有门和600s/CPU300s/RSS4GiB/round128MiB/fullartifact19,058,800预算见封存PROTOCOL。固定0.5来自论文，不扫权重/温度/offset/alpha；不按P/N好坏替换主臂。未过门则停止该配置。该实验只能形成历史暴露数据上的点证据，不能据此声称TMI创新、独立确认或完整CL未来收益。
