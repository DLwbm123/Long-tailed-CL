# 局部损失记忆：前瞻CPU审计

同矩反例已完整交付5d32e997。改变保存的信息而非继续改采样：每类保存当前类固定特征上softplus经验loss的局部二阶展开。该候选不是合成节点回放，而是局部损失替代项；尚无生产或分类训练接线。旧查询族/所有失败不改判。

L(v)=mean softplus(beta*x^T v)，beta8、||x||<=1。固定anchor a=2normalize(current class mean)，零mean拒绝。存a(d)、L(a)(1)、g(d)、H的14主方向(14d)、14特征值、最大遗漏特征值rho(1)，共16d+16 float64，与原16节点权重相同。小维数补槽。临时完整H/eigh只当前支持内使用、不部署；全成本/RSS计。梯度和曲率也可能包含低样本支持信息，本轮仅合成，不构成医学数据隐私保证或原特征回放授权。

替代loss=L(a)+g^T delta+.5 delta^T H14 delta，替代gradient=g+H14 delta。H PSD，rho=||H-H14||op。sigmoid导数形式给出softplus三阶最大绝对值1/(6sqrt3)，故固定特征单位域内T=beta^3/(6sqrt3)。任意delta有|L-Lhat|<=.5rho||delta||²+T||delta||³/6，以及||gradL-gradLhat||<=rho||delta||+T||delta||²/2。存储只提供局部调用：||delta||<=1/beta=.125，否则硬拒绝，不静默回退。半径由beta归一化一个logit单位前瞻确定，不查网格。界本身不保证足够小，原样报告。

每类48固定query：anchor一个，其余47个固定PRNG切向方向，在原norm2球面取弦长1/beta。种子case seed+1000+class，不query优化、不压力选择。class-specific局部邻域不是共享多类分类器可达轨迹；没有隐含未来头和encoder保持在域内的保证。低秩曲率适合性和local gate不能替代旧完整查询族门。

保留原random及moment48 Adam lr.02/16节点基线。另linear控制复用同anchor/loss/grad，不额外拟合，隔离曲率作用。moment和random均按旧identity-adapter归一化nodes解码，直接计算loss及相同v的梯度。候选无optimizer，baseline48不降低；eigh不称计算等价。三个规模原seed263/264/265、非高斯单位合成支持/N512K4d32、N9434K8d1536、N1038K23d1536及公开类比例。全合成，非医学fit。

4独立CPUcase：算术/三个规模。算术含loss梯度FD/Hessian-vector、anchor精确、排列预测、正遗漏谱的界、超域拒绝。所有scale核对每query的loss/gradient界及冻结state；所有类/方法mean max loss和gradient误差/anchor误差/rho/trace/界宽保留。主结论先局部数学实现是否通过；另两高维局部候选max<=.9各原moment/random、mean不增才局部有限信号。0新query优化/0医学images/features/checkpoint/val/test/GPU。没有序列/重压缩/完整分类训练或真实预算，不GPU-ready。

所有失败成本逐次保留，case1800/总5500秒资源界不变。先完整公开交付再分析。即使局部过门，仍需解释如何保持合法查询/固定接口并检验完整同架构分类训练，不凭局部近似通过直接医学GPU。
