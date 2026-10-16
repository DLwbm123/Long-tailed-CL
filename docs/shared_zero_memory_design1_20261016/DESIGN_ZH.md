# 共同零点损失记忆：完整合成分类序列前瞻协议

前轮d2dfa5f1已交付12stage类特定anchor球交集为空。停止该直接接入路线；本轮所有类共享零anchor和norm<=1/beta=.125的head参数域，不扩大原局部半径。将原anchor槽重分配为第15个二阶矩方向，每类mu(d)、U(15d)、lambda(15)、遗漏谱rho(1)，仍16d+16float64。Q=mean xxT的15主方向，Q15 PSD；rho=||Q-Q15||op。仅当前任务类支持可构造，旧memory只append，永不回访旧支持进行候选训练。低样本统计可能含支持信息，本轮仅合成不自动授权医学原始features回放/隐私保证。

单位固定x，L_c(v)=mean softplus(beta x^T v)。零点展开S_c(v)=log2+beta/2 mu_c^T v+beta²/8 v^T Q15,c v。softplus三阶在0为0、四阶绝对值<=1/8，因此loss余项<=beta²/8*rho*||v||²+beta^4/192*||v||^4；gradient余项<=beta²/4*rho*||v||+beta^4/48*||v||³。候选超域拒绝。该固定特征/小参数域保证不是原完整adapter门修复。

四相同线性架构轨迹：完整合成支持oracle_wide(norm<=2)、完整支持oracle_local(norm<=.125)、moment_local和quadratic_local(都norm<=.125)。oracle仅CPU科学对照可访问旧合成支持；两个memory训练函数只收当前任务支持和旧合成memory，不传old oracle。每类风险均匀加权，sum-head one-vs-rest BCE: L_c(w_h)-beta*mu_c^T w_c。分类argmax logits。四臂同初始化，新增head零初始化；每stage48全batch梯度步，固定lr1/16来自beta²/4 Lipschitz界，逐head投影。无网格/optimizer动量/权重衰减/参数挑选。非医学Core/LT训练，不声称复制Core或降低Core。wide oracle隔离小域限制成本，local oracle隔离memory近似误差。

memory方法训练时当前类风险精确，旧类使用各自memory。训练后只压缩当前类：原moment48 Adam lr.02/16节点及权重/原ball投影，使用时按旧identity法归一化；候选一次Q eigh无optimizer。两者存储相同，计算不等价。当前task moment联合目标均值和class-keyed seed+1000+class，不声称精确复现旧pooled优化轨迹；全部fit/eigh/训练成本保留。每stage旧memory逐tensor不变。head会更新，不能把memory不变说成无遗忘。

三个原非高斯unit合成fixture：small32d512K4序列2+1+1，ISIC1536d9434K8序列4+2+2，HK1536d1038K23序列13+2+2+2+2+2。原seed263/264/265/公开类比例，非医学fit。四臂全部stage训练封存后才生成seed+10000独立合成评价集，同分布/同counts；这是固定合成评估PRNG而非新增正式医学seed或val/test读取。所有stage accuracy/BA/oldBA/newBA/逐类recall全保留，stage1 oldBA=null。独立评价oracle在训练后检查真实旧类risk与quad risk、梯度、界宽，不反馈训练。

4CPUcase：小unit与三完整序列。unit验证风险梯度autograd、quad FD、loss/gradient界、首task无旧memory三步精确、超域拒绝(共6合成head更新)。scale每臂每stage48，总12stage*4*48=2304合成head更新；另当前memory压缩12*48=576，候选0压缩optimizer；全部不是正式医学预算。新分类阶段与旧查询检查不同，不声称计算等价或沿用旧效果门。

新前瞻有限分类信号：工程全部通过，两个高维每stage candidate BA/oldBA/newBA不低于moment_local，并各至少一处严格提升。无容差放宽，旧BA缺失不补0。所有oracle差/负贡献保留，不能用末stage覆盖中间失败。即使通过仍仅固定合成特征分类，不是ViT/医学/完整查询族/生产GPU证据。case1800/整体5500秒资源门不变；所有失败和cost先公开交付。输出48方法stage行、540匿名类recall、12界诊断/贡献、35谱统计、48训练成本行。无features/heads/memory矩阵/逐样本ID导出。
