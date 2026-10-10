# OrganAMNIST-LT IF100：CT跨模态配对冻结协议

Path IF100、IF10/50和平衡阶段完整交付后，执行用户已批准的OrganAMNIST跨模态阶段。固定原static_pc与同一local_transport候选，只2条真实配对；不依据Path val或CIFAR test改变候选、强度或种子。两条各212更新（128/54/4/12/14），共424，0策略更新。2个真实第一batch backward、零optimizer预检全PASS后才正式训练，两臂全部TRAINED且计数一致后才统一评价5个阶段官方val6491；test数组始终不打开。

官方OrganAMNIST224为11类灰度CT轴位图像；下载官方Zenodo NPZ（1803859544字节，8个精确206 Range，失败不自动重试），只读取train/val数组，与既有128 train/val labels精确核对。无损实体PNG存入独立图片根，既有Images加载器转RGB重复同一灰度通道，原ViT及增强不改。官方train34561，各类供给见LOCK；不读取test数组或复用它选择方法。原作者描述扫描级train/val划分，本实验只有样本索引，未独立重建患者/扫描分组，不能声称独立患者隔离/临床泛化。

频率rank=[4,0,7,8,5,2,3,9,6,10,1]，进入order=[7,8,4,3,0,1,6,9,2,5,10]，各由固定seed74003/74004的Python shuffle产生。任务3+2+2+2+2。仅用训练类供给确定nmax：100*floor(min(5000,min_c available_c*100^(rank_c/10))/100)=3100；不是按val结果调参，也不逐类截断。每类n=floor(3100*100^(-rank/10))，共8343，头3100尾31；同seed74002+c*2003无放回排列前缀。fit/meta原每类80/20身份索引划分seed74002，meta任务末进入原拟合，官方val不作meta。三频率组冻结为4/3/4类：head[4,0,7,8]，middle[5,2,3]，tail[9,6,10,1]；报告macro recall及全部11类n/recall。

两臂主体原ViT/adapter/global充分统计/解析竞争头/优化器、FD10、2epoch、batch64、workers4、lr.0003、WD.01、clip5、ridge.001、proximal.01不变。候选仍四区域CLS条件pool、去每图区域均值归一化、每类4中心各1/2入原P、旧局部中心当前类均衡区域共同位移，全局仍common_shift；推理原全局head。无附加读出、局部CE、可学习head、旧影像/逐样本特征回放。不要求无依据的历史逐类等价，两个新臂严格配对同数据/seed/order/budget。

数学：n_c由上述固定几何衰减生成；类别单位样本权重沿原static_pc，meta任务末进入充分统计。候选P_extra=[m_cq/2]，加入P_extra P_extraᵀ，4区域中心的类别列不直接进入推理类别得分。旧局部中心按当前类均衡before→after每区域均值位移运输；这是共同平移近似，不能保证旧类漂移准确。原竞争解析求解与全局主体完全保留。CPU检查原base head精确等价、44局部列先验正定、采样供给/固定计数、灰度RGB加载及dataset/test权限；真实GPU预检及正式有限loss/clip梯度/求解残差≤1e−8另审计。模块存在或激活不等于分类收益。

仅GPU0/1（启动前每卡≥32768MiB及NAS实际挂载/空间/小写读探针），GPU2释放、GPU3禁止，不处理其他任务进程。24h包含下载/CPU准备/排队/预检/训练/评价，独立新轮可有新截止，但历史成本不重置。正常2preflight+2train+2eval=6GPU子进程，每次成本出现分别保留；CPU准备/CPU检查/协调器墙钟单列，失败全部保存，不隐藏或自动正式重跑。

完整公开源码/协议/匿名六指标（最终/平均BA、tail、forget、old/new）与全部候选−base pp、Acc、5stage、11类recall/n、频率组、正负、审计/成本，通过代理GitHub核实远端SHA/匿名HTTP200及NAS交付回执。相对原base BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp全部通过才算本数据开发门；单seed/order、val开发证据，不能替代ISIC/HK自然长尾双目标，不能称官方MedMNIST test benchmark或临床验证。公开不含影像/权重/逐样本索引分数特征/矩阵/中心/原型/私有配置/原始日志。结束后不按本val选强度/追加seed；进一步科学阶段须独立冻结。

来源：[MedMNIST官网](https://medmnist.com/)、[官方数据说明与链接](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)、[官方Zenodo发布](https://zenodo.org/records/10519652)。

冻结前CPU检查首次发现总更新数手算错误（312应为212/arm）；保持阶段计数[128,54,4,12,14]不变，修正总和并保存CPU_CHECK_attempt1。修正后采样/供给/原head/44列先验/灰度加载/test权限及Path、CIFAR回归检查PASS；不是正式科学失败或GPU重跑。首次客户端检查墙钟7.429902秒单列，CPU执行耗时未独立测量，不伪造CPU精确费用。
