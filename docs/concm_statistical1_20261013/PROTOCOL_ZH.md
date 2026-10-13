# ConCM-inspired 视觉统计补全与动态结构匹配：固定三轮主四格

用户已授权两种机制全部设计并按优先级实验。仅ISIC/HK、GPU0/1、既有development val；test/reserved封存，RL暂停。保护原任务和历史模型，不重训原base/Core或local_transport，不调参，不新增seed或幅度/预算网格，不自动失败重跑。

## 方法身份与优先级

本轮为非官方ConCM-inspired边界统计/解析头实验，不是作者MPC/DSM完整复现。MPC是视觉记忆补全，无WordNet、语言模型、语义属性网络或元学习；DSM把最大匹配ETF作为解析head目标，无非线性projector、contrastive优化或高斯特征回放。机制来源为ConCM第3.2/3.3节；视觉原型融合也与TEEN相关，不宣称独有。

固定顺序：MPC → DSM → MPC_DSM；每轮ISIC_base、ISIC_local_transport、HK_base、HK_local_transport四条。A0/AL是原冻结控制，B/P为各自checkpoint加机制。原AL固定主方法，不能按val选择base或变体。三轮均执行，前轮正负不改变后轮参数/队列。每轮四条真实零更新预检全部PASS后统一评价全部stage；任何失败停止依赖并完整交付部分结果，不自动retry。

## MPC-inspired：支持量控制的视觉记忆补全

仅使用该stage已见类的原bank mu/Q/n，不读取未来类统计或任何train/test影像。q_c=n_c/(n_c+16)，供体权重softmax(16*cos(mu_c,mu_j)+log(q_j))，排除自身，不按标签名或val挑供体。eta_c=0.4*16/(n_c+16)，新均值=(1-eta)*原均值+eta*加权供体均值。0.4参考作者alpha=.6的补全份额，16个pseudo-count为前瞻工程选择，temperature16为既有视觉检索尺度；不是从当前val估计。所有旧/新类按支持量同式处理，不假设初始类充分可靠。该公式不估计旧类真实漂移，低n仅是可靠性代理。

保留原中心协方差：delta=新mu-旧mu，Q'=Q+mu delta^T+delta mu^T+delta delta^T；components同步平移，n/mass/radius不变。样本平移的矩一致性可精确证明，但分布校准正确性是经验假设。保持该stage原R和competition pairs，防止与local_transport先验重复叠加；只改求解输入统计。均值融合可能把细粒度疾病拉近，尤其少样本类需完整报告损失。

## DSM-inspired：最大匹配结构的解析头目标

用本stage类均值归一化组成M(d×K)，C=I-11^T/K，M C=SVD U S V^T，Delta=sqrt(K/(K-1))*U V^T C。要求中心秩≥K-1，退化则明确identity/skip，不伪造有效匹配。Delta^T Delta=K/(K-1)C；target=Delta*||W_control||F/||Delta||F，scale仅使用原控制头，不读取val。固定gamma=0.1*trace(mean_c Q_c+0.001R)/d。0.1为前瞻相对曲率选择，不按旧诊断幅度调参。

在原完整类均衡平方/competition/原R解析目标上加 gamma/2*||W-target||F²。Hessian加gamma I，RHS加gamma target；复用competition.solve的proximal入口，保留全部原竞争项。固定输入时仍为严格凸二次求解，不是另加CE训练或直接替换成ETF分类器。这个目标约束head而非作者projector，最大匹配旋转通过target实际参与求解，不是对所有统计/头作等价旋转。MPC_DSM先补全统计再匹配，target范数仍来自原控制。

## 原控制、数据与时间屏障

原科学源、交付SHA、order/task_sizes/seed74002/原训练更新/训练与val类频数完整写入LOCK。只读复用18个原阶段checkpoint；验证原CONFIG/DELIVERY、路径、seen/steps/module/n、原head重建和source文件元信息。原checkpoint没有完整嵌入source metadata时必须如实披露，由原配置回执共同验收。每个sidecar显式保存method/version/variant/dataset/control/source/order/seed/task/seen。原controls和AIR只复用已公开结果，新增费用0，不重评AIR模型。

CPU先一次准备18原stage的54个sidecar，全部封存再开启GPU预检。每轮4真实预检仅既有val首64图，无optimizer/policy更新、参数state精确不变、CPU/GPU分数容差atol2e-6/rtol2e-5、finite；不计算分类指标。正式阶段读取val前核对当前variant/phase、准备seal和本轮4PASS。evaluation采用原Resize224/Normalize(.5)、batch64/workers0、不shuffle，OFF/ON共享encoder特征，逐stage原控制Acc/所有类recall/分母必须精确复现。全stage均评价，不挑checkpoint。

新头只用于当前stage边界readout，不反馈下一阶段；0新的backbone/adapter/optimizer训练。不能称端到端增量训练收益。新类logit/预测/间隔允许变化，无旧新组别硬锁，开发门不是无损保证。原训练预算/预训练/存储资源原封，新增CPU统计求解和GPU推理单列。

整轮4小时从实际launch计时，包含CPU准备/预检/评价/等待，不重置。正常4CPU准备+12真实GPU预检+12GPU评价，共24GPU驻留；失败全部额外保留，每次一次。仅GPU0/1每次≥32768MiB，不借2/3，不处理其他任务。NAS写入前nfs/free/小probe。

## 结果与公开交付

两数据集18行主表：A0/AL/AIR和三变体各B/P，完整BA/avgBA/tail/forget/old/new、Acc/avgAcc/可得F1；旧缺失留null。全部变体相对各自控制、组合与单独、交互、AIR六指标正零负增量；每stage全部类n/recall、纠正/破坏/错→另错、所有旧新错误流和新类间隔正零负；小于5例不公开连续margin摘要，但计数与recall保留。原train频数完整披露。

主贡献门仍相对各自控制BA≥+1pp、tail≥−0.5pp、forget≤+1pp、old/new≥−1pp，两数据集同时满足才称通过本开发门。不因原型更分离/参数非零/CPU数学PASS称有效；不事后改门或挑主方法。单seed/order、反复development val不是独立患者/临床或封存test证据。

报告必须REPORT/RESULTS或PARTIAL_RESULTS/CONTRIBUTIONS/AIR_GAPS/STAGE_CLASS_RECALLS/PAIRED_COUNTS/MARGINS/CONFUSION_FLOWS/MECHANICAL/RESOURCES/AUDIT，保留全部失败及成本。只收匿名聚合，不外传影像/模型/逐样本分数/feature/indices/keys/means/Q/矩阵/私有配置路径/rawlogs/traceback。

实际结束显式127.0.0.1:7897代理推送DLwbm123/Long-tailed-CL feat/prototype-analytic，核对remoteSHA和精确commit报告匿名HTTP200后，NAS独立public_delivery/DELIVERY/HANDOFF防重复。聊天直接完整主表、所有贡献、新类变化/类别损益/费用与回执，不只链接。完整交付后暂停同一监测，不追加实验。
