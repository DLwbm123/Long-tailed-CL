# DECISION_ANCHOR：冻结判别方向保护的独立初始适配实验

本协议在读取新投影/拟合新头/任何实际预检前封存。用户2026-10-20明确要求结合A1–A3与Pro论文提出独立方向、说明假设/强对照/成功门后继续实验。本轮不调整或重启O，不覆盖旧负结果。研究问题来自DSDR的决策敏感方向保护思想与A1类2救回2/损害12、A3跨ridge/NCM损害、A2归一化交互弱；这些诊断不证明方向侵蚀或锚定有效。

## 假设、保护空间和固定目标
仅初始2555训练；原划分、canonical像素、原FP32 APART生产路径、1535固定坐标/网络池化、102912既有bias不变。不增加模块/训练整个adapterpool/换层。全原层bias绝对值覆盖411648bytes计部署容量，临时双head只训练前4logit并在终点丢弃。未来encoder冻结；本轮不读任何未来/测试，不论结果如何。

F为冻结原G；E为同批CE无保护；L为等迹全支持子空间保护；R为同秩随机方向；D唯一主候选冻结判别差方向保护。所有E/L/R/D从同一预训练bias与原head重置、同初始采样、独立重置optimizer，不从O/Bias或预检参数初始化。

先用2555训练和原类/组权构造固定F mapper与ridge alpha=.01（该值沿历史原G规则，已知选择，不新增搜索）；不读681开发构造保护空间。z=原mapper(g)维1535，自由截距不入投影。训练加权中心协方差C；S为C特征值>maxeig*1e-10的列空间，维m。若m<=3或数值非PSD/判别方向无法投影到S，工程未决停止，不换阈值。冻结F的前4类头W去列均值，其列空间经S投影/QR给Q_D，rank由singular>maxsingular*1e-10固定判定r<=3；与原W差空间偏差<=1e-8。Q_R由S空间内标准高斯(独立seed74251)QR取同r维；随机方向不使用标签/开发，不避开D或按性能重抽。记录D/R主夹角，重合不重抽。Q_S为S正交基，t=trace(C)。不把r、m或df叫信息秩。

对当前训练样本固定参考z0=mapper(g_F)，delta=mapper(g_theta)-z0；此mapper始终冻结F训练拟合，部署各臂仍另拟合自身原mapper。D惩罚=mean||delta Q_D||²/(t*r/m)；R同Q_R；L=mean||delta Q_S||²/t；E=0。等迹算子分别P_D、P_R、(r/m)P_S，共用分母，因此对S内各向同性漂移期望相同，不宣称真实参数梯度/强度完全匹配。目标=双head平均CE+2*惩罚，lambda=2固定一次端点，不搜索（受DSDR默认系数启发，但目标归一化/参考/任务不同）。不按开发调系数，不用训练梯度校准隐藏搜索。

这是对判别差空间的精确正交投影，区别于DSDR原文W_old行空间及其稳定化非精确投影；当前为首次监督适配，原文t>1历史模型场景不同，不称完整论文复现。保护的是非线性encoder更新中的方向，不是固定U读出重参数化；保留旧头差分不保证新ridge风险改善。F原g包含重复通道，因此R/L限于训练支持S，避免随机臂主要落在初始零方差方向的假控制。

## 训练与选择
训练E/L/R/D各固定800步、每步每类8个不同已知组各1图，共32槽/步=25600槽/臂。seed74250生成一次共同采样，微批8可checkpoint，整32任务CE/保护后反传；原encoder seed沿lock。AdamW lr.003, wd0, betas(.9,.999), eps1e-8，无scheduler/clip/lr/epoch/seed/参数子集搜索。每100步仅保存恢复点，不选中间模型/早停最优；nonfinite/越界/超资源未决，不加步。四臂各自重置初始bias/head/optimizer。F参考训练g直接只读旧cache，不重复图像前向，必须预检原batch32零更新回放<=1e-10，并披露微批舍入。

全部800步端点封存后，F/E/L/R/D各自仅用2555训练原类/组权拟合自身mapper/A/B；各自原681开发仅选ridge alpha[.0001,.01,1]，同分大者。所有15闭式解全报，F选.01且BA/correct回放原冻结值，残差<=1e-7。4个新初始头完整artifact含bias/norm/ids/PERM/三角A/B/W/元数据实测<=19058800。不借packing加宽，不免费teacher/projection：训练用Q/t/F头不部署；若后续推理仍需要则须另计，本方法推理只需一个适配encoder和新ridge。

## 预定开发门与机制判定
同原681已暴露开发集，非独立确认；类1仅3例，全部计数与离散阈值公开。D对F BA>=1pp、类2召回>=5pp；类0/1/3各成本<=1pp。D对E/L/R BA均严格>0、类2均>=0；D对E类2还须>=5pp。任何失败固定端点停止STOP_DECISION_ANCHOR_DEVELOPMENT，不挑L/R/E升主、不改lambda/空间/seed补试。

所有四臂终点和15头都已拟合封存后才汇总开发配对R/H，不能根据某臂结果修改后续臂。全部报告每类D对F/E/L/R R/H/去向、BA恒等式、全训练保护能量/总漂移及CE，低漂移或低loss不能当机制支持。

仅开发点门通过后，检查预设8项类内已知组配对bootstrap家族：D−F/E/L/R的BA和类2召回，共8项，10000draw、seed74252，同draw跨臂复用、Bonf分位[.05/16,1-.05/16]；全部调整下界>0才标DEVELOPMENT_DIRECTIONAL_PROTECTION_SIGNAL。未过则证据不足，不能以未调整95%/单格替代。它仍只开发探索性，不是独立泛化或SCALE-Med原类1未来J成功，method_success=false。报告原历史class1门并未被本研究问题替代为成功；本轮不读未来/测试，无未来J，不以换主指标冒充原总体方法成功。

## 预检与隔离
实际前向前合成：S/D/R正交/PSD、等迹期望和截距/共同logit平移不影响差空间、保护空间保存回读、zero-drift零梯度、projected-quadratic有限差分、随机控制可重现；开发点门成功/失败/缺失与8项JSON、R/H恒等式。实际固定32训练图：原G回放、fresh32与micro8/checkpoint梯度allclose(atol1e-5,rtol1e-3)，D/R/L保护项实际有非零梯度(固定合成bias扰动)且全部finite、参数边界与绝对override回读、恢复exact、实际各臂一步有限后丢弃。若需encoder有限差分沿已认证分档[.001,.0005,.00025,.000125,.0000625,.00003125]全报，最后两档各5%+1e-3且平均误差<=max(首误差/4,.001)，不得择档放宽门。mapper/保护参考一致性和同完整artifact序列化必须过。统一正式包装器注入失败四回执完整。所有实际probe参数丢弃不初始化正式。

只读inputs0和pixels0；禁止rawimages、inputs1/2、pixels1/2/3、任何历史测试预测/标签/特征、旧E/I/O learned checkpoint。既有预训练/冻结参考cache不是encoder-level OOF，预训练语料重叠未认证。模型生产者与缓存绑定按原已核流程，不冒称逐图hash。各臂最终重新编码3236行只一次，完整cache可供本轮CPU验收，不能盲目重跑无seal的残缺编码。

## 资源、回执与停止
本轮首实际预检开始共享4h总包络，所有重试/子阶段不重置；新增长硬限1GiB含本地/Git/helper16MiB，单文件512MiB、RSS12GiB，旧窗17GB软停/20GB硬限。预计新增<768MiB，只读复用pixels/model/env；GPU0启动前实际NFS/容量/写读probe/显存合理余量核。全部cache/tmp显式本轮NFS，PYTHONDONTWRITEBYTECODE=1，中性入口主子/GPU名。独立协调每2秒守卫，超时/空间/RSS保存最小回执终止本轮，不干预他人。至少预检+TRAIN+FIT的EXIT/FAILURE/资源/协调和driver回执；父子wait4/正文/driver重叠不重复计费。预算不是优化器收益，健康启动后唯一scale-med半小时跟进、不密集轮询。完成验收15fits/4artifact/4端点seal、覆盖/finite/parity/容量/退出退休、中文机制分析、全聚合/源码/协议/成本/失败摘要公开push核SHA/匿名HTTP200，私有DELIVERY双端，不公开模型/feature/身份/逐例/原私有日志。历史失败保留，不强推。
