# ISIC/HK 残余漂移约束：冻结贡献对照协议

用户明确批准在ISIC/HK验证，严格复用原base和local_transport，只新增漂移单独与组合。仅GPU0/1，RFMiD coverage3队列/部署/数据/配置/进程受保护；其PROGRAM终态且active空、两卡每卡可用≥32768MiB、NAS真实挂载/容量/小写读探针通过后才dispatch。24小时从本轮实际后台协调器启动计，包括等待/预检/训练/评价，失败不自动重跑。RL暂停，test/reserved始终封存。

## 四格与预算

|格|报告标识|原模块|漂移权重|本轮训练|
|---|---|---|---:|---|
|A0|base|base/static_pc|0|严格复用首轮已交付控制|
|AL|local_transport|local_transport|0|严格复用第二轮已交付控制|
|AD|residual_drift|base/static_pc|.1|ISIC468/HK176，新两条|
|ALD|local_transport_drift|local_transport|.1|ISIC468/HK176，新两条|

4条新正式轨迹、1288 adapter更新、0策略；复用4轨迹不重复计步或GPU成本。4个真实第一batch backward/零optimizer预检全部PASS后才TRAIN，4条新正式轨迹全部精确TRAINED/policy0后才统一固定development val评价。正常4preflight+4train+4evaluate=12GPU子进程，每条记录每次出现都保留，失败另全计。预检job带独立后缀，log/PREFLIGHT/STATUS不由正式train覆盖。

ISIC每stage100/72/296，共468；HK98/4/2/20/18/34，共176；epoch2/batch64/workers4/lr.0003/AdamW WD.01/clip5/FD10/ridge.001/proximal.01/competition_beta.5/原cosine eta_min1e-5/seed与split74002。原order/task_sizes见LOCK，fit/meta原影像身份80/20，meta任务末进入原拟合，val不作meta。两数据集均从相同原预训练真实训练；无旧影像/逐样本特征回放、无新head容量/局部CE/readout/选seed/调幅度/挑stage/放宽门。

## 约束

沿原common_shift Delta，以当前teacher-start与当前特征配对定义e=after-before-Delta，U=sum normalized_native_current_mass * e e^T。目标加(.1/2)tr(W^T U W)，系统加.1U；R_eff=R+100U（原rho=.001），不误加额外rho。仅fit估计epoch U，epoch内detach/freeze；boundary按原全current train含meta重估；不另重中心、不trace归一化、不累积历史U，不改原mu/Q/原型/区域运输公式。原reference/竞争/控制器/proximal/Woodbury/minibatch解与显式目标/boundary竞争头全部传同一R_eff。

局部组合保留既有四象限CLS条件池化/去图像区域均值/归一化、每类4中心各1/2入P、旧局部中心按当前类均衡区域共同位移；global仍原common_shift，推理原global head，无附加分数。此为受LCA邻域稳定思想启发的解析改编，不是原版LCA复现。当前fit漂移只是旧类漂移代理，不保证医学判别信息或召回安全。

第一次epoch before=after，U=0；真实零预检证明模型/梯度/预算与求解兼容，不能证明非零约束激活。正式epoch/boundary公开weight/trace/frobenius norm/残余均值范数/样本n/历史n0/系统扰动范数，以及finite与maxsolve≤1e-8；边界U非零为激活审计，不叫分类成功。首次epoch只允许数值零，不虚构旧类实际漂移测量。

## 严格匹配证据

A0来源module_additions1_20261009，科学源0cfe6b648f5de1b8e8783265e7a0b97b64939952，交付ecd783ffd7b32eea4e2e2a1311b9f033231f1677；AL来源module_local2_20261010，源0f0d8a0717c6da09109cc9205cccf44c490d3e84，交付4b6e0c87887b101ae2970738919760f00e19726e。实际NAS CONFIG核对全部科学字段，来源NAS与本地DELIVERY均已核实，已公开全stage聚合结果直接复用。

CONTROL_MATCH及CPU_PROTOCOL_CHECK记录核心统计/竞争/fit_split/loader/增强源码AST一致、同ApartFeatures、同既有只读manifest/images/权重路径、lambda0原头精确相同、局部池化/运输精确相同、真实train/val元数据与全部历史stage匿名分母/预算相同。其他新模块/权限分支不影响这两条默认路径。没有重新训练base、没有哈希影像/权重或断言未测量的逐字节内容身份/完整真实ViT轨迹重跑等价。模型/数据原资产受保护，发现匹配异常fail closed，不悄悄重训控制扩大矩阵。

## 全部贡献与评价门

每数据集完整A0/AL/AD/ALD六指标：最终BA、平均增量BA、尾类recall、forgetting、old/new recall；全部pp增量AD−A0、ALD−AL、ALD−AD、ALD−A0、交互(ALD−AL)−(AD−A0)。同时完整各stage Acc/BA、所有31匿名类n/recall、全部正负、小分母、ISIC新类0/3与HK新类8/3；不把组合全部收益归漂移，不把梯度/先验/U激活叫净效用。

相对原A0每数据集均BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp，分别判断AD/ALD。新类损伤区分无下降、仍下降但门内、仍违反门；保护门不是无损。双数据集都过门仅为单seed反复开发证据，本次不自动追加确认seed。不是患者隔离/临床验证/论文复现，医学test不打开。

## 交付和费用

只收匿名metrics/STATUS/PROGRAM每次cost、epoch与boundary纯标量、4预检标量；不传checkpoint/影像/逐样本分数特征/样本indices/means/Q/U/中心原型边矩阵/private config/rawlogs或traceback。全部实际费用含失败保留；CPU检查/CPU等待/并行墙钟单列，旧已闭合GPU各round加旧83019.61353103025一次，不相加cumulative，未闭合RFMiD另列直至其公开闭合。

实际结束先公开源码/协议/完整匿名正负和失败/REPORT/贡献表/逐stage逐类/审计/费用，显式代理http://127.0.0.1:7897推送项目GitHub feat/prototype-analytic，核对账号/范围/remote SHA/精确commit报告匿名HTTP200，再保存NAS public_delivery及独立DELIVERY/HANDOFF；未核实不称交付。失败部分不填0、不以单臂称配对，不自动retry或新增轮。

本轮独立中性协调器/worker，沿已有同一小时监测跟进；RFMiD的冻结源/训练/评价/交付流程保持原封。两轮都交付后暂停同一监测，不另开科学实验。
