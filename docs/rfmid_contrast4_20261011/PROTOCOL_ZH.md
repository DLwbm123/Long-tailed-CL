# RFMiD 正负差异与去共享第四轮贡献验证

用户已明确授权本轮：严格复用 coverage第三轮 base(A0) 与原 local_transport(AL)，只新增正负差异(AC)和差异去共享(ACS)。两臂真实第一batch backward/零optimizer预检全PASS才TRAIN；两臂各60更新/5checkpoint全TRAINED后统一official validation。test一直封存，RL暂停。

## 冻结设计与匹配

公式和边界继承已公开的 `rfmid_contrast_prior_design_20261011/PROTOCOL_LOCK.json`（设计提交1f4458d5d5174c99d9c28e226dba7ffa3a92438b），不按CSR或其他val调整。

每标签四区域正负中心先取原坐标差δ=μ+−μ−，每列p=δ/[2 max(||μ+||+||μ−||,1e−12)]。AC直接使用这四列；ACS去掉其在四个正负中点(μ++μ−)/2的完整数值列空间上的投影。SVD秩只用float64 epsilon×max(shape)×最大奇异值；两次数值去泄漏，残量不单位归一化。R=(I+PPᵀ)^−1，bias惩罚1/ridge.001，缺任一累计符号或零差异回到I。总能量≤1，R特征值在[.5,1]。所有29标签同规则，没有CSR专用强度、阈值或ROI。

保留原global正负n/s/G、signed RHS/等符号质量、BCE/FD10、共同运输及按计数加当前区域中心。改变的是binary解析feature penalty；正负朝向仍由RHS决定，二次先验仍sign-swap invariant。没有局部CE/额外读出/学习头/旧影像或逐样本回放，没有残余漂移组合。

A0/AL来源科学冻结8dec5e8dfd4795759403720a20ae4cd309f2a59c，完整结果17c7e1880a69d8abce5d9a953e9805c56d5a93ab。CONTROL_MATCH逐项核对数据、既有预训练、课程具体TASK_PLAN、mask、seed、优化器/更新量/loader及评价权限；CPU五任务与该冻结Trainer比较原base和原local_transport的参数、global统计、head及local统计精确一致。只复用已公开聚合RESULTS，不读控制模型重新评价或收费。

新增Trainer mode与checkpoint version必须一致，加载缺失/错误mode/version关闭。原模式默认不写新版本字段、默认计算路径保持。worker要求本轮专门授权/版本/v3划分，其他数据集、test、错误预算或旧控制携带新先验均拒绝。

## 数据、支持度与预算

只读复用原official train1920/val640与29多hot映射（Normal=1−Disease_Risk、固定27病种、OTHER其余18OR）。复用第三轮具体批准TASK_PLAN，不重新求解、抽样、复制影像或下载。

T1 Normal/DR/MH/TSLN/LS/CSR/ODC/ODP/AION；T2 ARMD/TV/PT/MHL/RP；T3 DN/ERM/ST/RT/EDN；T4 MYA/MS/AH/ODE/OTHER；T5 BRVO/CRVO/RS/CRS/RPEC。每task384、每图一次、累计known与future unknown清零。2470次阳性、650共病对、34010已知负例与19200未来unknown负例保持。Normal81/80/80/80/80；CSR每task P9/7/7/7/7，最终train37P/1883N，val11P/629N。全部29支持度及负例组成继承公共AUDIT，不伪称全binary监督或患者隔离。

原native A ViT双adapter/global1536、2epoch/batch64/workers4、seed74002、AdamW .0003/WD.01/clip1、每task cosine eta_min1e−5、FD10/ridge.001、无meta/controller均固定。AC/ACS每task12，共60/arm、总120正式adapter更新、0策略。A0/AL新增更新和费用0。

CPU兼容五任务TinyRegions检验原两路径精确、两新路径正负计数/4*K列/共享泄漏/finite/solve、unknown mask及checkpoint恢复和版本拒绝。CPU中的synthetic训练与零更新反传不是正式科学训练或真实GPU预检。报告fixture核对四格、五贡献、5stage×29label完整输出；使用旧匿名聚合填充测试，不是新结果。

首次CPU加载混用旧源搜索路径，缺新observation_weights导出而在检查前失败；调整CPU临时模块组装后检查通过，保留私有回执，0GPU/0正式更新，未改环境或重启实验。设计阶段已有CPU诊断只列历史，不重计本轮GPU。

## 资源、顺序与失败

冻结源代理公开后启动中性协调器进入WAITING_PROTECTED_DRIFT。仅在module_drift10的PROGRAM COMPLETE或INCOMPLETE、active空且其worker实际退出之后，查NAS nfs/空间/小写读探针和GPU0/1每卡≥32768MiB。任一不足等待，不借GPU2/3、不杀或修改其他任务。原残余漂移部署源/配置/数据/预算/进程原封。

24h从本轮协调器实际启动计，包含等待/预检/训练/评价，不重置。两个预检名及日志独立于正式训练，PASS/0optimizer/0policy/非零有限梯度/solve≤1e−8才开放TRAIN。新两臂全部60/0policy/5stage训练封存才开放EVALUATE；复用控制无需重训。评价从versioned新checkpoint恢复，错误版本拒绝。

正常2preflight+2train+2eval=6GPU子进程，每个驻留记录各计一次，失败额外全计。CPU检查、协调器并行墙钟/资源等待另列，数据准备0，旧控制/设计不重复计。历史仅各独立round GPU驻留+旧83019.61353103025一次，不能加各轮cumulative。失败停止依赖、保留所有输出/费用，不自动正式重跑或新增修复轮。

## 贡献与完整交付

四格A0/AL/AC/ACS，完整六指标(final/avg mAP、tailAP、forget、old/newAP)、AUROC及固定.5 macro/micro F1，报告AC−A0、ACS−A0、AC−AL、ACS−AL、ACS−AC全部pp。AC对AL同时改变方向/列数/能量；ACS对AC同时删除共享分量及降低能量，不能当等强度单因素因果对照。新设计较弱，可能只减少原先验损伤，不能预先称CSR会提升或超过base。

完整五stage、29label AP/AUROC/正负n、训练固定head10/middle10/tail9与旧新集合、所有正负均公开。重点CSR及AION/ODP/LS，同时报告其他25标签的得失。future AP/AUROC null，不当negative或混macro；forget允许负，不套单标签BA门。诊断输出raw/retained energy、shared rank/leakage、prior delta/4*K/finite/solve，激活不叫效用。单seed/order重复val、CSR只有11阳性、缺patient/eyeID/内容重复核对，仅开发证据，不称临床泛化或APPLE/MONICA复现。

COMPLETE后一次匿名收集metrics/STATUS/PROGRAMcost、每臂15条epoch/boundary纯标量及两预检标量，复用已公开控制/标签/数据/支持audit。不复制影像/checkpoint/逐样本分数features/indices/means/Q/局部中心/矩阵/privateconfig/rawlogs。INCOMPLETE保留部分聚合和全部费用，去标识公开具体原因/缺失，missing不当0、单臂不当完整配对。

真实结束必须完整GitHub代理公开源码/协议/全部正负与失败/REPORT/CONTRIBUTIONS/AUDIT/费用，核对账号DLwbm123、feat/prototype-analytic、remote SHA和精确报告匿名HTTP200；然后NAS public_delivery与独立DELIVERY/HANDOFF。不能只留NAS。所有GitHub用本机http://127.0.0.1:7897，大小写代理及NO_PROXY仅localhost,127.0.0.1，不直连/强推/覆盖其他改动。

同一tissue-nih-rfmid小时监测负责本轮与尚未交付的残余漂移轮，不重复监测或旧轮交付；健康/资源等待无实质变化保持安静。两轮都完整交付后停止此顺序跟进，不追加新实验。
