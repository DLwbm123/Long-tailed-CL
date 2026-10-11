# RFMiD 第二轮：保留共病与旧标签阳性的配对协议

用户已批准按审计过的 recurrent v2 划分补一轮。原 native 非RL多标签 A 与 local_transport 从同一预训练起点分别真实训练；旧轮代码、数据、模型、结果和交付回执保留。两臂采用同一已批准的具体私有样本分配，不重新求解、抽样或按val改变选择。

## 任务与监督

|任务|新标签|当前图像|adapter更新/arm|后续当前旧阳性次数|有旧阳性的图像|
|---|---|---:|---:|---:|---:|
|1|Normal, ARMD, ERM, MS, AH, PT, RT, EDN, MHL|384|12|0|0|
|2|DR, DN, CSR, TV, RP|384|12|39|37|
|3|MH, MYA, LS, CRVO, ST|384|12|142|129|
|4|BRVO, ODC, ODP, AION, OTHER|384|12|187|160|
|5|TSLN, ODE, RS, CRS, RPEC|384|12|308|234|

1920张官方train图像每张分配一次、每task384张，无旧图像或逐样本特征回放。累计已引入标签为known；未来标签unknown并清零，不伪造为negative。任何图像的阳性标签均已在该task或此前引入，保留全部2470次阳性（含Normal401）和650对共病阳性监督、458张多hot图。每个旧标签每个后续task都有正负支持；所有逐task逐label当前/累计正负n见TASK_SUPPORT_AUDIT。未来19200次负标注仍不进入训练，不能声称所有binary监督均保留。

新标签引入task的阳性最低min(5,ceil(train阳性/2))、负例最低5；TV/RP/ST阳性最低3，其余5。每个旧标签后续task阳性至少1且负例至少1；Normal/DR/MH/ODC/TSLN五锚点后续阳性至少20，其余旧阳性按LOCK中的供给公式。Normal在首task321、之后各20。稀少ST总5，T3为3、T4/5各1；TV/RP总6，T2为3、之后各1。并未生成额外标注、重复图像或扩大稀少类支持。

具体分配来源为上一轮只读设计的v2可行解，约束逐项独立验证；求解器60秒上限、gap约1.81%，不是全局最优声明。协议与源码冻结后不重新求解/替换seed或组。设计CPU历史另列89.817171秒，本次只做兼容检查，不重复收费。

## 模型、预算和屏障

沿原非RL多标签 A：ViT/双adapter，1536维global特征，逐label正负n/s/G和signed binary解析ridge+bias。masked weighted BCE；task2起FD10；AdamW lr.0003、WD.01、clip1；每task2epoch cosine eta_min1e-5；batch64/workers4；ridge.001；seed74002；全部当前task图optimizer及边界统计，原生无内部meta/controller。val不作meta。

local_transport只沿既有正/负四区域CLS pool、去每图区域均值/归一化；每个可用单位中心/2入各自binary P，R=(I+PPᵀ)^−1，bias penalty1。旧中心按原观察对权重的当前区域共同位移运输，global沿原common_shift。每个已见label累积两符号支持时8列；没有新增CE/readout/学习头/RL或优化器步。

每task 2×ceil(384/64)=12，5task每臂60，共120正式adapter更新、0策略。与旧66步历史并非等步控制：分配、引入组、监督mask和优化器步都改变，但图像总量与2epoch呈现次数相同；不能把跨轮差异全部归于一个因素。新轮候选相对新轮base才是同配置模块对比。

先CPU模型兼容与权限检查：原native基线的5task synthetic轨迹/解析头/统计精确一致；局部正负统计、8*K先验、checkpoint、unknown掩码、真实train PNG loader及新60步/旧66步隔离、评价屏障。随后两真实GPU第一batch backward零optimizer预检（workers0），adapter/策略更新0且有限梯度>0、解析残差≤1e-8；全PASS才两臂正式训练。两臂均60步/5checkpoint TRAINED且policy0才统一评价，禁止一臂先读val。正式失败停止依赖，保留所有输出与费用，不自动重训或放宽检查。

## 数据权限与报告

只读复用既有官方train1920/validation640和已冻结29标签映射，不重新下载/搬运或修改旧数据。Normal=1−Disease_Risk、DR至RP27病种、OTHER为余18旗标OR，固定来源见LABEL_MAPPING。多hot保留，不能声称精确MONICA/APPLE复现。原test未下载且继续封存，任何训练/预检/评价操作拒绝test；其他医学test也不解封。

全训练结束后仅official validation，五stage seen9/14/19/24/29、所有640图，未引入label AP/AUROC为null。主mAP；完整最终/平均增量mAP、tail AP、AP遗忘、旧/新AP、AUROC、固定threshold.5 F1、五stage及29label AP/正负n与全部candidate−base pp。频数组沿旧train head10/middle10/tail9，不按新val重分。AP遗忘=最终旧label的较早已引入stage最大AP−最终AP，允许负。不套单标签BA门；数学/局部先验激活不是净收益。

单seed、单order、重复official val仅开发证据；TV/PT/RP val仅2阳性，不夸大AP变化。无patient/eye ID且未做内容重复核对，不称患者隔离或临床泛化，也不替代ISIC/HK自然医学双目标。

## 资源、成本及交付

仅GPU0/1，GPU2释放/GPU3禁止；启动前真实NAS nfs/free/小写读写探针、每卡至少32768MiB。中性父子入口，保护其他任务。24h从实际launch计，准备/排队/预检/训练/评价均包含。正常2preflight+2train+2evaluate六次GPU驻留分别记录（同job/operation不去重），失败额外全计，CPU兼容和协调器墙钟另列，历史独立round只加一次、不相加cumulative。旧数据准备CPU不重计，新轮data preparation=0。

实际结束先公开源码/协议/全部匿名正负与失败/报告/审计/成本到项目GitHub feat/prototype-analytic，经显式代理核对账号、分支、远端SHA及精确commit报告匿名HTTP200，再保存NAS匿名public_delivery及独立DELIVERY/HANDOFF。禁止公开患者/身份/原CSV/图片/模型/逐样本特征分数/私有分配ID/统计矩阵/private路径/原始日志。聊天直接完整表格与正负，不只链接；受阻如实列未交付项。完成交付后暂停同一监测，不自行再开实验。
