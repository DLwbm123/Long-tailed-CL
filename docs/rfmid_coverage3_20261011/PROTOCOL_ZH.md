# RFMiD 第三轮：重点标签与负例覆盖配对协议

用户明确批准使用已核对的新划分，先冻结协议、模型兼容和两臂真实零更新预检，再以原 native 非RL多标签 A 与 local_transport 配对训练。只使用 GPU0/1，test继续封存。旧两轮数据、模型、源码和结果不修改，不重新训练或收费。

## 冻结任务与支持

|任务|新引入标签|图像数|Normal|更新/arm|当前旧阳性次数|
|---|---|---:|---:|---:|---:|
|1|Normal, DR, MH, TSLN, LS, CSR, ODC, ODP, AION|384|81|12|0|
|2|ARMD, TV, PT, MHL, RP|384|80|12|403|
|3|DN, ERM, ST, RT, EDN|384|80|12|346|
|4|MYA, MS, AH, ODE, OTHER|384|80|12|352|
|5|BRVO, CRVO, RS, CRS, RPEC|384|80|12|377|

具体分配直接使用上一轮 CPU 设计保存的可行解，样本身份仅留私有 NAS，不重新求解、重抽样、换seed或按val改组。1920张官方train每图一次，每task384，无旧影像或逐样本特征回放。原29标签映射与多hot共病保留。累计已引入标签known，未来unknown并清零；任何图像到达时其全部阳性均已引入。2470次阳性（含Normal401）及650对共病监督全部保留；后续当前旧阳性1478次，每个旧label在每个后续task均有正负支持。逐task逐label当前/累计n见TASK_SUPPORT_AUDIT。
Normal分布81/80/80/80/80。新标签引入时P≥min(5,ceil(trainP/2))、N≥5；后续旧标签P/N至少1，并沿v2供给规则。首task五锚点Normal/DR/MH/ODC/TSLN每task额外P≥floor(.8*trainP/5)；四重点AION/ODP/LS/CSR每taskP≥floor(trainP/5)。AION实际P为5/3/3/3/3，ODP13/13/13/13/13，LS9/9/9/9/11，CSR9/7/7/7/7。ST仍只有5个train阳性，TV/RP各6个，不增加稀少类供给。
四重点及首task锚点覆盖全部train正负组成。四重点N分别1903/1855/1873/1883，Normal负例各401；每个task都包含Normal、DR、MH、ODC、TSLN负例。条件负例疾病旗标可共病，不能相加。详细全部29标签负例组成、逐task四重点组成及相对v2变化见NEGATIVE_COVERAGE_AUDIT。
固定组规模/每task图数/累计mask下观察负例总数仍34010，未来19200次负例保持unknown。先引入重点标签有代价：MS/AH各少观察1152负例，ERM/CRVO等推迟标签减少负例窗口。不得只报告重点改善或把所有binary监督叫完整保留。设计相对目标gap约.099828%，约束逐项验证；不声称精确全局最优。

## 模型、固定预算与检查屏障

原 native A：ViT/双adapter、global1536、逐label正负n/s/G、signed binary analytic ridge+bias，weighted masked BCE、task2起FD10；AdamW lr.0003/WD.01/clip1，每task2epoch cosine eta_min1e-5、batch64/workers4、ridge.001、seed74002。原生无meta/controller；val不作为meta。
候选只沿已冻结 binary local_transport：四区域CLS条件pool、去每图区域均值/归一化；正/负各四单位中心/2入每label P，R=(I+PPᵀ)^−1，bias penalty1。旧局部中心按当前观察对权重区域共同位移，global原common_shift。每个已见label累积两符号支持为8列。无局部CE/附加readout/可学习head/RL或额外更新。保留既有共同位移近似，数学正确不保证AP安全。
每task2×ceil(384/64)=12，60/arm，总120正式adapter更新，policy0。两臂同起点、同样本、同seed与训练配置。从预训练真实配对，不把旧checkpoint作为新基线。CPU真实train标签与一张train PNG loader、五任务synthetic native等价/正负统计/局部先验/运输/直接解/checkpoint/unknown/test权限和paired barrier通过后才启动。CPU synthetic optimizer不算正式GPU训练。
两臂真实第一batch backward零optimizer预检（workers0）全PASS：0adapter/policy更新、有限非零梯度、solve≤1e-8，才进入两条正式训练。两条均60更新/5stage TRAINED、policy0，才统一评价。训练逐task严格核对冻结累计正负分母；有限loss/gradient与solve≤1e-8不放宽。失败停止依赖、保存全部错误/费用，不自动正式重跑。

## 数据权限、完整三轮报告

只读复用官方train1920/validation640及固定29标签：Normal=1−Disease_Risk、DR至RP27疾病、OTHER余18旗标OR。原45标签、原CSV、图像及私有样本分配不公开。test未下载、训练/预检/评价均拒绝test及跨dataset权限。旧数据不复制、不重新准备。无patient/eye ID，未做内容重复检查，不声称患者隔离或APPLE/MONICA精确复现。
全部训练后只official_validation，五stage seen9/14/19/24/29，每stage全部640图；未来label AP/AUROC null，不计macro。主mAP；完整六指标最终/平均mAP、tailAP、AP遗忘、old/newAP以及AUROC、固定.5 macro/micro F1、五stage与29label AP/AUROC/正负n、固定train head10/middle10/tail9和全部候选−base pp。AP遗忘允许负，不套单标签BA门或额外成功阈值。
完整结果与前两轮已公开RESULTS/LOCK比较：六臂全部六指标/AUROC/F1、第三轮对第一/第二每臂所有pp、各轮候选收益及收益变化、全部29label AP/同val P/N、固定频数组/各轮old-new固定集合、各stage共同已见集合。重点AION/ODP/LS/CSR与相对v2推迟引入标签固定集合全部正负，另列训练负例覆盖变化。报告不读取旧NAS模型/图像或重复收费。
v3与v2更新量同为60/arm，v1为66；课程、分配、mask不同。最终mAP/固定集合可以描述比较，平均增量/遗忘/各轮old-new依赖不同课程，不能当单因素因果。少量val阳性（TV/PT/RP各2）和单seed/重复val限制保留，不能替代自然医学双目标或临床泛化。

## 资源、费用、实际完成交付

只GPU0/1，2释放3禁止。启动前真实NAS nfs/free/小写读写探针、每卡≥32768MiB；中性父子入口保护其他任务。24h从实际launch起包含等待/预检/训练/评价。正常2preflight+2train+2eval六次驻留逐次记账，同job/train也不去重；失败另外全计。成功预检log可被同名train覆盖，独立PREFLIGHT/STATUS保留，披露该工程限制。
历史闭合GPU202975.5636281129秒保持，当前独立round GPUcost加一次；CPU设计34.119697秒是前期设计历史，CPU兼容与报告检查分别列，不加GPU、不重复子检查；新data preparation0。不能相加各轮cumulative。
实际结束先公开GitHub源码/协议/完整匿名正负及失败/报告/审计/成本、三轮对比，经显式代理核对账号DLwbm123/feat/prototype-analytic/范围/远端SHA/精确commit匿名HTTP200，再NAS public_delivery及独立DELIVERY/HANDOFF回执。禁止身份/患者/原CSV影像/模型/逐样本分数特征/私有ID/统计矩阵/private路径/raw logs。聊天直接完整结果与比较，不只链接；未核实不得称交付。完成全部交付后暂停同一监测，不另开科学轮。
