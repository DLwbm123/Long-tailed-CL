# RFMiD 29标签多标签首轮冻结协议

状态：FROZEN_PRELAUNCH_PROTOCOL。训练标签、任务、方法和预算已独立预注册；CPU数学、两任务小型训练与权限检查通过。尚未启动RFMiD GPU预检/训练，尚未进行分类评价。必须先核实NIH完整GitHub/NAS交付及其进程全部退出，再经GPU0/1和NAS准入启动本轮。仅此一轮两臂，不扩大数据/test权限。

## 数据和标签

用户提供的官方train1920/validation640全部实体PNG已准备，不重新下载、复制或抽样。Normal=1−Disease_Risk，DR至RP固定27疾病，OTHER=余18旗标OR，保留多hot共病。类别选择来源为已核实[挑战赛参赛作者固定代码](https://github.com/frankkramer-lab/riadd.aucmedi/blob/ec76be86809a8ae7e898d29c0d8d1e550aa2eef4/scripts/inference.py)，映射和全部正负支持度见[LABEL_MAPPING](LABEL_MAPPING.json)。MONICA实际映射未核实，不声称精确复现。原45旗标/CSV私有保留；所有test封存。只有split内样本ID，无patient/eye ID；未做内容重复检测，不能声称患者隔离或跨split无重复。

## 训练来源独立确定的增量任务

依据train阳性数量升序，先seed74004 shuffle再稳定排序打破同频次，固定进入顺序：ST、TV、RP、MHL、PT、RT、ERM、EDN、MS、AH、AION、RPEC、CRVO、CRS、OTHER、CSR、RS、LS、ODE、ODP、BRVO、ARMD、MYA、DN、TSLN、ODC、MH、DR、Normal。任务9+5+5+5+5标签。每张train图分配到含其任一阳性标签的最早任务；五任务样本95/114/199/389/1123，每张恰好一次，不回放旧影像或逐样本特征。当前图仍保留原multi-hot，但只有累计已引入标签参与loss/statistics；未来标签是unknown，经mask屏蔽并清零，不能作为negative。

这种无回放分配会使早期共病图上的未来阳性在后续标签引入时无法再用于监督：共324个阳性标签出现次数受到屏蔽，不是丢失同样数量的图像，亦不能称完整保留所有标签监督。逐标签数、每task正负和累计可见计数见[TASK_SUPPORT_AUDIT](TASK_SUPPORT_AUDIT.json)。本规则只依train确定，没有用validation改善分配。每个新引入标签在其任务均有实际正、负支持；此最早阳性分配使所有旧标签在后续task的当前图均无阳性：按原native A仅对当前可用binary组加权，旧positive聚合证据仍保留；后续负例训练及仅当前图的FD可能损伤旧label AP，不能称无损。缺失符号不能补成观测。

基线沿用实际已有`run_multilabel.Trainer(method='A')`，它没有内部meta控制器：本轮明确全部当前task训练图同时用于optimizer及任务末聚合，内部meta=0；official validation不作meta。没有为了套用单标签流程额外扣除20%图像。这里的基线是原多标签A，而非把单标签static_pc的softmax或竞争头直接套过来。

## 配对方法与预算

两臂同预训练ViT、双adapter、seed74002、1536维全局feature和逐标签signed二分类解析头；逐标签保留n/s/G全局充分统计，标签共病不互斥。原weighted masked BCE+任务2起FD10、AdamW lr.0003/WD.01、原clip1、每task cosine两epoch/eta_min1e−5、batch64/workers4、ridge.001全部保持。原多标签代码clip是1，不能误写成单标签代码clip5；没有单标签proximal.01竞争损失。head包含原正则化bias，任务初head固定至任务末再解析拟合。

每臂五任务更新4/4/8/14/36，共66；两臂132正式adapter更新，0策略。两真实第一batch backward预检workers0且0optimizer更新均PASS后才TRAIN；全部两臂66且0policy TRAINED后统一五stage官方val评价。每臂的固定optimizer步数和统计分母必须与冻结task plan一致。不自动正式重跑，不挑seed或按val调温度/先验幅度、选阶段/类别。

候选仅将原binary head的identity feature ridge penalty替换为四区域二分类先验：同一双分支前向捕获原norm token，按既有CLS条件pool、去图像区域均值和归一化，分别聚合每标签正/负的四region中心；每个可用中心单位归一化后×1/2作为P列，R=(I+PPᵀ)⁻¹，bias penalty仍1。原native A没有单标签global P，因此不虚构已有global prior；全局n/s/G及signed rhs原封，附加局部证据只改变ridge feature penalty。每个已支持binary符号提供4列，每标签最多8列；无支持符号不造中心。旧局部中心用当前观察对平衡权重的四region共同位移运输，然后按binary计数合并新中心；global仍原common_shift。没有局部CE、附加读出、可学习头、额外optimizer步或RL。

CPU检查见[CPU_ENGINE_CHECK](CPU_ENGINE_CHECK.json)和[报告/权限fixture](CPU_REPORT_CHECK.json)：与修改前native A在两任务fixture中adapter、head、全局统计逐项精确一致，局部正负计数与global一致，unknown不计negative，Woodbury与直接逆一致，运输与直接重算一致，candidate有限/数值残差、存取checkpoint及零更新encoder精确不变均PASS。test/跨dataset、预算或阈值越权与未封训练屏障的评价均拒绝。这是CPU工程证据，不能替代真实GPU预检或分类净效用。

## 评价、费用和公开交付

主mAP为累计已引入标签AP平均。逐label公开AP、positive/negative n、可定义AUROC和固定threshold.5的F1等；不以sigmoid为校准概率。五stage可见标签9/14/19/24/29，全部640val图；未来标签无观察，指标null，不当negative或混进macro。最终29标签均可定义AP/AUROC，极少val阳性TV/PT/RP各2，完整正负报告且不夸大AP变化。

固定train频数组head10/middle10/tail9，成员见LOCK；报告两臂最终/平均增量mAP、tail AP、旧/新label AP、遗忘以及所有百分点差。遗忘=最终旧label在较早已引入stage的最大AP减最终AP后平均，可以为负；不套单标签BA六门或声称自然医学双目标已通过。单seed/order、重复official val仅开发证据，非官方test benchmark/患者泛化/临床验证。

整轮≤24h从实际GPU队列启动计，正常2preflight+2train+2eval共6GPU进程；同job/operation=train的预检与正式成本每次保留，失败全计，GPU residence按独立round只加一次，CPU数据准备/两次检查和墙钟另列。NIH运行不受新文件影响，其NAS冻结科学源不修改。

结束先完成公开项目GitHub源码/本协议/全部匿名正负及失败/报告/审计/成本，显式代理推送并核实远端SHA/精确commit匿名HTTP，再保存NAS public_delivery和独立回执。公开不含原CSV/图像/权重/样本身份/逐样本预测features/统计中心矩阵/私有配置或原始日志。队列全部交付后暂停同一顺序监测，不追加下一科学轮。
