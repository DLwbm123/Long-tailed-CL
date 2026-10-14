# GAIN 联合概率同状态候选诊断：生产冻结协议

用户明确授权按准备版2d7a90a59032d30078e5902f070d8d97f4b74a23执行既有18stage checkpoint诊断。当前仅生产接线；实际GPU状态必须以STARTUP_AUDIT/PROGRAM为准。固定方向源d258fff32b804717ee17b81d4985f0920fbf843b；旧GAIN估计与选择只复用已交付d7687016d5b5b51ce947027de0733c71008224fa的匿名公开结果。旧实验不重评、不重计费。

## 科学参数保持

原base/LT checkpoint、Q/R/竞争图/头/order/task/seed74002/累计更新计数只读。原ray/统计借用/proposal/native PCG/半径1/原新类守卫保持。四候选 .125/.25/.5/1 的原float32量化保持；新概率Gaussian点模型、共享尺度Student-t预测、三个借用情景、65536积分节点/4096 batch/seed1074005/α.01/M432/未知质量规则与准备版完全不变。生产层不修改design helper。主控制local_transport，主选择JOINT_PRED_SELECTED，禁止按val切换。

四条轨迹ISIC_base→ISIC_local_transport→HK_base→HK_local_transport，仅GPU0/1。每个控制3/6stage，共18stage；实际侧文件只存四候选，共72。七个非OFF读数 C0125/C025/C05/C1/LEGACY_SELECTED/JOINT_POINT_SELECTED/JOINT_PRED_SELECTED；后面三项引用相同候选或0，不增设方向或拟合头。总32主行加AIR公开参考2行；RAW另列28行。

## 生产与权限屏障

1. 完整合成1536×23 CPU验收先PASS，包括实际native求解、非零proposal/ray、四候选与冻结方向一致、两R、四候选概率/两选择、roundtrip/来源拒绝、守卫、报告完整性、错误分母与早val拒绝。合成通过不是医学结果，不虚称GPU准入。
2. 冻结本轮科学源并通过代理push/remoteSHA/精确协议匿名HTTP200后，NAS真实nfs/free/小probe保留source。CPU准备只读取批准的原checkpoint所含统计，不读train/val/test图。全部18stage重建原头校验，solver≤1e−8，source/version/dataset/control/order/seen/seed/task/原steps精确核对，原checkpoint bytes/mtime前后不变；原公开ray norm另做一致性核对。四准备精确PREPARED才seal。
3. 四条真实finalstage第一val batch零更新预检分别PASS，encoder所有state精确不变，CPU/GPU OFF及四候选/三个引用头守卫分数一致，梯度/optimizer/policy更新0，指标读取false，test=false。任何一条失败停止依赖，不自动retry；失败成本全保留。
4. 四PASS seal后才统一每stage已有development val评价。训练图、test/reserved禁止；无重新抽样、feature回放、val方向拟合/参数选择。val接入前worker再次核对全部四状态及source seal。影像只共享encoder一次，同batch完成各读数。

## 成本与运行

当前CPU高维检查独立收费，pipeline包含于process residence。启动后4个CPU准备、4preflight+4eval=正常8GPU进程；失败额外计，每次cost独立保留。GPU01每次派发前free≥32768MiB；不借2/3、不干预其他进程。中性父子入口与完整ps/NVML核对；私有路径经环境变量传入。真实launch起4h含CPU准备/等待/预检/评价，不重置。已交付准备CPU29.468065921秒历史列示，不重复计入本轮。

原阶段val共享曝光总15966，4预检各最多64=256；七读数不乘曝光和GPU费用。encoder/model/checkpoint/stat矩阵/影像/样本/逐图分数仅NAS私有，匿名报告不外传这些对象。

## 全量结果与判断

32主行六指标(final/avg BA、tail、forget、old/new recall)/final与avgAcc/提供的macroF1；保留全部RAW、stage、31类n/recall、纠正/破坏/wrong→wrong、四类新旧混淆、间隔正零负、小n、守卫挡住的救回。无结果为null，不能填0。

全部88组贡献：七ON−OFF(28)，两新选择−旧选择(8)，预测−点选择(4)，八读数−AIR(32)，七读数双控制增量交互(14)，两数据集预测LT−预测base(2)。不依据其中最好项改主方法。

四候选×18stage×2estimator×3情景=432完整模型与实际比较。macro实际净=sum_c((rescued−harmed)/val_n_c)/K。完整中心和估计−实际偏差仅当unknown_mass=0提供；unknown>0时完整中心偏差留null，known-only减实际总净值单列并明确不是总体偏差。给出全部情景中心/数值半宽/未知旧新质量/净上下界、class实际和部分预测、原界、选择索引、support/prior/df/节点及舍弃trace。

**区分方向与准入**：四候选实际macro净值均不利，支持该状态这四候选无收益；存在实际有利而选择0，支持准入保守，再分未知质量/数值误差/预测尾部/模型中心排序。模型预测正而实际负属于模型失配；不能把identity直接当方向无效或把数值节点的区间当医学统计覆盖率。Student-t与三情景只是前瞻近似，不是完整后验或临床证书。

完成后匿名RESULTS/CONTRIBUTIONS/AUDIT/REPORT及全部正负CSV、失败与费用经代理公开GitHub、核实remoteSHA/精确REPORT匿名HTTP200，再NAS独立public_delivery/DELIVERY/HANDOFF。若INCOMPLETE则交付部分聚合/去标识原因/全部费用，不追加实验或科学修复。单seed/order反复development val，非封存test、临床、独立患者或统计显著结论。
