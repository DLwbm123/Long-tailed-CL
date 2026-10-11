# RFMiD 多标签首轮准备规格

状态：CSV_AUDITED / NAS_PREPARING / LABEL_MAPPING_NOT_FROZEN。已授权在Tissue、NIH先后完整交付后启动；已定位用户下载的官方train/validation两个包，CPU传输和实体化准备正在进行；目前没有RFMiD GPU进程、没有正式训练。只GPU0/1。本文是准备规格，不能当作已冻结标签映射、增量任务和训练预算的科学运行协议。

## 官方数据与权限

[原始数据论文](https://www.mdpi.com/2306-5729/6/2/14)、[官方全标签发布页面](https://riadd.grand-challenge.org/download-all-classes/)和[MONICA benchmark](https://arxiv.org/abs/2410.02010)为依据。只下载官方train1920和validation640；test640不下载、不读取。官方发布与论文的疾病数/风险字段、MONICA合并少样本类别后的29标签存在定义差异，必须从实际CSV和作者映射核对，不能仅凭表格数字生成标签。

- [Training_Set.zip](https://drive.google.com/file/d/1_pzNv3z3H1_ao8nOrp-B40S5TE1Rcx1W/view)
- [Evaluation_Set.zip](https://drive.google.com/file/d/1ORmTGxwXrU9QjufVL8H93yj8GZHjxrXC/view)

用户已下载两个包；按既有NAS规则传输一次，普通完成状态/退出码及一次必要可读性检查，不默认哈希或重复下载。原始标签和图片私有保存，公开只有类别聚合和标量。不能用未经确认的镜像补标签。

核对CSV列定义、binary值、Disease_Risk与疾病列的关系、ID唯一性、train/val图像交叉、图像覆盖及可解码性；患者/眼ID如不可得，明确无法独立确认患者隔离。保留multi-hot共病，不argmax、不删除多病图像，不制造未知标注为negative。所有合并及原类计数公开。现已核对真实CSV：45疾病旗标加独立Disease_Risk，train/val风险旗标与全部疾病OR无不一致，共病图像458/152，Normal 401/134。ID只在split内唯一，不能拿两边同号判断图像重叠；未获取患者/眼ID，也未做图像内容重复检测。完整聚合见[LABEL_AUDIT](LABEL_AUDIT.json)。

按train“>10例”只保留24个病种，加OTHER和Normal仅26标签，不能复现论文29。挑战赛参赛作者[固定输出列代码](https://github.com/frankkramer-lab/riadd.aucmedi/blob/main/scripts/inference.py)使用DR至RP的27个固定病种加OTHER；可构成Normal+27+OTHER的29标签视图。拟定Normal=1−Disease_Risk，OTHER=余18旗标OR，仍保留全部45原旗标私有来源；此视图所有29标签在train/val均有正负支持。类别列表来自已发布代码，不能依本轮val频次选类。MONICA作者目前仅发布README，未得到实际预处理映射，因此只将此视图记为有来源的提案，尚未生成训练标签或最终LOCK，也不声称精确MONICA复现。

## 拟保持的方法与实验范围

固定原多标签非RL工程主体为基线；现有单标签竞争头不能直接用softmax替代多标签。保留ViT/adapter、全局充分统计、逐标签正负平衡的解析二分类头和原FD10；候选仍测试四区域局部聚合先验及运输，无额外推理读出/局部CE或学习head、无旧逐样本回放。多标签正负局部统计与解析头的兼容实现必须先做数学CPU检查和真实零更新预检，不能把已完成的NIH14标签pilot当本轮RFMiD基线。

首轮仅同配置原基线/一个固定候选两臂，从相同预训练真实配对。使用原train自然长尾，不另合成下采样；原official validation不作meta，不看test。fit/meta只从train划分。实际标签支持度、标签增量阶段/掩码、训练样本分配、精确batch更新预算、任务顺序和全部参数必须在看本轮分类评价之前独立冻结；本次尚未生成最终PROTOCOL_LOCK，不能宣称这些已确定。

## 评价与交付

多标签主指标为mAP；同时给每标签AP、正负n、可定义标签AUROC、head/middle/tail AP、增量mAP/遗忘/旧新标签指标，固定支持度掩码并披露无正/无负标签，不将单标签BA和argmax召回作为主结果。标签组与阈值只依train和作者协议固定，不扫验证阈值/seed，不只报正标签。报告共病定义、不同方法原配置和全部候选−base增量；不套用单标签六指标门宣称多标签达门。

两臂真实零更新预检全PASS、全部固定正式训练完成后才评价；24h含实际启动后的准备/排队/训练/评价，费用不重置。实际失败完整保留，不暗中重跑。完整公开GitHub源码/最终协议/匿名正负及失败/报告/审计/成本，代理远端SHA/精确报告匿名HTTP200核实，NAS公开包/私有回执齐备后结束。图像、权重、原始CSV、身份、逐样本预测或统计矩阵不公开。

待解决：NAS图片解包/覆盖/可读性检查完成回执；29标签映射最终冻结与MONICA定义差异；训练来源的正负支持度与增量阶段；候选多标签兼容数学/工程检查；最终预算和协议锁。以上全部完成且NIH完整交付后才可启动。本轮不扩展其他RFMiD版本、数据集、test权限或RL。

## 本次准备工程记录

两个官方包已完成CSV和ZIP成员数本机核对，NAS CPU准备按原ZIP内置CRC读取并实体化图片，不扩大数据权限、不默认额外哈希或全量像素复测。一次最小CPU检查覆盖binary/schema/非法ID/重复或错行数拒绝。NAS准备结束将检查1920/640图像与标签精确覆盖，并各完整解码一张。未完成前不以文件存在宣称训练就绪。传输启动后本机回执写入脚本遗漏time导入，已按现存父子进程恢复回执；没有重启传输、没有GPU科学轨迹。CPU传输/解包时间单列。
