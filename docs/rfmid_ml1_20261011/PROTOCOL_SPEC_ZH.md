# RFMiD 多标签首轮准备规格

状态：WAITING_USER_DOWNLOAD / LABEL_MAPPING_NOT_FROZEN。已授权在Tissue、NIH先后完整交付后启动；目前没有RFMiD GPU进程、没有正式训练。只GPU0/1。本文是准备规格，不能当作已经核对实际CSV并冻结的科学运行协议。

## 官方数据与权限

[原始数据论文](https://www.mdpi.com/2306-5729/6/2/14)、[官方全标签发布页面](https://riadd.grand-challenge.org/download-all-classes/)和[MONICA benchmark](https://arxiv.org/abs/2410.02010)为依据。只下载官方train1920和validation640；test640不下载、不读取。官方发布与论文的疾病数/风险字段、MONICA合并少样本类别后的29标签存在定义差异，必须从实际CSV和作者映射核对，不能仅凭表格数字生成标签。

- [Training_Set.zip](https://drive.google.com/file/d/1_pzNv3z3H1_ao8nOrp-B40S5TE1Rcx1W/view)
- [Evaluation_Set.zip](https://drive.google.com/file/d/1ORmTGxwXrU9QjufVL8H93yj8GZHjxrXC/view)

用户下载后提供本机目录；按既有NAS规则传输一次，普通完成状态/退出码及一次必要可读性检查，不默认哈希或重复下载。原始标签和图片私有保存，公开只有类别聚合和标量。不能用未经确认的镜像补标签。

核对CSV列定义、binary值、Disease_Risk与疾病列的关系、ID唯一性、train/val图像交叉、图像覆盖及可解码性；患者/眼ID如不可得，明确无法独立确认患者隔离。保留multi-hot共病，不argmax、不删除多病图像，不制造未知标注为negative。29标签的Normal/OTHER规则须由作者定义确认，所有合并及原类计数公开；如果作者映射不可核实，记录差异并停在准备阶段，不冒称MONICA复现。

## 拟保持的方法与实验范围

固定原多标签非RL工程主体为基线；现有单标签竞争头不能直接用softmax替代多标签。保留ViT/adapter、全局充分统计、逐标签正负平衡的解析二分类头和原FD10；候选仍测试四区域局部聚合先验及运输，无额外推理读出/局部CE或学习head、无旧逐样本回放。多标签正负局部统计与解析头的兼容实现必须先做数学CPU检查和真实零更新预检，不能把已完成的NIH14标签pilot当本轮RFMiD基线。

首轮仅同配置原基线/一个固定候选两臂，从相同预训练真实配对。使用原train自然长尾，不另合成下采样；原official validation不作meta，不看test。fit/meta只从train划分。实际标签支持度、标签增量阶段/掩码、训练样本分配、精确batch更新预算、任务顺序和全部参数必须在看本轮分类评价之前独立冻结；本次尚未生成最终PROTOCOL_LOCK，不能宣称这些已确定。

## 评价与交付

多标签主指标为mAP；同时给每标签AP、正负n、可定义标签AUROC、head/middle/tail AP、增量mAP/遗忘/旧新标签指标，固定支持度掩码并披露无正/无负标签，不将单标签BA和argmax召回作为主结果。标签组与阈值只依train和作者协议固定，不扫验证阈值/seed，不只报正标签。报告共病定义、不同方法原配置和全部候选−base增量；不套用单标签六指标门宣称多标签达门。

两臂真实零更新预检全PASS、全部固定正式训练完成后才评价；24h含实际启动后的准备/排队/训练/评价，费用不重置。实际失败完整保留，不暗中重跑。完整公开GitHub源码/最终协议/匿名正负及失败/报告/审计/成本，代理远端SHA/精确报告匿名HTTP200核实，NAS公开包/私有回执齐备后结束。图像、权重、原始CSV、身份、逐样本预测或统计矩阵不公开。

待解决：用户下载包；实际CSV/作者29类映射；训练来源的正负支持度与增量阶段；候选多标签兼容数学/工程检查；最终预算和协议锁。以上全部完成且NIH完整交付后才可启动。本轮不扩展其他RFMiD版本、数据集、test权限或RL。
