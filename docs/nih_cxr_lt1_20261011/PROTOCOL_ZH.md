# NIH-CXR-LT 首轮配对协议

状态：标签、划分、现有影像覆盖已核对，CPU检查后冻结排队；Tissue完成并公开交付前不启动GPU。本轮为作者定义的20类单标签自然长尾，原static_pc与既有local_transport各重新训练一次，仅GPU0/1。

## 作者标签与划分

[作者论文](https://arxiv.org/abs/2208.13365)及[作者代码](https://github.com/VITA-Group/LongTailCXR)指向NIH官方Box标签。只使用single-label_train和single-label_balanced-val CSV；不下载/打开作者test标签，不使用原NIH14类pilot的划分或结果作为本轮基线。每行20列恰有一个1，subject_id是元数据；不从多标签argmax，不筛选共病样本来伪造该定义。

训练68058张、23379位subject；验证300张、112位subject，每类15张。两个划分影像与subject交叉均为0。全部所需影像在既有缓存中找到，缓存只读、不复制影像、不改原始文件。没有逐字节复核承诺。fit/meta和增量任务按原图像身份规则，因此不声称这些内部划分也患者隔离。

按论文阈值>1000、100–1000、<100计算，实际发布CSV为7/9/4，论文写7/10/3。本轮使用真实CSV分组，并保留这一差异；不调整标签或阈值以凑数字。

|编号|标签|训练n|验证n|
|---:|---|---:|---:|
|0|Atelectasis|3135|15|
|1|Cardiomegaly|800|15|
|2|Consolidation|868|15|
|3|Edema|435|15|
|4|Effusion|2875|15|
|5|Emphysema|229|15|
|6|Fibrosis|519|15|
|7|Hernia|49|15|
|8|Infiltration|6941|15|
|9|Mass|1665|15|
|10|Nodule|2036|15|
|11|Pleural_Thickening|811|15|
|12|Pneumonia|213|15|
|13|Pneumothorax|1485|15|
|14|Pneumoperitoneum|59|15|
|15|Pneumomediastinum|7|15|
|16|Subcutaneous Emphysema|32|15|
|17|Tortuous Aorta|339|15|
|18|Calcification of the Aorta|121|15|
|19|No Finding|45439|15|

## 固定预算与机制

进入顺序为`[13, 19, 6, 14, 17, 1, 9, 4, 15, 8, 18, 2, 0, 12, 16, 3, 7, 11, 5, 10]`（seed74004），任务10+2×5；训练/split seed74002。每类原80/20 fit/meta，meta仍进入原controller和任务末拟合，验证不作meta。全部训练样本保留，不再合成长尾。

每臂六阶段更新1504/26/84/12/22/58，共1706；两臂3412，0策略更新。2epoch、batch64、workers4、lr.0003、WD.01、clip5、FD10、ridge.001、proximal.01、原AdamW和调度全部不变。首任务No Finding样本很多，因此不能沿用MedMNIST的1000步上限；本轮上限2000、精确计数1706。

候选保持已选local_transport：四区域CLS条件池化、减每图区域均值并归一化，每类四中心各1/2加入原P；旧局部中心按当前类均衡区域共同位移运输，全局仍common_shift。推理原全局解析竞争头；无局部CE、附加分数或可学习头，不回放旧影像或逐样本特征。

两个真实第一batch backward、0 optimizer预检全部PASS后训练；两臂TRAINED且各1706/0policy后才统一评价六阶段author_balanced_validation。24h从实际启动计，准备CPU成本另列；不得用新轮重置旧成本。失败保留回执/费用、停止依赖，不原轮自动重跑。新配对不要求原NIH14类或自然医学历史召回相同。

## 判定与交付

完整发布最终BA、平均增量BA、tail、遗忘、旧类、新类和全部候选−base pp，平均/最终Acc、六阶段/20类recall及n、头中尾全部正负。相对原base门BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp。仅开发门，不能替代ISIC/HK目标或称临床泛化；每类val仅15张，一个预测对应6.6667pp，不夸大极少类变化。

工程审计检查2/2、2零预检、3412正式更新、0策略/test、全阶段每类配对分母15、有限损失/梯度、解析残差≤1e−8、候选4*K局部先验列及prior_delta>0。没有附加推理分数，不虚构读出改变预测数或逐样本等价。

实际结束先公开GitHub源码/协议/完整匿名正负及失败/报告/审计/成本，显式代理推送，核对远端SHA和精确提交报告匿名HTTP200；再NAS匿名public_delivery和独立DELIVERY/HANDOFF。不公开CSV身份、影像、权重、逐样本分数/特征、统计/中心/边矩阵、私有配置或原始日志。随后才准备并启动RFMiD；标签或权限未核对完成时保持等待。
