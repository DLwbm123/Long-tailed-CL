# D78：完整六阶段 TRAIN 水平翻转敏感性诊断

完整18个已封存端点（六stage各SFT、prototypeGRPO、distributionGRPO），87651行新翻转端点嵌入、87651行原图保存特征复用。原图opposite metadata-component-fold支持固定，以prototype/kernel各raw/natural测量原图与翻转query，共72条配对/144条单视图统计路线。0optimizer、0backward、0VAL/TEST、0新classifier fit。软件→两张真实TRAIN工程预检→完整测量通过；之后saved-only verify→analyze，结案0新增模型评分/优化。没有选择新性能候选或训练增强模型。

单一变换为PIL RGB图像水平翻转，发生于原确定性resize/center-crop之前。不是垂直/旋转/强度搜索或test-time ensemble。原图embedding直接来自R77保存的全部TRAIN支持，SFT支持原fold1后fold0拼接通过实际fold索引逆置还原；labels/counts和原head原型parity核对。所有参数及49152LoRA delta不变、grad=None。每个新翻转prefix与同视图完整冻结visual调用比较，最大绝对差0（相同权重与算子路径）；不是与原图旧512cache比较，也不是独立权重验证。

这些TRAIN数据均曾参与端点训练，componentfold不是独立患者保证。原图支持固定仅隔离query视图变化，不能代表带翻转支持集的再训练或部署结果。完整TRAIN manifest元数据可读，模型输入/支持/加载head仅已到达类；无未来类图像/特征和VAL输入。逐class聚合见FULL_CLASS_SUMMARIES.json，保留全部计数、正确转移、预测分歧、paired cosine/probability/margin均值，无逐例或分位数数组。

## 最终stage：原图支持下的配对结果

均为TRAIN描述性BA，宏为class等权；tail沿用TRAINcount<250的11类。不能替代R77已封存VAL报告。

|端点|readout raw|原图宏BA|翻转宏BA|原图尾BA|翻转尾BA|类等权预测分歧%|类等权paired cosine|
|---|---|---:|---:|---:|---:|---:|---:|
|sft_reference|prototype_raw|88.505737|74.264994|83.220751|61.863878|21.289219|0.872824|
|sft_reference|distribution_raw|82.935927|71.686572|75.553153|59.656125|20.575047|0.872824|
|prototype_grpo|prototype_raw|81.596113|73.335535|72.194357|61.817356|19.506493|0.886346|
|prototype_grpo|distribution_raw|79.951098|71.180233|71.981933|59.730665|22.782589|0.886346|
|distribution_grpo|prototype_raw|84.673313|75.570912|76.912587|65.061973|16.584761|0.875233|
|distribution_grpo|distribution_raw|82.722033|74.988439|75.013580|65.790741|16.401421|0.875233|

最终SFT的prototype raw宏88.505737→74.264994、尾83.220751→61.863878，类等权预测分歧21.289219%，paired cosine0.872824。原图正确而翻转错误587行，反方向117行。prototypeGRPO同读出原图宏81.596113→73.335535，分歧19.506493%，paired cosine0.886346。不能只因GRPO差距更小或cosine更高就称鲁棒性收益：原图起点正确率本身已下降。distributionGRPO原生kernel raw宏82.722033→74.988439、尾75.013580→65.790741，分歧16.401421%，原图正确转错514行、翻转改对158行，仍存在双向变化。

## 全阶段方向与反例

prototype_raw：18端点中宏下降17、相等0、上升1；宏差范围[-14.240743, 0.041676]pp；尾下降16/18。

prototype_natural：18端点中宏下降18、相等0、上升0；宏差范围[-11.623611, -1.075408]pp；尾下降18/18。

distribution_raw：18端点中宏下降18、相等0、上升0；宏差范围[-11.249355, -0.069777]pp；尾下降16/18。

distribution_natural：18端点中宏下降18、相等0、上升0；宏差范围[-7.660496, -0.589793]pp；尾下降18/18。

prototype raw存在一个微小宏上升端点，raw尾类也有例外，完整stage表和全部分母保留，不能说所有类别/阶段都受损。natural两readout的全部18端点宏与尾均下降，但不据此复活已负的先验调整路线。

## 主线含义与边界

这补齐了此前D76几何/R77支持分布未测的同图方向敏感性。表示和固定支持分类证据会随输入方向变化；这与已封存原图训练/VAL负增量并存，但不证明它是GRPO失败的唯一原因。图像方向、原训练视图、crop交互、器械/文字/边框等均可能贡献，未做患者或因果识别。没有证明每张翻转图的23类标签一定保持，也没有证明增加flip训练会提高原图VAL。Guo等2024内镜SSL论文使用多种增强，但改为16类、训练与数据不同；这里只是考虑方向因素的先例。[原论文](https://www.nature.com/articles/s41598-024-53955-8)。不引入其无标签数据/增强配方，不重复已有指定PDF全文审计。

若后续研究视图覆盖，须另行冻结一个可证伪训练机制，并控制图像/支持分布、优化更新、query曝光、sampling randomness与额外forward/backward费用。方向一致性loss、翻转支持融合或挑阶段不能由本表自动推出；也不能把低敏感作为短pilot淘汰门。已知图像增强不包装新RL，保留R77同读出小正增量及其原SFT最终/尾类代价。

## 工程、资源及交付

独立原钟462.827361秒（7.71分钟），worker及后代CPU623.574958秒，workerRSS峰6829465600B，独立采样group+controller峰6852259840B，新增实际磁盘峰30673864B。原20h起点/截止未重置；不冒称该worker费用覆盖后续分析/发布CPU。

独特翻转TRAIN图像6821行，prefix FP32 RAM4127960064B；每条新prefix编码都额外跑一条同视图完整冻结visual参考，不能以87651尾部行数冒称全encoder总forward次数或等原实验FLOPs。native两真实TRAIN工程预检51.097010s，翻转native parity0/strictrestore/49152/无backward通过。只用物理A1000端点/1prefix，共享排他30GB主机RAM/每GPUreserved<12GiB，rootcap0/noOSswap；root/remote spill均0。

仅声明可再生prefix清理，历史原图/权重/所有R77支持学习产物保留，D78无逐例特征或预测落盘。公开自有代码、冻结协议、全部class-stage安全聚合/分母、scope/checks/cost及本报告；不公开图像/身份/特征/权重/私有配置路径或原日志。单HK74002反复开发，不是三seed/三域BOTH/TEST/临床确认。无活跃D78 GPU任务，后续主线授权仍有效。
