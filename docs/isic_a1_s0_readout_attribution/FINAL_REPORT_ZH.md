# A1 首阶段表征与类别加权归因：部分交付

**状态：BLOCKED_ROUTING_PARITY，已停止。** 已实际运行资产核验、CPU解析工程检查、三个C-S0父模型的隔离GPU探针，以及A-U/A-CB全部顺序和阶段的拟合、validation评分和分析。三个父模型均未通过固定的原始FP32特征容差，因此没有释放S正式特征提取或S候选评分。不是完整72/432矩阵，也没有A1主比较的正负结论。

实际覆盖：主指标 **18/72**，逐类 **108/432**；核心分别 **18/36**、**108/216**。未完成的是S-J/M/F各自U/CB六项，共54行主指标、324行逐类。它们均标记未运行，不能当成零分、负结果或按性能剔除。

## 阻塞证据

恢复三个原C分支S0的epoch10最终checkpoint，文件和完整网络tensor哈希通过。所有参数/buffers冻结，eval和无梯度；原始训练配置、顺序、类别数量、训练源及协议元数据匹配。没有实例化optimizer。父1993/1994/1995各自仅使用对应顺序，不交叉父状态。

工程样本严格为每父S0基础类中sample_id升序的train前16张和val前16张。隔离探针只关闭两个已批准的batchwise_prompt旗标，保留native top-1；通过真实network(images, train=False, weight=None)读取pre_logits及pre_logits_few，未输入标签、频次、task或adapter ID。

比较规则保持 `abs(batch-single) <= 1e-5 + 1e-5*abs(single)`，reference为native B=1。下表是pointwise B32与同32张逐图reference比较，超差元素分母每路为24,576。

| 父模型 | main最大绝对差 | few最大绝对差 | main超差数 | few超差数 | 两路路由ID不一致数 |
|---|---:|---:|---:|---:|---:|
| 1993 | 5.7220459e-05 | 0.000213623047 | 21 | 165 | 0 |
| 1994 | 4.76837158e-05 | 0.000164836645 | 56 | 191 | 0 |
| 1995 | 4.76837158e-05 | 4.68492508e-05 | 18 | 33 | 0 |

反序、替换同行和预设B48也出现特征超差，但pointwise两路路由ID在所有这些布局中均与native B=1一致。同一个B32重复前向逐位一致；每父第一张图的pointwise B1与native B1抽查逐位一致。三个父模型的参数/buffers及CPU/CUDA/NumPy/Python随机状态均未改变。

算子/精度检查：参数float32，AMP关闭，matmul/cudnn TF32均关闭，float32_matmul_precision=highest。层级CLS记录显示差异从第一个Transformer block出现并累积；1993/1994两路在最后一个block首次越过容差，1995两路在最终norm越过容差。原始编码器也有批次尺寸数值差异，但其归一化train探针与锁定A缓存均通过容差，最大绝对差不超过1.066e-6。

这些观察支持“相同路由下的批次尺寸相关浮点差异”，排除了本固定探针中路由ID选错、父状态或预处理明显不符；尚未确定具体GPU kernel的差异来源，不能证明表征损坏，也不能以相对L2误差小来替代逐元素门槛。完整B32/B48、分层误差和精度见ROUTING_PARITY_FORENSICS.json。未按效果更改容差、路由、batch或精度。

作为背景诊断，原native batchwise B48在1994的few路有5/48个ID不同于逐图路由；其余两父为0。这与pointwise的FP32特征门槛失败是两个不同观察，不能混淆。已批准的pointwise旗标无需再次审批。

## 已完成的原始表征参照

仍使用V2锁定train18,718 / val295和8类、4→6→8；训练计数为[10529,3263,2835,1306,458,256,44,27]，实际不平衡比389.963。A缓存只由float32转float64，不二次归一化。U按样本均衡、CB按类均衡，总权重均为1；lambda=0.001、无bias、float64统计和直接求解。没有新数据、调参或重新抽取长尾。

| 方法 | Final BA % | tail % | 标签6正确/总数 | 标签7正确/总数 | AvgBA_all % | AvgBA_inc % |
|---|---:|---:|---:|---:|---:|---:|
| A-U | 26.993 | 0.000 | 0/26 | 0/24 | 29.450 | 26.084 |
| A-CB | 55.497 | 71.154 | 24/26 | 12/24 | 61.703 | 58.014 |

两个A最终拟合各是一个确定性参照，三个顺序最终预测一致，不报告伪造的三训练重复SD。A-U最终零召回原始标签为3、4、5、6、7，A-CB无零召回。A-CB的head/mid/tail recall分别为63.176%、43.830%、71.154%；A-U分别为73.188%、17.391%、0%。类别加权伴随头类召回下降，不能只引用BA提高。

| 顺序 | 阶段 | A-U BA % | A-CB BA % | A-U tail % | A-CB tail % |
|---|---|---:|---:|---:|---:|
| 1993 | S0 | 29.939 | 67.864 | 0.000 | 50.000 |
| 1993 | S1 | 19.959 | 64.731 | 0.000 | 71.314 |
| 1993 | S2 | 26.993 | 55.497 | 0.000 | 71.154 |
| 1994 | S0 | 36.915 | 58.559 | 0.000 | 54.167 |
| 1994 | S1 | 27.783 | 55.545 | 0.000 | 50.000 |
| 1994 | S2 | 26.993 | 55.497 | 0.000 | 71.154 |
| 1995 | S0 | 41.689 | 80.817 | 0.000 | 96.154 |
| 1995 | S1 | 27.783 | 61.315 | 0.000 | 92.308 |
| 1995 | S2 | 26.993 | 55.497 | 0.000 | 71.154 |

A-U/A-CB的18行指标及108行逐类正确数均重现G1高精度validation（数值检查1e-10），没有用R1 test作为复现目标。old/current与HM、base/post-base、尾类基础/后续划分、逐类margin、两种错误分解均在主表、逐类表和error_decomposition.csv中。forgetting.csv给出首次学完到最终的有符号变化；最后阶段才到达类的遗忘为NA。

## 可完成的固定配对分析

只对预定的A-CB−A-U作配对分析。按类别分层identity_component bootstrap 2,000次、seed43001，所有可用方法和固定顺序共用抽样；每次抽样平均三个固定配对，未重采样父模型。

| Final指标 | A-CB−A-U pp | 条件性95%区间 |
|---|---:|---:|
| balanced_accuracy | +28.505 | [+22.943, +34.033] |
| tail_rank2 | +71.154 | [+58.973, +83.333] |
| old_macro_recall | +34.298 | [+28.287, +40.432] |
| current_macro_recall | +11.125 | [+2.019, +20.163] |

这只说明既有原始A表征下的读出目标差异。**S-J-CB−A-CB、S相关全部次比较和交互差均为NA**，不能据A结果推断C-S0更好或更坏。没有挑选main/few/J中的最佳分支，也未生成任何S候选分数。bootstrap不校正反复使用validation、既有test反馈、未知患者关联或域变化，不是独立确认或临床非劣效。

## 工程验收和锁状态

已通过原始A统计流式/加权批处理、累计状态存取恢复、最终顺序不变性、复制样本控制、独立小型重复嵌入求解和实际A的重复嵌入分数/正则范数控制。真实联合特征不存在，不能把toy中的完整cross-Gram验证说成真实S的验收。当前类访问和test split负向guard、最小原标签tie规则通过；S真实特征的类内逆序/分块等价检查未运行。

协议定义先冻结于6830dab。CODE_LOCK_A1、PROTOCOL_A1、FEATURE_VIEW_LOCK与PARENT_LOCKS记录定义、执行源码和已核验父状态；**工程门槛失败，正式S候选release lock未释放**。完整矩阵吞吐准入没有到达，实际资源消耗通过不能替代完整预算准入。

首次工作器exit=1并保留原始失败证据；随后只做固定小样本数值取证，exit=0代表诊断完成，并不代表parity通过。初始ACCESS/RESOURCE报告单独保存，汇总报告纳入额外诊断及仅CPU的A结果分析。

## 资源、空间与访问

GPU工程上下文驻留观测合计 **28.08秒**；由两个日志创建时间至退出回执时间构造的保守整进程上界为 **43.87秒**，包含其CPU工作和退出。CPU解析/保存分数分析累计 **2.539秒**。峰值RSS **3.011 GiB**，GPU allocated **1.114 GiB**。单worker、4线程，未并行父模型。GPU现已释放。

新增文件和结束时磁盘精确读数见RESOURCE_REPORT.json中的最终存储盘点；正式S raw缓存为0。现有约2.10 GiB空闲足够保存本轮证据，无需清理。删除历史文件=0，删除本轮失败partial=0；没有复制原始A缓存、权重或checkpoint，没有安装包、下载模型或扩容。实现预定的raw双路只存一次、派生视图不落盘策略尚未进入正式提取。

实际解析阶段拟合 **18次**，含toy/恢复/最终等价检查的solve共 **48次**。工程图像读取事件 **128**（train64、val64），对应96个唯一父模型–图像组合；父1993初始探针和取证重复读取32张。正式新特征行 **0**。wrapper调用 **157次**、图像行 **1059**；内部original/main/few调用分别为 **160/157/157**，合计 **474**，内部图像行总计 **3225**。这些包含工程前向，不能声称encoder调用为0。A的统计累计读取缓存训练行56,154（每顺序18,718，U/CB共享），val不参与任何拟合。

新增神经训练epoch=0、optimizer step=0、S0训练=0；新增test图像读取=0、test特征读取=0、test预测=0。没有生成可选train重代入评分；S训练特征几何未运行。未使用旧test特征乘任何新W，也没有新增test分析。

## 结论与停止

观察事实：原始A基线及其类别加权收益已重现；原生路由ID通过、父状态和随机流保持；固定FP32特征门槛在三父模型均失败。机制推断限于批次尺寸相关数值差异，具体kernel来源未证实。A1所问的首阶段适配表征增益与交互差仍未验证。

**NEXT_DECISION=STOP，BLOCKED_ROUTING_PARITY。** 不自动降低batch、提高precision、放宽容差或换路由；不启动S0重训、GSR/K修复、融合、新backbone、新数据、test或监测。最小待决事项是是否为该数值一致性问题制定并锁定修订协议；本报告没有选择新的容差或绕过现门槛。

公开源码、协议、hash、工程和聚合表；raw个体特征、身份关联、W、逐样本分数、checkpoint及服务器精确路径仍私有。保留原始source-only目录、五处旧CSV修改和历史负结果，独立analysis分支不合并main。不能声称统计天然匿名或隐私安全。

```text
status=BLOCKED_ROUTING_PARITY
neural_training_epochs=0
optimizer_steps=0
new_S0_training=0
new_test_predictions=0
new_test_feature_reads=0
new_test_image_reads=0
routing_mode=S0_POINTWISE_PROBE
routing_configuration_changed=true
checkpoint_tensor_parameters_changed=false
validation_adaptively_reused=true
independent_confirmation=false
full_GSR_reproduction=false
further_experiments_started=false
analytic_fits=18
encoder_forward_calls=474
new_train_val_feature_rows=0
core_val_metric_rows=18
core_val_per_class_rows=108
all_val_metric_rows=18
all_val_per_class_rows=108
S_candidate_scores_generated=0
bootstrap_scope=A-CB minus A-U only
```
