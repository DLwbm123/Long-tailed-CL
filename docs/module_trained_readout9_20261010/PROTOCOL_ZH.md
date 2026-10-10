# 第九轮：监督局部表征是否支持同一固定类别读出

第八轮投影真实触发，改善较差控制但仍未达原static_pc：HK BA−0.0957pp、ISIC new−2.2727。继续加强损失或相同保护不能解决当前证据。独立检验类别关系是否在推理中被忽略：局部先验的PPᵀ不随局部类列置换而改变，监督CE学到的局部类别对应仅间接进入原全局解析头。第二轮未监督表征+.25cosine曾给HK小正例并伤ISICnew；第六轮有局部监督但未附加局部分数。固定相同.25读出，测试监督是否改变其分类效用，不扫幅度或再训练。

## 新候选与严格贡献控制

完全复用第六轮local_supervised在NAS的全部stage model/global head/module_memory，不修改checkpoint、不重训adapter、不拟合投影或新头。用同一次已有encoder输出四区域局部p，按原四区域CLS条件pool/去区域均值/归一化，s_c=mean_r cosine(p_r, stored_center_cr)，预测xW+.25s。.25沿第二轮冻结，所有数据集/任务/类同规则；无val拟合、开关类、幅度搜索或新数据。推理显式使用类别关联，可能重复全局信息、损伤新类，不能称先验方法已被证明正确。

主比较6条：两数据集×原static_pc/第六轮supervised无读出控制/新trained_readout。仅两条新零更新评价轨迹，不重跑控制；原static_pc始终效果基线。另从已匿名公开第二轮cosine与transport取六指标增量，报告交互差分(new−supervised)−(oldcosine−transport)，不把不同轮次当独立确认，也不只报正交互。必须保留训练模块与读出的贡献对照，不能把组合的全部增益归给读出或监督。

## 零更新源检查与评价封存

两数据集各一次NAS真实源检查覆盖所有stage：source module必须local_supervised，seen顺序、每stage来源步数及最终ISIC468/HK176必须匹配；stored head/local centers维度与类数/计数匹配，所有局部中心有限。用原bank、区域局部先验和stored boundary_pair_weights重建global competition head，重建结果转storedfloat32后相对误差≤1e−8，解析残差≤1e−8；仅给出标量，不搬运或发布统计矩阵/中心/模型。检查中心cos分数尺寸/有限/范围≤1+1e−6。.25固定，无新拟合样本/训练更新。

两者READY、adapter/steps/policy均0后才EVALUATE。每stage未加读出的global_per_class_recall必须精确复现第六轮supervised对应stage，否则fail closed；协调器全部2COMPLETE并steps0/historical_global_control_equal=true才COMPLETE，仅类级等价，不证明逐样本等价。不同于第八轮，无训练梯度或投影要求，不套投影/首任务检查。保留源训练seed/split74002、shuffled order、task_sizes、原2epoch/batch64/.0003/FD10与所有主体，评价使用同冻结开发集，无test/reserved访问、RL/CIFAR/多标签/新数据或旧逐样本回放。

授权池GPU0/1/2，按ISIC→HK队列，启动检查选显存足够的GPU0/1。新adapter/策略更新0；旧468/176仅来源计数，不重复收费。每轮24小时截止，正常2sourcecheck+2evaluate共4成功子进程，所有失败额外计；历史134837.49139174307秒不重置，CPU与并行墙钟单列。源检查重建失败不放宽，不自动正式重跑，工程修复另清楚尝试/源/协议/成本。

## 目标与交付

原static_pc效果门不变：ISIC/HK各BA≥+1pp、tail≥−0.5pp、forgetting≤+1pp、old/new≥−1pp；两者同时通过才冻结候选并一次预注册两新train seed配对确认，确认阶段需要真正重新训练原base/同一监督+.25读出候选，不能把旧checkpoint评价当新seed。全部是反复开发证据，非独立患者/临床验证或论文复现。

完成后完整公开base/control/candidate六指标、所有pp增量/交互、31类n/recall和全部正负，报告读出RMS与相对同checkpoint global路径改变预测数，局部项激活不等于净效用。核对ISIC0/3与HK末新类8/3的保护，损伤消除与满足−1保护门明确区分。公开源码/协议/匿名报告/审计/成本，通过代理推GitHub核实远端SHA与精确commit匿名访问，NAS公开包/私有DELIVERY留存后才决定后续。

禁止公开身份/影像/权重/逐样本分数特征/均值统计中心原型边矩阵/私有配置或原始日志。若组合仍失败，不继续扫.25附近幅度、重复平方保护或任意选类；重新评估固定四区域、当前类中心和极少类支持边界，没有新可区分执行假设时报告限制并等待新方向，不能制造成功。
