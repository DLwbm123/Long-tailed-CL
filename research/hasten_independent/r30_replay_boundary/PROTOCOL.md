# R30：原型边界一致的Gaussian回放，独立增量主线实验

## 依据与固定机制

R28统一FP64entailment后GELU完整训练但不通过主门；停止激活路线。D29固定TRAIN统计审查中，Gaussian生成类与当阶段原NCM全已到达类边界不一致：NIH6421/8960=71.66%、ISIC549/1280=42.89%、HK3061/10880=28.13%，三域及总触发量通过事前可行性规则。它是生成样本的构造标签一致性，不是实际图像准确率或失败因果证明。

R30保留R28 linear64的原线性结构与相同FP64entailment修复；只在原每旧类Gaussian的一次抽样后、输入Image_encode之前修正越界向量。不增减回放数量、不重新抽样、无额外RNG、不改变真实TRAIN输入。用原network.fc.weight[:C].detach()当前已到达全部类原始均值（原replace_fc在train前计算）定义native cosine边界。已属生成类的样本逐元素不变；越界g沿g→m_c线段解析求进入本类所有Voronoi半空间的最小t，公式见replay.py与D29，无优化/强度/阈值扫描。FP64求t后一次FP32 nextafter朝1取内点；最终同batch native评分仍非生成类时退回m_c，完整保留fallback计数。不是全空间最短欧式投影。

D29约48%的越界样本因边界舍入退到均值，约占全部样本23%；可能削弱回放多样性，这是固定方法的风险，不静默忽略。D29首轮HK因子批次与整批次FP32矩阵乘法边界舍入不同触发保护，原钟内只补整批最终核验/同一既定mean fallback，保留失败；不改数学半空间或扫描阈值。D29已完成NIH/ISIC复用，HK仅未封存域重放。D29 alpha字段是fallback前入口t，fallback实际t=1；R30 audit明确记录有效t，fallback记1。

## 共享初始阶段与配对控制

新协议事前明确复用R28 linear64各域各seed的9份已封存stage0 checkpoint和TRAIN回执（复制计入新预算，移除旧VAL metrics）；候选先恢复校验，无stage0优化。这是增量机制的共享初始状态设计，不冒称45个新训练stage，不复用GELU或R25。每域/seed后续原阶段种子设置保持；实际新训36个增量stage、12,120 optimizersteps，另外31,950初始步只作为已有共享状态。R28 linear64完整轨迹为同GPU/精度/共享初始参照，不重训对照或旧初始任务。

全部三域NIH6/ISIC3/HK6、seeds74002/3/4、原Biomed单位图文、exactsplit/order、原10epochbatch64AdamW.001、自然shuffle、Gaussian原mean/cov+1e-5I/每旧类1样本、树/模板/全部pairedcontrast/entail/hierarchy/mapreg损失与系数/原backward、冻结/原TRAIN_cov nullspace及postAdam实际delta保护不变。projection扩维/学习继续defer。无teacher/ridge/head/centroid辅助监督，无GELU，无融合参数调整。原NCM+hyper L2→softmax→.5/.5推理不变。

候选每dataset全部TRAIN头封存/restore后才用已有sameencoderR16 unit VALcache；exactIDs/unitnorm、不二次normalize。当前stage到达才读取对应TRAINcache，不提前未来features。物理搬入共享初始checkpoint不等于未来类别读取，stage0仅含初始已见类。共享stage0回执移除旧VAL metrics，候选末尾统一重新恢复评估；必须与R28初始三分支逐类计数一致。所有45评测记录、135分支检查，36新/9共享训练记录如实区分。历史开发集反复使用，无TEST/独立确认或临床证据。

## 约束与采用门

原finite/replay/oldadapter/fixedparents/actualdelta/strictrestore/zeroVALseal检查。新replay_boundary audit每增量stage calls==examples==old_classes*steps，by_class[c]==steps，offclass+unchanged==examples，0<=fallback<=offclass；第一次修正调用CPU/CUDA RNG状态相同，修正函数无RNG调用/重抽样。记录有效alpha、最大相对位移、fallback和全部类分母。NCMcorrect/count逐stage/seed与R28 linear64一致，步数相同；初始stage三分支correct/count同控制一致。

采用必须同时对历史H0和只读R28 linear64通过原主门：NIH阶段平均/最终各至少+1pp、TRAIN<250尾至少+2.857142857pp、至少2/3seed最终与尾同时改善；ISIC/HK最终总体与尾均值各最多降1pp。Fraction类级计数判断联合改善。失败拒绝本固定回放边界机制，不扫回拉强度/边界margin/替代mean/采样次数/seed/融合，不事后改门。过程审计或生成类一致不当效果成功，全部正负结果保留。

独立2h从首次preflight起，所有修复恢复共享原钟；4,000,000,000bytes含128MiB余量、预计新增<2.5GB（含9份初始checkpoint副本与小型参考聚合），CPU36000s继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro5000本地/data ext4，旧R23assets/official/TRAIN/VAL只读共享，不复制大骨干/原图、不清理旧资源；中性主子argv/独立guard/实际挂载容量write-readprobe/显存检查保留。结束仅在保存结果上分析并发布自有代码/协议/安全类级全正负结果；不公开身份/图像/逐例响应/特征/权重/原日志/私有配置。
