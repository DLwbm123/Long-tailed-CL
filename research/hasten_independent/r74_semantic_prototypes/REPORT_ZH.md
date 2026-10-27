# R74：HK74002 冻结文本伪样本原型的完整负结果

R67-R73跨实验审查未发现稳定采样GRPO增益后，本轮改检验一个结构假设：同一BiomedCLIP冻结文本特征能否补充少样本视觉原型。复用既有模型资产、标准类别名称与原固定提示模板，无下载/新SFT/提示搜索。四个分支完整六stage退出0，所有24head封存后才读取真实HKVAL。

语义原型=normalize(n_support*normalizedvisualmean+normalizedfrozentext)，固定1个文本伪样本，不搜索混合系数。TRAIN及ref用相反fold实际支持数，最终推理用全已到达TRAIN类数。原视觉图像编码器prefix/LoRA架构不变，文本编码器冻结，随stage仅编码已到达类别。它是启发式已知shrinkage形式，非最优贝叶斯估计或临床语义保证。额外semantic_sft_reference完全不训练，只给原SFT原型加相同文本伪样本，用于区分readout与GRPO收益。

| 方法，百分比 | 阶段平均BA | 最终BA | 最终尾类BA |
|---|---:|---:|---:|
| H0 |55.630100|56.454053|36.857416|
|sft_reference|66.840041|64.876782|44.924938|
|semantic_sft_reference|67.182120|63.974439|42.831274|
|visual_grpo|66.115954|63.655574|44.737644|
|semantic_grpo|65.870120|63.759645|43.464787|
|visual_exact|65.456019|63.954743|44.161723|
|semantic_exact|65.534434|62.630948|41.207178|

预先指定semantic_grpo对visual_grpo平均-0.245834、最终+0.104071、尾类-1.272857个百分点；对原SFT-0.969921/-1.117137/-1.460151个百分点，未形成整体增益。对semanticSFT平均-1.312000、最终-0.214795、尾类+0.633513个百分点，仍非同时获益；不能只报尾类恢复或相对H0较高。

仅加入文本伪样本、不做GRPO的semanticSFT对原SFT平均+0.342079、最终-0.902343、尾类-2.093664个百分点。文本readout本身在最终/尾类有负作用，不能把其后GRPO局部尾类恢复包装为优于原监督学习。semantic_exact对visual_exact平均+0.078415、最终-1.323794、尾类-2.954545个百分点，亦不支持整体改善。两个视觉控制重现历史同seed指标，不是新seed复现。

| 分支 | 实测TRAIN平均KL | 零优势组率 | 采样正确动作率 | clip比例 |
|---|---:|---:|---:|---:|
|semantic_exact|0.019378|不适用：精确积分|不适用：精确积分|不适用：精确积分|
|semantic_grpo|0.035829|74.924%|89.749%|5.171%|
|visual_exact|0.019433|不适用：精确积分|不适用：精确积分|不适用：精确积分|
|visual_grpo|0.046914|74.758%|89.572%|5.390%|

逐stage私有文本向量512维/有效、文本参数冻结/已到达类别数量、一个伪样本、完整head视觉原型与融合原型公式均核，干预生效；不是文本管线没有接通。文本与视觉类别结构是否不匹配是可能解释而非已证唯一原因，不能凭结果改提示词、类别别名或伪样本数量救援。极少类相似grade语义本身也不构成已证分类质量。

saved-only核验通过944optimizer/3696micro backward/24stagearmbranch/72curve/**36真实VAL读数**（24新+6原SFT+6semanticSFT）。额外6semanticSFT图像评测与按stage冻结文本编码成本公开，无额外optimizer；全部readout在新头封存后，结案verify/analyze0新增模型评分/优化。原参数冻结49152LoRA、strictrestore/fold、固定TARS表/oldref/未缩放KL/raw推理/真实VAL分母/seal/cleanup均核。

原钟2375.848383秒（39.60分钟），CPU3310.797047秒，workerRSS峰7.500GB，保守聚合峰7.522GB，新增实际磁盘峰59.027MB，包含新增私有文本向量等。仅A1000/1，GPU1冻结文本的内存计入资源guard；总RAM<30GB。root/remote prefix交换实际0，losslessFP32parity最大2.2352e-7，域末声明再生prefix已清，所有学习产物保留。共享783705670B原模型权重只读无新副本，不恢复停止R65；历史SFT1180/4620不计新增。

本次单HK单seed反复历史开发、少数VAL类分母可1或2，不能三域BOTH/三seed/TEST/独立患者或临床确认。保留阶段平均局部正信号与最终/尾类负结果，不换主候选/挑域seed/改旧门/只报较弱控制。R67-R74目前仍没有稳定采样GRPO额外收益，后续需要重新判断主线表示学习与类别风险，而非承诺再调一项即可让GRPO获胜。

公开自有source/protocol/跨实验review、全stage-class计数/曲线/文本管线元数据/诊断/费用/正负报告；原图/身份/逐例数组/token/文本及图像embedding/features/权重/原日志/私有配置均不公开。
