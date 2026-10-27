# R77：完整经验支持分布的平均正信号、最终/尾类代价与后训练小增量

本轮完整比较每类单个单位化均值与全部真实TRAIN支持的角核评分。原型score=20*z·normalize(mean f_c)；分布score=logsumexp_i(20*z·f_ci)-log(n_support_c)。固定原scale20，不搜索带宽/nearest-k/温度/混合。TRAIN全分支仍加原fullTRAINlogcount，最终推理不加该先验，各用自身raw评分公式；分布raw不是单cosine。它是已知经验角核classifier，不是新RL创新或医学密度校准。

D76类内离散/均值长度说明单位化单原型舍弃信息，不证明该信息一定有益或真实多模态。R59同类核评分在冻结特征+512共享角度参数/双边梯度/均衡CE下完整负结果保留；本轮区分为原生49152LoRA+相同成熟R67SFT起点、原GRPO/same-targetexact与oppositefold detached交替支持，不是R59带宽/秩/强度救援。

四分支均HK74002六stage完成，每stage从同SFT独立开始、不carry后训练delta。原binaryG8/popstd+1e-8/clip.2/TARSgamma.75固定SFTprototypeTRAIN错误表/KL.04不乘权重/anchor.01/2epochAdamW.001wd.001cosine2/batch256micro64和自然query顺序均保持。没有新SFT、R75macro权重、inverse、guideprojection、hardreward或text融合。新增distribution_sft_reference仅替换原SFT读出，0optimizer；由已编码两fold的SFT参考支持拼接全history，不增加TRAIN图像forward。

|方法，百分比|阶段平均BA|最终BA|最终尾类BA|
|---|---:|---:|---:|
|H0|55.630100|56.454053|36.857416|
|sft_reference|66.840041|64.876782|44.924938|
|distribution_sft_reference|69.291843|62.153597|41.216849|
|prototype_grpo|66.115954|63.655574|44.737644|
|distribution_grpo|69.458984|62.378647|41.736497|
|prototype_exact|65.456019|63.954743|44.161723|
|distribution_exact|70.019748|62.700126|42.617416|

预先指定distribution_grpo相对原prototype_grpo差值平均/最终/尾类 +3.343030/-1.276927/-3.001147pp；相对原SFT +2.618943/-2.498135/-3.188441pp。阶段平均改善，但最终/尾类下降，未形成相对原SFT的整体收益；不以H0差距或平均单项代替旧门。

零后训练distributionSFT单独相对原SFT +2.451802/-2.723185/-3.708090pp。因此固定SFT读出变化本身已有平均+2.451802pp，同时最终/尾类损失；不能将主候选平均+2.618943pp全部归于GRPO。上述是明确路径的条件差值分解，不是通用因果贡献比例。

主候选相对相同分布式读出SFT的平均/最终/尾类 +0.167141/+0.225051/+0.519648pp；这是单seed下三项同向的小后训练增量，需保留，不能忽略。distribution_exact对相同读出SFT +0.727905/+0.546529/+1.400567pp，也同向；但对原SFT +3.179707/-2.176656/-2.307522pp，仍有最终/尾类代价。exact不是预定采样主，不后改主候选或宣称三seed确认。两prototype控制重现历史同seed指标，不是新增独立复现。

最终尾类的主候选相对distributionSFT净+0.519648pp由大幅正负抵消构成：VAL2的class12少正确1个，贡献-4.545455pp；VAL7的class15多正确3个，贡献+3.896104pp；VAL21的class3多2个贡献+0.865801，VAL8的class13多1个+1.136364，VAL33的class18多1个+0.275482，VAL41的class9少5个-1.108647，其余尾类不变。不能把这个净值描述成广泛、稳定或临床尾类改善。完整class-stage正确数/分母/读出与RL差值在CLASS_STAGE_DECOMPOSITION.json；没有删小分母类或改门。

|分支|真实TRAIN meanKL|零优势组率|采样正确动作率|clip比例|
|---|---:|---:|---:|---:|
|prototype_grpo|0.046914|74.7578%|89.5722%|5.3905%|
|distribution_grpo|0.018925|65.9359%|86.4719%|4.9111%|
|prototype_exact|0.019433|不适用：精确积分|不适用：精确积分|不适用：精确积分|
|distribution_exact|0.015572|不适用：精确积分|不适用：精确积分|不适用：精确积分|

所有分支query平均固定权重1.426089177，分布分支没有重新拟合TARS错误表。分布采样零优势组率65.9359%低于原74.7578%、KL0.018925低于0.046914，但reference评分映射也改变，不能将KL下降当作同一参数距离、更稳定泛化或唯一因果。exact无随机组，不能把其零采样计数当零优势率0%。

完整saved-only核验944optimizer/3696micro backward/24stage-arm分支/72curve/**36真实VAL读数**(24新+6原SFT+6distributionSFT)，所有24head封存后才任何VAL。后两种控制均用同原SFT图像encoder，额外6VALforward和核评分额外计算如实计费。结案verify→analyze及saved_summary0新增模型评分/优化；fullsupport FP32单位范数/labels/counts/重建prototypes/SFTsupport parity、scoremode、freeze49152/strictrestore/fold/oldref/TARS/未缩放KL/封存/分母/cleanup均核。原始manifest全TRAIN固定元数据可读，但实际编码/支持/加载head仅已到达类，不读取未来类图像/features/head用于训练。

新增持久科学记忆不能忽略：24个头文件合计246004216B，6份SFTsupport文件60304592B，最大头14272725B。最大6821x512 FP32支持向量本身13,969,408B，原23个centroid向量仅47,104B；新kernel推理每query最多对6821支持评分，不是23centroid。prototype控制头也保存了用于审计的supports，但其读出只用centroid。不能称与原HASTEN等记忆，或由相同步数推断等FLOPs。全部支持特征/privateheads保留为source科学产物，不公开、不按cache清除。

控制器原钟2727.902920s（45.47min），worker及后代CPU3630.493491s，workerRSS峰7669895168B、独立采样group+controller峰7654436864B，新增实际磁盘峰657658700B。各RSS采样时刻不同；run费用不冒称后续全部分析/发布CPU。原20h时钟未重置，仅A1000/1，总RAM<30GB/GPUreserved各<12GiB。rootcap0/noOSswap，root/remote prefixspill实际均0，FP32prefix8519rows parity最大2.23517418e-07；结束只清声明prefix，全部学习特征/权重/历史和CLS保留。

本次是反复开发的单HK单seed探索，不称原三域BOTH/3seed/TEST/独立患者或临床确认。相对kernelSFT的小同向增量值得记录，但原SFT最终/尾类仍更高，不能宣称主线整体成功或忽略推理/记忆代价。不根据该结果扫kernel bandwidth/nearest-k/权重cap/LR/G/epoch/融合强度，不挑最好arm/域/seed或修改旧门。后续需结合完整曲线、class分母和D76证据形成另一个可证伪问题，不承诺GRPO必须赢。

公开自有source/protocol/全classseed聚合、曲线、诊断、存储/费用与全部正负报告。原图、身份、逐例数组、support/embedding/features、模型权重、原日志、私有配置路径和第三方全文均不公开。
