# R38：只改文本→图像的同类正样本集合

R37出生任务文本路由完整两门失败，停止该固定路由，不扫混合/adapter范围/冻结时机。原标准训练按实例配对，batch里同类提示文本完全相同，但文本→图像CE仍只把对角配对图像作为正例。R38检验另一监督假设：相同类别提示的任一同类图像均可构成正例。不是已证明的代码bug或唯一失败原因。区别R14只对图像→文本分母作类重复频次修正（其负结果保留），也不同于R08双向全已见类CE；不重启它们或扫描修正强度。

唯一变化：设原scaled text_logits为Z，行i文本类别y_i、列j图像类别y_j，将原文本→图像单对角CE改为mean_i[logsumexp_j Z_ij − logsumexp_{j:y_j=y_i} Z_ij]。图像→文本原paired CE逐字保留，原双向平均0.5和其余loss/coeff保持，仍原一次backward/optimizerstep。正负候选仅原batch真实行+原Gaussian回放，不新增all-seen负池/采样或类频次平衡；重复文本anchor行仍按原batch自然频次平均，不折叠为均匀类。

这是正集合概率质量目标，可能只顾较易同类图像而忽略难正例，不能保证覆盖多峰/改善尾类。同一batch全为同类时新文本→图像项恰0且梯度0，图像→文本无类区分信息，仍有原entail/hierarchy/mapreg；必须记录这种batch而不删除它们。不能将较低loss当性能改善。unique-label batch与原CE值/梯度一致；重复类时梯度一般不同，不是简单减log(m)常数。总loss日志含替换后项，original_text_ce_last仅detached参考，不向optimizer使用。

必须从初始任务全新训练，45newstage/44070newoptimizersteps，原三域exactsplit/order/stagesNIH6 ISIC3 HK6、seeds74002/3/4；不复用R28/R37已学初始状态。Biomedunit图文、原Image/Text线性且全部适配器求和（不带R37文本路由）、10epochbatch64AdamW.001、naturalshuffle/Gaussian原mean/cov+1e-5I/每旧类1行、原entail/hierarchy/mapreg及原coeff/树/prompt/parent/alpha/freezing、原TRAIN_cov nullspace与postAdam actualdelta保持。原公式FP64entail修复不改epsilon，projection扩维/学习仍defer。无GELU/teacher/ridge/head/centroid/10%guard/realmemory/边界纠正；原nativeNCM+hyper L2softmax.5推理保持，不调融合/温度。

TRAINcache每stage到达exactIDs，不提前未来features；每dataset全部候选头训练封存strictrestore后才用sameencoderR16unitVALcache，exactIDs/unitnorm/no二次normalize，无图像/骨干复制/下载/TEST/中间选择。原finite/replay/oldadapter/fixedparents/actualdelta/restore/seal保留；NCM各stage/seedcorrect/count对R28linear exact，各stage步数相同。

新增pooled_text_positive audit：calls==steps、rows==sum(seen_by_class)、positive_entries>=rows、duplicate_anchor_rows<=rows、all_positive_batches保留；old class seen_by_class[c]==steps（原每旧类1行分母不变）。保留original_text_ce_last与pooled_text_ce_last。合成CPU/CUDA检验unique-label loss/gradient exact parity、真实重复文本anchor结构下新旧梯度不同且finite、联合置换不变、全正batch loss/grad为0；实际pinned源码修改点唯一、NIHISIC展开runner编译。预检只证明实现，不证明科学效果。

采用BOTH门不变：对历史H0和R28linear64均NIHavg/final各>=+1pp、tailTRAIN<250>=+2.857142857pp、>=2/3seed final/tail联合改善；ISIC/HKfinal/tail均值各最多降1pp，joint Fraction exact。不择域/参照或放宽门。失败拒绝固定文本正集合目标，不扫pooling温度/正例加权/正例子集/对称组合/系数/seed/融合。

独立2h从preflight起含全部修复恢复，4GB含128MiB余量，预计<2.5GB，CPU36000继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro/data ext4容量/write-readprobe/GPUfree前置，旧R23assets/official/TRAIN/VAL只读共享不复制/不清理，R28安全参考小副本计预算。中性主子完整argv/GPUprocess名核验、独立guard不依赖SSH/心跳。全部三域exit0后仅savedresponses分析135branchchecks/GATE/POSITIVE_POOL_VERIFICATION/TAIL_TRAJECTORIES/BRANCH_TRANSITIONS/安全aggregate及中文全正负报告。全为反复历史开发集，无TEST/独立确认/临床证据；曝光/跨seed预测非独立患者。发布自有代码/协议/安全类级正负结果，不发布图像/身份/features/权重/逐例响应/原日志/私有路径。
