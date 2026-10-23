# R31：上一阶段功能软预测锚定（独立主线实验）

R30原型边界修正使NIH尾提高，但总体门与跨域约束失败，停止该机制，不扫描其强度或mean替代。R31恢复原Gaussian输入，检验旧模型功能保持，而非继续修正生成样本。

每个增量阶段，在候选自身上一阶段网络尚未update_fc/update_task之前，以原fc.weight[:old].detach()旧类TRAIN原始均值逐类一行，旧类文本走原提示/冻结骨干缓存和上一阶段Image/Text adapters+learned mapping，计算原exp(logit_scale)*负双曲距离。教师q为旧类softmax，固定detached保存该stage；无旧网络副本、额外拟合或外部ridge。准备前后network state和CPU/CUDA RNG精确不变。当前stage已到达的文本bank中仅索引旧类文本，不用未来类别/特征。

每个原自然batch/Gaussian回放更新前，旧类均值再走当前网络，配合本次原forward的全部已见类text_feas，计算原scale*负双曲距离。教师q在新类列补零；固定增加KL(q_old_padded || student_all_seen)，reduction=batchmean，等价逐旧类均值等权，系数1、无温度拟合、不扫描权重。原所有pairedcontrast/entail/hierarchy/mapreg目标和系数保留，原backward一次、原optimizer一步；没有R19–R23的10%辅助梯度截断或冲突过滤。新增项是标量loss，日志total含此项；base_loss_last另存原目标，避免混淆。

它与R19固定ridge类别判别教师、R23全类均值硬标签CE不同：此处上一阶段候选自身soft预测，只有旧类均值作行锚点，针对旧类功能与新类竞争；可能保留旧模型错误，不能当oracle。每stage教师必须来自当时候选旧模型，禁止从R28未来stage取teacher。旧均值能丢失多峰信息，本实验不证明是独立图像记忆。

三域exactsplit/order/stages6/3/6、seeds74002/3/4、Biomed单位图文、原线性Image/Text/树/提示/初始化/冻结/alpha、10epochbatch64AdamW.001、naturalshuffle、原Gaussian原mean/cov+1e-5I与每旧类1样本、原nullspace及postAdam实际delta保护、原native NCM+hyper L2softmax .5融合均不变。两类entailment仍使用R28同公式FP64修复，不改epsilon/系数。无GELU/ridge/head/回放修正/额外optimizer，projection扩维或学习继续defer。

事前沿用共享R28linear64的9份stage0 TRAIN checkpoint/receipt，去除旧VAL metrics、不搬响应；恢复校验，初始任务不重训，已有31950初始步不算新优化。36个增量stage、12120新optimizersteps，但每步多旧均值前向与KL，额外计算披露。R28linear64为只读共享初始/同硬件精度匹配对照；不重训或优化对照。每dataset所有候选TRAIN头封存/restore后才用sameencoderR16unitVALcache，exactIDs/unitnorm/no二次normalize；当前stage到达才读对应TRAINcache。无图像/骨干复制/下载/TEST/中间选择，所有45评测记录、135metric全类seed完整保留。

原finite/replay/oldadapter/fixedparents/actualdelta/strictrestore/zeroVALseal沿用；新functional_anchor audit calls==steps、examples==old*steps、coefficient1、teacher_detached/state_unchanged/preparation_rng_preserved、首batch有效非零且finite参数梯度证明、teacherentropy/maxprob、loss_last/max/base_loss_last。教师q固定不更新，旧均值detached。共享初始三分支counts与R28一致、全部NCMcounts与R28一致、步骤与原控制同。

采用门BOTH：对历史H0和R28linear64均NIHavg/final各>=+1pp、tailTRAIN<250>=+2.857142857pp、至少2/3seed final与tail同时改善；ISIC/HKfinal/tail均值各最多降1pp。Fraction类级计数判断联合改善。失败拒绝本固定功能锚定，不扫温度/系数/硬软目标/anchor数/seed/融合，不择域或放宽门。工程可达/教师immutable不等于科学成功，保留全部正负结果。

独立2h从首次preflight起含修复/恢复共享原钟；4GB含128MiB余量，预计新增<2.5GB（含9份共享初始副本），CPU36000s继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro5000/data ext4、GPU余量/挂载容量write-readprobe、中性主子完整argv、独立guard保持；R23assets/official/TRAIN/VAL只读共享不复制，不清理任何旧依赖。所有为反复历史开发集，无TEST/独立确认/临床证据。结束仅保存结果分析后公开自有源码/协议/安全类级全正负结果；不公开身份/图像/逐例响应/features/权重/原日志/私有配置路径。
