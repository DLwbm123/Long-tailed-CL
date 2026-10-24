# R42 固定初始恒等适配器（冻结）
D41未训练TRAIN探针支持检验初始对齐。仅在第一次原update_task完成原随机初始化后，Image/Text首adapter的原512×512 fc置I、bias置0；保留原alpha、全部随机流消耗、参数数量与学习。后续新增adapter全部原随机初始化，不是R06增量置零，也不扫描初始化幅度/单侧/alpha。原虚拟parent也经Image_encode，初始parent数值可能变化，不能声称parent初始化数值与控制一致；固定parent保护规则保持。

完整三域45全新stage/44070新优化步，3seed74002/3/4，初始任务也新训，不用任何已学初始。与R28linear64同Biomed单位图文/exact split/order/stages6/3/6、10epoch batch64 AdamW.001、naturalshuffle、Gaussian原mean/cov+1e-5I/每旧类1行、线性Image/Text原逐模块求和、原树/textleaf/prompt/alpha/冻结、原pairedcontrast双向CE/entail/hierarchy/mapreg及全部系数、一次backward/原optimizer不变。保留原公式FP64entail修复、原TRAIN_cov nullspace/梯度及postAdam实际delta保护；D13扩维/学习仍暂缓。无R40梯度过滤/teacher/head/visualleaf/realmemory/replay纠正/新辅助项。原nativeNCM+hyper L2softmax固定.5融合保持。

仅当前stage到达读TRAIN缓存exactIDs，NIH x与text均按原Hunit ensure单位化。每域所有头训练封存restore后才读同encoder R16unitVALcache，exactIDs/unitnorm/无二次normalize。原finite/replay/旧adapter/父节点/actualdelta/restore/zeroVALseal全部保留。NCM每stage/seed逐类计数与R28linear exact。audit初始identity一次、无新增参数/额外随机消耗、后续原初始化；恢复构造也安装hook再strictload。D41零步收益不冒称训练效果。

BOTH采用门：对历史H0与prior同GPU同精度R28linear64均NIH阶段avg/final各>=+1pp，tailTRAIN<250>=+2.857142857pp，至少2/3seed final/tail联合改善；ISIC/HKfinal/tailmean各最多降1pp。Fraction逐类计数联合改善；不择域/参照或事后改门。R28是先前参照不是本次同步control，不冒称位级H0复现。失败拒绝固定初始identity，不扫描矩阵/幅度/单侧/alpha/初始化阶段/LR/epoch/seed/融合。

独立2h/4000000000bytes含128MiB余量，从preflight首次实际起，含所有重试恢复。CPU36000/RSS8GiB/file1GiB/GPUreserved<8GiB/CPU4；预计<2.5GB。pro/data ext4容量/write-readprobe与显存先核、独立后台guard、中性主子完整argv。R23assets只读共享不复制，safe refs小副本计预算；不清理旧依赖。完整stage恢复/跳完整域，不盲重训或重置clock。完成45stage/exit0后仅保存结果分析135metrics与INITIAL_IDENTITY_VERIFICATION、GATE、全轨迹/分支计数，完整正负结果交付。

全部反复历史开发集，无TEST/独立确认/临床结论；保护历史和暂停comparator/Spatial约束。
