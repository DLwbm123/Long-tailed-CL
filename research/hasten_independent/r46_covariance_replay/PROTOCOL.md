# R46 有限样本协方差收缩回放（冻结）
目的：D45固定TRAIN内留出预测密度支持检验少样本协方差估计改进；不是准确率或遗忘因果证据。仅改旧类Gaussian回放使用的协方差，不改NCM、类均值、数据顺序或融合。

每增量阶段首次回放时，使用已有旧类原始S_c=self.cov[c]-1e-5I等权均值P。仅使用该候选此前已到达类的原TRAIN统计；当前新类尚未完成统计不纳入P，不读futurefeatures或旧真实features。C_c=(1-lambda_c)*S_c+lambda_c*P+1e-5I，lambda_c=512/(512+max(n_c-1,0))，n_c来自已到达TRAINcache exactIDs的原类计数。没有拟合/扫描收缩强度、目标或正则。每阶段固定derived covariance，无额外RNG；原每旧类每batch1行Gaussian、原随机调用顺序/数量/shape和旧类shuffle保持，只改变采样分布。原Gaussian输入不归一化、不回拉边界、不存exemplar。P可能模糊真实类间异方差，均值估计误差未解决；密度改善不保证判别收益。

原prototype/self.cov保存为未收缩原始统计，mean/cov精确不变；不把派生cov写回，避免递归收缩。派生cov仅RAM重建、每stage最多约C*512*512*8bytes（23类约48MB）额外内存，不复制原模型/特征。原TRAIN_cov nullspace及postAdam actualdelta全部按原统计保持，D13扩维/学习仍暂缓。原先stage保存恢复可从原cov+已到达counts确定性重建派生cov，无旧真实特征重读拟合。

事前共享R28linear64三域三seed9份stage0 checkpoint与TRAIN receipt：初始没有旧类回放，模型训练完全相同；移除旧VAL metrics、不搬responses，恢复核验不重训初始。31950初始步仅共享状态，36新增量stage12120新optimizer steps，45评测135branch checks，不称45全新训练。R28是prior同GPU/精度控制，非本次同步control、非旧mygpu位级复现。

原Biomed单位图文、exact三域split/order/stages6/3/6、seed74002/3/4、10epochbatch64AdamW.001、linear Image/Text全adapter求和与原随机初始化/alpha/freezing、原树/textleaf/prompt/parent、pairedcontrast双向原对角CE/entail/hierarchy/mapreg所有loss/coeff/一次backward/optimizer保持。原公式FP64entail修复不改常数。无teacher/ridge/head/identity/GELU/gradientpriority/新辅助目标；NCM+hyper L2softmax固定.5融合保持。

TRAINcache只在stage到达读取exactIDs，NIH原ensure对x和text都单位化。每域所有候选头TRAIN seal/strictrestore后读sameencoder R16unitVALcache，exactIDs/unitnorm/no二次normalize，无原图/骨干下载/复制/TEST/中途选择。共享初始三分支counts须与R28exact，NCM所有stage/seed counts与R28exact，原steps、finite/replay/旧adapter/固定parent/actualdelta/restore/seal全部保留。

新audit covariance_shrinkage：每增量stage calls==steps，examples==old*steps，byclass[c]==steps，preparations1/pooled_classes==old，lambda与n_c公式exact，regularizer1e-5，rng_first_prepare_preserved/raw_covariance_unchanged/means_unchanged为真。合成CPU/GPU验证SPD/固定公式/小样本权重较大/无RNG/原stats未变、MVN抽样RNG消耗同shape，pinned NIH/ISIC runner预检。不是科学成功。

独立2h/4000000000bytes含128MiB余量，CPU36000继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB/CPU4，预计<2.5GB含9初始副本；首次preflight原钟含全部修复/retry不重置，独立guard和中性主子完整argv，pro/data ext4容量write-readprobe/GPUfree先核。旧R23assets只读共享，不清理历史依赖。

BOTH采用门不变：对历史H0及R28linear64均NIHavg/final各>=+1pp、tailTRAIN<250>=+2.857142857pp、>=2/3seed final/tail同时改善；ISIC/HKfinal/tailmean各最多降1pp，joint Fractionexact，不择域参照不改门。失败拒绝此固定样本数/维度收缩配置，不扫lambda/pooled权重/正则/对角-full选项/采样数量/seed/融合。完整stage恢复/跳完整域不盲重训/重评分。exit0全齐后仅保存结果运行analyze.py，135checks/GATE/COVARIANCE_REPLAY_VERIFICATION/TAIL_TRAJECTORIES/BRANCH_TRANSITIONS/全class/seed结果中文报告公开交付。

所有反复历史开发集，无TEST/独立确认/临床结论；D45划分不是患者独立，训练曝光/seed非新增患者。
