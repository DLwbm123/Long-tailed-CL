# R40：原几何梯度服从原配对识别方向

R38固定文本正集合目标失败，停止其正例温度/加权/子集/对称组合/系数扫描。D39在18个封存TRAIN固定probe中，NIH三个seed均至少一个初始/最终cell投影后几何与contrast cos<−.01、几何norm>=.01contrast，事前可行性通过。但NIH几何norm仅原contrast约2.6%–12.6%，冲突不证明主要瓶颈或效果。R40据此测试原目标内部的梯度优先关系，不再新增teacher/辅助分类目标，也不改loss系数。

恢复原H0全部pairedcontrast（双向单对角）、.1anchor+.5hierarchy(entail/distance/root)+.1mapreg标量目标与一次原backward。backward前用autograd.grad保留原contrast参数梯度g_c（无.grad累积）；stage>0按原mapping nullspace算法投影其mappingweight分量。原backward和原mapping gradient projection后得到g_total，定义原加权几何梯度g_g=g_total−g_c。仅当dot(g_g,g_c)<0且||g_c||²>1e−24时，令g_g'=g_g−dot(g_g,g_c)/||g_c||²*g_c，使用g_c+g_g'更新；否则不重写.grad，原梯度逐元素保留。原None梯度仍None，防止对未用fc均值误启AdamW decay。阈值1e−24只是零梯度数值边界，非搜索。

所有比较在原已允许的参数方向内完成，不改变/扩大/学习mapping保护basis或rank，D13 defer仍保留；原postAdam实际delta投影继续执行。验证映射梯度相对再投影残差<=1e−4，几何与contrast余弦>=−1e−4；几何残差norm<=1e−6*max(original geometry norm,contrast norm)时余弦不可靠记0并计数。原负向几何分量被移除，而非限制全几何norm为原梯度10%；没有R19..R23辅助teacher的10%cap或新loss。此规则不保证Adam参数步长、准确率或泛化不受损。原标量loss日志不变，但更新已不是该标量的普通梯度；必须明确这一点。

所有45stage从初始任务全新训练、44070新optimizersteps，额外每batch一次contrast autograd.grad成本披露，不称与H0等计算。三域exactsplit/order/stages6/3/6、seeds74002/3/4、Biomedunit图文、原Image/Text线性与所有adapter求和（无R37路由）、10epochbatch64AdamW.001、naturalshuffle/Gaussian原mean/cov+1e−5I/每旧类1行、原树/textleaf/prompt/parent/alpha/冻结、原loss所有coeff及原mapping保护/actualdelta保持。FP64entail同公式同常数修复保留。无GELU/ridge/head/teacher/realmemory/边界纠正/centroid/pooled-positive/fusion标定，原nativeNCM+hyper L2softmax.5推理不变。不复用旧已学初始。

每stage到达才读TRAINcache exactIDs，不提前futurefeatures；每dataset全部候选TRAIN头封存strictrestore后才用sameencoderR16unitVALcache，exactIDs/unitnorm/no二次normalize，无图像/骨干复制或下载/TEST/中间选择。原finite/replay/oldadapter/fixedparents/actualdelta/restore/seal检查保持；NCM各stage/seedcounts对R28linear exact，各stage steps相同。

audit geometry_priority prepare_calls==calls==steps、conflict_batches/zero_contrast_batches/tiny_geometry_residual_batches、原几何/contrast最小余弦before/after、移除geometry分量norm占原geometry比例max<=1.0001、mapping残差max<=1e−4。首prepare前后CPU/CUDA RNG exact；额外autograd不改变采样。已对pinned实际网络测试已知反向梯度过滤、同向梯度exact保留、零contrast/原None保留、RNG/原空间和实际postAdam投影；NIHISIC展开runner预检。合成与固定probe不是自然实际轨迹或效果证据。

采用BOTH门不变：对历史H0和R28linear64均NIHavg/final各>=+1pp、tailTRAIN<250>=+2.857142857pp、>=2/3seed final/tail联合改善；ISIC/HKfinal/tail均值各最多降1pp，joint Fraction exact。不择域/参照/放宽门。失败拒绝固定原几何优先规则，不扫冲突容差/优先方向/组件组合/losscoeff/梯度cap/seed/融合。

独立2h从preflight起含修复恢复，4GB含128MiB余量，预计<2.5GB，CPU36000继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro/data ext4容量/write-readprobe/GPUfree前置，旧assets只读共享不复制/不清理，R28安全参考小副本计预算。中性主子完整argv/GPUprocess名和独立guard保持，不依赖SSH/心跳。三域exit0全齐后仅savedresponses统一分析135checks/GATE/GEOMETRY_PRIORITY_VERIFICATION/TAIL_TRAJECTORIES/BRANCH_TRANSITIONS/安全聚合/完整中文报告。全为历史开发集反复使用，无TEST/独立确认/临床证据，曝光/跨seed预测非独立患者；不公开图像/身份/features/逐例/权重/原日志/私有路径。
