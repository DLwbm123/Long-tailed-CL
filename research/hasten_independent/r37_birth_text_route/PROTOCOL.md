# R37：类别出生任务固定文本适配器路由

R35固定8真实特征循环回放完整三域失败，不扫memory数/选择/混合。D34完整旧类TRAIN显示old-only识别提高但加入新类有竞争；D36只读36边界495旧类提示确认，旧Text模块参数/输出匹配时，原新增Text模块仍使NIH旧提示欧氏输出相对变化0.177–0.817；屏蔽新模块可切断其旧提示梯度。它不是原自然batch损失梯度或hyper分类因果证明，不能预言本轮成功。

固定单一结构变化：已知类别c的文本提示，只累加在该类别出生task b(c)之前及当时的Text_Adapter[0..b(c)]，不再让后来新增Text适配器作用于旧类提示。birth=0若c<init_cls，否则1+floor((c-init_cls)/increment)，沿冻结原class order/task划分。全已见文本bank、原paired batch重复类提示和推理均显式传入类别索引；这是已知候选类的元信息，不用图像真值/TEST决定路由。相同出生task分组后保持相同累加顺序。原Text模块参数/alpha/随机初始化/扩展/冻结规则不改，当前新Text模块仍训练当前新类提示；没有新增参数、标量门、teacher、记忆或额外optimizer。

旧类欧氏Text表示在本stage固定当前缓存输入上应exact不变，原共享Lorentz映射仍按原nullspace约束学习，故不保证旧hyper文本/预测完全不变。原Image_encode继续全部Image适配器求和，不做标签条件图像路由；原虚拟parent仍提示文本走Image_encode初始化后冻结，原textleaf/hierarchy/tree/parent算法均不改。原loss语义/系数和一次backward/optimizerstep不变，梯度路由改变是本轮机制。区别R11冻结全部适配器、R31额外soft教师KL与R32视觉leaf，不恢复旧辅助方案。

共享R28linear64三域3seed9份stage0 checkpoint/TRAINreceipt，旧VAL metrics移除、不搬response，仅恢复不重训初始任务；单Text模块时原规则与候选相同。31950旧初始步仅共享状态，36新增量stage/12120新optimizersteps、45评测135branchchecks。三域exactsplit/order/stages6/3/6、seed74002/3/4、Biomedunit图文、原线性Image/Text、10epochbatch64AdamW.001、naturalshuffle/Gaussian原mean/cov+1e-5I/每旧类1行、全部pairedcontrast/entail/hierarchy/mapreg loss/coeff、原初始化/alpha/freezing、TRAIN_cov nullspace与postAdam actualdelta保持。FP64entail同原函数/常数修复保持，无GELU/ridge/readouthead/realexemplar/回放边界/centroid/10%auxguard，projection扩维/学习仍defer。原nativeNCM+hyper L2softmax.5推理不变，无融合/温度标定。

每stage到达才读TRAINcache exactIDs，不提前futurefeatures；每dataset全部候选头TRAIN封存strictrestore后才用sameencoderR16unitVALcache，exactIDs/unitnorm/no二次normalize，无图像/骨干复制/下载/TEST/中间选择。共享stage0末尾恢复重评三分支counts exact对控制，NCM逐stage/seedcounts exact控制，steps原控制相同。原finite/replay/oldadapter/fixedparents/actualdelta/restore/seal核查保留。

新audit birth_text_route：calls==steps、old_rows==old*steps/new_rows==(seen-old)*steps表示每batch全类文本路径，不是独立样本；explicit_class_ids、extra_parameters=0。本stage当前缓存旧类提示在start_audit捕获的欧氏输出每batchexact不变，maxchange0。首batch独立旧提示探针对当前新Text参数梯度零/None，原实际total对当前新Text参数梯度finite非零。探针不更新参数，不冒充原batch梯度分解。原Image路径没有额外路由；synthetic检验重复类索引/新旧类输出/当前模块一次实际更新后旧欧氏输出不变/无路由RNG/拒绝未到达类、原设备生命周期与展开NIHISICrunner。

采用必须BOTH：对历史H0及R28linear64均NIH阶段平均/最终各>=+1pp、tailTRAIN<250>=+2.857142857pp、>=2/3seed最终与尾联合改善；ISIC/HKfinal/tail均值各最多降1pp，joint Fraction类级计数。失败拒绝固定出生任务文本路由，不扫路由混合/adapter选择范围/冻结时机/初始化/seed/融合或放宽门，不择有利域/参照。

独立2h从preflight起含修复/恢复，4GB含128MiB余量，预计<2.5GB含9共享初始checkpoint副本；CPU36000继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro/data ext4容量/write-readprobe/GPUfree前置，旧assets只读共享不复制/不清理。中性主子完整argv与GPUprocess名检查、独立guard不依赖SSH。三域完整exit0后仅savedresponses生成135checks/GATE/TEXT_ROUTING_VERIFICATION/TAIL_TRAJECTORIES/BRANCH_TRANSITIONS及安全聚合/完整中文报告，所有class/seed/正负保留。全部反复历史开发集，无TEST/独立确认/临床证据，曝光/跨seed预测非独立患者；不公开features/图像/身份/逐例响应/权重/原日志/私有路径。
