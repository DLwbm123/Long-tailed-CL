# R32：TRAIN视觉均值作为层级真实叶节点

R28固定GELU、R30边界回放和R31旧模型软锚定均未通过冻结门，不继续扫描它们的激活、回拉或teacher参数。源码显示原层级真实leaf输入是text_feas；原NCM已计算的视觉TRAIN类均值没有直接作为该约束的leaf。R32检验把层级语义结构落到视觉类中心这一单一结构假设，区别于R09移除hierarchy、R07改变parent编码路线、R23均值硬标签CE；不增加独立辅助损失。

唯一变化：原all_real_prototypes_seen=text_feas，换为network.trans_hyper(network.Image_encode(network.fc.weight[:C].detach()))。原replace_fc在本stage train开始前已用当前到达TRAIN计算新类均值并保留旧均值；不新拟合、不归一化mean、不看未来classes/features。原树/父子边、原所有hierarchy entailment/distance/root及系数、mutable leaf与冻结旧leaf/虚拟parent规则保持。原虚拟parent仍从提示文本走Image_encode并首次建立后固定；算法路径不改，但后续parent数值可随候选已学网络不同，不能称所有parent初值与控制完全相同。图像映射分支获得新leaf层级梯度，Text分支不再从hierarchy叶路径获得梯度，仍保留原图文contrast/entail目标。

只有当stage新增的mutable real leaves会按原manager被替换；旧leaf缓冲仍原样冻结。因此seen*C前向行不是每个旧类的梯度曝光，更不等于独立样本数。保留对极少样本均值噪声/多峰丢失风险；不是已验证修复。

必须从初始任务全新训练，45newstage/44070新optimizersteps，seeds74002/3/4、NIH6/ISIC3/HK6原exactsplit/order保持；不复用R28/R30/R31已学初始权重，避免初始text leaf与后续visual leaf混合。相同Biomed单位图文缓存、原线性Image/Text、10epochbatch64AdamW.001、naturalshuffle/Gaussian原mean/cov+1e-5I/每旧类1样本、pairedcontrast/entail/hierarchy/mapreg全部原loss/系数/backward/optimizer、树/模板/alpha/冻结、原TRAIN_cov nullspace与postAdam实际delta保护全部保持。原entailment函数FP64修复保留，不改epsilon。无GELU/teacher/ridge/head/回放纠正/10%guard，projection扩维/学习仍defer，原nativeNCM+hyper L2softmax.5推理不变。

每stage到达才读TRAINcache exactIDs，不提前未来feature；每dataset所有候选头训练封存restore后才用同encoderR16unitVALcache，exactIDs/unitnorm、不二次normalize。无图像/骨干复制或下载/TEST/中间选参；全部历史开发集反复使用，非独立确认/临床证据。R28linear64为已有同硬件/精度原层级控制，只读，不重训优化；同seed参数初始化规则一致，不声称GPU逐位训练复现。

新增audit source=TRAIN_visual_NCM_mean、calls==steps、forwardrows==seen*steps、原manager mutableleafrows==(seen-old)*steps、detachedmean=true；每stage首batch hierarchy梯度有限，Image_Adapter梯度非零、Text_Adapter hierarchy梯度恰零，记录routing proof。这是路径校验，不是效果成功；原finite/replay/oldadapter/fixedparents/actualdelta/strictrestore三分支预测/先seal后VAL保持。NCM逐stage/seedcorrect/count与R28linear64精确一致；各stage步数同控制。全部45newstage/135branchmetric类级分母/所有seed及负结果保留。

采用必须BOTH：对历史H0与R28linear64均NIH阶段均值/最终各至少+1pp、TRAIN<250尾至少+2.857142857pp、至少2/3seed最终与尾同时改善；ISIC/HK最终总体与尾均值各最多降1pp。联合改善Fraction类级计数，不能择域/参照/放宽门。失败拒绝本固定visual-leaf机制，不扫描mean归一化、混合比例、parent路由、loss系数或融合，不事后救援。

独立4h从首次preflight起含重试/恢复共享原钟；4GB含128MiB余量，预计新增<2.5GB，CPU72000s继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro5000规定/data本地ext4，旧R23assets/official/TRAIN/VAL只读共享不复制、不删除旧依赖；R28安全控制聚合小副本计预算。中性主子argv、独立guard、挂载/容量write-readprobe和GPU余量检查保持。退出仅在保存结果上分析，发布自有代码/协议/安全类级全部正负结果，不公开身份/原图/逐例响应/features/权重/原日志/私有配置路径。
