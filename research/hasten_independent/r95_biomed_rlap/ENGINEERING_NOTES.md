# R95工程与启动边界

沿用D94最早准备时钟start，后继deadline=start+24h，D94自身工程已在1800秒内完成；本轮不利用独立工程编号排除相关准备。D94 workerCPU27.062167/wall76.313616作为关联准备纳入费用一次，不能再重复加其core。

31260正式step、3arms×6stage×20epochs、batch16/micro2、原CE/对比/balance/RLPO系数与参考EMA完全固定。加速不将微批对比改为大批对比；旧branch缓存仅在参数冻结和确定性预处理下使用，当前分支每组重算。相同更新内原型与文本图复用以leaf累计梯度，组末合并反传，浮点求和顺序不同，不能宣称逐bit相同。

原生FULL/CE_WEIGHT/UNIFORM三臂slow/fast：scores差0；loss最大约2.003e−8；完整梯度maxabs约1.416e−7。FULL AdamW参数归一化最大差4.376e−5，CE_WEIGHT4.466e−7、UNIFORM1.347e−8，均满足事前2e−4门。必须如实保留AdamW对极小梯度舍入敏感的差异，不把原生通过写成所有参数逐bit等同。所有比较是工程fixture，无分类效益主张。

6个distinct真实TRAIN图，native每组7rows含工程重复fixture，shape profile使用重复图和合成bank/标签仅测操作成本，非正式训练样本复制；native+profile累计172image rows/96text sequences/10engineering optimizersteps单列，正式训练模型从原seed重新初始化，不接这些更新。基座始终冻结。

完整profile初始13active文本与最终23classes/6branches各固定双pass，取预定第二pass最大step .395120144秒；frozen1/2row的最坏第二pass每row .010203838秒。31260steps、非stepvision上界199574rows、安全2和7200秒I/O/保存/分析余量，新全计划35975.753065秒≈9.993小时；检查时原remaining84151.087秒≈23.375小时，准入通过。估计不是完成保证，独立deadline/RSS/disk守卫生效。

旧LAION R88已闭合不重跑、其报告仅历史参考。新Biomed文本prompt每branch多1024参数、centroidmemory由医学底座同算法选，不能把跨run差值全归RLPO或纯骨干。FULL对两个同骨干控制的正负是本轮主要归因问题，final/tiny/oldnew门不弱化。0TEST/全部seal前0VAL。

源码与工程收据交付不是科学结果完成；最终报告需完整三臂及NCM/NCM_ALL的micro/macro/tail/全class-stage/费用。私有数据、缓存、端点与optimizer保留不公开。
