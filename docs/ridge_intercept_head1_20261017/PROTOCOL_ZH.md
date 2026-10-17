# 固定完整岭头，仅推导支持集截距

## 假设和唯一主要改动

单位列控制四片宏BA均低于完整岭，方向和相对长度均应保留。完整岭原目标没有截距；当前检查类别分数偏移是否仍贡献损伤。固定已封存完整W及全部列长度，只增加一个由同支持统计推导的截距，不重新拟合W。

m=(1/K)Σ_c mean_i z_ci是同一H/新S、raw单位特征的类平衡全局均值，直接复用已交付centered-centroid的center。固定W下，最小化(1/K)Σ_c mean_i ||z_ci W+b−e_c||²的条件最优解为b=1/K·1−mW。推理为8(zW+b)，已有metrics在8zW之后加bias，因此保存logit_bias=8b。保留共同1/K项；它不影响argmax或softmax，但用于完整平方损失方程。

仅数据推导类别截距，无新旧组单参数拟合、CE优化、温度、范数、λ或偏移强度扫描。它是固定W下的条件最优截距，不能称联合重拟合完整仿射岭，不能保证改善分类或CE，亦不是新方法。截距依赖支持数据，不能称零拟合。

## 来源和访问

完整W源码703dd54c9150a013a386a2b77bd996d9e8c08743、交付3b32f0d63280cd174502f821158d976063d16ad6；共享center源码6779e9d1361f28cdf44ba281fbeb1d116062c4b3、交付e91268d12338a36f6d20ab127cfb7d4d8200fcdc。两个历史CONFIG已核对同prior_root和bank_root；原代码均从同一H和新S构造raw单位表示及类平衡统计。原center使用每类均匀权重，原岭亦检查均匀权重。

保持seed74002/原类序/划分，ISIC旧4新2、HK旧13新2。只读原头、center及两者来源收据，检查后推导并保存新artifact，再读Qfit/Qgate和上一轮聚合。原头和原center不改。零adapter来自原头，center附带adapter须完全匹配。无H、新旧S、teacher状态、E、原图、新标签或GPU；不拷贝banks。读取center所在artifact会反序列化其旧head，但只使用center，旧head仅作形状检查，不用于预测。

## 判断和公开范围

raw_ridge为主参考，raw_centroid为必要参考。全部16参考为上一单位列轮15参考加raw_ridge_unit_columns，直接复用历史聚合；候选raw_ridge_support_intercept。每query总体BA严格正收益、旧新各不低于参考1pp，两query同时通过为对应描述性门槛。不按query选优部署或自动替换，不切换主要指标。

公开每类count/correct/recall/full-seen CE，all/old/new BA/CE及所有参考差值，微计数，原/新头norm（须不变）、正倍率1、每类unscaled_intercept/logit_bias，center_norm及均值方程残差，全部成本失败缺失。核对逐类差候选标签和正确数。query已反复用于设计、患者分组未知，只属开发机制证据，非独立医学、跨种子或全阶段CL。

## 实现、检查和有限预算

复用run_ridge_direction_head.py，新增support_intercept布尔分支，默认单位列分支保留；调用队列、lock、私有配置和中性入口一并更新。复用metrics对bias的既有语义。头1536×6/15、DT float64 finite、列norm>1e−12，全零model、center形状1536与finite、center来源收据和model一致性检查；mW+b−1/K绝对残差<=1e−12。查询finite1536和标签范围、每类非空，访问守卫先拒绝支持/query/ref提前读取。失败无回退或科学重试。

每dataset工程检查以类不平衡3样本、类等质量权重验证均值残差0、任意固定正负截距扰动增加平方损失恰为扰动范数平方、特征与center共同平移且相应截距改变后分数不变、零center给均匀截距。3个二维工程截距推导/数据集，共6；无工程矩阵或逐元素方程求解，无梯度。

正式2条件截距拟合，21个float64新参数168字节。复用32256个W元素258048字节；合计推理参数32277元素258216字节（不含原全零adapter），center不需要留在推理artifact中。0新头拟合/构造、0支持前向/Gram、2155查询前向，0梯度/adapter/teacher/encoder/memory更新，0GPU。缓存读取、范数和向量运算计入CPU，历史拟合不重计，客户端准备null。

两CPU任务ISIC_intercept、HK_intercept，各900秒总3600秒。启动一次检查后结束回合，现有半小时监测收尾。保留失败成本，工程故障最小修复未完成阶段；公开交付完整结果后推进下一步。
