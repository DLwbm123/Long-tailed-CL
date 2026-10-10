# OrganAMNIST-LT 首轮独立评价修复协议

原轮两臂均完成212 adapter更新及零更新预检，候选评价完成；原基线评价在分类推断后错误要求不存在的medical historical_metrics，退出码1。原轮完整失败已交付a40ed75e51df2bbb7f0047f26ee086ebb7a96946，不覆盖其数据、模型、日志或回执。

本尝试只修复共享评价函数的历史基线适用范围，Organ与既有Path/CIFAR新配对均不要求自然医学历史基线；ISIC/HK原历史检查仍强制执行。没有历史指标伪造或放宽类别召回匹配。仅复用原轮全部五阶段已训练base/local_transport checkpoint，0新adapter/策略/拟合更新，不重新抽样、下载、复制图片或训练。两臂重新统一官方val评价，候选原评价不作为新方法选择依据。

两个真实NAS源检查覆盖五阶段：module/order/累计来源步数128/182/186/198/212、原训练类计数及全局维度/有限值、候选四区域中心维度/有限值，原bank/局部先验/boundary_pair_weights重建存储float32全局head相对误差和解析残差均≤1e−8。两条READY且新steps/adapter/policy均0后方可打开EVALUATE。两条COMPLETE且新steps0/policy0才整轮COMPLETE，正常成本2source checks+2evaluate=4条成功子进程；全部失败额外计，不重跑失败原轮。

科学配置完整继承原科学源452af31b5d94ac485144e19ab7279d347bed808e及PROTOCOL_LOCK：11类CT灰度224，IF100/nmax3100，train8343、官方val6491原封，seed/split74002，order及3+2×4划分、2epoch/64/.0003/FD10与原ViT/adapter/竞争头均不变。只有四区域局部先验及共同区域位移的local_transport候选，推理仍原全局head。修复入口拒绝训练复用checkpoint。没有附加局部读出或可学习头。

只GPU0/1，2/3禁止；新启动检查NAS挂载/空间/小写读探针及每卡≥32768MiB，不干预其他进程。沿用原截止1791739830.6082594（北京时间10月12日01:30:30），不重置原24h或累计账本。原失败轮GPU3616.210987秒已计入已闭合160848.437431秒；本轮仅新增源检查和评价实际驻留，424训练更新与CPU准备成本不得重复计。

只official_validation，医学/Path/Organ test仍封存，test数组不打开；身份只有官方样本索引，未重建患者/扫描隔离。全部六指标、candidate−base所有pp、Acc、五阶段11类n/recall和4/3/4头中尾同时公开。效果门仍BA≥+1pp/tail≥−.5/forget≤+1/old,new≥−1；保护门不等于无损，单seed/order反复val仅开发证据，不替代ISIC/HK自然长尾双目标。

结束先GitHub代理公开推送源码/协议/完整正负或失败/报告/审计/成本，核实远端SHA和匿名精确报告HTTP200，再NAS公开包与独立回执。只匿名聚合和纯标量，禁止影像/权重/逐样本/统计中心原型边矩阵/私有配置/原始日志。未完成不另开新科学轮，不新增MedMNIST数据集/Organ IF网格或CIFAR轮。
