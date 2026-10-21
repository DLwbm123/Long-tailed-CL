# DAP/FROST正文审计后：固定双统计融合pilot

**STOP_FIXED_FROZEN_STAT_FUSION，method_success=false。** ISIC退步、类1未救回，HK有限BA改善但不满足tail及全类门。停止固定配置，不搜索融合系数，不将控制最佳升级。

## 为什么选这一项，而非照搬两篇

已读DAP核心方法与算法、证明/实现/超参附录及FROST完整核心方法、实现/消融附录，详见DAP_FROST_METHOD_AUDIT_20261021。两篇均为ICLR2027投稿，不称录用或官方代码复现。DAP包括base mean/covariance迁移、合成特征、3000–5000 RBF锚点、class-balanced ridge与base监督适配。其packed float64 Gram单独36,012,000–100,020,000bytes已超19,058,800cap，不能把kernel新名字/改小K当可直接复用；当前initial也无独立rich-base/novel划分。其校准不同于旧纯协方差距离，但不因此得到独立收益证据。

FROST原版也有base adapter、2000维随机ReLU映射及20%base holdout选lambda，并非全流程不训练/不选参。这里只复用Eq10固定0.5融合的已知估计器，原F不变、不复制额外随机映射/adapter。当前F每类质量已平衡，不能声称再次纠正base样本频率偏置。假设是全局逆矩判别与原生均值方向互补；不是新语义信息、原创函数族或完整FROST复现。旧mapped Euclidean NCM/FeCAM和Gaussian patch+global融合并非完全相同估计器，但其负结果仍保留。

## 封存与范围

源/协议5883be9e8bd531f79984aa36ec8f2fa96b21ceef先push/ref/匿名PROTOCOL200，才actual。唯一主臂M=.5F+.5P，其中P为同原class/group/image权重的原生g类均值余弦，N为原生Euclidean类均值强控制。所有4臂均完整报告，无扫alpha/权重/温度/offset/seed；不拼上已停止的组一致性C。

ISIC初始2555/4类，seed74290两producer已知病灶关联组整链互排；原F mapper/head/alpha.01不变。HK原S_old3140/13类，复用上一pilot的原F，E986与Qgate388两个互斥query cohort，不是两训练fold。未知patient只限制claim，不阻止匿名分类；无PAD/新数据/图像/GPU/encoder/optimizer。仅3次native prototypefit，0ridgefit/solve/mapperfit，所有完整artifact回读/HEAD/MODEL_SEAL先于query，未来类立即过滤不评分。历史评估已暴露，不是盲测或完整CL未来J。

## ISIC全类与两fold

| 端点 | BA % | 类0 | 类1 | 类2 | 类3 |
|---|---:|---:|---:|---:|---:|
| F | 64.713626 | 237/307 | 13/20 | 64/170 | 1626/2058 |
| P | 56.280891 | 209/307 | 11/20 | 67/170 | 1289/2058 |
| N | 56.440098 | 209/307 | 11/20 | 68/170 | 1290/2058 |
| M | 64.429282 | 235/307 | 13/20 | 64/170 | 1616/2058 |

M-F BA -0.284344pp；类0 R2H4、类1 R0H0、类2 R1H1、类3 R10H20。类1没被额外伤害，但也没有救回，不能写成tail改善。P/N单独均明显弱于F，混合胜P/N并不足以胜F。

| fold query | F BA % | M BA % | F类1 | M类1 |
|---|---:|---:|---:|---:|
| 0 | 73.741477 | 73.349328 | 8/8 | 8/8 |
| 1 | 59.042186 | 58.858455 | 5/12 | 5/12 |

两fold均退步；ISIC BA_GAIN/CLASS1_GAIN/CELL_BA false，其余冻结门true。全类pooled及每fold各<=1pp损失门通过，也不能覆盖主收益门失败。

## HK全部13类、两个cohort

| cohort | F BA % | P BA % | N BA % | M BA % | M-F pp |
|---|---:|---:|---:|---:|---:|
| E | 62.956731 | 62.710417 | 63.091674 | 63.131046 | +0.174315 |
| Qgate | 71.257239 | 64.095005 | 64.589511 | 72.102547 | +0.845309 |

| 类 | E F→M correct/total | E M救回/损害 | Qgate F→M correct/total | Qgate M救回/损害 |
|---:|---:|---:|---:|---:|
| 0 | 165→165/165 | 0/0 | 62→62/62 | 0/0 |
| 1 | 0→0/1 | 0/0 | 0→0/1 | 0/0 |
| 2 | 0→0/1 | 0/0 | 0→0/1 | 0/0 |
| 3 | 17→17/21 | 0/0 | 7→7/8 | 0/0 |
| 4 | 28→31/66 | 3/0 | 12→14/28 | 2/0 |
| 5 | 103→103/113 | 0/0 | 33→33/40 | 0/0 |
| 6 | 182→182/182 | 0/0 | 73→73/73 | 0/0 |
| 7 | 123→122/123 | 0/1 | 48→48/48 | 0/0 |
| 8 | 58→58/60 | 0/0 | 24→24/25 | 0/0 |
| 9 | 34→33/41 | 0/1 | 14→14/16 | 0/0 |
| 10 | 125→124/150 | 0/1 | 54→54/59 | 0/0 |
| 11 | 25→26/61 | 1/0 | 10→11/26 | 1/0 |
| 12 | 0→0/2 | 0/0 | 1→1/1 | 0/0 |

HK两个cohort均有有限BA改善，Qgate没有新损害，但均不到预设+1pp；类1各0/1、类2各0/1仍错。E中类7/9/10各损害1图；部分类成本超1pp。全类改善不能替代tail门，两个tiny query也不能作显著性/临床安全证据。HK BA_GAIN/CLASS1_GAIN/CELL_BA/CELL_CLASS1/ALLCLASS_COST/CELL_ALLCLASS_COST false，prototype比较和类1无损害门true。完整P/N全类、混淆和R/H见aggregate，不挑最佳cohort。

## 可证伪结论与下一步边界

本次固定融合没有支持两数据集少数类安全收益假设；不能将FROST整篇否证，亦不能从本结果推广任意原型或任意融合不可用。低开销和singleton可计算性只是可行性，未转化为类1收益。保持原F和前组一致性有限无害收益描述，不能追加某个融合权重/温度、prototype空间或C+P去救已停止配置。后续需另有独立估量或监督来源再封协议。无bootstrap，CI为NA，method_success=false。

## 完整性、成本、资源与隐私

3完整artifact最大逻辑9962904<=19058800；实际allocated见ALLOCATION（约逻辑2倍），max全计。A/B最大误差1.11e-16，F正规方程残差最大5.551e-17，ISIC/HK F query误差0.0/0.0，原计数exact。A/B/W/mapper/means回读exact，支持端回读score maxabs0.0。所有类别/cohort/fold覆盖、finite、已知组/path互斥、先seal后query通过。

firstactual=1792564429.005804；wall18.935803s/wait4CPU23.011190s/coord2.526741s一次并账，正文worker不重加；累计{'activity_wall_seconds': 31668.720306933712, 'known_worker_wait4_CPU_seconds': 40114.02294399999, 'coordinator_CPU_seconds': 1219.6878749999998}。准备、阅读、网络和报告CPU未知。RSS908578816<4GiB；round58729789+16MiB=75507005<128MiB；window13858518422+1001040384+16MiB<17/20GB。600s/CPU300s/file16MiB守卫均未触发；NFS4容量/小写读probe通过。

中性活动父子argv已捕获；父1098565/子1098566均exit0，退出检查为空，退休后不再监测/重启。无历史删除/像素模型环境复制。公开源码协议、正文审计、全类聚合/门/NA/成本及中文分析；第三方PDF与私有身份/逐例/features/means/model/Gram/原日志留私有。
