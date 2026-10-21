# METRIC_MEASURE_ALIGNMENT 原冻结F的目标测度检验

用户授权针对SCALE-Med重新核任务读出，不照搬CTTA。独立可证伪假设：在原冻结F表示、原support-only mapper和固定ridge规则下，将类内目标从组等权改为图像等权，能否产生通过类1优先和全类保护门的组外收益。该试验是拟合目标测度干预，不是新的无拟合解码器或原创方法。没有新增语义信息不是自动否决理由；目标不同也不保证收益。

## 先验核查与旧负结果边界

图像BA为四类图像召回均值，等价于每类质量1/4、类内每图1/n_c。原geometry.weights为每类1/4、组内每图1/(G_c*m_g)。两者类别先验都是1/4，组大小不等才有类内差异。固定score的共同正温度/单调函数不改变argmax；对已类平衡的目标再除原类别频次不是自动有效的BA修正。ridge分数也没有自然频率posterior校准认证。

核查producer_group_crossfit、matched_ridge_query_probe、full_history_ce、readout_support、initial_bias、kernel_readout、covariance_reference等既有协议：其训练均沿类/组权；此前matched probe作用于停止的T适配特征且保留同组权，readout_support改覆盖/重数。本轮只用原始冻结g，不读T特征或encoder、P/C/D、旧全2555 teacher；不修改原负结论。不称全部历史头都未尝试过，也不把结果称完整CL成功。

理论定义参照[BA官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)。[logit adjustment](https://arxiv.org/abs/2007.07314)摘要只支持该频次调整是既有方法，本文不声称复现其全文或用其保证本ridge收益。原始posterior p(y|x)/prior的BA Bayes规则须区分原频率分布与已平衡训练分布，不能无条件双重修正。

## 数据与强对照

只原initial2555及既定seed74290两fold，组互斥、每类保留。每个producer支持半组用于全部权重/拟合；query退出mapper、统计、求解和选择。只允许producer已封存PLAN cached_g及该fold Fteacher；原F mapper先拟合support已冻结，本轮exact复用，不重新估计。原681开发、future/test、rawimages、旧learned特征禁止。query已有历史暴露，非外部独立确认，不新分组/换seed。

强控制G：原Fteacher组权头，只检原正规方程、support A/B和封存query分数/计数parity，不重求。主臂I：相同原g、同mapper1535维+自由截距、同四类onehot目标（B/W六列末两列0），只将p_i=1/(4 G_c m_g)改为1/(4 n_c)。总质量1、各类1/4，alpha=.01只罚非截距，float64同solver。恰好两次真实solve，每fold一次，无其他fit/toy solve/温度/类offset/校准/mapper/alpha搜索或T救援。

support内报告每类组数/图数/组大小min-median-max、两权重TV=1/2 sum|p_group_cond-p_image_cond|、权比范围。TV只诊断测度差异，不作为按query选方案的指标；若所有support类TV恰0，则STOP_TARGET_MEASURES_EQUAL，0次真实solve。否则固定两折均完成，某类两权相同也不选择跳过该fold。

两完整artifact含固定mapper/ids/PERM、packed support A、B/W六列、原冻结encoder引用/无学习bias事实及元数据，各<=19058800bytes；原预训练模型只读共享不复制。两HEAD_SEAL及总MODEL_SEAL完成后，才允许读取原query分数、计算任何新query分数/风险。全部查询行覆盖一次，baseline exact计数、分数maxabs<=1e-8。

## 固定判定门与推断

报告G/I合并及逐producer fold的四类图像计数/召回、BA、CE/margin、混淆、组等权BA敏感性、配对rescue/harm及去向。合并从计数重算，不平均fold BA；CE仅描述，ridge分数不称校准概率。

点门全部同时要求：I-G合并BA>=1pp；类1至少多1个正确（>=5pp，20图）；两fold BA均>=G且类1正确数均>=G；G原正确类1H=0；其他类0/2/3合并召回成本各<=1pp。失败STOP_METRIC_MEASURE_ALIGNMENT，不把辅助、单fold或类3收益升主，不补CI0，不搜索权重插值、alpha、mapper、头、seed救本配置。

仅点门全过，做I-G BA和类1两项已知组配对bootstrap：每类原组有放回抽同组数，组内所有图同重数，seed74342/10000draw，两项同draw，Bonf百分位[.0125,.9875]，两下界均>0才DEVELOPMENT_METRIC_MEASURE_SIGNAL，否则STOP_METRIC_MEASURE_EVIDENCE_INSUFFICIENT。条件于固定fold/head，不估训练随机性、未知患者或外部总体确认。所有分支method_success=false，原类1和未来完整CL目标不替换。

## 校验和预算

合成0solve：相同组大小两权相同、异大小不同但各类质量相同；自由截距不罚/alpha不重缩放；门成功/失败/缺失；组配对bootstrap成功/失败；有限JSON与配对恒等式。实际support重算G A/B与旧teacher一致<=1e-10，原G正规方程残差<=1e-7。I正规方程残差<=1e-7、完整artifact exact回读/score<=1e-10、finite。关键误差先持久化再assert。任何异常或预算停止工程未决、不自动重试、不放宽容差。绝不监测/重启旧退休PID。

首次actual含synthetic/setup/拟合/评分/推断/等待共享600s，独立协调在590s TERM、5s后KILL；CPU300s、RSS4GiB、单文件逻辑64MiB；新增remote+本地/Git/helper16MiB严格<128MiB，按max(logical,allocated)计，预计remote<48MiB。旧窗17GB软停/20GB硬限，default1040384+localGit1GB外部预留及报告16MiB；所有cache/tmp在NFS，实际挂载/容量/小写读probe先核。GPU0调用0，无encoder前向/梯度/参数更新/新数据/像素模型环境复制。中性父子argv启动后核，唯一现有30min心跳不变。wall按协调完整进程一次，worker wait4和协调own CPU各一次，阅读网络准备缺口仍未知。完成公开源码协议全聚合失败/NA/成本中文报告，代理push核SHA/匿名HTTP200，DELIVERY本地NFS双端；身份/逐例/模型/feature/Gram/私有日志不下载公开。
