# MATCHED_RIDGE_QUERY_PROBE：两个固定匹配读出探针

依据用户最新明确要求，在任何新探针求解前独立封存；不是追加旧Q/R网格或为旧P/C/D追认成功。问题：在相同support组、相同F mapper、同ridge学习规则下，T特征是否仍可读出组外竞争力？优先检查类1，所有类和两个fold完整保留。method_success=false，query在研究历史中已暴露，本轮非外部独立确认。

## 固定资产与两个拟合

只复用producer_group_crossfit两fold封存PLAN、对应F teacher、T完整artifact/A/B、T一次编码featureseal/cache，以及封存query响应。只initial原训练2555，原681开发/future/test/rawimages不读取。每fold训练support=fixed fold k，query=另一半；不得跨producer接头、复用全2555旧teacher或用query选参。

每fold只求解一次，共**恰好两个新的真实ridge解，也只有两次数值solve调用**；不额外做toy拟合或F解重拟合。固定该fold原F mapper（exact核对T artifact mapper），不重新中心化、标准化、选择维度。原类/组权q每类质量1，总4；使用p=q/4，总质量1。T artifact A=X_support^T diag(p)X_support、B=X_support^T diag(p)onehot6已经归一化，重建A后不能再次除4。

W_probe = (A + .01 diag(1,...,1,0))^{-1} B；alpha=.01，自由截距不罚，原四类+两零列。没有搜索/调参/早停/新encoder/mapper拟合；查询标签不入A/B或求解。两完整artifact继承对应T bias/mapper/PERM/A/B，替换W与origin/alpha/源元数据，各≤19058800；不是只有小W的容量豁免。

在求解前核原F teacher的既有W正规方程残差（仅矩阵乘，不重solve）；核T缓存A/B固定5个坐标[0,17,767,1534,1535]的support子统计、截距质量A[-1,-1]=1/B末行每类.25、未来两列0、F/T mapperexact。原F_out/T_out分数重放≤1e-8、计数exact。两探针均HEAD_SEAL及MODEL_SEAL之后才在各query组评分；不产生seen性能来选模型。

## 事前指标和判定门

主指标仍原图像级BA；重点类1正确数/召回与F原正确query的救回/损害。报告每fold及合并的F、原T_CE、新T_probe三表，所有四类计数/recall、BA、CE、margin分位、混淆矩阵和配对R/H/去向；类内组等权再类间等权BA只作敏感性，不替换主指标。合并指标从合并计数重算，不平均两fold BA。

预定点门：
1. CLASS1_COUNT_RESTORED：probe合并及每fold类1正确数均≥对应F（合并13/20、fold0为8/8、fold1为5/12）。
2. CLASS1_F_CORRECT_PRESERVED：类1对F新增损害为0；这是本样本的描述，不是总体零风险/非劣证明。
3. BA_RESTORED：合并及每fold BA均≥F。
4. OTHER_CLASS_COST_WITHIN_1PP：合并类0/2/3各召回成本≤1pp；全部逐fold风险仍完整展示，不隐藏冲突。
四项全过才MATCHED_RIDGE_POINT_COMPETITIVE；此仍诊断、非方法成功/独立验证，不自动触发训练。

若合并及两个fold BA均低于F，则STOP_MATCHED_RIDGE_NONCOMPETITIVE：停止为该适配终点继续搜头，不称任意读出都不能利用。其余门未齐/方向冲突为MIXED_MATCHED_RIDGE_DIAGNOSTIC，保留真实局部改善，不择fold/类/指标升主，不追加第三次拟合。额外单列类1/类2每fold方向与对CE终点改善。

本轮不计算bootstrap或新增显著性族，不把此前Pro建议的四项CI事后称为预设，不改前轮两项reuse gap CI。该点门用于研究分支，不用降低seen性能追“缩小差距”。类3收益不能覆盖类1门失败。

## 预检、预算与交付

合成（零solve）：惩罚仅作用非截距、现有正规方程构造/已知W残差、自由截距平移等价、类/组权与独立query覆盖、点门成功/失败/缺失、配对恒等式及finite JSON。真实两个解均残差≤1e-7，finite/完整A-B-W-bias回读、同query分数parity≤1e-8/计数exact。失败保留不自动重跑或放宽容差，两个探针结果无论正负全报。

独立首次CPU worker含合成起≤600秒，coordinator时间/RSS/空间硬守卫，TERM5秒后KILL本轮组；新增远端+本地Git/helper16MiB<128MiB、单文件16MiB、RSS4GiB/地址空间8GiB、workerCPU300秒。预计新远端<32MiB，主要为两个完整artifact；明确计新增副本，原feature/pixel/model/env只读不复制。旧窗17GB软停/20GB硬限、max(logical,allocated)+defaultcache1040384+localGit1GB，报告增长另16MiB。

CPU无GPU初始化；真实NFS挂载/容量/小写读probe，中性父子argv，临时文件NFS，PYTHONDONTWRITEBYTECODE。验收2fit/2完整artifact/2headseal+MODEL_SEAL、两个query片覆盖2555一次、F/CE原端点parity、finite/正规方程/类组隔离/全类两fold/点门/退出退休/资源/一次成本后公开源协议全聚合中文报告成本失败摘要，核远端SHA/匿名HTTP200，DELIVERY本地NFS双端。模型/feature/身份/逐例/原日志/PDF私有不下载公开。唯一既有scale-med30min持续，不新建监测，完成不暂停整体研究。
