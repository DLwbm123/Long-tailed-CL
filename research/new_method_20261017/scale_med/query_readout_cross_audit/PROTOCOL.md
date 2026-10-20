# QUERY_READOUT_CROSS_AUDIT

独立CPU只读诊断，首次真实交叉分数计算前封存。延续用户要求的整个producer排query组；不复活旧P/C/D，不改既有CONDITIONAL_ADAPTATION_REUSE_GAP_SIGNAL，不升级交叉组合为主候选。

## 问题、数据与强对照

已有两producer各只在support组构建mapper/初始ridge并CE更新。只对各producer未参与适配的query组评分，同2555初始训练行每行恰好作为query一次。原681开发/future/test/rawimages均不读取；不读原始数据容器，F特征和y/groups直接复用已过滤且封存的PLAN.private.pt，T复用各2555 FEATURE_SEAL缓存。原artifact不修改，不调用encoder/optimizer/solver。

每fold完整2×2：F_on_F、F_on_T、T_on_F、T_on_T，记法为head_on_feature。F头来自该fold support-only teacher，T头来自该fold CE终点；两头原本共用同support F mapper，必须exact一致。F/T特征均经这个相同mapper，不能偷偷重新归一化拟合。所有fold/类全报，不在结果后选最佳读出。F_on_F/T_on_T须回放原F_out/T_out的逐类计数、query分数误差≤1e-8；不重新执行旧EVALUATE或编码。

统计4个合并OOF表及每fold4表，原分母307/20/170/2058；计数/recall/BA/CE/margin分位/R-H完整聚合。这里OOF仅指对应producer排组，非外部独立临床确认。原F/T头及图像量不同，跨fold差异不作因果。

## 事前描述门和分解

1. REFERENCE_RESTORES_CLASS2：F_on_T类2correct≥F_on_F且> T_on_T（原对应64及54，读取封存aggregate核对不硬改）；REFERENCE_RESTORES_BA：F_on_T BA≥F_on_F且>T_on_T；两项同时才COMPLETE_REFERENCE_RESTORATION_POINT_PATTERN，否则不称完全恢复。
2. FIXED_REFERENCE_CLASS2_HARM：F_on_T class2correct<F_on_F。另对全部类同样报告固定参考损害旗标；不以单类判断信息不可逆丢失。即使部分恢复也保留其点差，不按flag抹掉。
3. 不做bootstrap/显著性，不把事后新家族接到前轮CI，不拟合remapper。任意门只是描述，method_success=false，不触发旧方法重训或性能成功。

对每fold每类所有3个固定竞争类（2×4×3=24单元）做真实类对固定竞争类margin代数分解：m_FF→m_FT→m_TT（固定F读出下表示项，再换头项）；反向路径m_FF→m_TF→m_TT也完整报告。验证总差等于两项之和；interaction=m_TT−m_FT−m_TF+m_FF等于(Δz)·(ΔW_c−ΔW_j)。给出all、F_on_F原正确、T_on_T新增损害子集的有限值/mean/0,.1,.5,.9,1分位。F原正确样本上保留四个中间/终点对错类别及固定参考实际保留数，不能将路径分类比例称头因果贡献。分数尺度不同，margin幅度不作因果强度。

## 执行、预算与交付

合成先验：小维度共同mapper下2×2分数和固定rival双路径分解/interaction、分组query覆盖、四类计数/R-H恒等式及JSON有限；失败保留不自动重跑/放宽容差。正式先验只名单允许PLAN、两个teacher/artifact/feature及原responses、seals/aggregate，禁止其他私有torch文件。原metadata及分数parity通过才报告描述旗标。

独立首次实际worker含synthetic起≤600秒，coordinator独立守卫每次核deadline/RSS/存储，TERM后5秒KILL本轮子组；新增远端+本地/Git/helper16MiB<128MiB、单文件16MiB、RSS4GiB/地址空间8GiB/workerCPU300秒；旧窗17GB软停20GB硬限。CPU无GPU初始化，规定NFS实际挂载/容量小写读probe，临时文件NFS，中性父子入口。预计新远端<1MiB，无新模型/feature/逐例文件，无复制原pixel/权重/环境，无旧资产删除。

验收两fold×4表/合并4表、24margin单元、原4表中两端queryexact-count parity/分数≤1e-8、mapperexact/代数≤1e-8/finite/覆盖/退出退休/预算；coordinatorwall和wait4workerCPU/coordownCPU各一次，准备网络报告CPU缺口披露。公开源码协议全部聚合成本失败摘要，核远端SHA/匿名HTTP200，DELIVERY本地NFS双端；private模型/feature/组身份/逐例/原日志/PDF不下传公开。唯一scale-med继续30min，单轮完不暂停整体研究。
