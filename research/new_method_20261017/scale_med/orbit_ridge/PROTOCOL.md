# ORBIT_RIDGE：冻结编码器的方向扰动统计

真实新图像前向/拟合前封存。由用户明确要求结合GROUP_OUT负结果另找独立方向并实验；不是延长O。GROUP_OUT E/I/O与历史结果保持。主A仅一次固定端点；无神经参数更新、无O checkpoint、无lr/epoch/seed/层/网络池化/宽度搜索。

## 独立假设与已有依据
GROUP_OUT的O开发BA65.0951<F70.1546/E67.6236，O类1不变、类2损伤，不能把O>I当成功。Pro要求未来可学性/逐类风险与强简单控制；其“只有O成功才做伪未来”前提未满足，不启动伪未来元学习。另检验冻结预训练表示对皮肤镜方向变化的敏感性：在同一原图上固定旋转/镜像视图的训练统计，能否抑制无关方向引起的预测变化而保留诊断信息。

Cino等2025 Skin Lesion Classification Through Test Time Augmentation and Explainable Artificial Intelligence（https://pmc.ncbi.nlm.nih.gov/articles/PMC11766406/）讨论皮肤镜TTA；Raj等AISTATS2017 Local Group Invariant Representations via Orbit Embeddings（https://proceedings.mlr.press/v54/raj17a.html）给出已知群不变表示背景。本轮仅借用机制依据，不迁移论文性能，不称论文复现/原创。方向变换保持疾病标签是待检验建模假设，并非数据保证；文字/标尺等伪相关也可能被破坏。

## 固定操作与对照
原canonical224像素、FP32 APART G、原1535坐标PERM74010、原层/网络池化、原划分/类组权/平均类ridge不变。D4固定8元素：rot90^r(x)及rot90^r(horizontal_flip(x)), r=0,1,2,3；仅Tensor索引置换，无插值、缩放、裁剪、颜色变化。每视图按原FP32单位化→float64单位化，所有视图均权。图像仍一份原数据，8视图不是8个独立组。编码器永不更新，未来类不参与初始norm或alpha。

F：原单视图冻结ridge回放。T：沿F的norm/头，对八视图原始ridge分数均值（不是softmax均值），控制普通测试增强。
M：先平均八视图特征，再拟合ridge。I：M加等迹各向同性惩罚，强简单控制。A（唯一主）：八视图分别贡献训练统计，各图总权重仍q_i，部署平均视图分数。

M/I/A共享同一仅初始2555训练的expanded-view norm（两次原RMS形式，PERM不变），无平均后额外L2归一化。令x_iv含无罚截距，xbar_i=mean_v x_iv，A_m=sum q xbar*xbar'、A_a=sum q mean_v(x_iv*x_iv')、B=sum q xbar*onehot(y)'，C=A_a-A_m。C为同图方向扰动协方差之和，截距行列为零，理论PSD。I的统计A_i=A_m+trace(C)/1535*diag(1,...,1,0)。因此A−M隔离结构统计，A−I排除等量各向同性惩罚，A−T排除仅测试平均。不是此前类条件协方差Mahalanobis或固定RBF，不改名复活旧机制。该恒等式不是新算法。

各阶段(A/k+alpha D)W=B/k，截距不罚。F/M/I/A初始681分别从[.0001,.01,1]选择BA最大，同分大者；T继承F参数。初始alpha网格全报，之后冻结。不按开发结果更换主/变换/视图或重开网格。初始训练与验证组不交叠；未来仅当阶段到达才访问，统计增量，退休逐图输入不回读。

## 判定与访问
预检先合成验证D4闭合/恒等重复、增广与显式展开统计/解/截距一致、A_a=A_m+C、C半正定与截距零、I等迹、分数均值等于均值特征分数、容量序列化和完整成功/失败/缺失JSON及同包装器注入失败回执。真实固定initial训练每类前16共64图：身份方向vs旧G max<=1e-10（batch32），共享D4操作重排的平均输出误差<=1e-6；RMS方向扰动sqrt(mean||g_v-mean_v g||²)>1e-3才有必要进行性能实验，否则STRUCTURAL_NO_ORIENTATION_SENSITIVITY停止。此仅操纵门非风险证据。全部预检数据/模型丢弃，不按标签选择方向。

仅正式stage0提初始3236图×8视图；先完整F/M/I/A三个alpha共12fits，T不新增fit。初始开发门沿GROUP_OUT：A−F BA>=1pp，类1>=5pp，类0/2/3各>=-1pp；且A相对M/I/T BA均>0。类1验证仅3例，F2/3时必须到3/3；不放宽门。失败STOP_ORBIT_DEVELOPMENT_NO_TEST，报告全部控制，不进未来训练/测试。

初始通过才按顺序进入1812/28未来训练与442/8开发行，norm/alpha冻结，仅充分统计增量。开发J为stage1/2 BA均值，需A−F>=1pp、A−M/I/T各>=.5pp；类1三阶段各>=5pp，两个未来BA/类4出生转旧/类5出生>=F，旧四宏/最终旧五宏及类0逐阶段成本<=1pp。681/442/8均开发数据，仅评价与初始部署alpha选择，不训练编码器；复用选参非独立确认。

开发通过后封存全部9新阶段头（M/I/A）+3F头/norm/alpha；之后才打开586历史测试与pixels3。F回放G_Matched score<=1e-8。主A未来J对G/P/C_Matched及类1三阶段对G，共6项家族，10000已知组类内配对bootstrap seed74230，相同draw跨臂/阶段，Bonf分位[.05/12,1-.05/12]；全部下界正，J对G>=1pp/对P-C点正，类1各>=5pp，原逐类风险保护不变。仅主门全过后才检验A−M/I/T三个J机制对比：各点>=.5pp，3项调整95%下界>0（串行gatekeeping）。不以机制辅助门代替主门。救回/损害计数与BA恒等式全报。已暴露测试始终探索性，method_success=false，即使成功也只称冻结方向增强headroom，非原创/临床结论。

## 资源与停止
预估新增<=512MiB，硬限1GiB（含本地/Git/中性helper16MiB预留）；单文件512MiB。复用旧NFS pixels，不复制图像/模型/环境、不保存八视图逐例特征。所有新缓存/tmp/日志在本轮NFS目录，GPU0，实际挂载/余量/写读探针与显存先核，中性父子argv及GPU显示。全旧窗17GB软停/20GB硬限保留，写前预留并父进程2秒守卫；本轮总墙钟从首次预检<=4h含失败、子任务与正式，不重置，符合新全局<50GB/<=24h。部署每臂全norm/PERM/三角A/B/W/8视图元数据<=19058800，实际artifact回读认证；8倍推理成本明确报告，持久预算合规不等于计算免费。

非finite/隔离/回放/残差>1e-7/预算失败为工程未决，保留失败另版本修复；完整效果失败停止固定D4端点，不换变换集合、种子、alpha网格、颜色/分辨率或辅助主臂救。发布源码协议、完整聚合和费用失败摘要，不发布图像/身份/逐例/特征/模型/原私有日志。唯一scale-med心跳更新本轮，不新建。

## 首次预检序列化修正
源码c9487b7在模板三角A回读exact检查失败：预检模板未像正式路径那样将浮点矩阵(A+A')/2对称化，乘法舍入的非对称末位无法由上三角逐元素还原。已到该检查意味着身份/不变性门未报错，但不以此冒称完整通过。独立preflight_v2统一模板与正式对称化路径，原失败artifact/回执/费用保留，科学门/方向集/正则/数据/容量/4h起点不改；尚无正式分类结果。
