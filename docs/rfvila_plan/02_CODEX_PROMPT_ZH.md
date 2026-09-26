# 给 Codex 的执行 Prompt：NB2-RFVILA-12H-R1

你负责在 my-gpu 的既有 remote-home 项目内执行本包实验。先阅读 README、01、03、04、05、06 和 reference，之后完成真实实现、验收、运行和结果交付。不要只回复计划，不要把脚本启动或合成测试当作医学实验完成。

## A. 唯一任务

以 VILA 的固定随机非线性映射为新增机制，检验正常两类Task1 APART + Generic/Biomed VLM 的类别均衡解析持续学习能否优于同条件线性读出。

输入仓库 DLwbm123/Long-tailed-CL，基线提交 b01ba12f86706c44dfcdbed81c7305493ce354dc。
参考VILA代码固定ad4293af236d86e05d8e850d2ce33638fc16c7ed，只做算法来源核对，不直接运行作者默认训练脚本。
新分支 exp/nb2-rfvila-12h；若已存在且非本run，先审计，不reset覆盖；使用唯一新run_id。
论文式、公开代码式、本项目适配式必须分开，不声称本轮是VILA原始完整复现。

## B. 12小时与单GPU

第一个本任务远端命令开始即写不可重置的 T0/UTC/deadline 与43,200秒墙钟预算。
最多使用1个已获分配GPU，若未配置scheduler分配则询问/确认GPU分配，不擅占其他正在运行的卡。
不得沿用历史protocol中的“忽略时限” override 或 GPU 0/2/3 列表。历史授权不改变本轮限额。
所有准备、下载、编译、失败、恢复、备份、报告均计入。重启不重置时钟。
T+2:00前完成资源准入并锁定方法集合；T+8:30后不启动新fit单元；T+10:30后无新图像forward；T+11:00后只允许CPU报告/hash/归档；T+12:00必须停止。
新工作准入要求：实测预计耗时×1.5+必要保存/后续评价时间<剩余预算，并保留1小时报告。
只能结束本run PID/进程组，不能kill其他任务。提前完成即STOP，不增加模型或第二轮。
创建可恢复的真实控制器、worker状态和本run看门狗。5分钟heartbeat、每小时本地摘要；清理本run监测器，不恢复旧定时监测。

## C. 路径、依赖与备份

发现 my-gpu/remote-home 真正路径、旧run、原图train/val、manifests、class/order/component/tail锁、6份正常Task1父状态和共享参数、G/B权重及原生配置、已有val缓存、独立NFS副本。
不要写死/root/remote-home，不全盘搬家，不从不明镜像获取替代权重，不删除历史资产。
参数与文件依赖严格恢复，不仅检查文件名含Task1。每份父状态seen=2、正常Task1、order/manifests/core/delta等必须与原锁一致。
补做旧备份关键集合目标端SHA256对照：6父及共享依赖、G/B权重/config/tokenizer、数据协议锁、复用bank/W。实际restore一个父和一个bank/W。
大图像归档与关键集合验证状态分开；不把只读到完成回执说成全量逐字节核验。
关键备份无法验证时只运行非破坏CPU/合成检查，BLOCKED_BACKUP并交付，不新增医学forward。
每stage原子落盘并同步独立副本；归档失败时保存当期并暂停。计时不停。

## D. 不可变协议

ISIC=2+2+2+2；HK=2×10+3；seeds/orders=1993,1994,1995。
六份正常两类Task1父状态固定。无额外医学基础阶段，不重训Task1。
G=当前Generic OpenCLIP ViT-B-16 laion400m_e32。
B=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224，revision=9f341de24bfb00180f1b847274256e9b65a3a32e。
B必须使用原生PubMedBERT/text/tokenizer/context256/preprocess，不只换image权重。
G/B分别锁定，固定后断网运行。B预训练暴露保持UNKNOWN。
不尝试DermLIP，不添加第三VLM。
编码器、R和所有buffer冻结，新neural_epochs=0、optimizer_steps=0。没有ConCM/ACTM/FD/LoRA/训练投影层。
一切test/reserved文件、features、scores和身份表禁止访问。

## E. 必须新增可执行runner，不能假装reference已覆盖生产逻辑

复用并审计 b01 的 ct13_runtime/factories.py、medical_vlm.py、tools/run_medvlm.py、medvlm_math.py 等接口，不修改原历史文件来覆盖旧结论。
在新模块实现 RF map、class-balanced S/Q状态、Task1 grouped CV、稳定求解、CSE、单GPU流调度、原子恢复与隔离评价。
本包reference只做小规模数学oracle。生产runner需自己实现数据与系统边界。
每次实际运行前记录source hash、protocol hash、environment、具体命令。SOURCE_LOCK不能在fit后悄悄更新。

## F. 方法定义（严格按01）

a=normalize(APART main/few pre-logits)，dim1536。
u=normalize(对应VLM图像projection)，dim512。
h=[a;u]/sqrt2。

R_a/R_u用NumPy PCG64 SeedSequence([r,0])/([r,1])标准Gaussian生成，float64。r1=67101；r2=67102。保存实际矩阵哈希。
m=4096，不搜索宽度、不动态改变R、不加bias、不对RF输出再次L2归一化。

F.LIN=a；G/B.LIN=h。
F.RF=sqrt(2/m)*ReLU(aRa)。
G/B.RF=sqrt(2/m)*ReLU((aRa+uRu)/sqrt2)。
SPLIT总输出4096，每分支2048，sqrt(2/m)*concat(ReLU(aRa_half),ReLU(uRu_half))。
RPLIN=(aRa+uRu)/sqrt(2m)，无ReLU。
G/B及三个parent、两数据集使用同一r的对应Ra/Ru，不耦合parent seed。

每类S增量=Phi.T@Phi/n_c，Q列=mean(Phi)，全局W=solve(S+K*lambda*I,Q)。
等价于 class-averaged MSE+lambda ridge；不能省K，把普通固定正则RLS当成同目标。
生产优先Cholesky求解，不显式inverse。若用Woodbury需补ΔK*lambda的对角变化并验证batch parity。
FP32冻结图像encoder且无TF32/AMP；RF/S/Q/W/CV/score用float64。
正常方程相对残差<=1e-8，无自动jitter/无伪逆静默兜底。

## G. Task1 CV

所有新LIN、RF、SPLIT、RPLIN均有相同候选预算：
[1e-6,1e-5,1e-4,5e-4,1e-3,2e-3,1e-2,1e-1,1]。
只用当前Task1训练组件，最多3折；组件按SHA256('67026|class_id|component')排序round-robin；组件不可跨折。
最少类组件数<2或不能合法分组时，对该parent全部方法共同退回lambda=.001，明确报告，不用val代替。
评分为heldout class-balanced squared-error；损失tie<=1e-12取较大lambda。
CV后用全部Task1数据重拟合，锁lambda，在T2以后禁止重选。
模型特征已由全Task1适配，写明这是固定表征后的readout CV，不是端到端独立验证。
使用weighted-design的较小primal/dual系统，可对每fold复用谱分解评估lambda grid，避免重复9次大分解。
CV数字可写日志但不得用来删方法/Rseed；准入只看资源。
CSE复用父W；五个LOCK历史参照保留原固定lambda而非CV。

## H. 方法、优先级与预期行数

T0：F/G/B LIN + F/G/B RF1 + G/B的LIN/RF1各CSE1/CSE025 + 5个历史LOCK，共19方法。
必须完整两数据集3order，855阶段/8721逐类行；未完成只能INCOMPLETE。
T1：加入F/G/B RF2及其G/B CSE1/CSE025，共26方法，1170/11934行。
T2：依次添加G/B SPLIT1、RPLIN1、RF1.CSEperm1；最多32方法，1440/14688行。
实际唯一解/逻辑报告行/alias/CV解计数分开；预计矩阵在04中。
根据T+2:00前的资格测速预准入完整tier。优先完整主对照，不跑完ISIC后因HK慢悄悄放弃HK。
不能按分数挑seed、只报全类最好的模型或用RP2替换RP1。

## I. CSE接口与防混淆

文本使用既有两条医学模板、旧锁定类名，每模型原生text encoder，均值/归一化沿用b01。仅随类别到达构造文本。
TopK=min(5,K)，由原解析scores选择。
CSE1=s+mask*cos，α1（paper）。
CSE025=s+.25*mask*cos（代码0.8s+.2cos的argmax等价）。
不softmax、不乘CLIP logit_scale、不clip负余弦；TopK外仍为原s，并非硬删候选。
不采用作者代码初始W×.9，不混用作者普通样本ridge与本轮class-balanced目标。
RF1.CSEperm1只对sorted seen class IDs循环错位文本，α1、原候选集合。此项不是随机置换显著性检验。
不搜索更多alpha/topk/模板、不用历史val选template0。

## J. 训练特征访问与评价隔离

旧raw moments无法精确得到ReLU moments。禁止用ReLU(mean@R)、最终bank截列或伪样本冒充新的RF统计。
本轮每个parent/order是全新逻辑流，T1起顺序提取当期train；所有已准入映射在当期共同消费a/u，之后释放逐样本train临时缓存。
不同独立流可分别读原图，但单个流T2以后不得重读T1图像/逐样本特征。审计分账。
不要全量预提取所有train类供学习器任意读取。未来label/count元数据只能做已锁协议核验/排程，不参与提前调参。
生产fit/eval分进程；val caches/labels不可挂载给fit选择器。
全部已准入W/模型/文本/λ/R/score公式锁定后，才统一val。
已有val cache满足模型/变换/sample-id-hash/dtype/order全部一致才复用；否则评价阶段重提取。
每APART parent最多一次val提取、每固定VLM可以跨parent共享同一val特征；全部方法共享，不逐方法forward。
保存私有float64实值scores+真实样本列序与seen_ids；public只聚合。

## K. 生产测试、恢复与磁盘

跑reference 32测试，再额外完成真实loader、IO隔离、断点恢复、分组CV、分类顺序、hash/状态不漂移和资源门槛。
至少构造两类开始、后续含最后三类、长尾样本数不均衡的synthetic批量/递推一致性测试。
比较模型恢复和历史LOCK的W/可用score来源；缺原score只标未测，不能用报告三位小数当bitwise parity。
主Gram4096×4096 float64为128MiB，必须计入所有活跃矩阵/分解/备份，而非只数一个矩阵。
活跃流只保留新run的最新S/Q与一个回滚态；每stage永久存W/metadata/hash；每完整parent存最终S/Q。
严禁清理历史原实验。新临时回滚态只有新状态备份核验后可清理，路径必须限制本run。
基于实际文件容量预估1.25倍余量，至少5GiB自由空间；不足取消扩展，不覆盖旧文件。
错误退出要保留失败回执，恢复不能把同class统计加两次，也不能重置总时钟。

## L. 主分析与效用门槛

主随机投影固定67101。
逐dataset报告：G.RF1-G.LIN、B.RF1-B.LIN；同时RF1-F.RF1、RF1-F.LIN、B.RF1-G.RF1、F.RF1-F.LIN。
67102仅独立敏感性，全部原样报告；两投影均完成时另报均值，不能取max或implicit ensemble。

主utility gate：mean Final BA +1pp；2/3 orders提高；AvgBA_inc不降；final tail不降；final old/current均值损失各<=2pp、每order各<=5pp；所有阶段无新零召回类集合。
反映点估计门槛而非临床或独立显著性；BA/Macro-F1/AvgBA_all/AvgBA_inc/old/current/HM/tail/forgetting/每类n一并报告。
CSE统计true-class-topK coverage、correction/destruction、wrong-to-wrong、margin/语义扰动比例、topK外最终胜出，按类、old/current/tail分组。
按类分层component配对bootstrap 2000次seed68026，相同比较共用索引。固定parent与projection；多投影不是额外独立病例。
如组件跨标签，不拆分identity；明确整体component重采样处理和类别缺失限制。

只有完整有效比较才给UTILITY PASS/FAIL；其他是BLOCKED/INCOMPLETE_BUDGET/NOT_ADMITTED_RESOURCE。
不得整体FAIL掩盖ISIC正信号，也不得平均两个数据集掩盖HK负信号。

## M. 最终交付与停止

输出01第11节全部报告和日志，至少含：
FINAL_REPORT_ZH.md, FINAL_REPORT.json, METHOD_MATRIX.csv,
stage_metrics.csv, class_metrics.csv, summary_by_parent_and_projection.csv,
lambda_cv_scores.csv, lambda_selection.json, bootstrap_intervals.csv,
error_decomposition.csv, candidate_coverage.csv, projection_robustness.csv,
SOLVER_PARITY.json, ENGINEERING_REPORT.json,
PROTOCOL_LOCK.json, SOURCE_LOCK.json, RANDOM_MAP_LOCK.json,
ACCESS_LEDGER.json, RESOURCE_REPORT.json, BACKUP_REPORT.json,
completion_receipts.json, failure_receipts.json, NEXT_DECISION.json。

最终报告至少回答：RF是否超过公平调参LIN？VLM是否超过同容量F.RF？医学/G的差异是否改变？CSE的净贡献？两个projection是否一致？还有哪些缺失？
提供精确起止/墙钟、GPU型号及GPU进程驻留（不假装是纯计算）、CV/fit/forward数量、峰值显存/RAM/磁盘、备份核验范围。
源码、脱敏配置和聚合报告在新分支commit。仅在本轮有用户明确push授权时push，不能force-push。未授权则COMMIT_READY且STOP。
图像ID、私有路径、病灶/组件映射、逐样本scores/features、模型权重、S/Q/W/R不公开。
不得把论文PDF上传public。不能宣称未实际检查过的匿名访问成功。

全部完成或到达12小时即 NEXT_DECISION=STOP，清理本run监测器。
不启动ConCM、ACTM、神经微调、第二轮随机投影搜索或另一轮12小时。
