# NB2-RFVILA-12H-R1：随机非线性特征增强解析持续学习

## 0. 目标和来源边界

本轮以用户提供的 Zhao 等 VILA 论文（arXiv:2602.13670v3，2026-05-07）及项目提交 `b01ba12f86706c44dfcdbed81c7305493ce354dc` 为依据。

论文的主方法：正常首任务适配后冻结；分别归一化 Adapter 与 CLIP 分支；拼接后经过固定 Gaussian random buffer + ReLU；解析更新分类器；用候选语义增强 CSE 修正分数。见论文 p.3 式(1)/(2)、p.5 式(4)–(6)、p.14–15 算法。
本项目已完成归一化双视觉及线性类别均衡 ridge，本轮补上固定随机非线性映射。

**这是 VILA 风格的长尾医学适配，不是论文原始结果的严格复现。** 差异必须在报告列明：APART 1536 维而非论文 768 维；正常两类 Task1；BiomedCLIP 对照；4,096 维 buffer 而非论文 16,384；类别均衡目标；Task1 组件分组读出 CV 而非论文所述 LOOCV；医学模板；任务末统计解析求解。
不把论文关于 representation rigidity 的解释当作本项目已证明的唯一失败机制。

## 1. 研究问题与固定对比

H1：同一表征、相同 Task1 正则选择预算，RF 是否优于直接线性 ridge？
- 主比较：`G.RF1 − G.LIN`、`B.RF1 − B.LIN`，两个数据集分别裁决。

H2：RF 的收益是否只是通用非线性读出的收益，而不是 VLM 互补？
- `G.RF1 − F.RF1`、`B.RF1 − F.RF1`。
- 同时报告 `F.RF1 − F.LIN`。
- 差分中的差分：[m.RF1 − F.RF1] − [m.LIN − F.LIN]。这是固定系统差分，不自动证明因果交互。

H3：非线性融合下，医学模型是否改变原有的 ISIC/HK 差异？
- `B.RF1 − G.RF1`；对照 `B.LIN − G.LIN`。
- 不根据历史结果选择“ISIC 只用 B、HK 只用 G”。

H4：更好的候选解析空间是否让文本 CSE 有净收益？
- m.RF1.CSE1 − m.RF1；相应线性 CSE 对照。
- 固定 α=0.25 的代码风格灵敏度；不从两个 α 中选赢家。

H5（资源许可）：联合非线性响应、仅分支内非线性与线性随机重参数化有何差别？
- m.RF1 − m.SPLIT1；m.RF1 − m.RPLIN1。

随机投影 67101 是预定主分析；67102 是独立投影复验，不能替代失败的 67101。
同时完成两种投影时，另报告两个投影先在同 parent/order 内平均、再跨 parent 平均的稳健性结果；不做 logits ensemble，不把六个相关单元当六次独立训练。

## 2. 不变的协议

- ISIC：2+2+2+2，4 个任务、8 类。
- HK：2×10+3，11 个任务、23 类。
- 原 seed/order：1993/1994/1995。标签名、顺序、split、component、tail 不重建。
- 复用 6 份正常两类 Task1 APART 末状态及其共享依赖；禁止替换为大首阶段模型。
- APART 自 Task1 末冻结；G、B 始终冻结。随机投影也冻结。
- **新增神经 epochs=0，optimizer steps=0。** 随机特征拟合、CV 解析解不是神经训练，但要单独计数。
- 不访问 test/reserved。原 val 是开发评价集，所有已准入 W 和参数先锁定才释放。
- 本轮只用 G/B；不再尝试 DermLIP 授权或其他新模型。
- BiomedCLIP 预训练样本暴露继续标为 UNKNOWN，不声称“无污染”。

## 3. 固定模型与输入

G：当前锁定的 OpenCLIP ViT-B-16 / laion400m_e32；从原项目资产锁恢复，不重新从浮动 main 下载。
B：`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`，revision `9f341de24bfb00180f1b847274256e9b65a3a32e`，沿用已有文件 SHA256。
每个 VLM 使用自己的图像 encoder、配套 text encoder、tokenizer 和 native preprocess。B 使用已验收的 256-token 文本接口，不套用 G 的 77-token tokenizer。

APART main/few 分类前特征按已验证定义合为 1536 维并 L2 归一化，得到 a。
G/B 图像投影各 512 维，分别归一化为 u。若实际维度与锁不符，阻塞，不裁剪补零。
同一原始图像可解码一次，但各模型必须各用自己的转换，不强制共享预处理后的 tensor。
模型指纹在每个阶段结束核对，dropout/BN 处于 eval，所有 requires_grad=False。

## 4. 方法的完整数学定义

### 4.1 原始线性表征
F.LIN：q=a。
G.LIN / B.LIN：q=h=[a;u]/sqrt(2)。
没有额外的全数据白化、中心化、bias/intercept、特征选择或验证集驱动缩放。

### 4.2 主随机特征映射
m=4096，随机种子 r∈{67101,67102}。
用 NumPy PCG64/SeedSequence([r,0]) 生成 R_a∈R^(1536×m)，SeedSequence([r,1]) 生成 R_u∈R^(512×m)，独立标准 Gaussian 元素，float64。
保存矩阵实际文件与 SHA256，生产实现不得仅依赖“同一个种子”声称矩阵相同。
两个数据集、三个 parent/order 和 G/B 均复用该 r 的同一份矩阵；RP 与 parent seed 解耦。相同随机矩阵只控制随机映射差异，不假定不同 VLM 的第 j 个坐标具有相同语义。

F.RFr：phi_F=sqrt(2/m)*ReLU(a R_a)。
G/B.RFr：phi_J=sqrt(2/m)*ReLU((a R_a+u R_u)/sqrt(2))。

RF 后**不再次逐样本 L2 归一化**，无 bias，R 不学习，不按任务重采样。
这个尺度使单位长度输入的期望平方特征范数为 1（对随机矩阵的期望），不是声称每个样本恰好为 1。

### 4.3 扩展几何对照（均只用 RP1）
SPLIT1：sqrt(2/m)*concat(ReLU(a R_a[:,:m/2]), ReLU(u R_u[:,:m/2]))。
输出仍为 m，不能给两个分支各 m 维、无意让总容量翻倍。
RPLIN1：((a R_a+u R_u)/sqrt(2))/sqrt(m)，不使用 ReLU。
RPLIN 不能增加原输入的线性函数表达空间，但会改变有限样本的诱导正则；作为这种影响的诊断。
第一轮固定 4096，不追加 8192/16384 维搜索，优先把完整顺序与第二投影做完。

### 4.4 类别均衡解析目标（本轮权威定义）
对已见 K_t 类：
L_t(W)=(1/K_t)*sum_c [(1/n_c)*sum_i||phi_i W-e_c||²]+lambda||W||²。
S_t=sum_seen_c (Phi_c.T@Phi_c)/n_c。
Q_t 的第 c 列是 Phi_c.mean(axis=0)。
W_t=solve(S_t+K_t*lambda*I, Q_t)。

训练样本数 n_c 只在类别实际到达后进入学习器。每个类先完整累加再除以该类 n_c，不能按 batch 个数等权。
新类 Q 列按原 class order 追加；旧样本对新类 one-hot 列的贡献为 0。

只保存 S/Q、class counts/order、W、lambda、R hash 等充分统计和模型状态；这仍是解析持续学习，不是离线终端一次拟合。
**不要直接把固定 lambda 的普通 RLS 套进上述类均衡目标**：等价的未平均正则为 K_t*lambda，会变化。
本轮优先逐任务更新 S/Q 并 Cholesky solve，不构造显式逆矩阵；称为“递推充分统计＋任务末解析求解”。
如实现 Woodbury，必须先处理新增类带来的对角漂移 ΔK*lambda I，再按类样本权重更新，并通过小规模 batch equivalence。
“解析等价”仅指固定映射/目标/样本权重下的代数一致；不是旧类 recall 永不下降或所有未来目标可表示。

### 4.5 Task1 正则选择（所有新方法公平执行）
固定候选：[1e-6,1e-5,1e-4,5e-4,1e-3,2e-3,1e-2,1e-1,1.0]。
每个 dataset/parent/feature_kind/rp_seed 独立选一项，随后贯穿全部任务。
只使用正常 Task1 训练组件，最多 3 折；每类按 SHA256('67026|class_id|component') 排序组件，round-robin 分折。同组件不可跨折。
fold数=min(3,两类各自组件数最小值)。若小于2或组件跨标签无法合法分折，**该 parent 的全部方法**固定 lambda=1e-3，记录 TASK1_GROUP_CV_UNAVAILABLE，不用 val 代替。
每折用 support 类均衡目标（support 的 n_c）拟合，用 held-out 各类样本平均平方误差后再对类平均。折间等权。
分数并列绝对差≤1e-12时取更大的 lambda。九个候选全部保留日志，不追加边界搜索。
CSE 直接复用其父读出的 W/lambda，不另外选择正则。

可以在每折的 weighted design 上使用 primal/dual 较小系统，并复用一次谱分解评估整个 lambda grid。
不得为每个 lambda 重复大规模 eigendecomposition。正式时 float64，残差核验。
全量 Task1 特征在其任务活跃期暂存；CV 后用全部 Task1 数据重拟合并锁定，再进入 Task2。
该 CV 不是端到端独立验证：APART 父模型已经学习过 Task1。这一边界必须披露。

### 4.6 五个历史固定正则参照
F.LOCK1e3；G.LOCK5e4/G.LOCK1e3；B.LOCK5e4/B.LOCK1e3。
它们的输入均为原始线性 a/h，不经过 RF。分别对应 b01 的 F、G.J/G.J1、B.J/B.J1。
新生成线性 bank 可额外求解，也可复用来源严格一致的旧 W；计算相同的 λ 项去重但报告 ID 保留。
不把旧报告中四舍五入均值当作新配对分数。
这些参照说明 Task1 lambda selection 本身的影响，**主要 RF 比较仍使用同样获得 CV 的 LIN**。

### 4.7 CSE：固定两个来源定义，禁止选优
保留现有两条医学模板与已锁类名；沿用 b01 的原始文本 embedding 均值后单位归一化定义。
s=phi W，v=u T；K=min(5,已见类数)，按 s 取 Top-K。
CSE1：s'_c=s_c+1[c在TopK]*v_c（论文式(6)及其融合式）。
CSE025：s'_c=s_c+0.25*1[c在TopK]*v_c（本次核对作者代码0.8s+0.2v的argmax等价写法）。
不将 s softmax，不乘 CLIP learned logit_scale，不剪裁负余弦。Top-K 外保留原 s，不强制排除。
Top-K和argmax并列按 original class ID 由小到大打破；同样规则适用于全部方法。
这只是两个固定分数接口的对照，不是完整论文/代码复现；不复制作者初始分类器0.9缩放。

CSEperm1（扩展）：在已见 original class IDs 排序下循环错位一个位置的文本；候选集合仍由未修正的 s 产生，α=1。不能混入未来类文本。
不挑最好模板、不将历史 template-0 的较高分作为本轮选择依据。

## 5. 分层实验矩阵与覆盖数

T0 最小完整核心：
- F/G/B 的 LIN（3）。
- F/G/B 的 RF1（3）。
- G/B 的 LIN 和 RF1，各 CSE1/CSE025（8）。
- 五个 LOCK 历史参照（5）。
共 19 个输出方法×45 个 dataset/parent/task单元=855 阶段行、8721逐类行。
其中不含 LOCK 时核心14方法=630/6426行。

T1 预期完整目标：T0 + F/G/B RF2 与其 G/B 的两个 CSE（7）。
共26方法=1170阶段行、11934逐类行。

T2 扩展：G/B SPLIT1（2），再 G/B RPLIN1（2），再 G/B RF1.CSEperm1（2）。
全部完成32方法=1440阶段行、14688逐类行。
计数是逻辑报告记录，不是独立训练次数；实际解、CV解和alias去重要单列计数。

所有方法均为完整 ISIC/HK、三个原 parent/order。不能把HK8类前缀算完整23类。
在 T+2:00 前按实测资源预先准入一整个 tier（或其中列明的完整扩展组），之后不可按 val 或训练分类成绩选择。
优先级：T0 完整配对 > T1 第二随机映射 > SPLIT > RPLIN > CSEperm。
扩展不应使 T0/T1 的后续任务或报告被挤掉。未准入项标 NOT_ADMITTED_RESOURCE；中途被预算打断项标 INCOMPLETE_BUDGET，不填0，不宣布效果 FAIL。
无预算时不得把4096偷偷降到其他维度；交付阻塞/未完成清单。

## 6. 数据流与统计重建

旧 raw feature 的 S/Q **不能精确变成 ReLU 后 S/Q**。不得用 ReLU(mean@R) 或高斯合成冒充原样本的RF统计。
为本轮创建新的完整逻辑流：每个 parent/order 从正常 Task1 开始，按当前任务顺序读取原 train；所有准入映射在当期共同消费一次 a/u。
不需要重新训练 Task1，但需要当期特征重提取；必须如实计入forward/read。
每个任务只打开当前类图像。旧类图像或逐样本特征不能在后续任务重读；禁止先提取全训练集供各阶段自由使用。
不同 seed/order 是独立逻辑流，可分别读取同一原样本；不得把所有跨流读取误记为在线旧类回放。
当前 task 临时特征可存在仅当期可读的暂存区，完成全部当前统计/CV后释放；正式模型不保存逐样本 train 特征。
模型之外的验收/归档访问与学习访问分账。

验证缓存仅在全部拟合锁定后交给隔离 evaluator。每个固定 APART parent最多一次完整 val提取，VLM val特征可跨parent共享（相同encoder/预处理/图像哈希且无训练反馈）；已有缓存可复用但必须核对真实来源。
所有 W/CSE/投影共享这些 val特征；不按方法重复图像forward。按每阶段已见标签/候选列评估。
private scores保存的是解析实值分数，不能误称已经校准的概率。

## 7. 正确性与资源验收

已有参考测试只覆盖数学，不替代生产验收。生产至少验证：
1. 6个父状态、shared/core、原manifests/labels、G/B原生加载器及所有文件哈希。
2. 随机矩阵冻结且跨G/B、parent/order配对；RF能量与维度定义不漂移。
3. batch/recursive统计等价、Kλ缩放、不同最后任务大小、类别column padding。
4. 类内复制所有样本不改变类均衡目标；旧类不能重复append。
5. CV组件不跨折；无CV退回规则；全程lambda不重选。
6. CSE稀疏掩码、原class-ID ties、paper/code两个alpha、无未来文本。
7. Macro-F1不是BA；零召回按类集合，不只比较数量；无分母时null/N/A不补0。
8. 保存→恢复→继续的S/Q/W/R/计数/随机状态一致；中断不双重计入某类。
9. fit/eval进程隔离和旧/未来/test禁读反向测试。

编码器沿用已验证float32，TF32/AMP关闭；R、RF、S/Q、W、CV和求解使用float64。
建议Cholesky solve，优先在实测更快且有余量的CPU/GPU上执行。CUDA调用应同步计时。
残差 ||AW-Q||/max(||Q||,1e-30)≤1e-8；非有限值/负定异常不通过。禁止按需要自动加jitter改变λ。
每任务不需要完整eigendiagnostic；只记录足够的PSD/trace/规范残差，昂贵诊断在指定检查点执行。

m4096 的单个float64 Gram是128MiB；论文m16384是2GiB，不含分解工作区。
为避免保存每阶段稠密S造成空间爆炸：每个活跃流保存当前与一个回滚S/Q；每阶段永久保存W、R/lambda/类序、S/Q hash及收据；父序列完成后保留最终S/Q。
六个RF配置（F/G/B×2投影）×六个parent最终Gram约4.5GiB；linear、R、临时分解、扩展、备份另计。完整历史S非必需，本轮只保留新临时回滚副本，删除前必须有新状态的有效备份回执。
任何历史原实验资产、原图、父状态均不得删除。内存、磁盘实际准入需留1.25倍输出估计和至少5GiB空闲；不足先去掉扩展，不冒险覆盖。

## 8. my-gpu / remote-home / 备份

先发现SSH别名、remote-home真实路径、GPU分配、已有数据根、父状态、G/B资产和旧run缓存。
新分支：exp/nb2-rfvila-12h；新run_id，原b01和旧run只读。
不要写死/root/remote-home；不能重新搬整套数据；缺依赖只从已授权源增量复制并校验，不能从不明来源替代。

旧 backup_report 明确：独立NFS流式复制完成、回执可读，但未全量目标端rehash。
本轮新增计算前，对关键集合（6父及共享依赖、G/B权重/配置/tokenizer、锁、可复用W/R）做目标端逐文件SHA256比对，并恢复1个真实父状态和1个bank/W。
原图大体积副本仍按既有盘点记录，不为了声称“全量bitwise”耗尽预算；明确区分关键集合verified与全量image archive状态。
无可核实独立副本时，只做非破坏CPU/合成审计，输出 BLOCKED_BACKUP，不悄悄取消备份要求。
运行中每个阶段原子落盘、备份新状态、回执后才进入下一任务；同一物理存储另建目录不能当独立故障域。
备份故障可完成当期原子保存后暂停，不继续制造大量未备份新资产；总时限不重置。

## 9. 12小时墙钟执行表

0:00–0:45：发现路径和授权、时钟锁、输入依赖、关键备份复核。
0:45–1:30：新增runner、参考/生产测试、论文/代码差异清单。
1:30–2:00：只用固定Task1小样本和合成4096矩阵测吞吐；选择CPU/GPU求解后端、准入tier、锁定所有方法/参数。
2:00–7:30：按流完成Task1 grouped CV、全部任务特征→S/Q→W→备份；数据集中途不能按效果砍seed。
7:30–8:30：已准入拟合单元收尾、恢复/矩阵覆盖核对；此时不能新增未预定方法。
8:30–10:30：锁定参数与模型后统一val；scores、逐类、候选覆盖、纠错/破坏。
10:30–11:00：完成已有评分和bootstrap；不启动新图像forward。
11:00–12:00：结果审计、完整/缺失状态、报告、hash、备份、提交。

第一条本任务remote命令记T0，结束<=T0+43200。下载/重启/失败计时，monotonic与UTC双记录，掉线/重连不得改T0。
资格实验记录device型号、耗时、warmup；预测总时间采用实测×1.5安全因子，另保留1小时归档，不能用论文4090或上一轮多GPU时间推断本机一定足够。
T+8:30后不启动新的fit工作单元（含CV/新stage solve）；已运行单元只在预算允许下原子完成。
T+10:30后无新图像forward；T+11:00停止一切模型/解析/评分计算；到12小时必须结束。
控制器拒绝启动不能在截止前完成的工作；看门狗只能终止本run进程组，不能kill其他用户/项目GPU任务。
可每5分钟heartbeat、每小时本地资源汇总；不是创建无限期计划任务。退出时清理本run监测器，不恢复旧小时监测。
提前完成直接STOP，不空转、不追加搜索。

## 10. 评价与效用裁决

全部使用未四舍五入分数。报告Final BA、普通accuracy、Macro-F1、AvgBA_all、AvgBA_inc、old/current、HM、tail、逐类n/组件数、零召回集合、遗忘、old→new/new→old。
本文主指标是BA，不与VILA原文普通accuracy直接排名。AvgBA_inc排除Task1，与AvgBA_all分列，不混成同一列。

H1主候选各自相对对应LIN检查：
- Final BA平均≥+1pp；至少2/3 parent/order提高；
- AvgBA_inc非负；最终tail平均非负；
- 最终old/current平均损失各≤2pp，每个parent损失各≤5pp；
- 全部阶段均不新增相对对照的零召回类别（比较集合，不只数量）。
同表另报告RF vs F.RF、RF vs F.LIN，不把单一门槛替代所有机制分析。
两数据集和两VLM分别PASS/FAIL；不把ISIC正/HK负平均成一个“成功”。
FAIL=完整效用指标未达标；工程BLOCKED、预算INCOMPLETE、未准入NOT_RUN分开。

bootstrap：2000次、seed68026、按类分层identity-component配对重采样；相同比较使用同一重采样索引。
若component跨标签，采用整component抽样并明确方法，不把同component按标签拆开；保留每类是否有样本的诊断，不能伪造精确区间。
主区间固定RP1和3父；RP2单列。两RP均完成可给先按RP平均再按parent平均的条件区间，不重采样parent或把RP当独立病例。
报告3parent差异SD、两RP方向一致性，条件区间不是训练总体/临床置信区间，开发集反复使用且有多重比较。

CSE诊断：true-class Top-K coverage、correction/destruction、wrong→different-wrong、outside-topK跳转、原分数margin、语义扰动与margin比例，均按类及old/current/tail分组。
TopK外只是语义增量为0，负语义增量可能令外部候选获胜；不要误写成硬候选约束。
多类小分母特别标明。无可靠统计量的内容写NOT_AVAILABLE，不从hard predictions伪造logits。

## 11. 交付

公开或待公开：FINAL_REPORT_ZH.md / FINAL_REPORT.json，METHOD_MATRIX与缺失状态，stage/class_metrics，summary_by_parent_and_projection，lambda_cv_scores与选择（不含样本ID），bootstrap_intervals，error_decomposition，candidate_coverage，projection_robustness，SOLVER_PARITY，GENERIC/MEDICAL_MODEL_LOCK脱敏摘要，实际PROTOCOL/SOURCE/RANDOM_MAP_LOCK，access/resource/backup/completion/failure收据。
私有：图像/病灶/组件ID、路径、逐样本scores/features、模型权重、S/Q/W/R和实际含敏感路径配置。
别把PDF、权重或dataset上传到公开仓库。记录论文SHA和引用即可。
每阶段W包含lambda、实际Rhash、seen_ids、source/protocolhash；不足完整矩阵不能只输出COMPLETE。

默认只在新分支本地commit；有本轮明确push授权才推送脱敏源码与聚合报告，不force push/不覆盖旧分支。无法推送时报告COMMIT_READY，不伪称匿名访问成功。
最终 NEXT_DECISION=STOP，不启动ConCM/ACTM/第二轮训练。

## 12. 预定解释

RF胜LIN且RF-VLM胜F.RF：支持在当前固定输入上更有效地利用VLM信息。
F.RF也同幅提高而VLM额外差值小：更多是非线性读出的通用收益。
Joint RF胜Split且超出RPLIN：与跨分支非线性响应解释相容；不是严格因果证明。
只有重新选lambda的LIN提高：应归给正则选择，不能归给VILA random buffer。
医学B只有ISIC改善：保留模态依赖，不全局选医学模型。
RF有益而CSE无益：保留随机特征解析路线，文本机制仍未证实。
仅RP2有效：报告投影敏感性，不能换主种子。
全部无改善：这是有限维数、固定模型/目标下的负结果，不证明全部解析或VLM方法无效。
