# DFD-T4-P：现有方法的方向性特征蒸馏试验

版本：DFD-T4-P-v1；日期：2026-09-28。
状态：PROPOSED_NOT_RUN。本文件是实验设计，不代表已经执行、分配了服务器或验证了私有资产可恢复。

## 0. 决策摘要

先改现有方法，但不重复已完成的冻结参照和弱 FD 实验。本轮只验证：相同 Task3 父状态、相同 Task4 数据及训练配方下，保护旧类均值差异方向，是否比随机选择同样多的方向更有效，并能否改善已观察到的旧类保持—当前类学习取舍。

本轮名称为 DFD-T4-P，不占用已有 CT 编号，不覆盖现有 route_a/route_b 或历史实验目录。
只新增 D（旧类方向）与 R（同秩随机方向）两个训练条件。复用三个已完成的 T 读出参照：FD10_T=C3，FD1_T=B1T，FZ1=冻结正常 Task1 编码器。
不训练路线 B，不搜索 beta，不扩任务，不访问 test/reserved，不重新读取旧类 fit。

## 1. 已核实的研究背景与来源边界

### 1.1 已完成的对照，不列为待执行

CT4-F 已完成冻结正常 Task1 编码器的六类前缀参照。ISIC F1 的 prefix_terminal_BA 为 62.654%，C3−F1 为 +0.869pp，固定模型的条件性区间为 [-0.086,+1.925]；HK 终端 F1/P/C2/C3 分类预测相同。因此不能把相对弱无 FD 控制的全部改善归因于持续适配。[S2]

CT9-P 已在同一组 CT3-P Task3 父状态上，只将 Task4 的 FD 系数从 10 改为 1。HK B1T−C3 为 0pp；ISIC 为 +0.537pp，条件性区间 [-0.238,+1.370]，当前类 +2.624pp、旧类 -0.159pp。它不是 beta=1 从 Task2 开始的全程训练。[S3]

CT10-O 更正说明：真实 CE 网络使用原 batchwise_prompt=True；FD 和解析 eval probe 都使用 pointwise 路由。不要把 batch=48 误写成批次路由，也不要自动“修复”原 SUM/MEAN、assignment、pull 或路由。[S4]

CT11-H 已有同模型神经头/解析头诊断，不能再把换成神经头作为本轮救援措施。[S5]

### 1.2 论文只提供动机

CLTR（上传的 arXiv:2306.13275v2）以完整数据集的频次排序构造头到尾课程，使用现有 CL 方法。它不直接提出本文件的方向 FD，也不证明本方法必然改善 BA。本轮不重排外部任务，不把旧/新类别当作头/尾类别。[S7]

### 1.3 本轮的证据上限

即便 D 成功，也只能支持“条件于既有 beta10 前缀、Task4 一次干预”的开发性证据。ISIC 的 Task4 已覆盖 8 类，但不是 D 从首次增量开始的完整方法训练；HK 只覆盖每个原顺序各自前 8/23 类。
保留所有有效旧类均值方向的投影，也不是类别重加权的独立实验。满秩跨度在很多正权重重加权下不会改变，不能宣称本轮已验证“保护上的类别均衡”。

## 2. 固定问题与实验矩阵

研究问题 H1：D 相对同谱、同秩 R 是否改善 Task4 all-seen BA？
辅助问题 H2：D 是否同时超出简单弱 FD 的效用，并在接近原强 FD 的旧类保持下恢复当前类学习？
辅助问题 H3：D 是否提供超出 FZ1 的实际持续适配收益？

固定数据集：HK、ISIC；seed/order：1993、1994、1995；仅 Task4。
所有新条件从每个数据集/顺序对应的、已核实 CT3-P FD Task3 末父状态分叉。
严禁 D 从原 U 父状态起步、R 从 F 父状态起步，或混用 Task1 冻结父状态。

| 新标识 | 历史标识/来源 | Task4 保持方式 | 本轮训练 |
|---|---|---|---|
| FZ1 | CT5-F 的 Task4 F1，来源锁须在 P0 解析 | Task1 后固定编码器，合法增量统计 | 0，复用 |
| FD10_T | CT6-F 的 Task4 C3 | 10 × 全空间 FD | 0，复用 |
| FD1_T | CT9-P 的 Task4 B1T | 1 × 全空间 FD | 0，复用 |
| D_T | 新 D 状态 + 固定 T 读出 | 旧类方向强保护，其余弱保护 | 每个 case 10 epoch |
| R_T | 新 R 状态 + 固定 T 读出 | 同秩随机方向强保护，其余弱保护 | 每个 case 10 epoch |

主机制比较：D_T−R_T。辅助比较固定为 D_T−FD1_T、D_T−FD10_T、D_T−FZ1；另报告 R_T−FD1_T 以区分一般各向异性效应。
本轮只评价 T 读出，不在 A/T 之间择优。为兼容原 checkpoint，底层可以继续维护原 A 状态，但 A 不构造 Q、不作为新成绩、不反馈本轮训练。

新增正式训练：2 条件 × 2 数据集 × 3 顺序 = 12 条 Task4 轨迹。
新增 task-epochs：120；预期 optimizer steps：2 × 2750 = 5500；新增终态 checkpoint：12。
最终 T 表：30 行阶段指标、240 行逐类指标，其中新增 12/96，复用 18/144。
2750 来自原 CT9 固定六个 Task4 的步数总和；P0 必须从真实任务计数、batch/drop_last/epoch 配方复核，不可硬改计数凑表。

## 3. 方法的精确定义

### 3.1 特征空间

沿用 tools/ct3p_core.py 的 features：学生和冻结教师分别输出 pointwise 的 pre_logits 与 pre_logits_few，先拼接，再整体 L2 归一化。维度应为 1536，若实际不符先 BLOCKED，不擅自改变空间。

令每个样本的行向量差为 delta = J_student(x) − stopgrad(J_teacher(x))。
使用同一真实训练增强张量 x，不为 FD 再抽一次增强。学生路径可微，教师 eval/no_grad。

### 3.2 旧类方向 Q_D

在 Task4 正式更新前，从该 Task3 父状态的合法 T bank 读取 mu，预期形状 (1536,6)，对应六个已经到达的旧类。必须核实维度、标签映射、space_version、父网络 SHA、统计 SHA。
mu 是历史合法统计及既有输运的估计，不是旧图像在当前教师下重提取的“真实均值”。不得用 Q11/oracle 替换。
不要再将每列 mu 归一化。按类别中心化：

    mean_mu = mean(mu, axis=1, keepdims=True)
    M = (mu - mean_mu) / sqrt(6)

在 CPU float64 上做 thin SVD：M = U diag(s) V^T。
保留 s_i > max(1e-12, 1e-8*s_max) 的全部方向；Q_D = U[:, kept]。
数值有效秩 r 应在 [1,5]。r=0、非有限值或维度异常 => BLOCKED_SUBSPACE；r<5 如实记录，不补随机列，也不改为输出头方向。
不按 validation 调秩，不用累计能量阈值网格，不按特征值再加权。
Q_D 在整个 Task4 内固定；即使中间探针产生新的 bank，也不得重新估计 Q_D。

### 3.3 随机方向 Q_R

Q_R 与 Q_D 同维度、同秩，使用独立的局部 NumPy PCG64 Generator，不消耗任何训练/loader/augmentation/synthesis RNG。
种子固定算法：SHA256(UTF-8("DFD-T4-P-v1|random|{dataset}|{seed}|4")) 的前 8 个 digest bytes 按 little-endian 无符号整数解释。
dataset 的规范字符串只用 HK 或 ISIC，不用机器路径。
从标准正态生成 (1536,r) 矩阵，float64 reduced QR；将 R 对角线符号规范为非负（0 按 +1），相应调整 Q 列。
不将 Q_R 刻意正交化到 Q_D 的补空间，不试多个随机种子择优/择劣。保存 Q_R、随机种子、Q_D 与 Q_R 的 SHA、正交误差及主角重叠统计。

### 3.4 固定损失

    L_D = L_real + mean(||delta||^2 + 9 * ||delta @ Q_D||^2)
    L_R = L_real + mean(||delta||^2 + 9 * ||delta @ Q_R||^2)

等价于 beta_parallel=10、beta_perp=1。
D/R 的特征空间惩罚谱相同：r 个特征值为 10，其余 1536-r 个为 1，trace 都是 1536+9r。
这是特征空间度量匹配，不是参数梯度、AdamW 位移或实际保持效果匹配；须记录这些量，不能用同 trace 代替实际测量。
FD1_T 是 floor 相同的弱正则参照，不是严格 trace-matched 控制。D−R 是隔离方向选择的主比较。

实现必须用上面的加法形式，避免计算 total−parallel 时的数值负残差。不得再除以 1536、r 或 3，不做 trace 归一化，不增加 gate、置信度、margin、随机 mask 或新类别权重。

## 4. 保持不变的训练和读出

真实 CE 的 all-seen 范围、三项 CE、main/sum mean 与 few/assignment sum、/3、assignment 的原定切换、负 pull 项、路由及采样均不变。[S4]
继承实际入口的 AdamW：pool lr=0.0003，其他可训练参数 lr=0.003，wd=0.01，batch=48；每任务重建 optimizer，10 epoch cosine 到 1e-5。配置 scheduler 文本不替代真实入口。
不启用 AMP/TF32，不改变 backbone 权重，不加参数，不冻结额外 keys，不把推理 Linear 补丁用于 backward。
训练 CE 网络保持其原路由；FD/probe 使用 pointwise。每次 FD/诊断恢复 module.training、pool.batchwise_prompt、adapt_list、RNG 等原行为。

Task4 教师为该 case 对应 CT3-P Task3 模型的独立冻结拷贝，不复用可能被 extract 同步的 probe。
D/R 分别完整恢复同一父状态、训练 RNG、loader RNG、synthesis RNG，进入相同 task_setup。专用 Q 随机数不得改变这组状态。

保持原 T 计算、完整 S/mu/v/e 输运和 class-balanced ridge(lambda=0.001)。旧类均值不再次归一化。Q 只读 task-start 的 T 快照，不改变 bank 数值。
新增 Q 对训练的影响是本次明确的干预，不得回写或重新归因 CT3-P 结果。

## 5. P0：资产、源码与工程准入

### 5.1 先检查真实可访问的资产

需要：六个 Task3 父 checkpoint；原 AugReg 同 SHA 共享权重；每父原配置/代码锁/协议锁；Task3 合法 T bank；当期 Task4 fit；原 validation manifest/identity components；三个历史参照的 Task4 sealed scores 与锁；对应 Task3/更早的锁定逐类结果。

从 CT9 锁追溯 CT6/CT5 的真实 private 资产引用和路径，不从公开聚合数“还原”逐样本预测。
当前服务器/归档可用性须实际查验。历史报告中的地址、PID、临时路径不视为现成可用资源。[S8]
若父 checkpoint、必要 bank、权重或配对参照缺失，完成能做的代码/合成测试后输出 BLOCKED_ASSET 及精确缺项，停止正式训练；不自动重训 Task1–3、不换数据协议、不新租服务器。

### 5.2 实验锁与 fork

创建独立 worktree/branch；保留 dirty changes，不 reset/clean，不覆盖原报告。
建议基于 CT9 的已核实实现作小范围扩展，科学依赖逐文件绑定原 hash；把新 worker/Q 代码及 design 另列。
普通父恢复保留原 CT3-P 的所有源码、参数、RNG、共享权重和协议检查。先原样 restore，后显式 fork 成 D/R。
不能通过修改旧 checkpoint 的 beta/source/protocol 字段绕过恢复。新 checkpoint schema 记录自己的 method/Q/coefficients/design_hash/source_hash 和父谱系。
执行时生成 PROTOCOL_LOCK.json、SOURCE_LOCK.json、ASSET_LOCK.json、CONTROL_COMPATIBILITY.json，冻结后才能开始正式更新。

### 5.3 必须通过的工程检查

CPU 合成检查：Q^TQ=I；两种投影表达式一致；空 Q 恢复 beta1、Q=I 恢复 beta10；beta_parallel=beta_perp 时恢复对应全空间 FD；特征梯度与解析式一致；D/R 的秩/trace/谱一致；公共平移不改变中心化旧类跨度。
float64 合成数学检查建议 atol/rtol=1e-10；float32 Q 正交误差 max_abs <=1e-5；真实 FP32 旧 FD 回归采用 atol=1e-6、rtol=1e-5。原 ridge/恢复容忍保留原值，不能为通过而放宽。

在合法 Task4 current-fit 小批次上核实：学生 FD 可微且实际可到达 adapter；教师独立、不变、无梯度；完整 real loss 保持原行为；FD 前后 flags/RNG 完整恢复；D/R batch ID 与增强 checksum 对齐。
验证新状态 roundtrip、中断恢复及错误 method/Q hash/coefficients/source/protocol 拒绝。工程可用最多 16 个隔离 optimizer steps，全部计入工程账并销毁其派生状态；正式训练必须重新从原父分叉。
初始学生=教师导致第一 batch FD 和 FD 梯度为 0 是预期，不应误判实现失效；非零梯度回归可用合成扰动或隔离工程更新。

所有六个 case 的准入与全矩阵资源预估通过后才训练。不得先看一个 seed 的效果再决定是否跑其余 seed。

## 6. P1：固定 Task4 训练

单 GPU 串行执行 12 个单元。固定顺序：dataset 按 ISIC、HK；seed 1993、1994、1995；每个 case 按 D、R。顺序只是调度，任何新 val 均不能提前释放。
每单元固定 10 epoch；不 early-stop，不选 best epoch，不搜参数。只保存最后一个科学终态，并用 rolling checkpoint 支持同锁恢复。
不得从 D 的训练终态继续训练 R，也不得将 D/R 结果作为新的教师互相传递。

建议资源边界：本次总 GPU 进程驻留 <=6 小时（工程、训练、评估及失败尝试均计），新增持久/归档资产 <=3 GiB，新增本地活跃/临时 <=2 GiB，最低剩余空间 >=1 GiB，CPU 后处理 <=2 小时；线程沿用 torch/BLAS 4、loader 4。
这是一组本轮建议预算，不是历史实测运行承诺。以 P0 的实际硬件吞吐、内存与可恢复归档测试判断能否容纳全矩阵。
不为预算缩短 epoch、改 batch 或改精度。预算或资源不满足时 BLOCKED_RESOURCE 并保存进度，不启动未准入单元；中断只能在同设计/同锁下恢复，不换 seed 或参数。
旧的周期性监测授权不自动扩展为新任务；本轮不另建 cron/周期任务，普通训练心跳与资源日志可写。

## 7. 诊断量：解释保持与学习，不能用于在线调参

每 epoch 首 batch、main/few adapter 分组记录：完整 real 梯度范数、加权 FD 梯度范数、二者 cosine、AdamW 实际更新范数、parallel/perpendicular 的特征位移能量。
同时记录 Q 的 rank、谱、hash；真实 loss 各分项及 reconstruction residual；每个任务的累计参数变化和教师 fingerprint。
若分母 real norm <=1e-12，梯度比记 NA，保留两个绝对范数，不把极大比值当作自动减弱 FD 的依据。
所有 probe 必须不污染 .grad、optimizer、RNG、flags；120 个 epoch 首 batch 记录是探针，不是全轨迹梯度统计。
负 total loss 可能来自原 pull 项，单凭符号不判失败；非有限值或损失账不一致才按工程错误处理。
允许聚合当前 fit 的类别级 loss/能量诊断，但不重读旧 fit，不测新的 Q11/oracle。
不能将当前输入上的漂移减少表述为旧类真实漂移已经减少；也不能把参数变化小表述为泛化改善。

## 8. P2：锁定后评价

所有 12 个终态、T bank/W、训练与访问账本锁定后，再统一释放本轮新 val 预测和结果。不要按 val 选 epoch、方法、rank、随机种子、阈值或读出。
参照的逐样本分数只能用于兼容性核查和固定配对评价，不能反馈新方法构造。

主指标：Task4 all-seen BA；每个数据集单独汇总 3 个原 order/train-seed，不把两个数据集混为一个总体。
必须报告：old/current recall、HM、tail recall、旧尾/当前尾、头中尾分组及其有效分母；每类 correct/n/components、所有零召回；最差 seed；每个配对的差值。
频次分组沿用原锁定定义，仅报告本阶段已见类。没有有效类别时记 NA，不当作 0。中频类别不可强并入头类。未来类别信息不得进入学习。
沿用 all-seen、old/current restricted 和 old→current/current→old 错误分解。restricted 只是额外诊断，不替代主 CIL 成绩，也不单独证明是表示层原因。
首次学习质量与后续变化用既有到达时锁定结果和当前新结果报告；Task4 单任务干预不等于修复此前没有学会的类别。

区间：每个数据集按原 CT9 已验证的 class/component 配对 bootstrap，2000 次，新种子 64201；同一批抽样权重用于所有方法，保留跨顺序共享样本的相关性。不重抽训练 seed，不把 12 轨迹/类别当独立重复；完整保留 singleton 和多次 validation 开发限制。
这些是条件于已训练模型的开发性区间，未作多重比较校正，不是训练总体的确认性置信区间。

## 9. 预设继续/停止规则

全部 pp 指百分点，全部差值先在同 dataset/seed/layout 配对后取三顺序平均。
以下是本计划的资源决策阈值，不是论文参数、功效分析或临床重要性界限。不可在结果出来后修改。

### 9.1 每个数据集独立判定

机制门 M：D−R 的 BA >=+1.0pp，且至少 2/3 个配对 BA >0。
若机制门通过但 D−R 的条件性 95% 区间下界 <=0，记 MECHANISM_TENTATIVE；下界 >0 也只记开发性支持，不写独立确认。

弱 FD 效用门 U1：D−FD1 的 BA >=+0.5pp，old >=0pp，current >=-1.0pp。
原强 FD 取舍门 U2：D−FD10 的 current >=+2.0pp，old >=-1.0pp，BA >=0pp。
冻结参照门 U3：D−FZ1 的 BA >=+1.0pp，HM >=0pp。
安全门 S：每个 seed 的 D−FD1 BA >=-2.0pp；若 tail 有有效分母，D−FD10 的平均 tail >=-1.0pp；对 val n>=10 且 identity components>=5 的类别，不得出现 FD10 非零而 D 新变为零召回。
所有小分母零召回照常报告，不能因未纳入 S 就忽略。

### 9.2 决策解释

M、U1、U2、U3、S 全通过：REVIEW_FULL_HORIZON（只建议为该数据集另写全程计划；本轮停止，不自动开跑）。
效用通过但 M 不通过：REGULARIZATION_EFFECT_ONLY；不能主张旧类方向选择有效，不将这项模块直接包装成独立方法。
M 通过但任一效用/安全门不通过：MECHANISM_SIGNAL_NO_PROMOTION；说明局部方向信号不足以形成可用改善。
主要效用与机制均未通过：CLOSE_DFD_T4_PILOT；只关闭该候选和该条件实验，不据此否定全部子空间方法。
任何资产/恢复/访问/数值错误：BLOCKED，不能以科学负结果结案。

跨数据集不互相抵消：ISIC 通过、HK 未通过时，只能为 ISIC 建议后续全程验证，不能宣布两数据集方法成立。
不因未通过而追加 beta、rank、random seed、A/T、新损失、路线 B 训练或 holdout 访问。

## 10. 路线 B 的触发边界

路线 A 成功：下一步应先验证从正常 Task1 后开始持续使用该规则的完整序列，仍保留 random/weak/frozen 对照，独立锁协议。
路线 A 失败：先按 M/U1/U2/U3/S 及诊断分类，不自动认定“类别均衡软子空间保护一定更好”。
只有当已保存证据提示最终特征均值不足以代表旧知识、或当前输入上的 FD 覆盖受限，才为路线 B 另写只读设计与预算。
路线 B 需要层级 classwise 未中心化二阶矩，不得冒充现有最终特征 S/mu 就是各层 R_c。缺少历史层级统计时，应另立从数据合法到达开始收集的实验，不能回读旧 fit 补齐。
路线 B 的“学习均衡 × 保护均衡”2×2 矩阵、层选择、统计陈旧问题和与既有软投影方法的差异，均需另行确定；本 prompt 不授权执行。

## 11. 交付与恢复

必须交付：
- FINAL_REPORT_ZH.md：固定所有正负结果、效用/机制结论、条件性区间、证据限制、NEXT_DECISION=STOP。
- PROTOCOL_LOCK.json、SOURCE_LOCK.json、ASSET_LOCK.json、CONTROL_COMPATIBILITY.json、FORK_AUDIT.jsonl。
- SUBSPACE_AUDIT.json、ENGINEERING_GATE.json、RESOURCE_LEDGER.json、ACCESS_AUDIT.json。
- STATE_W_LOCK.json、PREDICTIONS_LOCK.json、CHECKPOINT_MANIFEST.json、DELIVERY_AUDIT.json。
- task4_metrics.csv（30行）、task4_per_class.csv（240行）、paired_differences.csv、conditional_intervals.csv、decision_gates.json。
- epoch_records.jsonl（120条科学 epoch 记录）、gradient_probes.jsonl（120条 epoch 首 batch 记录；可将 main/few 放在同一条）。
- 12 个可恢复终态 checkpoint；Q、W、私有 sealed scores 的可读性和 SHA 验证。

聚合结果与完整 manifest 必须逐单元增量同步到持久存储，并在最终退出前从另一端重新读取校验，避免只留在租期服务器。原始图像、样本身份、特征、W、逐样本分数和 checkpoint 不公开；公开发布需遵循已有授权，没有授权不新建公开分享或自动 push。
中断保留真实已执行更新、图像访问和资源账；不得重跑已完成单元，也不能将未知计数写为零。

## 12. 来源索引（历史事实与新设计分开）

[S1] CT3-P，提交 6f6e15fb63bb0ac4b4034d14f4ef0c5a16d3fb34，docs/ct3p_memory_compatibility/FINAL_REPORT_ZH.md 与 EXECUTION_PROMPT.txt。
[S2] CT4-F，提交 46c372794195ced17fe138328497b85b7d75c83e，docs/ct4f_task1_frozen/FINAL_REPORT_ZH.md。
[S3] CT9-P，提交 082302817113a385922cbc4847fe9c2a6e13f226，docs/ct9p_fd_perturbation/FINAL_REPORT_ZH.md；其锁引用的 CT6-F、CT5-F 是 Task4 控制资产来源。
[S4] CT10-O，更正提交 9c60f601ad687070be8177cdb1b453ff2df16208，docs/ct10o_objective_accounting/FINAL_REPORT_ZH.md、ROUTING_CORRECTION.md。
[S5] CT11-H，提交 915bd2d96d02f6c5cb62254a040b6b713ff64ef4 的固定头诊断交付。
[S6] CT9 提交下 tools/ct1_statistics.py、tools/ct3p_core.py、tools/run_ct9p.py；源码实际执行口径优先于配置中的未使用字符串。
[S7] Molahasani, Greenspan, Etemad，On The Relationship Between Continual Learning and Long-Tailed Recognition，arXiv:2306.13275v2，2026-01-30；上传 PDF 第23页 Algorithm A1 与第24页附录 F.1。
[S8] CT13 资产可用性风险记录，提交 cb763ad075474e2b20a57771e8341d9f1fad6504，docs/CT13_ISIC_RESULT_REPORT_ZH.md。该报告不用于本轮效果对照，只用于提醒持久交付与服务器到期风险。

来源入口为仓库 DLwbm123/Long-tailed-CL 的上述固定提交，不以 main 的旧摘要替代各实验分支。方法公式、随机数规则、资源预算、决策门槛和本轮矩阵均为本文件的新设计，不是已有实验结果。
