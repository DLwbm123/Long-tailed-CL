# NB-RL-A1：无独立基础阶段的 RL 辅助训练试验

版本：2026-09-30-v1。状态：DESIGN_NOT_RUN / ASSET_UNVERIFIED。
目标时长：12–16 小时；单卡；墙钟与累计 GPU-process residence 均不超过 16 小时。
本文件是实验设计，不代表已启动训练、获得服务器或私有资产访问权限、安排监测或获得推送授权。

## 1. 研究问题与证据边界

主问题：在正常 Task1 起步、无旧图像回放的 ISIC 类别增量流中，向统一的“解析引导 CE + 弱 FD”训练接口加入真实的 on-policy 类别动作策略梯度，是否提高完整序列的最终类别均衡性能？

本轮取 RaPO 的保持奖励与跨任务奖励尺度平滑思想，做 categorical contextual-bandit adaptation；不是 Qwen2-VL 生成式 RaPO 的忠实复现，也不检验语言推理能力。保留监督 CE，检验辅助 RL 的增量效用，不声称以纯 RFT 替换 SFT。

单步类别动作缺少同一正确答案对应不同文本轨迹的多样性。因此，本轮负结果只关闭本具体迁移，不能否定原论文的文本轨迹机制或所有 RL 方法。

区分三个结论：
1. R 相对 S/H/K/F_S/F_R 有效：保持奖励/策略优化辅助训练具有开发性效用。
2. R 相对 G 有效：支持保持奖励这一组合干预，而不是单独验证 CTAN。
3. R 相对 E 有额外收益：才有采样式策略梯度优于无采样期望优化的初步证据。E 本身也是奖励目标优化，不应笼统称为完全不含 RL 思想。

## 2. 参考论文与采用范围

[P1] 用户上传 Lou 等《Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning》，arXiv:2605.09640v1。
- 第4–5页，公式(2)–(6)：动作/轨迹相关的截断 log-ratio、detach 的指数保持奖励、组中心化、跨任务 EMA 尺度。
- 第6页：n=8, alpha=20, lambda=0.5, beta=0.999；保持奖励从 Task2 启用。
- 第7–8页：原分类实验为 Qwen2-VL-2B、每类5样本；不是本项目的自然长尾协议。
- 第10–11页：理论针对理想化 detached surrogate，不是正式无遗忘保证。

[P2] 用户上传 Luo 等《RLAP-CLIP》。
- 第4–7页：样本权重构造原型、分类间隔奖励、双模态提示、MoE；主实验每类20个 exemplars。
- 第25页：零 exemplar 时关闭 exemplar-dependent 组件，退化为 CLIP 零样本基线。
- 本轮不加入 exemplar、MoE、原型权重策略或间隔奖励，不把全部组件增益归因于 RL。

[P3] 用户上传 Lu 等《QPrompt-R1》。
- 第2、5–7页：GRPO-inspired 的组相对 query 监督，argmax 相似原型分组，分割损失与 GRQA 共同训练。
- 第7页：先监督训练前2/3，再 GRQA 后1/3；这不是本项目的增量协议。
- 只借鉴训练期辅助、推理期不增加组件的部署方式；不移植分割查询架构，不将确定性相似度分组自动视为 on-policy RL。

[P4] 项目固定源码：DLwbm123/Long-tailed-CL，152ea12，tools/ct3p_core.py 与 tools/ct1_statistics.py。
- 可复用 pointwise 联合特征、FD、公共平移、合法统计追加、class-balanced ridge。
- 原解析 W 不直接反馈原神经 CE。本轮新增共同的解析引导训练接口；S/H/K/G/R/E 全部使用它，不能将这一共同改变的收益归因于 RL。

## 3. 数据与任务锁

- 主数据集只有 ISIC；完整任务 2+2+2+2，原始顺序1993/1994/1995。
- 沿用已核实的医学数据划分与身份组件；不得换成旧4+2+2、扩大 Task1、重排头尾类、读未来 fit、访问 test/reserved。
- 从锁定的外部 AugReg 预训练权重与统一的未训练 APART 特征模块初始化开始。不得从历史 C-S0、Task3、DFD 或 NB2 适配后 checkpoint 开始。
- 每个方法从 Task1 就使用自己的预定目标；R/G 在 Task1 即使用准确率策略梯度，只有保持奖励在 Task2 才开启。
- 每个任务使用全部合法 fit，自然频次抽样，不截断头类、不将长尾改为均衡5-shot。
- 类别权重只使用已经到达的当前任务计数：w_c=N_t/(K_new*n_c)。损失按 batch size 求均值，不按当前 batch 的权重和重新归一化。
- 所有方法使用相同真实样本序列、增强种子、任务步数；策略采样使用独立局部 RNG，不能消耗 loader/增强 RNG。
- 本轮不增加 HK 实验；不得在看见 ISIC 结果后临时选择 HK 类别或顺序。

## 4. 共同模型与解析训练接口

### 4.1 表征

J_theta(x) 为现有 main/few pre_logits 拼接后整体 L2 归一化的1536维向量。
使用 pointwise、无随机 dropout 的特征路径进行全部训练与评价；eval mode 不等于 no_grad。学生路径保留 autograd，教师 no_grad。

原神经 main/few 分类头冻结并不用于本轮训练目标。核心视觉预训练权重冻结；共同的可训练特征模块及参数清单在 P0 锁定。只扩张解析分类器的已见类列数，不新增或重置任务专属特征模块。若现有入口必须调用会改变特征映射的 task_setup，则该接口必须先被显式审计；不能隐式新建模块或把解析记忆放入未经核实的新空间。

这是新的、所有条件共享的训练接口，不是对历史实现的静默修复。不得把新 S 称为历史 CT6/CT9 的逐位复现。

### 4.2 每任务开始

1. 冻结当前任务起点的特征网络 theta_anchor；Task2以后继承该条件自己的上一任务末网络。
2. 只在当前 fit 提取起点特征 J_anchor。旧 bank 已处在对应上一任务末坐标；要求任务切换不擅自改变特征映射。
3. 复制旧 bank，临时追加当前类起点统计，求 W_start。该 bootstrap 副本不提交为最终记忆。
4. W_start 在整个任务内固定，不反传 through solve、不用 val 选择、不按 epoch 重新拟合。

定义 p_theta(c|x)=softmax(J_theta(x) W_start / tau)_c，tau=1；候选集为全部已见类。
锚点 q_t(c|x)=softmax(J_anchor(x) W_start / tau)_c。

q_t 使用“上一任务表征 + 当前已见类公共解析头”，不是上一任务旧输出头未经扩张的原概率分布。这是本迁移的明确修改，使新类具有合法非零支持；不能用零概率旧头作分母，也不能人为屏蔽旧类。

### 4.3 每任务结束

- 冻结学生与最终参数。
- 在当前 fit 的相同确定性预处理下取得终点特征；与起点特征按样本身份配对。
- 按原公共平移 T 规则从当前类估计漂移，只对旧 bank 快照输运一次，然后只追加一次当前类终点统计。
- 不对旧类均值再归一化，不反复输运累计同一漂移，不把 bootstrap 当前统计与终点当前统计重复计入。
- 使用原 class-balanced ridge，lambda=1e-3，float64，求 W_final；这是主评价头。
- 保留 W_start 作为训练接口/重拟合差异诊断，不选择两者中成绩较好的头作为主结果。
- 起点/终点逐样本训练特征仅当前任务内临时存在；离开任务前释放，不作为旧特征 replay。

两个固定表征参照分别继承各自正常 Task1 的网络及 bank：F_S 来自 S，F_R 来自 R。随后只随数据到达累计统计、求解析分类器，不更新神经参数，也不另行训练基础阶段。

R 与 S 从 Task1 就使用不同目标，所以仅比较 R 与 F_S，不能隔离“RL在Task1学习得更好”与“后续持续RL有效”。必须比较 R 与其自身 Task1 后冻结的 F_R。两个冻结条件都不得从历史S0/Task1模型替代。

## 5. 真实策略采样与奖励

对每个当前样本，在采集参数 theta_k 的同一个可微前向中计算 p；pb=stop_gradient(p)。

G=8；独立有放回采样 a_g ~ Categorical(pb)。8个动作只是同一概率向量的8次抽样，不是8次图像编码或8次反向。

任务奖励 r_acc(c)=1[c=y]。
Task2起，d(c)=max(log pb(c)-log q_t(c),0)，r_ret(c)=exp(-20*d(c))。
R/E 使用 r(c)=r_acc(c)+0.5*r_ret(c)；G 使用 r_acc。
所有 reward、anchor、baseline、normalizer 都 detach。单动作 log-ratio 的正部只是漂移代理，不是精确KL，也不是真实旧类召回。

禁止用对同一图像所有动作都一样的完整 KL 标量充当组内差异奖励；这样的常数会被组中心化消掉。单独KL对照按全类别求和计算，不能用 argmax 项冒充完整KL。

### 5.1 组相对策略梯度

采用 leave-one-out baseline：b_g=(sum_h r(a_h)-r(a_g))/(G-1)。
A_g=(r(a_g)-b_g)/s_prev。
L_PG=-(1/(B*G))*sum_i sum_g w_yi*A_ig*log p_theta(a_ig|x_i)。

每个真实 minibatch只做一次 optimizer step，随后丢弃动作。没有 PPO多轮复用；不添加恒为1、实际不起作用的伪 importance ratio 或 clipping。
本实现应称“RaPO-inspired on-policy group policy-gradient auxiliary”，不能称原版 GRPO/RaPO 复现。

### 5.2 CTAN-inspired尺度（明确的离散化变体）

- 每条轨迹 hat_sigma 初始0.5，beta=0.999，跨任务持久化。
- 当前更新使用旧状态 s_prev=max(hat_sigma,0.05)，避免分母依赖本次随机动作。
- 离散类别至多8类，计算准确的 batch 奖励二阶矩：
  mu_batch=mean_i sum_c pb_i(c)*r_i(c)
  var_batch=max(mean_i sum_c pb_i(c)*r_i(c)^2-mu_batch^2,0)
- 完成本次更新后，以采集时已冻结的上述矩更新：hat_sigma <- 0.999*hat_sigma+0.001*sqrt(var_batch)。
- Task切换、断点恢复不得重置 hat_sigma。

原论文用采样奖励批次标准差、先更新EMA，并用组内均值。这里为了可验证的精确期望对照，使用滞后一拍的尺度、可枚举矩和 leave-one-out 基线。这三点全部属于本实验修改，不能追写为原文。

类权重 w 在优势归一化之外作用；不把同一图像所有 reward 同乘一个类别权重，再声称组内标准化后保留了同样的长尾修正。

## 6. 精确期望对照 E

令 b_i=sum_c pb_i(c)*r_i(c)，全部 detach。
L_exact=-(1/B)*sum_i w_yi*sum_c p_theta(c|x_i)*(r_i(c)-b_i)/s_prev。

对于给定采集状态、奖励和 s_prev，leave-one-out L_PG 的期望梯度等于 L_exact 梯度；两者共享同一个解析奖励方差EMA规则。
E无策略梯度动作抽样，但仍在优化同一个奖励目标。它区分“奖励目标本身有用”和“Monte Carlo采样是必要的”，不是证明任何RL范式无效。
仅有0/1类别奖励时，E[r_acc]=p_theta(y|x)，其梯度相当于按当前正确类概率缩放的CE梯度。此数学关系必须写入最终解释，不能把所有提升包装成推理能力。

## 7. 固定实验矩阵

共同 CE 为上述 p_theta 的类别加权交叉熵；共同 FD 为同一类权重下的 J_student 与 J_anchor 平方距离均值。
Task1无旧知识：FD与额外KL关闭，保持奖励关闭；G/R/E的准确率奖励优化仍开启。

| ID | 训练目标 | 作用 |
|---|---|---|
| S | CE + 1*FD | 共同的弱FD监督基线 |
| H | CE + 10*FD | 强FD保持参照 |
| K | CE + 1*FD + 0.1*full_KL(p||q) | 直接损失级KL参照 |
| G | CE + 1*FD + 0.1*PG(r_acc) | 任务奖励策略梯度，无保持奖励 |
| R | CE + 1*FD + 0.1*PG(r_acc+0.5*r_ret) | 预定主候选 |
| E | CE + 1*FD + 0.1*exact(r_acc+0.5*r_ret) | 同奖励精确期望控制 |
| F_S | 复用S正常Task1后冻结，后续仅解析统计 | 监督Task1冻结参照 |
| F_R | 复用R正常Task1后冻结，后续仅解析统计 | 隔离RL首任务收益与持续RL净效用 |

K的0.1与保持奖励0.5不是数学等价的正则强度，不作梯度强度匹配声明。H/K是两个预先固定的实用保持控制，不代表穷尽正则超参。

六种训练条件*三个顺序=18条完整轨迹；不删负结果种子、不把次要条件升格为主候选。

## 8. 训练超参与规模

- 单卡，FP32；不启用AMP/TF32；不重新下载/替换主干；可用显存门24 GiB起，实际峰值P0验证。
- AdamW；所有锁定可训练特征参数 lr=3e-4，weight_decay=0.01；每任务重建optimizer，cosine到1e-5；global gradient norm clip=1。
- batch=48，drop_last=false。
- epoch tier只能在P0根据吞吐/资源选择EPOCHS=5或3，不根据loss/BA择优；所有条件与任务相同。
- 5epoch：360 logical task-epochs、72 logical任务末训练状态；3epoch：216 task-epochs、同样72状态。
- 按既有18,718张ISIC fit估计，六方法三顺序的总更新约3.51–3.54万（5epoch）或2.106–2.122万（3epoch）。正式值必须用每任务真实ceil(N_t/48)求和，不硬凑数字。
- S/H/K在Task1以及G/R在Task1可作逐位一致性检查；默认仍按完整逻辑矩阵计数。只有事先声明、严格一致的公共前缀才可物理复用，不能用S的Task1替换G/R的Task1。
- F_S/F_R均无新增神经训练。包括两个冻结参照的主表96阶段行、480逐类行；训练头诊断另表，不混入主表。
- 新增持久化产物含独立备份建议上限16 GiB，须测量72个紧凑状态与统计、rolling恢复状态、评价分数总量后准入。不得继承M1旧3 GiB额度或删除历史资产腾空间；实际无法准入则BLOCKED_STORAGE。

## 9. 12–16小时时间预算

单卡墙钟与累计GPU进程驻留分别计账。计入初始化、正式/工程更新、前向、解析求解、备份等待、失败与恢复；重启不重置T0。

| 时间上界 | 工作 |
|---|---|
| 0–2.5h | 资产与源码锁、数学/梯度验收、真实T1吞吐、恢复/备份测试、选定5或3epoch |
| 2.5–12.5h | 完整六条件三顺序四任务；逐阶段锁定状态与W，并异步备份 |
| 12.5–14.5h | 锁定全矩阵后统一all-seen验证，统计与梯度诊断 |
| 14.5–16h | 配对报告、独立备份核验、停止与交付 |

P0最多180个工程optimizer steps，使用可丢弃副本和当前Task1 fit；不看validation；正式训练重置为锁定初始状态。对六条件都测吞吐，覆盖非零FD/保持奖励路径；用合成8列头测最大类别维度，不读取未来图像。

选择最大的epoch tier，使当前耗时加1.3倍实测训练/统计/评价/IO预测满足16h硬上限，且训练预测在12.5h前完成。评价用样本数量与无标签/合成输入计时估计，不提前解封val图像测效果。若3epoch也不满足，输出BLOCKED_BUDGET；不删对照、不仅跑最佳顺序、不改few-shot。

12.5h停止新训练，14.5h停止新图像前向，16h停止全部本轮进程。未完成矩阵标记INCOMPLETE_BUDGET，完整方法效用NOT_EVALUABLE；不得把缺失条件当0，不以局部子集替代主比较。16h是硬上限而非必须消耗完，提前完成即停止。

## 10. P0 必须通过的验收

1. no_base=true，4任务、8类、三个原顺序；初始外部权重SHA与未训练特征模块状态明确。历史S0/Task3/DFD父状态拒绝。
2. Feature dimension=1536；训练/预测均pointwise；无任务ID或真实标签进入特征/路由；当前fit不得泄露到未来任务。
3. W_start只用已合法到达的统计和当前fit，临时副本不污染最终bank；float64 ridge残差与旧求解配方一致。
4. 真实Categorical采样、G=8、局部RNG；动作与log_prob严格对应同一采集策略；一批一更新。
5. 保持奖励、teacher、W_start、advantage/EMA全部无梯度；单独RL loss反向对可训练feature参数有梯度，不能只有原神经分类头有梯度。
6. 常数奖励=>PG梯度0；动作相关奖励=>相应梯度方向；精确枚举小型分布验证leave-one-out期望梯度，原组均值的有限G缩放不能被漏算。
7. p=q时保持奖励常数1，组优势中的常数贡献消失；Rret位于(0,1]；新类q>0；不把maxcos或低熵当分类正确。
8. 全KL按全部已见类别求和；不用单argmax项冒充；不对零概率人为截断改变类别支持。
9. 单独统计CE、FD、PG/Exact、KL梯度以及最终总梯度/clip；不能只凭总梯度判断RL分支接通。
10. 合成测试验证公共T输运、全二阶矩、身份映射、任务统计只提交一次；无旧fit重提取。
11. 全状态恢复后下一次真实更新一致，包括theta、optimizer、scheduler、hat_sigma、task/batch游标、数据/增强/策略RNG和bank；独立备份目标端hash/恢复核验。
12. 所有科学设定在正式第1次更新前写PROTOCOL_LOCK，实际服务器、显存、预计时间与磁盘准入齐备；目前均未声明已通过。

## 11. 评价、预定门槛与统计

主指标：W_final的Final all-seen BA；完整4任务全部报告AvgBA_inc、Macro-F1、old/current recall、HM、尾类宏召回、每类正确数/分母、零召回、旧新错误流向、遗忘。
尾类集合遵循既有锁定定义，不按本轮成绩改类；全局频次统计只可由隔离评价器使用，训练不得借此读取未来分布。

主比较R-S，次比较R-H、R-K、R-F_S、R-F_R、R-G、R-E。不得从所有条件选赢家重新定义主候选。

以下是新试验的开发性决策阈值，不是功效分析/临床非劣效界：
- 主效用：R-S平均Final BA>=1.0pp，至少2/3顺序正，AvgBA_inc不下降。
- 安全：相对S和F_R，平均old/current各不下降超过2pp，平均tail不下降；任一顺序Final BA不下降超过2pp；不新增val支持数>=20的零召回类，同时报告所有支持数类别。
- 强控制净效用：R对H/K/F_S每个Final BA平均差至少非负，且R-F_R>=1.0pp。未达到时只能称局部辅助效用，不晋级为稳定更好的完整方法。若R优于S却不优于F_R，不能把改进归因于Task2以后的持续RL。
- 保持组合信号：R-G>=0.5pp且不以当前类/尾类破坏换取；这不是单独CTAN因果结论。
- 采样附加信号：R-E>=0.5pp且2/3顺序正；条件区间跨零则仅初步，不宣称采样必要。E不劣于R时保留“奖励目标可用、采样无额外证据”的结论。

全部阈值基于未四舍五入值。2000次按类身份component配对bootstrap，共享重采样，固定三个已训练模型；不把多个actions/epochs当独立样本，不把三顺序当三独立患者队列。主比较为一个，次比较描述性；无独立外部确认，多重开发限制全部保留。

若R-S不通过但CI跨过有意义正收益，结论应为“本预算证据不足/未过晋级门”，不是已证明等效或普遍无效。训练不足、极少有效奖励组等必须独立披露；不据此自动续训或改奖励。

## 12. 必须记录的机制诊断

每任务/epoch：总样本与各类暴露、CE/FD/PG/Exact/KL原始与加权值、各项梯度norm和夹角、总clip频率、参数更新、reward均值/标准差、hat_sigma轨迹、全对/全错/混合组比例、有效非零优势组、动作熵、q与pb支持、保持奖励量级与饱和比例。

相同正确类别的重复动作有相同保持奖励，这是类别动作模型的预期局限，不伪称复现“全对文本轨迹组仍能排序”。保持奖励变大或KL下降不能替代旧类/当前类实际评价。

W_start 与 W_final的两个固定读出：记录同一模型重拟合前后BA/错分变化，不择优；主成绩始终W_final。旧fit oracle、restricted-current 和输出头择优均不作为合法CIL成绩。

## 13. 交付与停止

交付至少：PROTOCOL_LOCK、SOURCE_AND_ASSET_LOCK、METHOD_MATRIX、P0_ENGINEERING_REPORT、stage_metrics、class_metrics、paired_differences、bootstrap_intervals、reward_diagnostics、gradient_diagnostics、ACCESS_LEDGER、RESOURCE_LEDGER、BACKUP_REPORT、FINAL_REPORT_ZH、NEXT_DECISION。

状态分别填写：engineering_status / matrix_status / utility_status / retention_signal / sampling_added_value / frozen_comparison_status。

无自动push；无自动改超参、扩大HK、增加样本回放、引入新VLM、释放test、延长预算或创建小时监测。单独授权的执行器可以按本协议运行，但本次回答本身只交付设计。
