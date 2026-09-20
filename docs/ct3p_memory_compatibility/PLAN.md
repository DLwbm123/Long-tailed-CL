# CT3-P：无独立 S0 的记忆兼容性小规模在线干预

## 0. 状态、范围与研究问题

**状态：新提出的实验计划，尚未执行。**本计划基于 CT2-D 已完成诊断，不把离线 oracle 恢复当成在线方法成功。

本轮只运行 CT1 的 **Task 1→2→3 前缀**，两数据集、三个原有 order/train-seed 配对。取消独立大基础阶段的设定保持不变：Task 1 仍仅两类、10 epoch；Task 2/3 持续更新同一套 adapter，不能在 Task 1 后冻结。

本轮只检验两个因素：
1. 不再缩放旧记忆、仅作公共平移，是否减少不必要的几何收缩？
2. 在当前任务图像上加入上一任务教师的 pointwise 联合特征蒸馏，是否减小实际漂移和在线读出退化？

这是**基础机制干预，不是预先成立的新论文算法**。不恢复 Risk 参数搜索，不新增 ConCM 回放神经训练，不同时修改学习率、原 CE reduction、采样器或路由配置。

### 0.1 本轮新增训练授权

发送配套执行 prompt 即批准：从六个合法 CT1-U Task1 父状态分叉，每个分叉训练 Task2 和 Task3，各10 epoch。共 **6 条新续训轨迹、12 个新任务 checkpoint、120 个新增 task-epoch、预期 6,620 个正式 optimizer steps**。Task1 为复用的正常两类任务前缀，不是新增预训练/S0，不计入新增训练数。

对照 U 的 Task1–3 已在 CT1 完成，复用原状态和分数，不重复训练。所有新读出保持任务访问边界。

### 0.2 数据访问边界

- 在线学习只读取当期 train/fit 图像。Task t 的旧类仅通过其合法历史统计和上一任务模型进入学习器。
- 禁止读取 CT2-D 的 oracle 特征、oracle 均值/协方差、oracle W、其逐样本分数来初始化/校准/训练任何候选。
- CT2-D 允许离线读旧图像的特批不自动延续到在线 learner。
- 仅在所有候选状态、W 和预测锁定后，按 §9 另开隔离进程进行四个预定 Q11 诊断；允许该进程读取已到达的历史 fit 图像。该权限不授予 learner，诊断结果不回写。
- 新 test/reserved 图像读取、特征读取、旧逐样本预测读取、模型前向及预测，均为0。不重新进行 reserved 哈希/解码审计。
- validation 仅用于固定评价和事后机制分析，不做蒸馏、先验、映射拟合、温度/阈值/权重选择或 best-epoch 选择。
- 原 validation 已多轮用于开发，本文结果仍非独立确认。

## 1. 已有证据与不可误读之处

CT2-D 交付：`7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb`。
CT1 交付：`345bd7714662a8b7404c4609770df4bba4808451`。
CT1 绑定执行源：`8e63785896d17126114a436027d1cb0890b4658f`。

先读以下公开材料并核对本地来源：
- `docs/ct2d_forensic_diagnosis/FINAL_REPORT_ZH.md`
- `BUGS_VS_DESIGN_LIMITS.md`、`IMPLEMENTATION_AND_VALIDATION.md`
- `oracle_summary.csv`、`transport_generalization_and_geometry.csv`
- `routing_change_diagnostics.csv`、`task_step_recalculation.csv`
- CT1 `DATA_AND_TASK_LOCK.json`、`MODEL_LINEAGE.json`、实际配置与代码绑定。

关键观察：
- HK/U Q00→Q10→Q11：13.638→51.273→55.638 BA；ISIC/U：46.401→56.958→61.502。
- Q01 单换旧类协方差和并未恢复：HK/U 13.158、ISIC/U 39.093。
- Q11-Risk 相对 Q11 在 HK/ISIC 为 −0.082/−0.279 点，无可靠正向证据。
- HK/1993/U Task10→11 的真实旧类 between scatter 从0.0333236到0.0397738；把当前任务对角映射施加于真实旧 pre 特征却得到0.0167769。它说明该步近似与真实变化存在差异，不足以证明所有任务均同一机制。
- 四个指定末任务转移的 pointwise 路由几乎不变；不能继续把大幅 routing switching 当作已经确认的主要原因。

U 表示不含合成回放的训练流，R 表示 ConCM 回放训练流；CT2-D R/Q00 是 R 轨迹的联合特征 CBRidge，不是 raw ConCM heads 得分。

## 2. 数据和任务前缀固定

完整原始划分保持不变：ISIC train18718/val295；HK fit6821/val1698。但本轮每阶段只访问/评价已经到达的类别。

三个 order 的前六类（规范 original_label）固定为：

| 数据 | seed | Task1 | Task2 | Task3 |
|---|---:|---|---|---|
| ISIC | 1993 | [4,0] | [3,7] | [5,6] |
| ISIC | 1994 | [3,7] | [4,5] | [1,0] |
| ISIC | 1995 | [1,3] | [0,6] | [4,5] |
| HK | 1993 | [0,4] | [12,8] | [2,11] |
| HK | 1994 | [22,16] | [12,13] | [21,4] |
| HK | 1995 | [9,13] | [16,14] | [15,5] |

完整 order、head容量8/23和计数仍读取原锁，不把head改成6行，不重新初始化已有父状态。

- 只训练/评价2→4→6类。本轮的 Task3 BA **不是**整个8类/23类协议的 Final BA，字段用 `prefix_terminal_BA`。
- 各 seed 的六类集合不同，PT-CB 的 Task3 结果不能强制设为相同。
- head/middle/tail 沿原全协议固定定义，阶段尾类指标仅对已见尾类求平均，写出有效类别数及每类分母；无尾类时为 NA，不填0，也不重排组。
- 此前全8/23类的 PT-CB 55.497/61.128 不能直接充当本轮六类性能基准。
- seed同时包含类别顺序和训练随机性，不当作纯初始化方差。

新训练的核验计数：

| 数据 | seed | Task2 n/steps | Task3 n/steps | 新增steps |
|---|---:|---|---|---:|
| HK | 1993 | 821/180 | 997/210 | 390 |
| HK | 1994 | 829/180 | 644/140 | 320 |
| HK | 1995 | 286/60 | 491/110 | 170 |
| ISIC | 1993 | 1333/280 | 300/70 | 350 |
| ISIC | 1994 | 714/150 | 13792/2880 | 3030 |
| ISIC | 1995 | 10573/2210 | 714/150 | 2360 |

计数来自 CT2-D `task_step_recalculation.csv`，每格steps为10×ceil(n/48)，本地再次核验。新正式steps总计6620；工程步数另记。任一清单不一致应阻塞，不能为了匹配数字改数据。

## 3. 唯一固定矩阵

| ID | 神经训练流 | 旧统计更新 | 角色 |
|---|---|---|---|
| P | 原始冻结预训练 | 精确累计原始特征统计 | PT-CB 强参照 |
| C0 | CT1-U 原轨迹 | 原对角仿射 A | CT-J-CB 原控制 |
| C1 | 同一 CT1-U 原轨迹 | 仅公共平移 T | 取消缩放的读出干预 |
| C2 | U+FD 新轨迹 | 原对角仿射 A | 仅加入表示保持 |
| C3 | 与 C2 同一 U+FD 轨迹 | 仅公共平移 T | 两者组合 |

唯一新神经变量为 FD。C0/C1 共享同一原网络；C2/C3共享同一新网络。两套统计bank不能反传或回写训练heads，不能根据读出表现改变下一任务。

预设比较：
- **主机制比较：C3−C1**，固定T下加入FD。
- 输运干预：C1−C0，固定原U网络。
- 复核FD：C2−C0，固定A。
- 组合系统：C3−C0，明确为两因素变化，不称单因素消融。
- 交互：`(C3−C2)−(C1−C0)`。
- C2/C3对相同阶段P，检验是否仅仅超过弱失配控制。

Task1没有旧统计，且FD尚无上一任务教师，C0/C1/C2/C3必须对应同一父状态与同一读出。这里的共享记录不可当独立训练重复。

## 4. 父状态、恢复和原训练配方

每个数据/seed只使用对应的 **CT1-U Task1末**，该状态源自原始锁定AugReg。

固定公共模型：`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`；原SHA256：
`c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800`。

禁止使用历史医学S0、CT1最终模型或CT2-D刷新统计来启动前缀。

实现专门的 `fork_from_ct1_task1`：完整核对父delta、共享core、类别、数据和RNG；保存父SHA、原args、子args及允许的协议差异。不关闭通用恢复检查，不通过篡改父元数据伪装为普通resume。

共同训练保持：
```
每任务10 epoch；batch48；drop_last=False
AdamW；pool lr=0.0003；其它原可训练组lr=0.003；weight_decay=0.01
每任务重新初始化optimizer和cosine(T_max=10, eta_min=1e-5)
同一套adapter/keys/assigner/heads持续继承；core冻结
原main/sum/few三项all-seen CE、原SUM/MEAN、/3、assignment与pull全部保留
自然shuffle；不追加均衡采样/重加权CE
FP32，无AMP/TF32；原训练算子及batchwise路由
```

不减少学习率，不冻结keys，不选M代替J，不新增head-norm。

仅任务末分类器使用CBRidge：1/C类别平均、lambda0.001、无bias、float64直接solve，所有J cross-Gram保留。正则不随新trace调整。

## 5. FD：上一任务教师的可微 pointwise 特征保持

### 5.1 定义

Task t开始，复制其**自身最新Task t−1**的特征网络为教师，整任务固定，无EMA。教师不使用CT1原轨迹的后续状态。

对当前batch已经完成随机增强的同一个图像张量x，取两套raw特征，构造：

`j_theta(x) = concat(main,few) / ||concat(main,few)||_2`。

蒸馏目标：

`L_FD = mean_over_samples(sum_over_1536_dims((j_student - stop_gradient(j_teacher))**2))`。

`L_total = L_real_CT1 + 10.0 * L_FD`。

- **10.0是本轮新锁定的单一试验值，不是文献最优值或已验证参数**。不搜索，不根据val或梯度大小自动调整。
- FD加在完整原real loss之后，不能再次整体除以3。
- 不使用默认逐元素MSE造成额外除以1536；记录逐样本平方距离的mean/p95。
- 这是当前图像上的局部约束，不是看到了旧类图像，也不保证旧类域漂移受控。
- 仅选择当前样本标签用于原real CE，FD本身不需要标签或真实类别频次。

### 5.2 保留原CE路径，不偷改训练路由

FD需要一个额外的**可微 pointwise**学生前向：
1. 先保存所有module training flags、两pool的batchwise flags及额外前向前RNG。
2. FD学生前向临时eval、两pool batchwise=False；但必须开启autograd、不能inference_mode/no_grad，不能调用原 `extract()` 返回numpy的路径。
3. 教师eval、pointwise、所有参数requires_grad=False，在no_grad中前向；输入同一个x，不重新采随机增强。
4. 两者均调用输出pre_logits/pre_logits_few的实际路径，`train=False, weight=None`，不传oracle task ID。
5. FD使用原生FP32、相同批形状的算子，不把仅推理Linear补丁带入神经训练。
6. 恢复学生所有flags和RNG；不改变原CE的batchwise路径、dropout日程、数据顺序或梯度图。

推理/统计提取仍使用已审计的A1/CT1 pointwise FP32提取路径；对本轮可微pointwise特征做数值兼容检查，不把两条代码路径默认视为等价。

原 `probe` 会在每次提取时load当前学生，**不能兼作固定教师**。教师必须独立且不被提取同步覆盖。若共享只读core以节省显存，须证明与学生可训练参数无别名，且教师全状态hash不变。

### 5.3 必做梯度验收

- student=teacher时FD接近0；在隔离副本中进行一个真实更新/微扰后FD可有非零adapter梯度。
- 教师梯度为None，公共core梯度为None；至少被选中的main/few adapter组获得可用FD梯度。
- 离散top1的key不一定有FD直接梯度，不能强制每个组件均非零。
- beta=0并跳过FD额外前向时，原下一步loss/gradient/更新应与CT1控制一致。
- FD附加前向不得改动模型buffer、CE前向路由状态或后续采样RNG。
- 记录真实梯度范数、FD梯度范数、加权比例与cosine；不能看到比例很小就本轮自动加beta。
- 这不是LwF原logit蒸馏的逐项复现；仅借鉴教师保持思想，不声称此损失原创。

## 6. T：公共平移记忆更新，作为“不收缩”的控制

### 6.1 只能拟合当前fit

在更新前/后对当期fit图像提取确定性J特征，身份顺序严格对齐。令当前类数K，权重 `w_i=1/(K*n_current_class_i)`：

`b = sum_i w_i*(z_after_i-z_before_i)`，`a = ones(d)`。

`r = sum_i w_i*(z_after_i-z_before_i-b)**2`。

不估计逐维缩放、不改0.5阈值、不用旧fit/val拟合、不优化按类的b。

这是公共平移的简单控制，不是已验证准确的旧类漂移模型，也不是原SDC的完整复现。它不能描述真实旋转、非线性或类别特定变化。

### 6.2 完整统计公式

设bank `S=sum_old E_c[zz^T]`，`mu`为旧类均值列矩阵，旧类数m，`u=sum_c mu_c`：

```
S_new = S + outer(u,b) + outer(b,u) + m*outer(b,b)
mu_new = mu + b[:,None]
v_new = v
error_proxy_new = error_proxy_old + r[:,None]
```

然后加入当前真实特征统计，求解CBRidge。不能仅移动mu而保留未中心化S不变。

公共平移保持旧类两两均值差与中心化协方差不变。这只是该变换的代数性质，不意味着它一定更符合真实旧类分布。

输运后虚拟特征矩可偏离单位球面；报告trace/均值范数，不额外归一化旧均值、不clip、不加jitter或自动调整lambda。

### 6.3 A与T的时序

- 每个任务开始为两种bank分别存immutable pre-task状态。
- 每个epoch的映射都从同一pre-task当前特征到该epoch特征拟合；只用于日志/临时诊断。
- 第10epoch最终映射分别施加于各自pre-task bank，提交一次。
- 两bank独立跨任务累计，不能Task3时把T bank重置为A bank，也不能每个epoch反复累加完整任务变换。
- U原轨迹的T统计用CT1-U合法任务状态重建；T1从合法存储矩开始，随后按Task2/3顺序，每步只读取当期图像。不能用CT2-D最终刷新矩替代。
- A采用原fit_map与transport定义。不因本轮新诊断再修改A。

## 7. P0：只做必要工程，不重复全项目审计

复用CT2-D通过的来源和标签审计，新增检查以下项目：
1. 六个T1父状态严格恢复、两套bank初始化、共享Task1读出。
2. 原C0/P在原T1/2/3评价中的逐类正确数和身份一致；近tie差异按预设容差记录，不能选更好实现。
3. 完整T二阶矩与显式样本平移等价；b=0恒等；旧类两两均值距离和中心化协方差不变；cross-Gram未被丢弃。
4. 从三个合法任务依次构造统计，与显式按对应历史映射变换的虚拟数据目标一致。它不是Q11真实重编码等价。
5. FD梯度、flags、teacher隔离、RNG、beta0回归和恢复同下一步。
6. 统计/W不反馈到神经训练，C2与C3权重指纹一致。
7. 在线文件白名单、future/test/oracle资产拒绝；诊断另进程单独许可。
8. 先测一个当前batch的真实train+FD吞吐和checkpoint大小，确认完整剩余预算。

工程optimizer steps限定最多16步，全部在废弃的隔离副本中，另计、不回流正式状态。相同技术问题最多两次修复，不放宽科学规则后宣称原实现。

锁定 `CT3P_PROTOCOL_LOCK.json`、新源码commit/hash、数据/任务、父状态、参数、比较、访问和预算，再进行新候选评价。若实现阻塞，保留完成项并报告，不启动自由搜索。

## 8. P1/P2：固定前缀实验

P1：恢复原U的T1–3、重现C0/P，合法顺序重建C1。可复用身份/模型/特征配置完全相符的CT1正常缓存，禁止oracle缓存。

P2：六个F分支按HK1993/1994/1995，再ISIC1993/1994/1995执行。每个分支从对应T1父状态出发：
```
Task2: teacher=parentT1；训练10epoch；提交A/T bank和两种W
Task3: teacher=this_F_Task2；训练10epoch；提交A/T bank和两种W
STOP
```

不重置F Task3为CT1-U Task2。不得使用CT1任务最终模型选择停止时间。训练每个任务均有adapter参数更新，不隐性冻结。

所有候选的任务末状态/W生成后统一释放validation性能；运行中只看技术、train与资源日志。不以第一个seed表现删方法、加超参数或取消其余配对。

交付 **90行合法在线前缀阶段指标、360行逐类结果**：5方法×2数据×3配对×3阶段；每配对每方法逐类数2+4+6=12。

同时保存12个新增任务末compact checkpoint、120行正式epoch日志，预期6620个正式更新。C2/C3共享每个checkpoint，不造双倍训练数。

## 9. P3：四个预定、严格事后的Q11复核

为了防止“在线分数偶然提高，但表征或记忆更差”被误读，预先固定：
- seed1993，HK与ISIC；每个数据检查U Task3和F Task3，共4个模型。
- 所有在线候选和分数锁定后，隔离重编码**仅已见前六类**fit图像，并用相同lambda拟合Q11；不能读Task4以后的未来fit。
- 不载入CT2-D已用全部8/23类拟合的任何oracle结果。
- 每个模型的Q11与其A/T在线读出比较，量化新的online–Q11 gap；还报告旧类均值误差、类间scatter与协方差trace差。
- 只增加4行离线最终诊断、24行逐类结果，不合并到90/360合法表。
- 无Risk求解，无Q10/Q01扩展，无新test。
- 旧fit只用于这个已锁定的事后进程，计数为`offline_diagnostic_old_fit_reads`，不写`all_old_reads=0`。在线字段仍须为0。

若诊断环境不能隔离，报告P3被阻塞而保留完整在线结果，不将旧图像权限开放给learner。

## 10. 指标和判断

固定记录：每阶段BA、accuracy、Macro-F1、prefix_terminal_BA、前三阶段平均BA；old/current召回及HM、restricted-old/current、双向跨组错误；已见tail及逐类正确数/分母/component数；first-to-terminal与max-to-terminal（Task3才到达类为NA）。

机制记录：逐维缩放的分位数/clipping（A）、平移范数、当前映射残差；FD逐样本距离、分项梯度大小和方向；实际路由使用/切换；旧记忆类间距离/centered scatter、均值范数、矩PSD及solve residual。

配对bootstrap：2000次，seed47001，按类component抽样，固定模型，不重采样run。每个seed只在其自己已见集合上形成配对；跨seed用规范类别/sample_id对应的共同基础抽样保持共享样本相关性。不从未到达类别读取图像/特征建立评价；可以使用锁定元数据和既有component ID。无可靠组件则区间NA。单组件类别区间不代表病例总体不确定性。

**不设“达到阈值自动跑完整协议”的流程。**报告采用以下解释：
- C3优于C1，实际FD梯度非零，且在线–Q11/均值误差减小：支持此固定FD干预改善兼容性。
- C1优于C0：支持取消缩放有帮助，不意味着公共平移是真实映射。
- C3改善old但损伤current：报告取舍，不称全面修复。
- 只超过C0却仍落后同阶段P：仍未证明持续适配的净价值。
- Q11也下降：保持项可能限制塑性，不能为通过门槛改beta。
- 两因素均无明显帮助：报告负结果，停止；下一步再决定更换参数隔离/对齐策略，不能本轮追加。

前缀6类的正结果也不证明HK11任务末或全部尾类的长期保持。不同seen集合的分数不直接与原全协议表相减。

## 11. 资源、恢复与交付

- 单GPU训练worker；torch/BLAS4线程、loader4worker，沿原精度；禁止购买/新增算力。
- 新增GPU-process累计预算4小时（含工程、额外FD前向、提取、事后诊断）；CPU解析/报告额外2小时。
- 活跃新增文件含tmp峰值≤2GiB；新持久/归档产物总计≤3GiB；每个运行盘空闲≥1GiB。
- 初始可用空间未知，必须实时测量。只逐个恢复需要的父状态，不复制全部90个历史checkpoint/整库图像。
- 原始权重、图像、历史checkpoint和报告不修改、不删除、不移动。仅回收本轮登记tmp。新产物可用既有授权归档通道搬运，必须hash与可读性校验，禁止迁回全部旧resume。
- checkpoint保存所有非共享参数、teacher来源、双bank、optimizer/scheduler和全部RNG；仅保存一个活跃rolling resume，任务末严格恢复校验。
- 中断仅同锁恢复，不换seed/epoch/参数；无法精确恢复的时长和调用数报告范围，不补造。

建议交付：
```
CT3P_PROTOCOL_LOCK.json
PARENT_AND_FORK_LINEAGE.json
P0_ENGINEERING_TESTS.json
FEATURE_DISTILLATION_AUDIT.jsonl
TRANSPORT_A_VS_T_AUDIT.jsonl
TRAIN_EPOCH_METRICS.csv
validation_prefix_metrics.csv                 # 90行
validation_prefix_per_class.csv               # 360行
paired_differences.csv
bootstrap_intervals.csv
ORACLE_P3_ACCESS_AMENDMENT.md
oracle_prefix_terminal_metrics.csv             # 4行，独立表
oracle_prefix_terminal_per_class.csv           # 24行，独立表
RESOURCE_AND_ACCESS_LEDGER.json
FINAL_REPORT_ZH.md
NEXT_DECISION.json                            # STOP
```

最终实填正式120 epochs/6620steps与工程步数、12新checkpoint、实际encoder调用、6个父T1复用；test全部0；在线旧/未来fit访问0；离线诊断旧fit访问实数。公开仅代码/协议/聚合，不公开图像、身份映射、特征、W或逐样本分数。

完成P3后停止。禁止自动恢复Risk、ConCM/GSR/DSM、M读出挑选、PT融合、新数据集、全量Task4–11训练、test或每小时监测。

## 12. 来源与外部背景

项目事实以以下固定链接为准：
- https://github.com/DLwbm123/Long-tailed-CL/blob/7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb/docs/ct2d_forensic_diagnosis/FINAL_REPORT_ZH.md
- https://github.com/DLwbm123/Long-tailed-CL/blob/7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb/docs/ct2d_forensic_diagnosis/transport_generalization_and_geometry.csv
- https://github.com/DLwbm123/Long-tailed-CL/blob/7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb/docs/ct2d_forensic_diagnosis/routing_change_diagnostics.csv
- https://github.com/DLwbm123/Long-tailed-CL/blob/7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb/docs/ct2d_forensic_diagnosis/task_step_recalculation.csv

已有方法背景，不作为本提案的效果保证：
- Li & Hoiem, Learning without Forgetting: https://arxiv.org/abs/1606.09282 （当前数据上的教师保持思想；本轮FD不是其完整原实现。）
- Yu et al., Semantic Drift Compensation for Class-Incremental Learning, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Semantic_Drift_Compensation_for_Class-Incremental_Learning_CVPR_2020_paper.html （用当前数据估计历史漂移；本轮公共平移不是其完整原实现。）

本轮beta、前缀长度、对照矩阵、预算和停止条件均为新实验设计，不能写成CT2-D的已观察结果。
