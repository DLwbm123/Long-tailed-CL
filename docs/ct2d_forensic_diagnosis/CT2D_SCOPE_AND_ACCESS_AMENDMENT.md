# CT2-D：CT1 代码复核、当前空间重估与失效归因

## 0. 定位与本轮授权

状态：待在用户服务器执行。本文是 CT1 完成后的新诊断协议，不覆盖 CT1 结果，不是 CT1 成功修复报告。

目的：区分 (a) 实现/标签/状态错误；(b) 持续适配后表征本身不适合统一线性读出；(c) 历史统计输运失配；(d) Risk 代理/受限读出无效；(e) ConCM-lite 梯度路径与分类器训练失衡。

**本轮新增神经训练 epoch=0，optimizer.step=0。**允许在隔离模型副本中求分项梯度，但不执行参数更新。允许预先固定的离线解析重新拟合。这不是“所有形式的拟合都为零”。

用户仍选择无独立 S0 的持续学习研究目标。冻结已有 checkpoint 仅为控制变量的离线诊断，不是把候选方法悄悄改回 S0 后冻结，也不建立隐藏基础阶段。

### 明确的数据访问修订

CT1 训练时禁止回看旧类图像，该限制继续适用于任何合法在线方法。**CT2-D 仅在隔离的事后诊断进程中，允许重新读取既有 train/fit 中的旧类图像，测量同一已训练模型当前空间下的真实统计。**

- 这是新增授权；发送执行 prompt 表示同意这一范围有限的诊断访问。
- 这些结果统一标为 `OFFLINE_REENCODING_DIAGNOSTIC` 或 `ORACLE_MOMENT_DIAGNOSTIC`，不是合法无回放 CIL 候选，也不是数学性能上界。
- 新统计、新 W、旧类特征、诊断标签和查询结果，绝不回写 CT1 checkpoint、任何未来在线 learner 或新的训练过程。
- 新 test/reserved 图像、特征、逐样本预测读取及新预测全部为 0。本轮不做保留集哈希/解码审计。
- validation 可以用于固定诊断评价，不能用于统计拟合、选择超参数、选择映射或挑选 checkpoint。
- 若执行环境不能建立独立诊断访问边界，完成其余只读审计并报告 BLOCKED_ORACLE_DIAGNOSTIC_ACCESS；不能静默绕过原 learner guard。

## 1. 固定来源和证据等级

交付提交：`345bd7714662a8b7404c4609770df4bba4808451`。
绑定执行源：`8e63785896d17126114a436027d1cb0890b4658f`，须与 SOURCE_COMMIT_BINDING、CODE_ENV_LOCK、checkpoint 指纹逐项核验。

主要源码：
- `tools/run_ct1.py`
- `tools/ct1_statistics.py`
- `tools/report_ct1.py`
- `tools/report_locked_holdout_r1.py` 的 predict_columns / bootstrap 辅助
- `tools/run_medical_v2.py` 的 Images / 预处理
- `third_party/APART/models/apart.py`
- `third_party/APART/backbone/vision_transformer_adapter_pool_a.py`
- `third_party/APART/utils/medical_v2.py`
- A1 的 `check_isic_a1_linear_native.py` 推理补丁及相关加载器。

必须读取实际配置与以下审计：DATA_AND_TASK_LOCK、MODEL_LINEAGE、TRAJECTORIES_LOCK、P0_ENGINEERING_TESTS、ACTUAL_CONFIG_*、TASK_TECHNICAL_AUDIT、transport_audit、risk_solver_audit、train_epoch_metrics。

分类报告使用以下证据等级，不合并：
1. `IMPLEMENTATION_CONFIRMED`：实际代码、运行状态或独立回归明确验证。
2. `SPECIFICATION_WEAKNESS`：实现符合 CT1，但设计不提供所需保护。
3. `SUPPORTED_MECHANISM`：受控对照支持某项机制，仍限本协议。
4. `UNRESOLVED`：当前材料不足以裁决。

代码 PASS 不等于科学假说成立。负性能不等于发现软件错误。存在实际偏离时，列出影响范围；不得事后将原结果覆盖成 corrected 结果。

## 2. 不变的数据与任务

ISIC：原 CT1 train=18,718、val=295，任务 [2,2,2,2]，最终 Task 4。
HK：原 CT1 fit=6,821、val=1,698，任务 [2,2,2,2,2,2,2,2,2,2,3]，最终 Task 11。

沿用 CT1 三个 1993/1994/1995 order/train 组合、HK M1 规范 ID、既有 head/middle/tail 分组。不重分数据，不调整任务，不改标签语义、不增补图像，不选择更好 fold。

严格区分 original_label、head index、类到达时间。所有 argmax tie 按 CT1 最小 original_label 规则。

## 3. P0：只读审计与结果复算

### 3.1 覆盖与来源

复核 12 条神经轨迹、90 个最终任务状态、900 epoch 行和 32,340 个记录更新。用每任务样本数、batch48、10epoch 重新计算期望步骤，不仅检查总和。报告 U/R Task1 配对和后续连续继承。

逐项核对 source hash、共享预训练核心、全部非共享参数、pool_few、keys、assigner 和 heads。原来 nonzero-adapter 的检查只覆盖部分命名模式时，指出其覆盖范围，不据此认定另一个分支没训练。

不要直接实例化原 `Run` 或调用 formal/main，避免自动写回原公共报告、访问计数、归档队列或启动训练。

### 3.2 结果复算

从已封存 validation logits 和明确标签映射独立重算全部 270 行主指标、2,754 行逐类结果。原 CSV 只用于最终核对，不作为独立重算的输入答案。

补上 CT1 主 reporter 未明确输出的：
- restricted-old BA/accuracy 与 restricted-current BA/accuracy；只作诊断。
- old/current 的样本加权及类别宏平均错误分解。
- 逐类、逐到达年龄的 first-to-final 和 max-to-final 下降，最后到达类 NA。
- Macro-F1 可从相同预测补充，标明新增指标，不改变原主终点。

错误分解必须同分母满足 correct + to-other + wrong-within = 100%。总体 BA 必须等于全部逐类 recall 的平均，也应等于按 old/current 类数加权的两组 recall。

PT-CB 的最终结果在三个顺序下是确定性参照。不要将共享预测当三次独立训练。

### 3.3 逐函数与独立数学验收

必须实测：
- labels 与 columns 往返、任意类别置换后指标不变、seen slice 正确、future 列不参与。
- 模型每任务没有意外重置，core 不变，router/adapter/head 可训练白名单正确。
- U/R 初始化同源；真实输入 RNG 与独立合成 RNG 不互相污染。
- epoch anchor→epoch 映射每次施加到 immutable pre-task bank，任务末只提交一次。不要把重复施加原本没有的 bug 写入结论。
- affine aggregate S 更新与显式 transformed samples 的 full Gram/均值/对角方差一致，包含交叉项。
- class-balanced ridge 目标与明确加权样本目标一致；risk 的 PSD factor 方向正确；eta=0 恢复 W0。
- Risk `optimal` 与 normalized violation、primal/dual residual 一致；但同时统计 xi>=delta，不能把含松弛可行性当无风险保证。
- 记录实际 loss reduction 和 all-seen CE，不把当前继承的 SUM 直接“修正”为 MEAN。

已发现候选错误时，先在独立最小样例中复现。只有指标实现错误可在新目录输出额外 `RECOMPUTED_*` 结果；影响学习轨迹的错误不在本轮全量重训。

## 4. P1：固定最终模型的当前空间统计反事实

### 4.1 12 个最终父状态

选择全部2数据集×3固定组合×U/R两流的最终 checkpoint，共12个，不按效果删选。

它们均已完成无S0完整训练；本轮冻结这些最终参数，只测量表征。

恢复后核验模型 SHA。使用原 CT1 的 pointwise probe、FP32 encoder、禁用 TF32/AMP、确定性 Resize224 bicubic antialias、mean/std0.5、batch48、sample_id 升序。

保留 J = raw main/few 拼接后整体 float64 L2。不得换为分支分别归一化、不换 M、不加 whitening、不改路由、不加原始 PT 融合。

每个最终模型只重新提取一次全 train/fit 和一次全 validation；如果已有**同模型同路由同预处理、身份与哈希可核验**的缓存，优先复用。该流程约 165,192 条“模型×图像”记录（无复用时），内部三路编码器调用另计，不将样本数写成 forward call 数。

### 4.2 先完成当前类一致性检查

当前类在 CT1 任务末用真实当前模型提取，因此它的计数、均值、对角方差和与总统计有关的二阶量，应与重新提取版本一致到锁定数值容差。

若当前类本身已明显不一致，应先查恢复、row identity、预处理、router、feature definition、task/version 与交叉项，不能直接解释为“旧类漂移”。

初始容差采用原工程 FP32 特征 atol/rtol=1e-5；需要基线逐样本预测与逐类正确数一致。近 tie 变化需逐例记录 margin 和预设tie规则，不根据分数选择实现。

### 4.3 2×2 旧类均值/协方差矩阵

以下四种状态对同一个最终模型与同一 validation 特征求解，全部lambda=.001、无bias、float64，原head顺序，完整Gram。当前类始终保留真实任务末统计。

| 模式 | 旧类均值 | 旧类中心化协方差和 | 身份 |
|---|---|---|---|
| Q00 | CT1输运结果 | CT1输运结果 | 原部署统计复现 |
| Q10 | 当前模型重新编码旧fit所得 | CT1输运结果 | 均值替换反事实 |
| Q01 | CT1输运结果 | 当前模型重新编码旧fit所得 | 协方差替换反事实 |
| Q11 | 当前模型重新编码旧fit所得 | 当前模型重新编码旧fit所得 | 完整刷新统计反事实 |

Q10/Q01/Q11 都不是合法无旧样本访问的 CIL 方法；Q11 也不是性能必然更好的上界。

### 4.4 明确的统计构造

令当前集合 N、旧集合 O，C为已见类数。对每个真实类计算population统计：

```
n_c
mu_c = mean(z_c)
M_c = z_c.T @ z_c / n_c
V_c = M_c - outer(mu_c,mu_c)
```

CT1保存 S00=sum_c M_c，mu00。先用准确当前类统计求：

```
Snew = sum_{c in N} Mnew_c
Mold00 = mu00[:, O]
Vold0 = S00 - Snew - Mold00 @ Mold00.T
Vold1 = sum_{c in O} Vfresh_c
```

对 a,b in {0,1}：

```
Mold = stored_old_means if a==0 else refreshed_old_means
S_ab = Snew + Vold[b] + Mold @ Mold.T
mu_ab = old means Mold followed by exact current means in the locked head order
G_ab = symmetric(S_ab/C)
R_ab = mu_ab/C
W_ab = solve(G_ab + .001 I, R_ab)
```

校验 Q00 与原保存 W/分数/指标一致。Q11 与独立逐样本类均衡批量解一致。Q10/Q01 要连贯重构二阶矩，不允许只改 R 或改mu却错误保留其旧均值外积。

Vold0/Vold1 应半正定到浮点误差。明显负谱不能任意截断修好，应报告统计或恢复错误。仅允许来源容差内的对称化；保留未修饰数值与残差。Q10/Q01 的虚拟统计可能不再有单位trace，明确报告，不调整lambda弥补。

采用类流式矩累计，不需持久保存23份1536²矩阵。该2×2只区分CB目标可见的旧类均值与**类内协方差之和**，不是对每类全部分布的因果分解。

### 4.5 固定 Risk 诊断

仅对6个U最终状态，在Q11真实当前空间矩上重求一次原CT1受限Risk：

- 完整G/R来自Q11；mu、v、n来自当前真实fit。
- e=0只表示这个诊断不再用输运残差，不表示均值估计无不确定性；u仍含v_tilde/n。
- eta=.1,rho=1,kappa=1,delta=.1,同对角收缩、W=W0B、同求解器与容差。
- 不增加eta/lambda/margin搜索，不改全空间，不改协方差先验。

对比Q11-Risk−Q11-CB，检验原Risk在较准确的统计下是否仍无益。保留45个旧Risk日志，补计算B−I、W变化率、预测翻转数、目标两项、xi、无松弛风险margin和类误差关联。

### 4.6 正式输出规模

12模型×4统计模式=48行最终诊断指标；另6行U/Q11-Risk，共**54行**。
逐类 `(2流×3组合×4模式+3个U Risk)×(23+8)=837` 行。

这54/837是**离线诊断表**，不能并入合法CIL性能排行榜。Q00是原模型控制，不代表又训练了一个模型。

输出相同val上的PT-CB引用，来源相符才比较。不能用旧test的PT值减本轮val值。

## 5. P2：四个预先固定末任务转移的机制检查

选择不依据表现：seed1993，HK Task10→11和ISIC Task3→4，各U/R，共4个转移。

额外只需4个前任务checkpoint；最终状态复用P1。仍无新神经训练、无optimizer步骤。记录这是一个固定组合的机制检查，不是多种子确认。

### 5.1 输运跨类别外推与几何收缩

以当前任务fit样本重新提取 pre/post 特征，严格复算原fit_map。只能用当前fit拟合a/b；旧fit、全部val仅作独立诊断查询，不参加映射拟合。

报告：
- 每坐标a分布、floor/ceiling比例、b范数；不只报告a平均。
- current 与 old 的真实pre/post map残差；mean-coordinate MSE 与RMS L2均报，避免1536维均值小掩盖总误差。
- 真实旧类均值在两个时点的变化，与“将真实pre均值做一步原映射”后的误差；再与累计输运记忆相比。
- 类间均值距离、centered between-class scatter、类内trace、均值范数；按类到达年龄分组。
- two-pool pointwise路由在pre/post对同一old/current图像的切换率与使用占比。切换相关性不自动是因果结论。
- 对有历史记忆的早到类，比较真实post分布与原输运分布的偏差；不可因均值范数接近1宣称类别信息被保留。

不需要重放全部90个encoder。各任务完整总结使用既有transport/technical日志；本节只追加4个转移的实际模型探针。

### 5.2 零更新分项梯度审计

在隔离副本上，对固定当前fit的同一组变换后张量，逐项计算：main CE、sum CE、pool-weighted few、assignment、pull、synthetic CE以及总loss的梯度。

- 先从当前类各取最多8张（按sample_id排序）建立诊断batch；记录实际分母。用重复同一batch得到2B控制，证明SUM和MEAN的尺度差；不据结果改变原loss。
- 分组保存梯度范数与内积/cosine：main/few adapters、keys、assigner、main/few heads；core必须无梯度。
- 显式验证 synthetic CE→adapter 为0/None、synthetic CE→head可非零。
- 显式验证 CT-Risk 不在神经训练图中；求解成功不产生adapter保护梯度。
- 不执行optimizer.step，不保留梯度副作用，不改变任何已封存模型。
- R在pre/post各用匹配版本的原old raw memory采样，独立固定生成器；不把统计刷新结果送入这项历史代码路径。
- 跟踪pool_id实际分布、接近0/1比例、few分支原始未加权CE。原日志few加权项很小不能单独证明few分类已经学好或该pool完全没有学习。

这些是受控分项探针，不是声称精确重放了历史某个minibatch。若缺历史增强RNG，明确不提供“历史逐步完全复现”的说法。

### 5.3 ConCM 合成分布的最后层适配检查

在6个R最终模型上，用原存储raw old memory固定每类生成32对特征，只经过heads，不更新、不用生成数据重新拟合W。比较：
- 存储旧均值被各类head判为何类；
- 合成旧特征的all-seen CE/召回/old→current；
- 同模型真实旧fit特征的相同指标；
- raw main/few配对相关性和独立高斯假设的差异（只描述，不新增full-cov方法）。

若合成点分类好、真实旧样本坏，更支持记忆分布失配；两者都坏，则还需检查优化和目标取舍。这不是唯一根因证明。32是诊断配额，不改变CT1训练时每类4/cap48。

## 6. P3：归因与下一步决策，完成后停止

预设解释：

| 观察 | 支持的解释 | 不能声称 |
|---|---|---|
| Q11显著恢复而Q00低 | 最终表征仍有可用线性信息，部署记忆失配是主要可修复因素之一 | 旧类特征完全未损坏或可以合法刷新历史图像 |
| Q10恢复明显、Q01小 | 旧均值错位相对更重要 | 任意均值平移都能在无旧数据条件下修复 |
| Q01恢复明显 | 旧协方差汇总估计值得检查 | 单类不确定性已经得到验证 |
| Q11仍低且fit好/val差 | 适配后特征或读出泛化可能是主要限制 | 信息不可逆丢失或所有持续学习均失败 |
| Q11-Risk仍无增益 | 原固定受限Risk本身未获支持 | 所有分布鲁棒方法无效 |
| synthetic梯度只到heads | 现有ConCM-lite不直接保护上游适配器 | 回放毫无间接作用 |
| 发现明确代码偏离 | 提供最小反例和受影响artifact范围 | 未重训就宣称修复后有效 |

平均差、分解交互和每个固定组合都报告，不设效果好才继续/效果差换参数。按原固定类分层component做2000次配对bootstrap，seed46001；只用保存的val分数，固定父模型不重采样。singleton类不能提供病例间变异，明确列出。

**本轮到此停止。**后续可提出一个无S0 Task1→Task2的稳定性干预计划，但不自动执行。候选优先级应由本轮结果决定：若统计失配主导，先限制漂移或验证补偿；若真实当前空间读出仍低，先引入对adapter实际有梯度的保持约束，再谈Risk/高斯增强。不将学习率、路由、loss reduction和蒸馏一次全改。

## 7. 资源、存储、可重复性

默认一worker、最多8 CPU核、batch48，FP32 encoder，FP64统计求解。
- 新GPU进程累计驻留上限 **3小时**，包含全部工程与P1/P2提取。
- CPU解析与报告额外wall上限 **2小时**；先做小样本吞吐预算，不作效果gate。
- 新持久诊断产物≤512MiB；活跃含checkpoint/特征临时峰值≤1GiB；实时磁盘空闲≥1GiB。
- 逐个恢复模型，使用现有跨服务器checkpoint归档，核验源/目标SHA；不复制全部90个状态，不下载新权重，不删除历史资产。
- 每模型raw临时缓存约百MiB，可在该模型固定矩阵完成并校验后回收；只删除本轮登记临时路径。
- 不保留每类全协方差的多份永久副本；矩按类流式累计，必要诊断特征单模型处理。
- 运行前记录环境、求解器版本和风险参数；不得升级torch/cuda或放宽数学/身份门槛。
- 真实技术阻塞有限2次修复，不建立小时监测/自动搜索。预算不足如实PARTIAL/BLOCKED，不删方法或缩样本伪称完成。

## 8. 交付

```
CT2D_SCOPE_AND_ACCESS_AMENDMENT.md
SOURCE_AND_PROTOCOL_AUDIT.json
RESTORE_AND_REPRODUCTION_AUDIT.json
BUGS_VS_DESIGN_LIMITS.md
recomputed_ct1_validation_metrics.csv       # 270行，独立复算控制
recomputed_ct1_validation_per_class.csv     # 2754行
restricted_candidate_diagnostics.csv
transport_log_summary.csv
CT2D_ORACLE_LOCK.json
oracle_factorial_metrics.csv               # 54行，离线诊断
oracle_factorial_per_class.csv             # 837行
oracle_paired_differences.csv
oracle_bootstrap_intervals.csv
risk_counterfactual_audit.jsonl
transport_generalization_and_geometry.csv
routing_change_diagnostics.csv
gradient_and_reduction_audit.json
synthetic_vs_real_feature_diagnostics.csv
RESOURCE_AND_ACCESS_LEDGER.json
FINAL_REPORT_ZH.md
NEXT_DECISION.json                         # STOP
```

公开代码、协议、聚合审计；私有图像、identity映射、原始特征/W/logits不公开。

最终准确填写：
```
new_neural_training_epochs=0
optimizer_steps=0
backward_probe_calls=实际值
new_analytic_fits=实际值
encoder_forward_calls=实际值
diagnostic_old_train_image_reads=实际值，允许非0
new_test_image_reads=0
new_test_feature_reads=0
new_test_model_forwards=0
new_test_predictions=0
oracle_diagnostics_are_valid_online_CIL_results=false
original_ct1_artifacts_modified=false
further_training_started=false
monitoring_tasks_created=0
NEXT_DECISION=STOP
```

## 9. 本轮制定依据与静态检查边界

制定前已经核对CT1执行源绑定、训练/统计/读出/报告源码和部分实际日志。静态审阅未发现必须推翻全部CT1结果的明确标签错位、重复epoch输运或risk因子转置错误；这不等于已在私有checkpoint上完成独立复现。

以下行为来自原计划与代码，不应假称新发现的软件bug：
- U没有作用于旧样本表征的保持loss；Risk不回传；synthetic CE直接进入heads。
- fit_map输出的共同对角仿射可能收缩旧类距离；均值范数正常不保证类间可分性。
- soft slack是Risk的一部分；optimal不保证未松弛的margin。
- few与assignment为SUM而另外CE为MEAN，这会影响有效梯度，但CT1明确继承此配方。
- batchwise训练和pointwise提取是已声明差异，应诊断，不事后称为严格同一映射。

作者方在本地仅完成独立合成代数/算术检查，仿射S/均值/方差误差最大约3.6e-15，受限目标差误差约4.4e-16；未运行医疗模型或用户服务器训练。这些只是公式检查，不能替代本轮P0/P1的实际复核。
