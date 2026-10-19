# 新一轮文献筛查：哪些候选改变信息，哪些只是已有头的组合

## 本次范围与结论

从用户提供的43,013条公开投稿标题/摘要快照重新筛查，聚焦analytic CIL、long-tail、few-shot与支持外风险。快照不是录用名单，也不是穷尽优先权检索。标题规则命中12个analytic相关候选、116个long-tail/imbalance标题；这是词法筛选数量，不是主题全集，也不按投稿编号靠前程度推断重要性。

筛选后重点核对六篇：InnerCIL、FROST、RIDER、Input Anchoring、CCSF与Spatial Moment。InnerCIL与FROST在已有本地全文中核对了具体方法；Spatial Moment局部方法段亦可读取。RIDER/CCSF/Input Anchoring本次只有摘要依据，不能称其完整实现已核实。在线OpenReview PDF访问返回不可访问或浏览器验证页，未绕过验证；已有缓存正文与在线检索的同名异文明确区分。

结论：没有形成一个已经排除相关工作重合且足以启动的新算法。两个明确改变信息或训练函数族的候选分别是原生神经元读出和Input Anchoring，但前者已有具体方法，后者本次正文尚未核对且不是零成本替换。其余融合、空间池化、坐标修复也已被现有论文覆盖。继续分析，不为维持运行而增加现有20行query上的试验。

## 1. FROST：不能把原型融合当成新方向

[Exemplar-Free Few-Shot Class-Incremental Learning with Frozen Statistics](https://openreview.net/forum?id=yKWeZIY0Sn)

正文§3.3–3.6区分了基类训练后冻结的adapter、随机ReLU升维后的ridge相关统计，以及在adapter特征z上保留的归一化原型。两臂固定凸融合，文中实现alpha=.5，ridge惩罚由base留出选定。

重要限定：原型用z、ridge用h(z)，不能简单说原型必定只是同一个Q矩阵列的重复。反过来，在完全同一特征坐标和同一权重中，类均值可从标签交叉矩与类质量重建；“新增原型记忆”并不自动提供独立样本。

其类质量论据依赖未归一的raw sums使base类样本量占优。我们weights按类/组/图像平衡，各类总质量均1；所以不能直接把原FROST的base-mass问题当成现有类1初始损失已被解释。融合仍可能改变归纳偏置或引入另一表示，但这是已有方法及强控制，不是可直接命名的新机制。该论文自报收益不作为我们的实验结果。

## 2. InnerCIL：新的访问位置，但已有人提出

[InnerCIL: Internal Neuron Representations for Analytic Class-Incremental Learning](https://openreview.net/forum?id=RoJiIFcc1v)

正文§3.3在冻结MLP隐藏神经元非线性之后、W2输出投影之前采集响应，用CLS-to-patch注意力池化；与多深度CLS组成字典。checkpoint参数注册候选原子，§3.4使用累计类矩做Fisher筛选，再从累计Gram/标签统计拟合ridge。

它与我们此前的patch均值/方向二阶及block6 CLS不是同一函数。隐藏响应位于W2的零空间部分可能在输出合成时消失，因此这是一种不同的信息访问，不能用“固定U残差化只是广义ridge”将其一概排除。原文的隐藏响应干预定理是架构可访问性论断；不能由此推出真实皮肤图像中该部分一定含有可泛化的类1标签信息。

因此，“直接加原生神经元局部字典”已有明确重合，撤回其作为独立创新的宽泛主张。若以后做这类实验，只能先标为已有方法启发的机制核对/强基线，不能宣称新方法。

## 3. RIDER：支持有效的分支证据

[RIDER: Support-Aware Analytical Evidence Construction](https://openreview.net/forum?id=6ZDr9wRsXx)

仅摘要核对：它分解跨任务与任务内错误，结合原型、全类解析全局参照与支持有效的分支证据，另有LoRA漂移/方向幅度等设计。与我们固定编码器单个全局读出不同，不能宣称等价；但“只让拥有类支持的分支给证据、再用全局参照校准”这类方案已存在明确候选重合。我们已完成的决策归因又没有通过共同新竞争损失门，故本次不借该论文复活校准路线。正文不可访问部分保留未核实。

## 4. CCSF：坐标不变性的宽泛口号也不是空白

[Equivalent Predictors, Different Updates](https://openreview.net/forum?id=DtaC93MmTo)

摘要研究多策略梯度协调中私有坐标导致的更新非不变性，并在规范坐标中融合/更新。这不同于我们闭式ridge惩罚的同余等价，不能说它直接给出我们的定理；但“坐标等价预测却产生不同优化”的宽泛叙事已有相关方法。当前特定固定U方案已因广义ridge重参数化被排除，不另立实验。

## 5. Input Anchoring：改变模型族，不是小修复

[Input Anchoring Makes Your Model a Deep Analytic Continual Learner](https://openreview.net/forum?id=MafOc3dJKT)

仅摘要核对：input-anchored多项式网络与Loss-Anchoring Points近似将解析记忆推广到更深的函数族。它确实超出固定线性坐标等价类，但代价涉及更高阶统计、近似锚点及训练/存储增长；与QUILT/功能记忆/coreset的边界仍需正文核对。不能在未明确预算与新的支持外评测前直接上模型，也不能将它与同名其他FROST检索结果混用。

## 6. Spatial Moment：可控孔径已有具体池化设计

[Narrowing the Aperture for the Tail](https://openreview.net/forum?id=S2J0Z2xpQz)

已有正文方法段显示多专家的广义均值池化阶数、偏置及推理路由等设计。它不是当前同核心冻结ridge的直接控制；但“tail专用局部孔径”“换池化阶数”等宽泛方案已有方法覆盖。此前局部二阶阴性并不否定所有内容依赖池化，却也不足以支持我们直接扫描pooling阶数。未做新实验。

## 可保留的独立问题，但尚非已封存实验

目前最清楚地超出固定坐标变换的访问点，是“输出合成前、真实数据上的额外标签信息”。如果后续选它，问题必须是可否证的：在固定全局核心、相同留出组与支持权重下，原生隐藏响应中被输出合成压掉的部分，是否比输出可见部分、固定随机函数和普通池化提供额外组外风险收益？这是InnerCIL启发的诊断问题，不是新的创新声明。

至少要预先解决：

- 零空间架构干预不等于真实图像干预，不能拿理论不变性替代真实标签证据。
- 隐藏宽度通常3072而不是768；新增投影、注意力缓存、峰值与拟合成本必须全部计入。不能仍宣称旧19,058,800字节同预算，或借删全局核心制造收益归因混杂。
- query必须退出原子筛选、Fisher/归一化/子空间/头所有拟合；不能把测试标签用于原子注册或null/range选择。
- 至少比较相同1318核心的纯全局、普通输出可见特征、原生额外部分和固定简单特征；不靠多扫层、维度或pooling增加命中机会。
- 新诊断仍受同一16组信息量上限限制，不能把重用query称独立复验；若不能建立合理风险证据，应停止而非增加bootstrap。

这些尚未完成预算可行性与正文全面重合核对，因此本次不封存为实验，不启动服务器任务。现有失败、成本与停止条件均不改变。

## 状态

本次仅标题/摘要筛查、本地方法段审读和有限在线原文访问，没有SSH、数据读取、拟合、测试重评分或新增GPU运行。没有新的算法有效性结论，也没有证据要求重跑已退休任务。下一步可继续核对上述候选的预算与可证伪边界，若无法区别于InnerCIL或形成独立风险证据，就撤回候选，不强造连续实验。
