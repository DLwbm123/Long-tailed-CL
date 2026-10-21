# DualTrack 最小迁移准入卡：当前方案不启动

**结论：REJECT_MINIMAL_MIGRATION_CURRENT_PROTOCOL。** 这是否决“在当前固定CLS/patch充分统计路线中，只加冻结routing＋可训练specific支路即可合法提升”的迁移提议，不是DualTrack原论文的效果否定。没有实现或启动DualTrack实验，没有新训练/模型/统计重建/图像或特征读取/评分。只复用已读全文笔记，并针对方法§3.2–3.5与附录B的函数和配置做定位核对，不重复阅读18篇全文。

论文：DualTrack: Accessible Knowledge Transfer for Continual Learning under Long-Tailed Distributions，OpenReview KbIzI01PYR（投稿，非接收结论）。原文链接：https://openreview.net/forum?id=KbIzI01PYR 。本卡基于本地已提供原文，不声称复现作者代码或数字。

## 1. 原文函数与当前拟议不能混称

原文固定ViT，但early shared上投影和晚层General仍可更新。classification流包含General与Specific，routing流排除Specific，routing递推的token停止梯度。它只在共享参数固定时对Specific变化保持不变；stop-gradient不等于共享参数不变，也不等于下一任务routing坐标与旧Gaussian登记坐标一致。

原文按类保存每层对角Gaussian（均值/方差/count）及class→expert映射；专家匹配取已登记类中最小能量，再Top-K。它与当前最终CLS＋patch的患者等权完整Gram不是同一个状态。把现有最终CLS当作完全冻结router是另一个设计，不是原文该路径的直接复现。

当前准入提议必须明确选择：

- 保留原文shared训练：承认旧routing统计可能过时，不能继续称冻结routing或旧Gram精确有效。
- 真正冻结所有routing祖先：routing坐标可以固定，但同时改变原文共享适配；specific输入/历史分类状态是否固定仍需另证。

两者都不能靠改一个名字略过。旧PMT/FD已经研究过相关坐标保持问题，本轮不重新拟合映射或追加保护项。

## 2. 新支路的历史量从哪里来

设固定router表示r(x)，新可训练支路sθ(x)，若最终仍拼接后用当前解析联合头，则每个旧类需要

E[r]、E[rrᵀ]、E[sθ]、E[sθsθᵀ]、E[rsθᵀ]。

当前旧记忆只有既有CLS/patch坐标的一二阶矩，不能一般恢复新增非线性支路的后三项。Specific的零初始化能令起点输出为零，不使训练后的历史期望和交叉项保持零。没有读取旧图像/逐图特征的权限就不能在各任务末重算这些量，也不能用当前类拟合出的关系冒称旧类真实统计。

如果按router条件选择专家，历史所需量进一步包括E[1_route=k r]和E[1_route=k rrᵀ]等条件矩；已知路由函数与总体均值/协方差一般不足以决定这些截断分布量。原文类Gaussian可定义一个近似生成假设，但它不是对真实旧类条件矩的精确恢复。当前每层routing Gaussian也未在原样本到达时保存，不能事后由最终层Gram变出。

另一种做法是完全改为原文专家分类器/原型推理，放弃原统一解析头的矩闭合要求。这在另立协议时可能成立，但会同时改变模型、历史状态、推理与类别竞争，属于完整新路线而非本提议的最小机制对照；不能继承原27.33%参照的访问/记忆或把旧对照当等模型消融。

## 3. 扩秩与保护到底保证什么

原文Specific LoRA为α/r·BA。扩到r′时旧B列乘r′/r、新B列为零，才能补偿分母变化、保证扩展瞬间同一输入的输出相同。随后新切片受训练，函数可改变。旧前缀梯度mask还须结合optimizer state和weight decay处理才保证旧参数确实不动；该代数恒等式本身不是长期旧类函数或分类保护。

原文也明确：共享祖先/输入改变会影响冻结专家的输出；即便旧专家函数不变，新类分数和路由竞争仍能改变旧类预测。其方向梯度投影不产生额外旧样本信息。附录reference配置的历史expert学习率乘子为零，不能以“执行了投影”宣称这些旧expert实际持续适配。

若只在固定最终特征后接线性低秩Specific再接完整线性头，可逆时不增加线性函数族；rank/penalty几何变化另算。若把它放进真实Transformer的后续非线性层，则可能改变信息，但回到上一节历史统计缺量的问题。不能在这两个位置间偷换解释。

## 4. 真实执行路径的归因限制

原文给出prototype-only中fusion权重1、neutral模式Specific尺度0的配置。该路径仍可能计算route和训练共享层，但route refinement不会通过已置零的Specific输出直接改变最终neutral特征。不能据路由指标或扩秩激活就宣称它们直接贡献了该路径的top-1；需要一个确实消费Specific的固定预测路径及相应对照。

迁移本路线若依赖原型迁移、count shrink、logit融合或路由门控才能运行，又会进入已关闭的source3/risk-shrink/分数融合邻近机制。source3微小增益并不证明论文原型迁移无效，但不允许把同一输入输出功能换成DualTrack名字再试。旧APART AdapterPool已有基于CLS/key的专家混合，原方法有regular/few双支路及global/local统计；双流这个名称本身不构成独立机制。

## 5. 准入条件逐项判定

| 必须满足的问题 | 当前可证明状态 | 决定 |
|---|---|---|
| routing全祖先是否固定 | 原文不固定；完全冻结变体未定义完整训练函数 | 未满足 |
| 新specific的旧类均值/二阶/交叉项来自何处 | 原旧Gram一般不足；禁止偷偷读全历史缓存 | 未满足 |
| router条件历史统计是否已合法保存 | 当前最终层统计不能恢复每层/每专家条件量 | 未满足 |
| 预测是否实际消费specific | 原文neutral prototype-only路径不直接消费；本迁移未另冻结路径 | 未满足 |
| 是否区别线性正则与真实非线性增量 | 支路位置未冻结，不能同时声称线性闭合与新信息 | 未满足 |
| 是否区别既有双支路/PMT/响应/融合 | 只按当前口号不能区别 | 未满足 |
| 是否使用未来类/旧逐样本/任务ID推理 | 原则上均禁止，尚无可满足前三项的唯一实现 | 不凭原则声明通过 |

因此没有进入性能实验的唯一候选，不冻结科学预算、不创建guard/进程，不设“跑了再看”的参数网格。拒绝的是此提议的当前实现完备性与解释，不是要求分布无关准确率定理才准实验；一个明确承认近似、来源合法且能被最小对照证伪的新设计仍可重新准入，但不能事后沿用本卡称已获批准。

## 6. 什么可以保留

保留“路由坐标与适配坐标分离”“扩容起点保持函数”“保护必须看真实推理路径”作为设计原则。它们尚不是本项目已采用、已验证的机制，不新增融合论文计数。18篇全文已读状态不变。

下一具体工作应将研究对象切回原方法已定义的训练函数：先明确其稳定特征基、可训练分支和实际推理路径，再比较任何新增机制需要的记忆与现有状态。若选择另起独立专家路线，必须明确离开原统一统计读出的哪些约束、保留哪些公平对照和预算；不能把本卡的拒绝理解为默认允许完整DualTrack复现。持续自主研究保持有效，不为填满设备或增加引用而启动不完整方案。

## 证据与交付

- 原文§3.2：shared/General与两个递推；§3.3：类Gaussian与expert能量；§3.4：扩秩瞬间恒等式及方向保护；§3.5与附录B.4–B.6：原型/neutral/routed真实预测路径。没有重新访问网上页面或下载第三方全文，全文不公开。
- 当前代码：third_party/APART/backbone/vision_transformer_adapter_pool_a.py（Adapter/AdapterPool），tools/module_additions.py（regular/few支路局部统计/评分），tools/paired_moment_transport.py、tools/affine_moments.py（旧运输），docs/representation_memory_audit1_20261021/COMPATIBILITY_ZH.md（函数族/矩闭合）。
- 本轮纯本地阅读与文档推导，0服务器/模型/图像/特征/拟合/评分/工程自检运行，无新实验时钟；客户端阅读时间未单测，不填作0。没有新增科学效果或修改任何旧门与成本。
