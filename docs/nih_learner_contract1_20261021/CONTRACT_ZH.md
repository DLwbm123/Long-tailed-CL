# NIH learner调用链与状态契约

状态：SOURCE_RECEIPT_BOUND_NO_NEW_EXPERIMENT。仅核对现有源码、协议与封存聚合回执；为补本地缺失的local_evidence父节点，远程只读其run.py/config/status文本。未读取模型/统计张量、逐图特征/图像/预测，未启动实验、矩求解、训练或评分。

## 1. 最重要的术语澄清

NIH确实存在完整可训练原方法的历史实测，不能说可训练双分支只在ISIC/HK运行。2026-10-11的author20配对：base最终24.6667%、平均28.3192%，local_transport最终24.0000%、平均28.0924%，3412正式adapter更新（每臂1706）。local_transport最终及尾类门失败，已关闭。

当前反复引用的27.3333%是另一条**冻结CLS/patch强解析参照**，不是上述完整可训练原方法的同一checkpoint/损失。后续称A为“原可训练NIH方法”，B为“冻结强联合参照”。两者使用相同author20计数和显式类序，但特征定义、归一化、权重、损失、内部fit/meta与频率分组不同，不能直接把82−74张归因给冻结、patch或去掉某个模块。

原可训练NIH尾类按n<100定义4类/60图；冻结分支tail7为频率排序7类/105图。原base尾25%与强联合20/105不能作同组差值。作者train/val患者不交叉，也不代表旧训练内部80/20 image-identity fit/meta患者隔离。

## 2. A：原可训练NIH真实调用链

执行源由旧NIH `results/AUDIT.json`绑定到41e6545c514b90b2b771d3eb7661cc9042a7977d；结果报告提交f871dba41122df54f55ad9509827c2024c82eedd。核对历史源，不把后来加入的PMT/anchor/guard等可选分支都当NIH已执行。

```text
NIHCXRLT author配置/显式类序
  → tools/run_module_additions.py: run / author_nih
  → ApartFeatures（tools/run_multilabel.py）
  → ct3p_core.features：pointwise调用，regular/few pre_logits拼接并整体L2归一
  → 当期canonical特征与image-identity fit/meta
  → prototype_coherent.seed_components / memberships / sample_weights
  → common_shift → translate旧bank → append当期fit统计
  → modules.head：原型metric R，local_transport另加四区域证据
  → core1_competition + edge.Controller(static_pc)：固定竞争，无策略更新
  → proximal_head / competition.solve（PCG）
  → 当前mini-batch平方拟合+pair+旧平方风险+ridge+proximal+10×FD
  → backward / clip5 / AdamW，仅两个adapter池可训练
  → 当期epoch末重抽特征，任务末全部当前train（含meta）重拟合
  → stage checkpoint封存后，单线性全局解析头推理
```

ApartFeatures只令`.pool.pool.`与`.pool_few.pool.`参数requires_grad；编码器eval模式不等于adapter不更新。Task-start teacher拷贝冻结，FD在当前图像上比较1536维联合特征。原始预训练backbone与非adapter参数不按这个训练循环更新。1536维是regular/few各768维，**不是final CLS＋patch mean**。

原分支每类质量1，零动作时类内图像均匀；component用于原型证据与先验，不能把它们称患者随机效应。内部meta参与controller/任务末拟合，不是独立验证。2epoch/任务、AdamW3e-4/WD.01，原已封存更新1504/26/84/12/22/58，各臂1706；本次没有重跑。

### A的持久状态与不变量

- bank：每类mu、原始Q、样本n；components保存label/count/mass/center/radius。当前样本到达时构造，旧原图/逐样本特征不回放。
- 旧统计使用当前类均衡共同平移；给定shift的矩公式精确，不声称真实旧类变化恰为该平移。
- prototype metric是R=I−P(I+PᵀP)⁻¹Pᵀ；有向平方竞争已进入原训练/解析求解，不能称新机制。
- local_transport的四区域中心通过当前区域公共位移运输，并进入先验；NIH该臂无新增local CE、local logit融合或额外推理分数。
- checkpoint含adapter、head、bank、seen、steps、boundary pair weights，以及对应model/局部memory等。它们属于该动态表示坐标；不能与B的Gram/head互换或默认可作新分支初始化。
- inference按保存seen/head在相应阶段encoder特征上argmax，既无真实任务ID oracle，也不是DualTrack类Gaussian路由系统。

## 3. B：冻结强联合27.33%的实际依赖链

```text
local_evidence_v1（已封存父节点，源文本本次只读核对）
  frozen timm ViT-B/16 ImageNet augreg，FP16 forward_features
  保存原始CLS768的增广Gram/target与128投影统计（不是adapter更新）
  → docs/matched_heads1_20261018/run.py
    补每图patch mean768 Gram，核对CLS边际；CLS768/128与patch读出对照
  → docs/joint_heads1_20261019/run.py
    补CLS×patch跨矩，拼1537维Gram；原联合/duplicate等控制
  → docs/joint_controls1_20261019/run.py
    仅前10基类能量固定patch尺度；每阶段训练df匹配CLS alpha1
    → scale_joint / df_matched_joint
  → 保存六头后打开query，原stage scores直接复用
```

这里encoder.eval().requires_grad_(False)，inference_mode前向；原始非归一化CLS与patch mean，不经过ApartFeatures regular/few池。每类总质量1，患者等权、患者内图像均分。数据Resize224、Normalize.5/.5；alpha包括偏置罚。local_evidence父源远程status为COMPLETE，seed74002/68058/300/同显式order，源码明确冻结；仅查文本，没有重新读取权重或张量。

B每类保存CLS增广769 Gram/target、patch mean增广769 Gram/target、768×768跨矩；联合向量为[CLS/√2,s×patch/√2,1]，s只由基类能量决定。阶段头仅依赖该seen前缀统计，未来类不参与基类尺度/早期目标。实现可以先顺序收集各类统计后统一重解六个前缀头；这不是连续训练出六个encoder。所有阶段头封存才统一评价，不能把最终20类头用于早期CIL。

B的tail7/中6/头7按全20类训练频次排序，只作固定诊断，不混用A的7/9/4阈值分组。B是本轮新表示/固定读出探针共用强参照，旧强参照的创建费用不因复用归零。

## 4. C：静态参考与离线读出不属于A/B的同协议训练

静态IN/CLIP系列是joint20、cosine-LA头、自然采样、随机增强/30轮；用户停止并清理，不恢复。ImageNet full best31%/last18%，非稳定上界。离线CE则读取B全部历史逐样本特征，94/300但未收敛且尾门失败。两者不能初始化A/B早期任务，也不能当A/B相同权限下的增量收益。

## 5. 已运行与仅存在于代码的边界

| 组件 | NIH证据 | 准确表述 |
|---|---|---|
| regular/few双adapter＋FD/共同平移/原型先验/平方竞争 | A两条训练及审计3412更新 | NIH历史实测；非只在ISIC/HK |
| 四区域local_transport先验 | A候选臂，local CE/推理融合关闭 | 实测但未过门，不恢复 |
| 原始CLS＋patch mean联合df匹配 | B完整六阶段82/300 | 冻结强参照，非A完整原方法 |
| 当前源码后来增加的PMT/decision-anchor/其他可选模块 | 历史41e6545路径及NIH锁未启用这些选项 | 不能因import或有函数就称NIH已测 |
| APART AdapterPool | A确实调用其feature wrapper；完整第三方训练算法另论 | 不能说整套APART在NIH已复现 |
| DualTrack独立routing/扩秩/类Gaussian | 没有本路线启动/运行记录 | 准入卡拒绝最小迁移，不列已实现 |
| Spatial signed-p3/native hidden等 | B坐标下固定新统计探针 | 已执行但门失败，非A的神经训练机制消融 |

## 6. 后继决策契约

当前冻结强联合B继续作为探针参照，A保留为历史完整原方法证据；不把两者统称同一个“原方法”。若研究要回到A，必须明确改动是在可训练adapter、旧坐标更新、R/竞争还是推理；不能把B后面的CPU小改称A适配改进。

若在B中接入A的原型先验/平方竞争，也不是默认新贡献：这些函数已在A和旧EDGE-BUDGET存在，必须先检查新假说是否只是重跑已关闭先验/竞争/读出，不能为维持运行把迁移数据集/坐标当机制新颖性。若改成真非线性神经训练，则不能沿用B旧矩而忽略坐标变化，须明确新旧记忆信息来源。

本次完成的是调用链/契约，不新增可启动候选。下一科学协议必须只选择A或B一个明示基准，写清相对哪一个已运行函数的不同、训练证据、最小对照与停止条件；没有新证据时继续设计，不启动重复轨迹。持续授权与全部已关闭限制保留。

## 来源与限制

A：docs/nih_cxr_lt1_20261011/{PROTOCOL_LOCK.json,results/AUDIT.json,results/REPORT_ZH.md}及绑定41e6545源；B：matched/joint_heads/joint_controls的PLAN、run、config、PARITY/VERIFICATION和封存报告。local_evidence父节点源码/配置/status仅远程文本读取，未重新验收其模型内容。源与回执绑定不等于全量历史tensor内容重新核验，本轮不声称后者。

本轮0图像/特征/模型/预测读取，0训练/求解/评分/实验启动；远程文本只读检查2次，客户端时间未单测。公开函数和聚合状态契约，不公开机器路径、身份、模型、统计矩、样本级结果或私有配置。两个原有未跟踪tools脚本未改。
