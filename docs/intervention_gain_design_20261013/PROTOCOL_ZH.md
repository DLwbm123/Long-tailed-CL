# 可干预净类均衡收益模块：贡献协议冻结

本轮仅设计与合成CPU验收。production/GPU/medical/checkpoint/val/test/training授权字段均false，0正式更新、0策略更新，不创建/恢复monitor，不修改已交付科学源，不调用已有collect/deliver或读取旧NAS模型。

设计依据ec24c67完整诊断和用户提供GPT Pro分析；数学、假设与限制见DESIGN_ZH，固定参数见PROTOCOL_LOCK。代码独立于现有workers，不冒充官方MPC/DSM。

## 如将来另行授权：固定资产与主四格

严格复用concm_statistical1冻结LOCK的base/local_transport两科学源、delivery、原18stage checkpoint、train/val counts/order/tasks/seed74002和AIR各stage公开结果。固定主方法local_transport，不按val改主控制。

ISIC18718/295、3stage；HK6821/1698、6stage；原weights zero-action均匀，支持量/provenance需精确验收。只读合法聚合，0训练图像/0optimizer/policy/0旧图回放/0生成旧特征，test/reserved封存。当前不授权上述资产读取。

四个固定读出：GAIN（主模块）、GAIN_NO_BORROW、GAIN_RESCUE_ONLY、NEUTRAL_MATCHED。两个控制共18stage×四头得到72 delta sidecars。主四格A0=base、AL=local_transport、B=base+GAIN、P=local_transport+GAIN；其他头是前瞻贡献项，不挑最好替代主模块。GAIN_NO_BORROW同时改变提案几何和风险情景，因此只能归因于完整借用路径，不能称准入门单因素消融。两AIR参考只读本地公开结果，无重评、费用0。

## 固定贡献与同状态评价

未来每个头相对各自OFF；GAIN−GAIN_NO_BORROW、GAIN−GAIN_RESCUE_ONLY、GAIN−NEUTRAL_MATCHED的全部六指标和Acc/F1，B/P交互；与AIR的BA/avg/tail/forget/old/new及Acc/F1差距。40组比较固定为：16组各头−OFF、12组GAIN−各贡献头、2组主交互、2组组合−单独、8组主四格−AIR。所有正零负完整，不把模型证书当实际收益。

原控制+四头共20行主表，AIR2参考行。raw/guarded同步完整报告。各stage全部31类n/recall和纠正/破坏/错转错，新类raw和几何margin正零负、旧旧/旧新/新旧/新新流；n<5连续摘要抑制、离散n保留。原val分母仅用于事后评价，算法类权重统一1/K。

预测与实际对照：分别报告每类救回/破坏上下界、macro净下界/上界、实际对应事件及BA；界是否包住实际仅是development描述，不是真实分布覆盖证明。记录不可修复质量、old-mean必要条件、unknown/abstain质量；不以全错误计作可救回。保存lambda各情景差异、原支持/供体先验/矩误差范围、供体标量权重及有效数、原始/接受delta作用量、native能量、neutral匹配误差、全部5候选统计和选择原因。

forget必须分解历史峰值变化和最终旧类变化；峰值下降造成的forget改善不称保持能力。阶段数值不是新的训练收益。所有旧图像/feature/患者IDs、逐样本分数、均值/covariance/矩阵和原始错误留NAS私有。

## 将来准入与费用（当前没有启动）

必须另行接入生产worker/launcher/collector，完成原源码/数据/budget/版本/metadata、uniform支持量、native double optimum与实际float32OFF回归、72 sidecar roundtrip、stage-one/zero-force identity与matrix-free缓存验收，再发布新的production science source。不得直接复用此CPU-design SHA当GPU部署源。

4条CPU准备全部封存；4真实零更新预检分别ISIC/HK×两个控制，真实首64 val影像、四头分数finite/CPU-GPU兼容/冻结encoder state精确/0optimizer/policy/testfalse，不输出分类指标。4/4 PASS后统一已有val，不能提前选checkpoint/步长。

GPU仅0/1，正常4预检+4共享评价=8独立费用；CPU准备4独立记录。只对18stage提取一次特征，val曝光15966+preflight256=16222，不能按四头重计影像/GPU费用。临时模型置信计算可能占主CPU开销，前瞻固定4h全程序上限（14400秒）从未来实际launch计时，不在准备阶段制造deadline或历史GPU费。stage缓存只有合法统计，不生成旧features。

失败保留全部实际驻留和私有原因，停止依赖，不能自动retry/放宽置信门/改变内部候选或读out。全部拒绝为identity也是结果，不能报机制成功。

## 科学判定与完成交付

固定开发门相对各自控制BA>=+1pp、tail>=−.5pp、forget<=+1pp、old>=−1pp；当前新类recall不下降为机械验收。主模块须两个数据集通过，不能只挑一个控制/数据集/极小分母类或以“净纠正数正”代替宏收益。每次支持极少n推动的门通过单列，不称稳定收益。

这仍是单seed/order及反复development val，非独立患者、临床或封存test证据。新增seed不能代替独立数据；独立确认须另行授权/冻结，不在此自动追加。

将来真实结束后完整源/协议/正负/missing/报告/结果/贡献/cost明确代理push feat/prototype-analytic，remoteSHA与精确匿名HTTP200，再NAS独立public_delivery/DELIVERY/HANDOFF，不仅存NAS。本轮只能public_preparation，不能创建实验RESULTS或COMPLETE_DELIVERED。没有医学性能结论。
