# 同状态四固定候选诊断协议（未启动）

当前授权仅设计、CPU 检查和准备冻结。没有医学 checkpoint/val 读取、生产 launcher 或 worker、真实预检、GPU启动、训练、参数选择；test/reserved 封存。未来执行需单独授权，接生产入口并冻结实际运行源，不能把本准备 SHA 当作已经部署。

复用 base/local_transport 的 ISIC/HK 18 个 stage checkpoint（四条控制×数据集轨迹）；统计、标签 order/task/seed74002、源与原更新计数须精确匹配 PROTOCOL_LOCK。只允许读取既有 checkpoint 所含聚合统计，不重看 train 图、不重训、不修订方向。不能重复 collect/evaluate 已交付轮。

每 stage 同一状态生成同一原 ray 的四候选 .125/.25/.5/1（72 sidecar），原量化和旧类对比约束、R、能量帽、守卫保留。报告八种 readout：OFF、C0125、C025、C05、C1、LEGACY_SELECTED、JOINT_POINT_SELECTED、JOINT_PRED_SELECTED。选择头只是候选索引或 identity，不重新拟合头。首 task 无两旧类时全部 identity。旧准入与源必须匹配原公开诊断；不存在旧证据时标 missing，不补造旧界。主方法固定 JOINT_PRED_SELECTED，主控制固定 local_transport。

## CPU 和真实准入屏障

设计 CPU 验收拒绝医学/GPU/test/改步长等权限；合成正负 oracle、23类相关赢家、singleton/零散度/重尾、native方向一致及1536×23→46维接口。当前检查不替代高维实际生产验收。未来补全 18stage 私有 CPU sidecar 后，4真实零更新预检必须都 PASS，源、metadata、零 optimizer/policy、test=false、头/银行不变、新列与守卫、新旧标签及分母一致、有限/solver≤1e−8；通过后四轨迹统一 val，不提前读取分类指标。

未来仅GPU0/1，GPU2/3禁止，不影响他人任务。4预检+4共享评价=正常8GPU进程，每次cost独立保留；统一一次encoder/sharedval曝光15966，预检256（每条最多一个64batch），不按八个头重算影像曝光。具体实际 batch 不足须据实记账。没有 optimizer、回放、feature/val方向拟合。未来实际 launch起4h包含准备/预检/评价，不伪造当前 deadline。失败保存私有原因/全部费用、匿名缺失范围，不 retry。

## 必须交付的正负结果

1. 2dataset×2control×8readout=32行主表；全部六指标(final/avg BA、tail、forget、old/new recall)、final/avg Acc、macroF1及原提供项。缺失为 null。AIR 两行只复用公开结果/成本。
2. 四候选逐步长相对 OFF、两个新选择相对 OFF/LEGACY 的全部六增量；两控制上对应增量的交互差。列全部stage和31类 n、recall、纠正/破坏/净计数、wrong→wrong、不变，宏净纠错=sum_c((rescue_c−harm_c)/n_c)/K，严禁以总样本准确率代替类均衡收益。分母为固定 val_counts。
3. 旧旧、旧新、新旧、新新混淆；逐类原正确新类是否破坏、原错误新类是否救回、新类间隔分布及负尾；RAW→GUARDED 纠正/破坏/被挡救回单列。不得用 val 给候选选择权或修改守卫。
4. 每候选 JOINT_POINT/PRED 三模型中心净值/数值半宽/未知旧新质量/情景净下界、实际 val宏净、误差方向、原联合界；分开判断方向是否实际有益、估计是否排序一致、准入是否失去判别力。模型选中但 val负也完整公开；identity 不等于没有有效方向。
5. 同四方向 native能量/量化/solver/元数据，prior可用/数据识别/真实support/不增加ESS、借用情景、预测df、score积分节点/舍弃covtrace。所有成败费用逐次驻留，CPU pipeline包含于驻留不相加；公共控制/AIR仅复用不重计，未知模型cost留null。

独立 public RESULTS/CONTRIBUTIONS/AUDIT/REPORT、逐stage/class/flows/误差与成本；禁止公开权重、影像、逐样本预测/分数/ID/features/means/矩阵/private配置/raw日志。GitHub显式代理 push，remote SHA及精确REPORT匿名HTTP200核实，然后 NAS独立delivery。上述是未来协议；当前只交付 design/CPU/PREPARATION，不制造实验结果。

单 seed/order、反复development val、无独立患者校验，既非封存test结果也非统计显著或临床证据。本轮诊断不追加种子、步长、方向、预算或新机制；完整后再决定是否改模块。
