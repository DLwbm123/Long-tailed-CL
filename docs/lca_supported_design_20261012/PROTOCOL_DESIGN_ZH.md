# LCA-inspired 旧类残余判别风险：设计协议

状态：仅设计与合成 CPU 检查，未授权 GPU 实验。新增模块不由任何生产 worker 导入；没有训练入口、launcher 或自动排队。DAT/Hao 的部署源 `6dabf8fc4c618ac004dc46a73b8e50a8a535119d`、数据、配置、预算、进程、评价时序与监测 scope 均不修改。test/reserved 封存；不读取当前 val、医学影像或旧 checkpoint。

参考：[LCA 原论文](https://arxiv.org/html/2603.09888v1)，特别是 §3.3 的类邻域损失敏感度和 §3.4 对原型变化的限制。原方法包含模型合并、Gaussian 特征采样和分类器再训练。本设计没有这些操作，也不移植其泛化定理。我们只借鉴邻域预测稳定性，采用统计矩与二次解析改编，不能称完整 LCA 或作者实现。

## 上一版的证据与问题边界

复用已公开 `module_drift10_20261011/results`，不重评：ISIC 漂移单独最终 BA +0.893847pp，组合相对 local_transport +0.297077pp；组合新类平均召回 -1.25pp。HK 相应两臂各项指标与原控制相同。U 非零不是净效用证据。以下是待验证的设计假设，不是已确认的失败因果：

1. 当前类残余二阶矩不一定代表旧类残余风险。
2. 对所有 W 列使用同一额外惩罚，可能牵制新类学习。
3. 新惩罚可能抵消 local_transport 在特定方向放宽的原先验。

## 1. 旧类估计：可观察范围内的仿射外推

仅使用任务开始 teacher 与当前 encoder 对同一批 current-fit 图像的配对全局特征。权重使用原 native current class mass，归一化后为 a。原共同位移 Delta 原封不动，先计算 e_i=z_after_i-z_before_i-Delta。

令 m=sum(a_i z_before_i)，e_bar=sum(a_i e_i)，X_i=z_before_i-m，C=X' diag(a) X。固定 tau=.001，B=Pi_C(C+tau I)^-1 X' diag(a)(e-e_bar)，其中 Pi_C 是当前协方差的数值支持子空间，float64 机器精度定秩。r_hat(z)=(z-m)B+e_bar。没有沿未知方向人为补全漂移。

输入旧类是 **task-start 原 bank** 的 mu_c,Q_c，同 before 特征坐标；不能输入已经 translate 的 bank 又加一次 Delta。旧类协方差 Sigma_c=Q_c-mu_c mu_c'。预测风险 U_c=B' Sigma_c B + r_hat(mu_c)'r_hat(mu_c)，U_old 是旧类等质量平均。没有旧图像、历史逐样本特征或 Gaussian 采样；既有 mu/Q/n/components 不被替换，不引入推测的新类漂移数据。

这不是直接测到的旧类漂移。必须报告每个旧类关于当前支持空间的能量覆盖 trace(Pi_C E[(z-m)(z-m)'])/trace(E[(z-m)(z-m)'])、支持 rank、fit 残差 MSE、map/intercept 范数。覆盖低不能称该类风险估计可靠。没有旧类时 U=0；纯共同平移时只允许数值零。旧类仅位于未观察方向时可能 U=0，必须报告未覆盖能量，不能把零估计解释成旧类无漂移。

共同仿射漂移、二阶矩传播只是模型假设：当前类非线性/类别特异漂移不必可迁移到旧类。本 CPU 的显式样本 oracle 只证明在指定合成映射下矩公式正确。

## 2. 去重：区分位移与先验，而不是删掉所有区域中心

global common_shift 已在 e 中减去。local_transport 的区域共同位移仍仅用于原区域中心运输；它不自动成为全局特征的逐样本漂移校正，不能把所有局部方向叫“已修复方向”。

在同一 candidate bank 上分别计算原 R_base 与 R_local。Delta_R=R_base-R_local 是新增局部证据带来的先验放宽，理论上 PSD。取其数值列空间基 V，P_perp=I-VV'，U_filtered=P_perp U_old P_perp，两次投影消除数值泄漏。只删除实际先验放宽方向的重叠，原全局证据方向不盲目删除，保留能量不重新归一化。

必须报告 Delta_R rank、raw/retained risk trace、leakage、全去除情况。这个操作的准确解释是“避免额外风险惩罚抵消局部先验放宽”，不是证明局部运输已校正漂移。可能删除有用约束；删除方向同时减少强度，所以 AD-AO 不能证明等强度下的方向单因素因果。

## 3. 有界尺度与新类保护

参考曲率 H_ref 固定在 epoch/boundary：candidate 的类均衡 Gram+rho R；epoch 还加原 proximal .01I，boundary 无 proximal。rho=.001。用原始和去重后风险的 H_ref 白化最大特征值 v_raw,v_retained，S=.1 U_used/(1+max(v_raw,v_retained))。未去重臂两者相同。固定 .1 是相对曲率上界，不是旧版绝对系统系数 .1；不声称数值等强度。S<=.1 H_ref，消除方向后不放大残留风险。原型、正则化与原配对损失参数不变。

完整臂令 M 的 old block 为 I-11'/K_old，新类行列全部零。新增目标：

`J = J_native + .5 trace(W' S W M)`

这约束旧类之间的 score 差异对预测残余的敏感性，保留旧类共同分数自由度。新增项对新类输出列的直接梯度严格为零；无旧类或只有一个旧类时 M=0。它 **不保证新类 recall 不下降**：原竞争项仍耦合旧/新输出，旧类分数变化仍可能改变 argmax；也没有保护所有 old-vs-new margins。全部新类得失必须报告。

每个 epoch 内 B/U/S/M 全 detach/freeze，boundary 按原全 current train（含原 meta）重估。没有 val gating、幅度搜索、推理修正或新学习头。初始 before=after 时约束数值零，仅为路径检查，不冒充激活。

## 4. 解析头接口与适用范围

由于右侧有输出空间矩阵 M，不能通过 `R_eff=R+constant*U` 把此项折叠成所有类别共享 prior。新增 Hessian 算子是 `native_H(W)+S@W@M`，原监督 RHS、竞争 RHS、原 mu/Q/n 和 pair mass 不变。两项均 PSD，原 ridge 保证严格凸。用已有 PCG，复用原 Cholesky/Woodbury 作为预条件器；不构造 (1536*K)^2 全 Hessian。

原 reference head/pair 构造继续走原 control 路径，以防把改变竞争参考混进贡献。正式生产接线必须让 controller 中的候选解、每 batch competition/proximal 解、显式 loss、boundary 解共享同一 frozen S/M；zero 模块应返回原算术路径。现有 CPU 只核对独立 solver 接口和原函数，尚未接生产 trainer。

单标签检查包括显式 Hessian oracle、beta0 与 .5、proximal direct/Woodbury 一致、frozen-prior envelope gradient、1536维合成解、HK23列分区及 checkpoint 版本/形状拒绝。CPU 合成 checkpoint roundtrip 为容差一致，不声称正式医学 checkpoint 已验证。

多标签只做兼容数学检查：逐 label 正负 signed ridge 仍独立，不对不同医学标签作互斥 centering。旧 label 可以在其 feature block 加 S/rho，新 label 原 prior 精确保留，bias 不加风险项，n/s/G、signed RHS、unknown mask 原封。估计必须按旧 label 的两个 sign 等质量 moments，而不是把单标签旧类平均直接搬过去。本协议不安排 RFMiD 或多标签生产接线。

## 5. 贡献对照与预算（全部是未来提案）

主四格：A0=原 base，AL=固定主方法 local_transport，AB=base+完整旧类 margin 风险，AP=local_transport+完整风险。严格匹配时复用 A0/AL 全阶段公开结果，不读取旧 NAS 模型或重新评价。

机制分解用三个额外 local_transport 臂：

|格|风险来源|局部去重|输出矩阵|
|---|---|---|---|
|AC|旧版 current-fit U，采用新版统一有界尺度|否|I，所有已见类|
|AO|新版 old-affine U|否|I，所有已见类|
|AD|新版 old-affine U|是|I，所有已见类|
|AP|新版 old-affine U|是|旧类 centered block|
|AB|新版 old-affine U|无 local 增量方向|旧类 centered block|

AO-AC 检查旧类外推相对当前代理；AD-AO 检查去重和随之减少的能量；AP-AD 检查输出保护范围（同时改变约束秩，非等强度比较）。AB-A0、AP-AL、AP-AB、AP-A0 和 `(AP-AL)-(AB-A0)` 报告独立贡献与组合交互。AC/AO/AD 对 AL 全报。上一版 residual_drift/组合只作为额外历史背景，因为尺度/输出范围改变，不能单因素归因。

ISIC 每臂 stage 更新100/72/296=468；HK98/4/2/20/18/34=176。沿用 seed/split74002、原 order/task_sizes、同预训练/增强、2epoch/batch64/workers4、AdamW lr.0003/WD.01/clip5/FD10/ridge.001/proximal.01/beta.5/原 cosine eta_min1e-5、原 fit/meta80/20，不添加 val 反馈。原控制增加费用0。

只主四格：4条新轨迹、1288更新、4真零预检、正常12次 GPU费用。完整机制设计额外6条轨迹、1932更新；合计10条新轨迹、3220更新、10真零预检、正常30次 GPU费用（preflight/train/eval各10），失败全部另计。**这两种规模尚未授权，不能把准备任务变成3220步的自动实验。** 数学操作和额外 feature-stat 求解算力另列，不声称等计算预算。

每臂相对自身 control 报最终/平均BA、tail、forget、old/new recall、Acc、可得F1、每 stage 及31类n/recall、全部正零负；不按 val 选主方法。分别对 AB/AP 检查原 base 门 BA>=+1pp、tail>=-.5pp、forget<=+1pp、old/new>=-1pp；另要求 AP-AL 最终BA>=+1pp 且新类>=-1pp。所有条件双数据集达标才称开发贡献；新类下降即使门内也披露，门不是无损或临床保证。

## 6. 后续启动前必须完成

DAT/Hao 完整 GitHub/NAS 交付并实际退出；用户明确批准具体矩阵；独立生产 train/controller/proximal/boundary/checkpoint/eval/collector 接线和 CPU 回归；真实 control matching；最终公开源码冻结；NAS nfs/free/probe 与 GPU0/1准入；全部真实零 optimizer GPU 预检通过；全部轨迹精确 TRAINED 后才统一 val。test/reserved/RL/新seed/网格/自动失败重跑均禁止。真实 launch 才建立24h deadline，不在设计阶段制造运行时限或 GPU费用。

公开仅源、设计、匿名 CPU 标量与错误摘要。未来不得公开 mu/Q/B/U/S/M、权重、影像、样本ID/特征或 private runtime。单 seed、固定 order、重复 development val 仅开发证据，不代表原 LCA 复现、患者泛化、统计显著或临床结论。
