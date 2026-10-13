# 可干预的净类均衡纠错：Guarded Class Gain（设计与合成CPU验收）

状态CPU_DESIGN_ONLY。独立数学helper；没有生产worker/launcher，没有医学影像、医学checkpoint、val读取或GPU。test/reserved封存。固定主控制local_transport。非官方统计读出设计，不称原ConCM或临床有效方法。

## 1. 从本轮证据改变研究目标

依据完整交付ec24c67的四控制状态、七头及126阶段×头诊断，以及用户提供的GPT Pro分析：

- 主组合最终BA四格全部为负；样本净纠正为正不等于宏召回净收益为正。HK/base三纠正的贡献+.065876/+.071276/+.027693pp，被类22一破坏的−.207039pp抵消。
- ISIC/local TAIL的forget减少.450450pp由阶段2类5的22/37→21/37造成，终局仍15/37；它降低历史峰值，没有改善终局保持。
- TAIL单独与加在MPC上的贡献会反向。总体RMS小不能解释关键边界，不能由此直接加大强度。
- 原供体核的相似性部分最多3:1，rank补满不代表信息可信。新版借用只形成风险模型的先验和压力情景，不修改原Q/R/pairs。
- 现有guard额外纠正/破坏计数为0，不能把失败归于它；但它严格限制可修复错误集合。

目标改为每类条件概率的等权净收益，而不是把原错误质量、借用量或ETF作用量当效用：

    U(delta) = 1/K sum_c [P(OFF错且ON对 | c) - P(OFF对且ON错 | c)]

当前只构造这个量的统计模型区间和一个受限候选方向，不保证真实U为正。不使用val分母、实际val混淆或已看到的个别类别来设权重。

## 2. 守卫的精确可干预范围

保留原新列和旧类contrast增量，Delta_new=0、Delta_old 1=0。读出复用上轮old-max单向守卫。

令OFF分数s，RAW=s+x^TDelta，b=max(s_old)，n=max(s_new)。在相同argmax规则（旧列在前、相同分数选较小index）下，有：

    OFF赢家为新类 => ON赢家严格保留OFF赢家
    OFF赢家为旧类 => ON赢家等于RAW赢家

证明：原旧赢家意味着b>=n；raw旧最大值被裁为min(max(raw_old),b)，仍不会把raw旧赢家改成新赢家。原新赢家意味着n>b，裁后旧最大值<=b<n，新列不变。

因此救回只统计“原赢家a为旧类且a!=真实类c、RAW赢家变为c”；破坏只统计“真实类c为旧类、原赢家为c、RAW赢家变为别类b”。新类破坏概率严格0。旧→新原错误、新→另一新原错误排除，不能虚报为可救回质量。

更强的必要条件：旧类零和增量保留原旧类平均分。守卫后的旧最大值也不会低于原旧平均分；若新类最高分<=原旧平均分，则这个输入的新→旧错误也不可由本结构救回。这是数学范围，不是对当前val逐样本比例的实测。

这是非线性readout保护，不是纯线性分类头；当前新类变旧后，不再受这一保护。它不保护所有旧类或新类precision。

## 3. 联合赢家转移同时表示救回与破坏

对真实类c的组件r，均值u_cr保留，构造原经验共享形状的高斯矩近似，以及借用产生的模型情景。没有抽取/生成旧特征，也没有旧图回放。

给定候选W1=W_saved+Delta，定义联合事件：

    E(a,b) = {W_saved下赢家为a AND W1下赢家为b}

每个赢家事件包含对全部其他已见类的比较。对较小index竞争者使用严格大于，以正确处理确定性组件的tie；非退化高斯下tie概率为0。不同(a,b)赢家转移互斥，避免把重叠pairwise尾部直接相加。

统计投影采用全类Helmert contrast C：

    T = [W_saved C, Delta C]
    joint_mean = u_cr^T T
    joint_covariance = T^T Sigma_cr T

原/候选联合score空间最多2(K−1)，HK最终44维；不会把23×1536的全参数Hessian构造出来。原始特征covariance仍来自合法聚合银行，用于投影，不生成数据。

救回概率sum_{a<old,a!=c} P(E(a,c))；旧类破坏概率sum_{b!=c} P(E(c,b))。组件按原质量混合，各类别最终固定1/K；n小不降低它在宏目标中的重要性。

## 4. 模型概率界与可信度边界

当前没有真实旧分布、组件完整covariance或独立患者计数。完整联合Gaussian积分也不是必需的CPU依赖。本版用保守界，避免给不可靠代理过高精度的外观。

对于赢家转移的全部半空间约束，利用一维正态边际：

    lower P(intersection) = max(0,1−sum_i (1−lower P_i))
    upper P(intersection) = min_i upper P_i

这是Bonferroni/Fréchet界，无须假设不同margin独立。上界再与必要边界反转的概率界取min。设原a−b间隔M、候选变化D；转移必须满足M>=0、M+D<=0，因此对任意epsilon>=0：

    P(转移) <= P(0<=M<=epsilon) + P(D<=−epsilon)

内部epsilon只取固定的0/.25/.5/1/2/4/8乘候选变化的模型尺度，不使用val。实现包括区间概率对均值、方差的内点极值；Delta=0或a−b完全不变时转移严格0，不把数值残差当救回。

每组件情景还包含明确的矩误差范围：每margin均值偏差界2*sqrt(envelope_variance/n_r)，covariance统一尺度区间[max(0,1−f),1+f]，f=2*sqrt(2/max(n_r−1,1))。这些尺度来自Gaussian/iid图像和均匀组件权重假设，属于预先固定的压力范围；不是经过医学校准的95%置信区间，不能声称分布无关覆盖或独立患者ESS。

固定原控制源码0cfe6b6/0f0d8a0的boundary代码都用zero actions调用sample_weights，因而alpha=1/n_c；组件内权重均匀，n_r可以作为该假设下的图像支持量。未来必须重新验收这项来源及组件质量=count/n；加权或换控制不能直接套用。患者/眼底重复和相关性未知时，实际有效支持量可能更小。

模型或矩误差范围以外，概率界没有保证。组件count=1且没有可识别的方差先验时，救回区间置[0,1]、旧类破坏置[0,1]，不伪造确定性。判别所需统计不可识别时报告拒绝，允许原控制identity。

## 5. 统计借用提供压力情景，不改原解析系统

用原控制实际W_saved的全类score contrast张成的特征方向B，对全类类内covariance T_c=Q_c−mu_c mu_c^T作投影。供体只有其他类、支持>=2且投影trace>0的类。投影shape normalize后：

    w_cj ∝ [n_j/(n_j+16)] exp(−||shape_c−shape_j||²/.1)
    eta_c = .4*16/(n_c+16)

这是前瞻固定的方向核，没有top-k/val温度选择；核锐化本身不构成可靠性证明。保留全部供体标签/标量权重、有效数以及情景差异用于将来匿名审计，不能只报告rank。

供体prior用目标组件经验radius作尺度；零经验radius时用合法银行非零类内trace的中位数作未识别方差先验。该prior是借来的假设，不是目标类新观测。风险情景固定lambda=0/eta_c/2/eta_c；组件中心、质量、支持量不变，原bank/Q、原R、原pairs完全不改。

envelope=sum情景covariance，PSD上支配每个情景；计算矩误差范围时不将供体count加到目标n。情景分别计算救回与破坏界，再取救回最坏下界和破坏最坏上界。方向不一致会扩大风险区间而非制造确信。

新版区分：样本离散程度、供体先验、估计误差假设、患者独立性缺失。当前仅最后的声明诚实且数学界在声明范围内可核验；真实可信度仍需训练侧校准。

## 6. 解析兼容：方向提案、原曲率和有限步长准入分开

方向提案由旧-旧winner facet构造。对c,j旧类边，采用各组件原间隔的平滑边界密度、条件均值，以及其他全部类别不越过该边的联合赢家概率界中点。平滑score噪声固定为.1倍原判别score RMS，只用于提案，不用于接受风险模型，也不生成特征。

条件赢家概率中点×条件均值没有包含全部联合截断一阶矩，因此g明确是方向启发式，不声称精确U梯度。只在旧-旧facet提出方向；新类可能因旧竞争者降低而恢复，但不会以缺失的旧→新或新→新可修复性为由造力。

原H0包含全部native平方损失、.001R及.5 competition；W0需为精确double原最优解。原保存float32头仅用作实际OFF和风险状态，不把float重构误差当g。

    Delta=Z V_old^T
    A0(Z)=H0(Z V_old^T)V_old
    A0 Z = g V_old

这是一条SPD线性系统的方向提案，matrix-free PCG沿用128步、停止1e−9、拒绝残差>1e−8，CPU用独立稠密二次目标oracle验证。原W0/Q/R/pairs不改；没有optimizer、训练反馈或ETF。

不沿用上轮0.1小能量硬限制。提案ray最多具有原旧contrast头相同native能量（relative radius1）；它只是预先定义的ray域，不是对完整受约束净收益优化的闭式解。随后只评估五个固定候选scale=0/1/8/1/4/1/2/1，先量化到float32 delta，再在模型中验收。

    certificate_lower = macro rescue_lower − macro harm_upper

选择下界最大且>1e−12的候选；相同分数保留较小步长。全部非正就严格identity。内部五步选择只使用训练聚合，不是val网格或五个医学预算臂；计算次数与CPU费用需公开。保守界可能过宽、使真实有益修正也被拒绝，这是明确失败风险，不能在结果后改用点估计放行。

最终还经过单向max守卫，因此整个算法是“原曲率线性方向＋模型有限步长选择＋非线性读出”，不能称直接求解真实BA，也不保证真实单类或整体BA提升。

## 7. 与local_transport互补和贡献检查

local_transport负责原R及共同统计运输。本设计在其自己的W_saved和原H0下判断尚余的可干预净收益；不重复运输、移动均值或改R，不根据base/val选择控制。图像/optimizer预算0，只增加统计计算、delta侧车和一次统一特征推理（当前尚未授权/接入）。

未来主四格A0/AL/B/P，B/P固定GAIN。另固定三个贡献头：GAIN_NO_BORROW（原情景；矩误差和净门保持）、GAIN_RESCUE_ONLY（同提案/同借用，准入只看救回，明确允许模型预测的旧类破坏）、NEUTRAL_MATCHED（固定seed1074005随机旧contrast方向，匹配GAIN接受delta的guard前native能量，不使用风险门）。稳定项单独为原控制identity，只CPU检查，不加医学臂。

NEUTRAL_MATCHED匹配的是native能量，不是guard后的分数RMS/患者实际影响。GAIN拒绝时neutral也为0，不能据二者相同声称方向有效。必须报告作用量和非零案例覆盖，以及主组合相对local_transport本身的贡献；正交互是额外、更强的命题。

## 8. 已知风险与下一准入屏障

统计模型错误、共同运输不能代表真实旧类漂移、组件共享covariance假设、矩误差未校准、稀有类小分母、score舍入、保守概率界太宽、方向提案不等于真实净收益梯度、旧类保护不足、反复development val引导结构选择均保留。

当前CPU合成检查即使PASS，也不是1536×23医学完整求解、GPU零更新预检或分类效果。新的生产准备/共享缓存、元数据/原控制float32精确回归、风险区间与实际救回破坏对应、病例独立性和将来验证范围都未完成。本轮不接任何生产/GPU入口。
