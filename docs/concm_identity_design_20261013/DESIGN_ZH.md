# MPC/DSM-inspired v2：保留类身份的协方差补全与错误尾部边界修正

状态：CPU_DESIGN_ONLY。只有合成数学检查；0医学影像、0医学checkpoint、0 GPU、0正式optimizer。没有生产worker/launcher或医学评价入口，本设计不会影响现有实验。固定主控制仍为 local_transport。非官方 ConCM-inspired，不是作者 MPC/DSM 的等价复现。

## 1. 本次诊断怎样改变设计

依据 `49ad4136066d41de0fc7c0ae0d5d31c1cbc3fb6c/docs/concm_mechanism1_20261013/results`：

- 供体和错误类别有时重合，有时并不重合；类14借18/20/13却更多错到5，说明只移动目标类均值不足以解释联合解析头改变。
- ISIC base的零目标正则化比原ETF更好；HK的ETF则相对零目标救回2例新类，证明正则化和结构方向必须拆开。
- 公共能量占比低，改用判别范数没有改变预测，所以范数修正只作为一致性处理，不作为性能根因或主要贡献。
- 原始margin平均下降包含分数尺度因素；仍有真实对→错，必须报告逐类纠正和破坏。

新版把问题分成：借用什么统计信息、怎样定位错误边界、正则化与方向怎样分别作用、怎样保护当前新类决策。不增加语义属性网络、RL、回放或训练参数。

## 2. MPC-v2：借协方差形状，保留类身份

原类c有均值mu_c、二阶矩Q_c以及组件中心u_cr、质量pi_cr和半径r_cr。严格验收组件质量和为1、加权中心等于mu_c、组件计数和支持语义相符。令

    A_c = sum_r pi_cr u_cr u_cr^T
    S_c = Q_c - A_c
    tr(S_c) = sum_r pi_cr r_cr

S_c是组件内部的合并协方差，A_c保留组件中心及其多模态间结构。只借内部形状，所有mu/u/pi/n/r保留。

非零tr(S_j)的其他类为供体；Sbar_j=S_j/tr(S_j)，q_j=n_j/(n_j+16)。冻结无温度核：

    w_cj ∝ q_j / (1 + ||Sbar_c-Sbar_j||_F²), j != c
    eta_c = .4*16/(n_c+16)
    S'_c = (1-eta_c) S_c + eta_c tr(S_c) sum_j w_cj Sbar_j
    Q'_c = A_c + S'_c

无供体/零内部方差时该类严格identity。1e−12只是固定零方差判据，不是val阈值。

性质：S'_c半正定、迹不变；均值、全部组件中心、质量、计数和半径不变；类间中心差和组件间结构不变。原prototype R及boundary competition pairs因此直接固定复用，不用补全后的Q重新选pair或修改local_transport的局部中心。

这保留的是统计层面的类别身份，不保证最终分类决策不变。支持量q只控制借用强度，不是已测供体可靠性；保留协方差形状本身是否有益仍是待验证假设。新版不再声称恢复了真实中心，也不估计真实旧类漂移。

## 3. DSM-v2：在分量的错误尾部修正边界

与旧版所有均值margin向固定1靠近不同，新版固定原控制W0，在原训练聚合统计上构造每个旧类组件的局部高斯近似：

    x|c,r ~ N(u_cr, Sigma_cr)
    Sigma_cr = r_cr / tr(S_c) * S_c

零内部方差时为确定性组件。这个共享形状混合模型的类均值、Q、组件中心/质量/半径与银行一致，但分量真实协方差并未保存，模型仍是假设。L2归一化特征上的高斯是局部二阶近似，不是精确数据分布。

只考虑旧-旧竞争对c,j。v0=W0_c-W0_j，m=u^T v0，s²=v0^T Sigma v0，z=m/s。错误区域固定为v0^T x<=0；该区域由控制头决定，绝不根据val真标签或修订后的头反复改图。

p=Phi(-z)、phi=标准正态密度(z)、a=Sigma v0/s。错误尾部的**未归一化**一二阶矩为

    b_bad = p u - phi a
    Q_bad = p (Sigma+u u^T) - phi(u a^T+a u^T) + z phi a a^T

不生成旧特征、不保存逐样本数据、不除以极小p。不同竞争对的错误事件会重叠，不能把pairwise p直接相加当总分类错误率。p低于float64机器精度时按数值零处理。确定性错误组件p=1，目标tau=max(0,-m)；其他tau=s，用原控制margin的标准差定义目标尺度，避免重新套固定unit margin。确定性零margin且零方向没有人为新增信号。

冻结权重

    omega_crj = .5/K * q_c * original_pair_cj * pi_cr

给定修订头W，边界损失

    L_tail = 1/2 sum omega [v^T Q_bad v - 2 tau b_bad^T v + tau² p]

包含风险区域内的全部二阶信息，而不只是类均值残差。其Hessian H_tail是半正定，Q_bad与其补集二阶矩均半正定，故H_tail不超过原native competition曲率，且不超过全native H0。不再把输入修正限制在旧类均值张成的特征子空间，也没有先投影掉local_transport整个span或套旧0.1能量硬上限。

这里的边与p是**训练统计推导的混淆代理**，不是已测真实混淆概率。未来必须逐边与val混淆进行预测对应核验；val只用于报告，不能用于选边、重加权或决定是否回退。旧统计被共同运输后的准确性、共享组件形状与局部高斯假设，仍可能让预测边界错误。

## 4. 解析头：完整原目标、固定新列、独立正则化和ETF力

原native目标L0包含全类平方读出、原.001R和原.5 competition，H0包括全部competition曲率。每个控制条件分别得到精确double最优W0。保存的float32原头只作OFF读出，不用重构误差生成修正。

V是只覆盖旧类、列正交的Helmert contrast基，新类行严格为0。解

    W = W0 + Z V^T

Z保留全部1536个输入维度。新列变化严格0，旧列公共分量不变；旧类少于2时严格identity。

MPC开启时原目标里的Q替换为Q'，其mu/R/pairs保持原值。Q'_c >= (1-eta_c)Q_c、eta<=.4，所以修改后的native H_M >= .6 H0，保留正定性。

正则化单独定义在判别部分和完整原native曲率中：

    L_reg = .1/2 * ||Z||²_A0
    A0(Z) = H0(Z V^T) V

这是围绕原控制的增量稳定项，不是把整个W向0拉的gamma I。REG单独作用时严格identity，不作为独立医学性能臂；其作用只通过TAIL_REG−TAIL、MPC_TAIL_REG−MPC_TAIL衡量。与早期CPU设计的整体旧contrast收缩相比，这避免无边界依据的整体分数压缩；它不复现上一轮普通ridge的ISIC增益，也不能据此声称保留了那部分效用。

ETF只对旧类构造最大匹配simplex，尺度用||W0_old C_old||。投影位移E=(T-W0)_old C_old，新列0。方向力单独取

    F_ETF = H_tail(E)
    L_dir = -.1 <Z V^T, F_ETF>

因此只经过冻结错误尾部二阶几何施力，不加入额外ETF Hessian，也不强制整个头变成ETF。目标是原头时E=0，或尾部曲率为0时，方向力严格0；ETF rank不足时单独标记该方向不可识别，保留其他项，不选备用方向。

总目标是 L_native(Q或Q') + L_tail + L_reg + L_dir，严格凸于旧contrast空间。尾部、正则化、方向有独立固定开关。D有/无ETF的Hessian完全一致，差别只有线性方向力。用原Cholesky基底作matrix-free PCG预条件，复用128步/1e−9相对停止、拒绝残差>1e−8；不构造(1536*(old−1))²的巨大Hessian。CPU稠密Hessian只用于小型合成oracle。

## 5. 新类保护：允许恢复错误新类，不允许夺走已有正确预测

原分数s=x^T W_saved，raw=s+x^T Delta_float32；Delta新列为0。每输入旧类最大值b=max(s_old)、a=max(raw_old)。

    s'_new = s_new
    若a<=b：s'_old=raw_old
    若a>b：s'_old = b - (a - raw_old)

所以max(s'_old)<=max(s_old)。所有当前新类的raw true-class margin不下降，原来预测为新类的输入保留原标签；真实新类正确性不会下降。旧类最大值下降时，原来错成旧类的新类可以被救回，也可能让真实旧类错成新类。

与前一版始终恢复原旧类最大值不同，本版不冻结旧新组别。它仍不能改变新类之间的排序；不能修复新→另一新错误；不保证old/tail/F1或新类precision无损。新类成为后续旧类时不再受该阶段保护。stage1全部类都是新类，所有候选为原控制identity。

这个max操作是无标签的非线性readout。解析求解和Delta本身是线性的，但完整受保护方法不是单一线性解析头。必须同时报告raw和guarded的纠正/破坏、幅度保留、旧→新false positives，不把guard避免损失当作模块效用。

## 6. 与 local_transport 的互补及创新落点

local_transport保存四区域类中心，加入R=(I+PP^T)^−1，并共同运输历史统计。新版不再运输均值/局部中心，不修改R，不以局部span整体投影删除修正。

MPC改变组件内部协方差形状；DSM根据各控制自己的W0、原组件/矩信息挑出可能错分的旧-旧尾部；local_transport已解决的边有较小的错误质量，理论上减少重复施力。两控制的图不同是前瞻允许的状态条件化，不是按val挑边。MPC与DSM同控制内使用同一原图，避免MPC一边改统计、一边事后改目标图。

这只是结构上的互补理由，不保证参数更新正交、共享误差已消除或存在正交互。未来必须报告(MD_P-AL)-(MD_B-A0)、组合相对MPC/DSM及全部负值。

潜在方法贡献是：类身份和多模态间结构保持的二阶借用；不依赖旧图像的错误尾部矩边界目标；保持曲率不变的ETF方向分解；允许新类恢复的单向决策保护。它们解决的是统计读出问题，不是训练中的表征保持；真实医学效用和与原方法的创新关系仍需独立证据。

## 7. CPU证据与边界

最终：两个控制×八个数学开关（其中REG为identity检查）与独立显式二次目标的Hessian/梯度/稠密最优解一致；MPC中心/质量/半径/迹一致、共同平移不变；错误矩与补矩PSD、数值积分正确；REG identity正确、ETF两开关Hessian一致；checkpoint往返、非法授权/输入拒绝、新列固定、float32/64新margin保护及允许新救回通过。

压力样例：使用独立合成重叠类，不改协议参数，验证非零尾部作用、修正超出旧均值span、八头类别置换等价；另做1536维ETF和23类guard接口。不是完整1536×23医学求解验收。

容易分离的fixture尾部力很小；早期CPU草案的全contrast正则化掩盖了尾部，已改为增量稳定项并重新验收。重叠fixture非零不意味着真实医学资产有可靠尾部信号。没有旧真实中心、语义属性、真实医学数据或独立患者评价；不能从CPU PASS声称GPU-ready、分类提升或真实新类收益。未来是否能命中真正混淆是首要失败风险。

补充准入：所有信息项都inactive时，明确返回零delta，不把原PCG停止残差当成模块作用；component count和n不一致、非对称Q拒绝。首个准入fixture误用了float质量而原native.evidence要求Tensor，发生在合成控制构造中；已修正fixture，不改原native模块，失败费用和去标识原因保留。
