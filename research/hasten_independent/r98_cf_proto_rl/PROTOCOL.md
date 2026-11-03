# CF-PROTO-RL：实施、执行与实验总 Prompt

你正在处理 Long-tailed-CL 项目的原 BiomedCLIP / 强 SFT / RL 路线。本任务不是 HASTEN 路线，也不是 R97 的调参续跑。

请实施一个新的独立实验：强监督锚定、受限原型加权、外部训练查询奖励、类别级反事实信用的真实策略梯度，暂称 CF-PROTO-RL。完成代码、必要的新机制单测、资源准入、六阶段正式实验、封存评估、结果分析和安全交付。不要只交方案或反复停留在诊断。

以下数值是本轮一次性、事前固定的试验配置，不是已验证最优超参数，也不保证取得正结果。除无法恢复的必要资产、数据契约不满足、正确性错误或预算不足外，不再要求用户逐项确认；在授权环境内执行。不得改变已有实验的结论或验收标准。

## 0. 研究问题与边界

预定主候选始终是 CF；不能读完结果后将表现最好的其他臂改称主方法。

本轮回答：
1. 在冻结的原强 SFT 表示上，原型策略持续保留 CE，并加入外部查询效用后，是否改善真实部署分类？
2. 在相同目标下，反事实策略梯度相对普通全局奖励策略梯度表现如何？
3. 与同一连续随机目标的路径导数优化相比，采样式策略梯度是否存在条件性优势？

本轮使用单个既有教师 seed 74002、完整六阶段。属于阶段式原型后训练机制实验；不声称端到端在线 RL 持续学习、独立患者验证、严格小记忆 CL、多骨干种子确认或原创性已经成立。

反事实基线本身不改变期望效用目标；风险偏好来自效用函数 U。不能把 CF 的效果解释为“只有反事实分支才看到了旧尾类损害”。

## 1. 先读取并复用已有资产

仓库：DLwbm123/Long-tailed-CL。

必须读取：
- commit 971fa262b266df3d2a3b5f83316b36419b3deddb 下 research/hasten_independent/r97_strong_supervised_joint/ 的 PROTOCOL.md、REPORT_ZH.md、strong.py、CONFIG_TEMPLATE.json、losses.py、model.py。
- commit 0b6ba17e7c6740bdf7c23a1ed3af6982894e92a5 下 research/hasten_independent/r91_cpc_prototype_construction/ 的 PROTOCOL.md、REPORT_ZH.md，以及实际 S/B/F 角色绑定实现和私有清单。
- R88 的 RLPO 实现与保存结果；只作已有目标移植依据，不重训 R88。

复用原 R67 对应六阶段的强 SFT LoRA 与原型头，不使用 R97 新 SUP/JOINT 权重作为本轮起点。复用合法、版本匹配的冻结特征；必须核对 stage、教师权重 SHA、类别映射、图像 ID 顺序、预处理与特征单位化。不同阶段 LoRA 产生的缓存不能混用。

在 research/hasten_independent/ 下创建未占用的新实验目录，编号按仓库当前实际状态确定，不覆盖 R91/R97 或假定 R98 尚未使用。冻结 PROTOCOL.md、CONFIG.json 和资源预算后再启动正式优化。准备阶段只绑定元数据、S/B特征和教师资产；F/VAL特征不得进入模型或指标计算，缓存容器包含其他行也不能解除访问mask。

## 2. 数据、角色及访问约束

沿用 HK、seed 74002、23 类、13+2+2+2+2+2 到达顺序；每个阶段只使用已经到达的类别及对应阶段教师。

沿用现有不可变 S/B/F component 划分，不重新挑选有利分组。已有报告的总量为 S=3405、B=1713、F=1703；tiny 类1的 S/B/F 为1/1/1，类2、12为3/2/1。以实际可验证 manifest 为准，若绑定不一致则先解决契约问题，禁止默默另造划分。

本轮角色固定：
- S：构建原型、状态、支持几何描述。
- B：全部新增模块的监督训练查询及 RL 奖励查询。
- F：所有模型封存后的模块级迁移读出；不参与损失、调参、早停或决定是否继续某个臂。
- VAL：所有模型及部署规则封存后统一一次读出。
- TEST：不访问。

S/B/F 不交换角色。Q 在以下公式中一律指当前阶段全部 B，不另造奖励集。

每个优化步骤有两个 episode：一个使用全部当前 S；另一个各类无放回取 max(1,floor(m_S/2)) 张，按事前种子和 step 确定。每个集合内不得复制图像或把 tiny 补成较大类。两 episode 都使用全部当前 B，损失先等权平均再做一次 optimizer.step。所有臂共享 episode 索引和 B 顺序。

必须明确披露：原教师已见完整 TRAIN；F 也曾用于历史开发，所以以上隔离只针对本轮新增模块，不是未污染的独立数据确认。B 被用作奖励后就是策略训练数据，不能再称独立验证集。

正式部署支持集在全部训练和 F 读出完成后，按事前规则固定为当前阶段全部已到达 TRAIN，与原强 SFT 的原型信息预算一致。此时 F 可以成为部署支持，但不能据此再训练或调参；F 的模块迁移指标必须在并入前、只用 S-full 支持计算。

## 3. 模型：冻结表示，只训练共享原型策略

冻结原强 SFT 的全部骨干和 LoRA，f(x) 为512维单位特征。本轮不新增视觉 prompt、文本 prompt、MoE、可训练投影、类别偏置、温度搜索或额外数据。

当前支持 S_c 的均匀原型：
    p0_c = normalize(mean_i f_ci)
部署原始分数：
    z_c(x;P) = 20 * f(x)^T p_c

对每类均匀原型找余弦最近的三个其他类原型 d1,d2,d3。邻居只由当前 S 的 p0 确定，不随动作变化，也不用 B/F/VAL 的查询响应。并列按原型向量字典序解决；完全相同向量之间的次序不得改变输出。实际阶段 C>=13；单测的小 C 可补零并显式记录缺项。

每张支持图像的原始4维描述：
    h_raw_ci = [cos(f_ci,p0_c), cos(f_ci,p0_d1), cos(f_ci,p0_d2), cos(f_ci,p0_d3)]
每类、每列按支持图像中心化，除以 max(population_std,0.05)，再截断至[-3,3]。m_c=1 时 h_ci 必须全0。h 和所有支持统计停止梯度。

策略状态 s_c 共1032维：
- p0_c：512维；
- S_c 无偏逐维方差：512维，m=1置0；
- 三个邻居原型余弦：3维；
- 5个标量：log1p(n_raw_c)/log(6822)、log1p(m_c)/log(6822)、方差可估计mask、age/5、new flag。

共享 MLP：1032 -> 64 -> tanh -> 4，最后线性层权重及偏置初始为0。均值输出：
    mu_c = 2 * tanh(raw_output_c / 2)
不得加 class-ID embedding、按类独立可学习参数或查询分布输入。

条件独立动作：
    a_c ~ Normal(mu_c, sigma^2 I4), sigma=0.25 固定
给定支持状态，不同类和不同 rollout 的 epsilon 必须独立。禁止加入动作间自回归依赖，否则本轮反事实梯度推导不再适用。

动作到权重：
    u_ci = 2 * tanh((a_c^T h_ci)/2)
    q_ci = softmax_i(u_ci)
    w_ci = 0.75/m_c + 0.25*q_ci
    p_c(a_c) = normalize(sum_i w_ci*f_ci)

因此 w>=0.75/m，sum(w)=1，w<=0.25+0.75/m。记录 ESS=1/sum(w^2)、ESS/m、最大权重、相对均匀原型的角度变化。

m=1 时原型严格退化为唯一支持特征，该类不计入策略梯度和 Gaussian KL 的 active 类集合。仍保留在所有分类、宏/尾效用及损害计算中。不能声称此动作空间可以主动修正单支持类本身的原型。

动作 a=0 在数学上产生均匀原型；用固定容差核对数值实现。不要用“if mu==0: return p0”截断初始监督梯度。全 TRAIN 均匀原型与历史教师原型的核对在全部训练封存、F读出完成后执行，容差沿用2e-5，并记录实际误差；不得承诺跨硬件逐位一致。若发生可能改变结果的差异，保存并审计，禁止用 VAL 调整路径。

## 4. 持续保留的监督目标

确定性动作使用 a=mu_phi(s)，构建 P_mu。

B 为自然完整查询分布，因此监督项固定为：
    L_CE = mean_(x,y in B) CE(z(x;P_mu)+log n_B, y)
其中 n_B 为当前阶段 B 的实际逐类计数。推理及奖励分数不加 log n_B。

这里将先验与实际监督查询分布 B 对齐，保留 logit-adjusted CE 的训练原则；不另外乘逆频次权重，也不暗中更换为重采样分布。

共同 warmup 只优化 L_CE，完成固定预算后保存该阶段 SUP0。阶段0从零均值策略初始化；之后共同 warmup 策略参数跨阶段携带，每阶段重置优化器。各后训练臂不影响下一阶段共同 warmup。

封存六个 SUP0 后，对每个阶段分别从对应 SUP0 复制出全部后训练臂。固定参考网络 phi_ref 为该阶段 SUP0，不做 EMA。每个 episode 都在自己的 S 上重新计算参考均值和参考原型，不得误用另一支持集的缓存。

公共保持项：
    L_KL = mean_(active c) ||mu_phi(s_c)-mu_ref(s_c)||^2/(2*sigma^2)
所有后训练臂均保留 L_CE + 0.04*L_KL。没有 active 类时KL定义为0。

不添加第二个教师 KL；不要无意复用 R97 的双 KL 或路由正则。

## 5. 外部训练查询效用 U

所有动作及反事实在同一份 Q=B 上评分，类别集合、query行、mask和分母完全相同。使用 raw 20cos 分数。

定义逐类连续指标：
    v_c(P;B) = mean_(x in B_c) sigmoid((z_c(x;P)-max_(d!=c)z_d(x;P))/tau_m)
    tau_m=1.0

H 为该阶段已到达的 old 类与 tail 类之并集。tail 沿用完整TRAIN计数<250，不用 VAL 数量重新定义。

参考 P_ref 为 SUP0 在当前支持集上的确定性均值动作原型，v_ref 必须 detach。

    U(P;B) = mean_all_classes v_c(P;B)
             - mean_(c in H) relu(v_c(P_ref;B)-v_c(P;B))
H为空时第二项为0。lambda_h=1固定。U使用0到1指标尺度，不乘100；百分点只用于报告。

U、正部及差分累计尽量使用float64，防止小差分被消掉；不能为方便计算在 PATH 臂中 detach 这些可微操作。

这个连续效用是训练代理，不等于真实宏准确率或尾类召回。必须并列记录argmax真实指标。风险惩罚相对于SUP0，不能宣称自动保护原教师；正式效果门同时检查原教师和SUP0。

## 6. 主候选 CF：真正的反事实策略梯度

每个 episode 采 G=8 组完整联合动作；各组覆盖所有 active 类，非active类保持其固定原型。

对每组g：
    P_g = build(a_g)
    U_g = U(P_g;B)
对每个active类c，其他动作保持不变，仅把原型c替换为当前支持上的固定SUP0参考原型：
    P_g_minus_c = P_g with p_c replaced by p_ref_c
    D_gc = U_g - U(P_g_minus_c;B)

CF损失：
    L_CF = - mean_g sum_(active c) stopgrad(D_gc)*log pi_phi(a_gc|s_c)
    L_total_CF = L_CE + L_CF + 0.04*L_KL

实现契约：
- log_prob 对4个动作维度求和，得到[G,C]；类维求和，组维求平均。
- 不得把类维再次求平均；U已按类别平均，额外除C会改变RL相对CE的强度。
- 不做组内标准差归一化、不做TARS再加权、不裁剪D、不强制成功动作、不对负reward重采样。
- 只做一次on-policy optimizer update，然后重新采样；第一版不引入PPO旧策略、多次复用或GRPO裁剪。
- 动作必须在score-function分支detach，D与全部reward计算detach；但mu不能detach。
- 禁止把 log(w_ci) 冒充高斯动作log_prob。

推荐等价实现骨架：
    eps = torch.randn((G,C,4), generator=..., device=...)
    action_sf = (mu[None] + sigma*eps).detach()
    logp = Normal(mu[None], sigma).log_prob(action_sf).sum(-1)
    with torch.no_grad():
        U, D = counterfactual_rewards(action_sf, ...)
    loss_pg = -(D[:,active].detach()*logp[:,active]).sum(-1).mean()

不可直接使用未detach的rsample结果去计算本分支log_prob，否则会错误引入路径导数甚至抵消固定方差高斯的均值score梯度。

本轮应称“条件独立原型动作的反事实REINFORCE”，不能宣称标准clipped GRPO、COMA原样复现或多步长期回报已建立。

## 7. 固定五个匹配后训练臂

所有臂共用状态、网络、动作空间、SUP0起点、episode清单、CE、参考KL、学习率、更新步数和梯度裁剪。预定主臂为CF，不可事后选择。

A. SC：监督继续训练
    L_SC = L_CE + 0.04*L_KL

B. RLP：受限结构中的RLPO-style移植控制
    使用确定性a=mu及同一受限w、P。
    对每个支持特征i定义：
      r_i = cos(f_i,p_yi) - max_(d!=yi)cos(f_i,p_d)
            - 0.1*mean_(d!=yi)cos(p_yi,p_d)
    对全部支持行做一次population mean/std标准化，epsilon=1e-8，并detach：A_i。
    L_RLP_aux = -mean_support_rows A_i*log(w_i)
    L_RLP = L_CE + 0.1*L_RLP_aux + 0.04*L_KL
    原实现的额外样本权重reference-KL不再重复叠加。
    这不是完整RLAP重现，也不是随机策略梯度；它是共享受限空间中的辅助目标移植控制。

C. CF：第6节主方法。

D. GLOBAL：同目标、无类别反事实基线的真实策略梯度
    b = U(P_ref;B)，固定且detach。
    L_GLOBAL_aux = -mean_g [(U_g-b).detach() * sum_active_c logpi(a_gc|s_c)]
    L_GLOBAL = L_CE + L_GLOBAL_aux + 0.04*L_KL
    GLOBAL和CF的期望效用目标相同，区别是梯度估计及基线。

E. PATH：同连续随机目标的路径导数控制
    a_path = mu[None] + sigma*eps，保持重参数化梯度。
    L_PATH = L_CE - mean_g U(P(a_path_g);B) + 0.04*L_KL
    不用log_prob，不detach动作/原型/U，仅detach固定参考。
    PATH也使用G=8；它仍是蒙特卡洛期望优化，不叫完整exact积分或零方差优化。

CF/GLOBAL/PATH使用配对epsilon种子；支持子集RNG不含arm，三臂epsilon流共享同一(stage,step,episode,rollout)键且不受其他分支调用次数影响。策略分叉后动作不会相同，只有基础噪声对应。SC/RLP不必执行无用随机前向来伪装等计算量。各臂实际原型构建、评分、reward和反向成本单列，不能以等步数宣称等成本。

必须从本轮开始保留PATH和GLOBAL，不能先查看VAL后才决定是否增加这些关键控制。

## 8. 固定训练预算与阶段语义

固定配置：
- teacher/data seed：74002；本轮只有一个策略训练seed，独立RNG流按stage/step/episode/用途派生。
- 每stage共同warmup：512 updates。
- 每stage每个后训练臂：256 updates。
- 每update：2个episode，loss平均后只做一次优化。
- AdamW，warmup lr=3e-4，后训练lr=1e-4；betas=(0.9,0.999)，eps=1e-8，weight_decay=0。
- 固定学习率，不设数据驱动schedule或early stopping。
- 全部臂global grad norm上限1.0；记录裁剪前范数及裁剪频率。
- 网络训练float32；本轮不用AMP/量化；reward归约和差分float64。
- warmup每stage重置optimizer；五个后训练臂均从fresh且完全一致的optimizer初态开始。

总正式optimizer steps：
    6*512 + 6*5*256 = 10,752。
共同warmup6个最终端点，后训练30个最终端点，共36个正式模型/优化器最终端点。另保存后训练step128中间端点用于恢复审计和训练诊断，但不读其VAL、不择优部署。

六阶段warmup先按顺序完成并封存；之后五个臂分别复制阶段SUP0。后训练结果不carry到下一阶段warmup。本轮回答的是stagewise增量，不声称RL已经改善在线长期记忆。

不设置“TRAIN效果足够好才允许RL”的新科学前置门。只要实现、数据和资源准入通过，就完成五臂六阶段，不因前两个阶段不理想提前换方法。安全故障、非有限值或资源上限触发才停止并如实结案。

## 9. 必须实现的接口与测试

建议新增文件：
    data_contract.py       # 只读清单、stage特征绑定、角色边界
    episodes.py            # 配对support episode和独立RNG
    prototype_policy.py    # state / action / bounded weights
    utility.py             # classwise utility / harm / column counterfactual
    losses.py              # SC / RLP / CF / GLOBAL / PATH
    train.py               # warmup与五臂固定预算
    evaluate.py            # seal后F和VAL
    analyze.py             # 逐类分解与冻结效果门
    verify.py              # saved-only重算
    tests/                 # 只测试本轮新增机制

核心接口至少包括：
    build_state(support_features, labels, metadata)
    sample_actions(mu, sigma, generator, G)
    build_prototypes(actions, support_features, h, labels)
    classwise_utility(raw_logits, labels, ref_values, protected_mask)
    counterfactual_credit(sampled_logits, reference_columns, labels, ...)
    loss_for_arm(arm, model, reference, episode, eps)

必须通过：
1. a=0均匀原型、权重上下界/和为1；单支持类动作不改变原型。
2. 支持行置换、类别重命名及对应输出重排一致；不依赖类别ID参数。
3. 用“替换分数列”计算的CF与暴力重新构建反事实原型一致；所有被竞争类必须重新计入utility。
4. 奖励/状态/支持特征无梯度，score-function的mu有梯度；PATH动作及utility有路径梯度。
5. 在可解析的低维高斯二次效用上，用固定65536组MC验证CF、GLOBAL和PATH梯度均值与解析梯度一致；逐坐标容差固定为5倍MC标准误+1e-5，禁止失败后换seed或扩样救援。此测试不使用梯度裁剪，不能要求三个随机样本梯度逐次相等。
6. 人工构造“自身类别改善但伤害另一旧尾类”的例子，验证本轮U会计入损害，CF与GLOBAL目标一致。
7. 学习率0时仅训练参数不改变，严格保存/恢复可重建输出；所有teacher参数无梯度。
8. S/B/F交集、query进入prototype、未来类/错误stage缓存访问必须有断言；seal前F/VAL/TEST loader禁用。

这些是软件与机制实现测试，不是性能实验；工程参数和随机状态全部丢弃后再按正式seed初始化。不得反复运行已经关闭的R90/R91数学目录作为新进展。

## 10. 诊断与计数

每个正式step记录train loss分项、U、D的均值/std/正负比例、KL、权重ESS和最大值、mu饱和比例、裁剪前梯度范数以及实际动作数。按stage/arm聚合，避免把每次重复暴露当成独立病例。

在固定step 0/128/256、预定S-full+B上下文上，额外记录CE与辅助项梯度范数及夹角。诊断不得反向更新参数；其计算计入总预算。

另在每个阶段相同SUP0点、固定S-full+B上下文上，用32个独立G8噪声组比较CF与GLOBAL的原始策略梯度方差、均值范数和方向一致性。这个比较必须在同一参数点进行，不能比较训练分叉后的不同模型来宣称降方差。方差下降是待验证结果，不写成保证。

必须区分：optimizer updates、episodes、joint rollouts、active class actions、prototype builds、原始score rows、counterfactual utility calls、image encoder rows、缓存命中、额外diagnostic forward/backward。复用分数列不算新的视觉编码。

## 11. 封存与部署

所有36个最终状态、配置、源代码版本、RNG状态、support规则和部署规则先封存，再进行F/VAL读出。

固定主要部署：
    a_c = mu_phi(s_c)
    使用当前阶段全已到达TRAIN支持，raw20cos分类。

不得在验证集上选择最好采样动作、挑G、挑温度、挑epoch或切换成随机ensemble。随机策略期望效用只作TRAIN诊断；确定性均值动作好坏必须以最终部署读出为准。

顺序固定：
1. 用S-full构建原型，在F读出本轮模块迁移；各臂及参考同support、同query。此处均匀参考记为T_S，不能把含F支持的全TRAIN教师头混入并宣称同支持公平比较。
2. 不根据F改变任何配置或决定是否启用某个臂。
3. 用全TRAIN构建所有部署原型，封存；F此后不能继续充当独立查询评估。
4. 对T（原强SFT）、SUP0、SC、RLP、CF、GLOBAL、PATH进行六阶段统一VAL读出。
5. TEST始终为0。

## 12. 冻结的效果门与结果解释

数值门只是本轮单seed开发门，不是统计显著性阈值。

方法效果门A：CF相对原强SFT T，必须同时满足：
- 平均macro >= +0.5pp；最终macro >= +0.5pp；
- 平均tail >= 0pp；最终tail >= +1.0pp；
- 至少4/6阶段macro和tail同时非负；
- tiny类1/2/12不新增T原本正确样本的错误，另对SUP0原本正确样本执行相同保护检查。

RL增量门B：CF分别相对SUP0、SC，必须同时满足：
- 平均macro >= +0.3pp，最终macro >= +0.3pp；
- 平均tail及最终tail均非负。

A、B全部通过才可称“本轮CF主候选通过开发效果门”。高于SC但低于T不是主方法成功。不得选RLP/PATH的正结果替换CF失败。

另列机制证据，不混入总准确率：
- CF对GLOBAL：同参数点梯度方差与最终指标分别报告；一个改善不能代替另一个。
- CF对PATH：严格说明同目标路径导数控制。若PATH相当或更好，可以支持外部查询目标/原型动作有价值，但不能证明score-function RL有独立优势。
- CF对RLP：回答新奖励和信用设计是否超过本轮受限结构的RLPO-style目标，不能外推原论文整体。

必须报告所有臂及T/SUP0的micro、macro、tail、各阶段old/new/head/tail及交叉组、每类正确数和分母；同时提供相对T和SUP0的rescue/harm。

必须按原指标分母做可加贡献分解：
    对平均macro的 class-stage 贡献 = delta_recall_pp/(6*C_stage)
    对平均tail的贡献 = delta_recall_pp/(6*tail_class_count_stage)
分别列tiny1、2、12和其余类别贡献，且总和应重建主指标差值。再报告独立图像/可用component层面的影响敏感性；同一图跨阶段重复不是多例。不得从主指标中删除不利tiny病例。

置信区间若可行，按合法独立component配对重采样，并在跨阶段保持同一component抽样权重；没有患者ID不能叫患者bootstrap，极小分母不能伪装充分统计证据。

## 13. 工程与预算

优先复用已验证的冻结特征与部署环境；不要重新下载模型或无故重编码全部图像。缓存缺失才在合法TRAIN上补编码，并计入预算。VAL特征即使已存在，也只在seal后读取。

沿用项目授权机器与data盘。默认最多1张空闲GPU，CPU线程4；不得占用他人GPU、杀他人任务、改系统环境或删除历史结果。CPU可完成时不强行占GPU，但必须使用真实profile判断完整预算。

采用新实验自己的单次24h硬预算，包含准备、排队、测试、全部正式训练、评估和交付；不得换attempt或换机器重置。继承已有更严格资源上限；若无更严格约束，采用总RAM30GB、worker28GB/group29GB、GPU reserved<12GiB、实验science8GiB、每服务器新增总存储<50GiB、单文件<1GiB、CPU累计400000s。数据与大文件不写root盘。

正式训练前对初始/最终规模做固定两次profile，以第二次最坏值×2估算全部10,752次更新及诊断、原型构建、VAL，另预留7200秒保存、分析和交付。预算不能覆盖完整五臂六阶段就以BUDGET_NOT_ADMITTED退出，不偷偷降G/砍PATH/少跑阶段或修改数值配置。

只允许随本轮进程退出的本地资源保护器，不创建cron、自动化任务或周期监测。此前已关闭的监测保持关闭。

安全故障或非有限状态立即保留现场及费用，不盲目重试。修复saved-only分析可以不重训，但必须保留修改记录；改变训练数学或正式结果需要新的协议版本，不能覆盖失败记录。

## 14. 交付与验收

至少交付：
- PROTOCOL.md、CONFIG.json、ASSET_MANIFEST_SANITIZED.json；
- 新源码与tests、PRECHECK.json、BUDGET.json；
- TRAIN_COMPLETE.json、SEAL.json、EVAL_COMPLETE.json；
- STAGE_METRICS.csv、CLASS_STAGE.csv、GROUP_STAGE.csv；
- CONTRIBUTION_DECOMPOSITION.csv、POLICY_DIAGNOSTICS.csv；
- GRADIENT_ESTIMATOR_DIAGNOSTICS.csv、COSTS.json；
- VERIFICATION.json、REPORT_ZH.md。

saved-only独立验收必须重算：逐类正确数→macro/tail→阶段平均→A/B门；核对全部阶段、臂、更新次数和失败记录，不重新选端点。

费用区分新增编码/缓存命中、CPU/wall、峰RAM/GPU、磁盘、失败attempt、准备与发布；不能把已包含worker的子项重复累加。历史教师费用单列为借用资产，不冒充本轮新增或免费总成本。

只将获准公开的代码、脱敏配置、聚合指标和报告提交到既有授权仓库。原图、患者或图像身份、逐例预测、features、权重、optimizer、私有路径和rawlogs保持私有。核对远端提交与报告可访问性，不能只报告本地commit。

最终答复必须直接给出：
1. CF是否完成、是否通过A/B；
2. 相对T、SUP0、SC、RLP、GLOBAL、PATH的平均及最终macro/tail差值；
3. 哪些类别/阶段贡献收益与损害，tiny独立样本数量；
4. 证据支持的是原型结构、外部查询目标、反事实估计器，还是仍未成立；
5. 全部新增成本、失败项、报告路径和commit。

不要以“实现正确、奖励上涨、运行完成”替代准确率效果；也不要因结果为负删掉实验。完成本轮后停止，不自动追加搜索、下一轮实验或恢复监测。


## 执行绑定澄清（优化前冻结）

实验编号R98，原BiomedCLIP强SFT/RL路线。每server新增<50,000,000,000B，science cap8,000,000,000B加512MiB早停余量，继承更严格边界。tiny的T及SUP0新增正确损害检查覆盖全部六stage。F仅报告不设新的科学前置门。CF/GLOBAL同目标和无偏解释仅指裁剪前估计器，裁剪后实际更新不保证相同期望或降方差。原971fa26代码用途保留，最新3c695c3逐类归因作为解释修正，原效果门不变。固定Gaussian均值部署，随机训练不声称初始NOOP。SHA核仅用于附件明确要求的6个小教师头，不扩展为全数据哈希。结束后停止，不自动追加实验/恢复监测。
