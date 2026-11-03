# R99：动作坐标保持与分类行为保持的匹配机制实验

用户已明确要求在pro5000直接执行第一步。唯一预定主候选CF_BHV；九臂同时冻结，不读结果后选主方法。R98保持完整负结果，不重启、不改门，不把新增约束当已证明有效或创新成立。无周期监测，完成本轮后停止。

## 问题、资产与信息预算

本轮检验：R98的动作Gaussian KL是否限制了有用分类变化？在局部尺度匹配后，用分类行为KL替代动作坐标KL，能否在持续CE下产生真实macro/tail收益？SC排除继续监督的解释，PATH检验同目标可微优化；NONE用于识别保持项整体作用。行为KL与动作KL仅起点局部尺度匹配，不能据此宣称整个轨迹严格等强。

固定HK74002、23类13+2+2+2+2+2、S/B/F不可变3405/1713/1703。每类均非空，component互斥；tiny1=1/1/1，2/12=3/2/1。原强R67 SFT已见全TRAIN，F也有历史开发暴露，均须披露，不声称患者独立/严格小记忆CL/独立确认。本轮借用R98的6个SUP0_512模型作为相同初始化和固定参考，R98 warmup3072更新为借用历史成本，不算本轮新更新；不接R97新SUP/JOINT，也不重新训练R67或R98。使用R97已验证的对应stage冻结R67特征；只在合法stage读取S/B。六个小教师头SHA沿用R98既有核验收据，不重哈希全部资产；manifest顺序、filled/class前缀、角色及特征单位化仍断言。

模型/动作/score完全沿R98：冻结512维单位表示；共享1032→64tanh→4策略；mu=2tanh(raw/2)；sigma=.25、G8条件独立高斯；u=2tanh(a·h/2)，w=.75/m+.25softmax(u)，单位原型，raw20cos。m1无动作自由度。状态、邻居、h描述/标准化及支持来源不变。每step两episode（S-full与各类无放回floor(m/2)至少1），均查询完整B，loss等权平均后一次AdamW更新；各臂support/噪声键配对，与R98同(seed,stage,step,episode,purpose)规则。

## 唯一改变：保持项

所有臂始终保留L_CE=natural B mean CE(raw_logits+log n_B,y)，不改采样、先验、reward或动作幅度。监督参考SUP0固定，每episode在该episode支持上重新计算，不用跨支持缓存。

ACT：K_A=mean_active ||mu-mu_ref||²/(2*.25²)，系数.04，与R98一致。

BHV：令q_ref=softmax(raw20cos_ref)、q=softmax(raw20cos_current)，不加log n_B。K_B=(1/C) sum_c (1/n_Bc) sum_{i:y_i=c} KL(q_ref_i || q_i)。正向参考KL，ref停止梯度，所有已到达类参与。损失.04*kappa_s*K_B；概率/KL归约float64。行为保持不保证argmax或正确病例保护，必须另报实际翻转、救回和损害。

NONE：保持项0，用于区别约束类型与移除约束；不事后将NONE改为主候选。

### 固定局部尺度校准（不优化、不按效果选择）

每stage只在相同SUP0参数点，取(step0,S-full)及(step0,half)两上下文。每上下文32个固定独立标准Gaussian方向，seed=74002+9000001*stage+1009*episode+7919*71+j。对参考网络raw output r构造mu_tilde=2tanh((r+.01*epsilon*active)/2)。计算对应K_A与K_B。kappa_s=全部64扰动K_A之和/全部64扰动K_B之和，之后固定。B只用于训练期尺度校准，无F/VAL。若分母<=1e-14、非有限或kappa不在(0,1e8]，按CALIBRATION_NOT_IDENTIFIABLE不准入，保留现场，不换幅度/方向/seed或裁剪比例救援。该比例不是VAL最优参数，也不保证全程梯度相等。原始BHV与加权BHV/ACT数值及梯度均报告。

## 九个固定臂

SC_ACT/SC_BHV/SC_NONE：CE+各自保持项。
CF_ACT/CF_BHV/CF_NONE：CE+R98反事实REINFORCE+各自保持项。动作与U停止梯度，真实Gaussian log_prob，类sum/组mean；不做TARS/std归一化/PPO重用。
PATH_ACT/PATH_BHV/PATH_NONE：CE−同G8随机utility均值+各自保持项，保留路径导数。CF/PATH的U、旧尾类保护集合及lambda_h1、margin_tau1与R98完全相同；baseline/ref始终为SUP0，不随约束臂改变。

不重跑GLOBAL/RLP：本轮不检验新的估计器或RLPO移植；保留R98结论但不拿跨run差异替代本轮匹配控制。所有九臂从stage对应SUP0复制，fresh AdamW lr1e-4/betas(.9,.999)/eps1e-8/wd0，每stage每臂256步、无schedule/earlystop，global grad norm1。网络FP32、不AMP/量化；utility/保持项float64归约。共6*9*256=13,824本轮正式更新，54最终+54个step128中间模型/optim端点，另6个借用SUP0参照。后训练不carry至其他stage；不称长期回报改进。

不设TRAIN效果前置门。只允许正确性/资产/资源/完整预算问题阻止入场；一旦入场九臂六stage完整执行，正负均保留。

## 诊断、封存与效果门

固定step0/128/256，在S-full+B上记录CE、纯RL目标、加权保持项三者参数梯度norm/cos；记录raw deterministic micro/macro/tail、相对SUP0真实类别翻转/救回/损害（聚合，不发布逐例）。step128仅诊断不评VAL不择优。正式逐step记录保持项原值/加权值、CE/U/D、grad clip、ESS、mu、原型变化及成本。规范区分factual utility和replacement-only counterfactual逻辑上下文；复用分数列不计新编码。

全部54最终模型、固定kappa、配置/源版本/RNG及部署规则先seal。随后S-full→F统一模块读出（T_S同支持）；不据F改变配置。再以全已到达TRAIN重构所有部署原型（核T对原教师<=2e-5），封存后T/SUP0/九臂一次VAL。最终部署固定mu确定性动作，raw20cos；TEST0。随机期望与确定性规则不假定等价。

主方法A门同R98：CF_BHV对T平均/最终macro各>=+.5pp、平均tail>=0、最终tail>=+1pp、>=4/6stage宏尾两非负、tiny1/2/12对T与SUP0在全部六stage无新增正确病例损害。B门：CF_BHV对SUP0、SC_BHV平均/最终macro各>=+.3pp，平均/最终tail非负。A/B全过才称本轮主候选通过开发门，非显著性证明。

约束机制C另列：CF_BHV对CF_ACT平均/最终macro严格正，平均/最终tail非负；同时报告SC/PATH下同样替换及CF-NONE比较。差分中的差分[CF_BHV−SC_BHV]−[CF_ACT−SC_ACT]逐指标报告。若BHV在SC和PATH同样有效，不归给RL独立贡献；若NONE同样好，不能只称行为形式优越。若PATH相当/更强，不证明采样策略梯度独立优势。只改变分数无有效类别翻转、utility上涨或工程通过均不算效果成功。

全class-stage/micro/macro/tail/oldnewheadtail/rescueharm、tiny/其余可加贡献及可用component敏感性均报告。重复阶段不当独立病例，不新增病人bootstrap/不删不利tiny。任何结论均是历史开发数据上的单seed机制证据。

## 工程/预算/交付

新机制测试：行为KL零点、正值/有限梯度、对共同logit平移不变、参考stopgrad、classmean与重复head-query不改类权重、校准公式/固定噪声可复现；ACT损失及梯度对R98原函数在配对真实/合成输入上相等至预定tol2e-6/2e-5。不重复已经通过的R98 Gaussian65536 MC测试。严格加载/保存/恢复与LR0，teacher冻结与S/B/F边界持续断言。

初始C13和最终C23真实合法S/B上下文，每臂两次完整step（含两episode）profile，固定第二次最坏×2；诊断、校准、F/全TRAIN部署、VAL编码上界10188及7200s保存分析发布余量全计。完整九臂计划不够则BUDGET_NOT_ADMITTED，不砍臂/步数或换原clock。复用环境、冻结特征/heads/images，不下载/复制底座，不重编码全TRAIN。拟合无需新encoder；六张S/B原生缓存核验仅在确有必要时执行，不重复已验证的同路径；新约束native主要核真实S/B目标、梯度、restore和ownGPUUUID。

pro5000 data盘、最多1张显存余量充分的GPU（可共享），reserved<12GiB；RAM64GiB/freeGPU14GiB准入，CPU4/400000s/文件1GiB/worker28GB/group29GB每2s，science8,000,000,000B+512MiB早停余量，每server新增<50,000,000,000B。独立24h含准备、校准、队列、全部工程/fit/eval/分析交付，硬保护原clock，不重置/不盲partialresume。saved-only分析修复单列费用。中性入口及PID→物理GPU UUID核验，保护全部历史与他人。

公开仅安全源码、脱敏配置、工程及聚合正负/成本，走本机显式7897代理且NO_PROXY空，exactremote/匿名报告200及source+local交付收据。图像、身份、逐例结果/Q逐类风险、features、weights、optim、rawlogs与私有路径不发布。监测保持关闭。完成本轮停止，不自动追加新实验或搜索。

## 0正式更新资源准入修订

首次选定的GPU5在启动复核时已出现其他进程，launcher在创建STARTED/controller之前退出；0GPU native/0formal/0F/VAL。依用户现行servers授权，非空GPU只要显存充足可共享，不动他人进程。本轮改物理GPU3，超过14GiB空闲准入且保留reserved12GiB限制；仅资源绑定改变，原FIRST/deadline、方法、校准种子/比例、九臂/更新预算和效果门不变。该次准入shell CPU非0未单采样，不当科学失败或新attempt重置。
