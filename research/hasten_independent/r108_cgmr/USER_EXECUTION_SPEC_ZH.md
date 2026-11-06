# 给 Codex 的执行 Prompt：R108-CGMR

你是本项目的实验执行与独立核验负责人。实现并执行下面冻结的新实验，不仅写建议；但所有数据访问、资源、停止条件都必须满足。这个任务只授权本轮 A→条件 B，不授权后续实验、调参救援、TEST 评估或自动监测。保持 monitor OFF。不要修改 R107 的方法、结果或失败判定。

## 0. 起点、范围和实验身份

仓库：DLwbm123/Long-tailed-CL。
基准提交：c7d5db2d51b77a6c1b1efa92c6612c12f6762680。
必须先阅读该提交 research/hasten_independent/r107_survival_set_lt/ 下的 PROTOCOL.md、ANALYSIS_ZH.md、manager.py、memory.py、meta.py、rl.py、learner.py、worker.py、SAVED_POLICY_AUDIT.json。
新工作名：R108-CGMR，Conditional Group-relative Memory Repair，中文“集合条件组相对记忆修补”。若 R108 已占用，使用新的明确目录名并披露，不覆盖现有实验。
这是待验证的新组合，不声称原创性、性能成功、标准 GSPO 或 RL 必要性。这里的学习动作是“选择一份完整的局部修补记忆”，不是逐 token 长序列；更准确地说是使用延迟回报的有限候选决策及 GRPO-like 策略改进。
不得混入独立 HASTEN、RaPO 或此前强监督模型路线。
本轮有意把影子输入从 R107 的 current-only 扩展为本臂合法历史记忆＋当前到达数据；这是新的组合方案，不是单变量消融，不能将跨 run 改善归因于某一个模块。

## 1. 不改变的实验条件

沿用 R102/R107 的 HK 数据绑定、Ordered 与 stratified order0、五阶段 5+5+5+4+4、6821 TRAIN 和 1698 final VAL；TEST 始终为 0。
只用原始通用预训练 BiomedCLIP，六视图、FP32 单位化 512 维。不得导入旧 HK 监督模型、旧策略、旧价值网络、旧优化器或旧特征缓存作为新实验训练起点。
正式分类器、类别均衡采样、AdamW、KD 和每阶段 200 次 new+old 更新、选记忆后 100 次 memory-only 更新，完全继承 R107。正式分类器各自沿自身历史推进。
记忆仍为最多 256 tokens，每 token 2064 bytes，加 184 bytes 的 observed count/birth 表，总上限 528568 bytes/arm。所有局部 token 必须有保留的父 global，每个已见类别至少一个 global。模型/优化器/策略参数单独记账，不充当隐藏样本缓存。
每臂任何时刻只能使用自己的上一阶段实际记忆与当前到达 TRAIN。不得读未来到达数据、取另一臂旧记忆、恢复已丢弃旧图像/视图/特征。影子轨迹同样执行不可逆淘汰，不能从之前的影子支持池恢复被丢弃 token。
正式验证前冻结协议、源码和全部随机种子。模型训练、候选构造和正式选择都禁止使用 VAL/TEST。所有正式端点全局封存后才统一评估 VAL。

## 2. 参照记忆与候选生成：固定机制，不依赖可训练分数

对当前合法池 P，先使用 R107 ZERO_ACTOR 对应的合法下一动作均匀过程得到 M0。沿用覆盖锚点、完整可行性掩码与父子约束；这不是对所有最终合法集合均匀采样。
同一状态下所有候选固定总 token 数和 global/local 总数。这样本轮只检验内容修补，不学习 global/local 配额。
保护 M0 的每类 coverage anchor。构造最多八份额外候选：
- LOCAL 类型四份，目标操作数依次 1、1、4、4：移除一个已有 local，加入一个当前 P 中尚未选择、且父 global 已保留的 local。
- BUNDLE 类型四份，目标操作数依次 1、1、4、4：移除一个非锚点父样本及其当前保留的全部 k 个 token，加入一个尚未保留的父 global 及 P 中合法可用的 k-1 个 local。可以跨类，但不得破坏任何类别覆盖。新增 local 在合法可用视图内按预注册随机规则选择。
每份候选最多四次操作，移除 token 总量不超过 min(24, floor(B/10))。禁止重复撤销/恢复来伪造编辑次数；禁止读取旧样本缺失视图来凑足一个 bundle。
每个目标候选最多 64 次预注册随机尝试；无法完成目标操作数就缺失，不降级后挑好结果。按最终 token 集合去重；重复候选不能重复占概率。不足八份时使用实际候选数；只有 M0 时是合法 no-op 状态，不伪造训练信号。
最终 A(s)={M0,...,Mm}，m<=8。候选生成只依赖合法 P、固定规则和候选 RNG，不能使用 actor、监督排序器、训练 reward、audit 或 VAL 过滤候选。
完整候选都是单个动作。内部 1~4 次修补不是需要估计 likelihood 的可训练动作序列。

## 3. 稳定的集合条件输入，不使用漂移的价值编码器

删除 actor 对可训练 ValueNet encoder 的依赖，不再把一次集合回报差拆成两个 token 的固定 +/-d 价值。
使用确定性、detach 的 48 维集合描述，MLP 为 Linear(48,32)-Tanh-Linear(32,1)，最后一层初始化为 0，共 1601 个参数。
定义每份候选的 12 维 psi(M)：
1. 当前合法 global 池到同类保留 globals 的宏平均覆盖，cosine 映射到 [0,1]；
2. 稀缺类别集合 T 上的相同覆盖；
3. 对保留父样本，当前合法池可用 local 到已保留 local 的平均覆盖，无 local 为 0；
4. 类内已保留 globals 的平均成对冗余，singleton 按 0 并在审计报告其占比；
5. global 类别分布熵 / log(C)，C=1 时为 0；
6. 最小类 global 数 / 平均类 global 数；
7. 属于 T 的 global 比例；
8. 旧类 global 比例；
9. 固定当前分类器在候选 bags 上的类均衡真实标签概率；
10. 相同 bags 的宏平均真实标签 logit margin / 40，裁到 [-1,1]；
11. 相同 bags 的宏平均预测熵 / log(C)，C=1 时为 0；
12. global 样本所属类别 age/(1+age) 的平均。
T 使用当前已见类别累计可用训练计数的下三分位阈值，包含阈值并列类别。影子计数只能取影子实际已到达支持计数；正式计数来自合法 observed 表。不把 T 称为正式 HK tail；正式 tail 仍沿用 <250。
48 维输入为 [psi(M), psi(M)-psi(M0), psi(M0), state8, edit4]。
state8：C/23、B/256、B/|P|、Ng(M0)/B、P 中本次新到达父样本比例、clip(log(1+min_c accessible_parents_c)/log(257),0,1)、mean_c log(1+observed_count_c)/log(1+max_c observed_count_c)、stage/(1+stage)。
edit4：是否 M0、移除 token 数/B、替换 global 父样本数/max(1,Ng(M0))、操作数/4。
所有统计只在当前合法支持池及候选记忆上计算，不能访问任何 query 的特征或标签。覆盖相似度分块计算，不创建全历史巨大两两矩阵。
按字段写 FEATURE_SCHEMA.json，并提供缺失/边界值测试。可训练分类器随阶段变化是状态变化；不能另设不断更新且未受 actor 约束的输入编码器。

## 4. 策略、训练目标与监督对照

先验 pi0(M0)=0.5；m>0 时每个修补候选为 0.5/m；m=0 时 M0 概率为 1。
pi_theta(k|s)=softmax(log pi0(k)+f_theta(xk)-f_theta(x0))。
主方法正式执行从该分布抽样一次；不得评估多个正式结果后择优。
每个影子根状态枚举全部合法候选的真实训练后回报 Rk，最多九个分支。这是候选全反馈，不是假装九个动作都是 actor 独立采样。
在更新前冻结 p_old。定义：
  b=sum_k p_old[k]*Rk
  sigma=max(std_unweighted(R),0.02)
  Ak=stop_gradient((Rk-b)/sigma)
  rk=p_theta[k]/p_old[k]
  J=sum_k p_old[k]*min(rk*Ak,clip(rk,0.8,1.2)*Ak)-0.01*KL(p_old||p_theta)
最小化 -J，每根状态两次 AdamW 更新，lr=1e-3、weight_decay=1e-4、grad clip=1。第二次仍使用同一组冻结 p_old、回报和优势。所有候选回报相等时优势精确为零。
注意 p_old 权重不可省略，不能给枚举动作错误使用采样均值；没有序列长度归一化；不把同一最终集合的多个构造路径重复计数。
REWARD_SHUFFLED 使用自身根状态与分支，把整组最终 R 在所有排列中均匀置换，允许固定点及恒等排列，不重抽。置换后重新计算 b，但 sigma 使用未加权 std，具有置换不变性。一次未裁剪条件梯度的零期望不代表完整 Adam 训练是严格零过程。
SUP_GREEDY 使用相同 48→32→1 容量、同类根状态数量、同样候选生成和同样全反馈预算。回归 (Rk-R0)/0.05，预测用 g(xk)-g(x0)，M0 预测固定为 0；每根两次同配置 AdamW MSE 更新。正式执行选预测收益最大的候选，最大收益<=0 时选 M0；并列优先 M0。不得追加更多训练或窥视 VAL。
有限编辑和 no-op 先验只是结构约束，不保证正式风险不增加；不能把它们称为统计安全改进保证。
训练臂后期各自旧历史不同，因此各自生成合法根状态，禁止为追求“相同数据”互换旧证据。报告端到端配对对照与同状态目标函数验证的区别。

## 5. 影子环境：只用当时合法证据，匹配竞争规模

不能把正式已训练分类器复制出来，再把它已训练过的样本当成独立 query。影子分类器从通用初始化重新训练，prototype/梯度/候选/蒸馏的全部路径都排除该影子 query。
当 identity component 首次合法到达时，在每类内按预注册独立 RNG 划定 S / Q_fit / Q_audit 的固定身份角色；贯穿本轮不互换。>=3 components 时，Q_fit 和 Q_audit 各取 min(4,max(1,floor(n_components/4))) 个互不重叠组件，各最多八个父样本，其余为 S；不足三个组件则只能支持，不制造独立 query。
identity component 的角色必须全局一致，跨类、跨阶段都不能重新分配；已分配为 query 的组件，不能因为后来可访问组件变少而改为支持。类内分层分配必须服从这一全局约束；不足以形成 S/Q_fit/Q_audit 时报告缺失，不能复制或跨角色借样本。对已分配的历史组件，“不足三个组件只能支持”不表示可以把其 query 角色改成支持。
本角色划分用于影子机制；正式分类器仍按原协议使用全部合法 TRAIN。因旧记忆经过历史筛选，Q_audit 也不是整套算法的独立临床测试集；必须披露选择条件性。
旧父样本只能使用上一记忆实际保留视图，不能补齐六视图。记录 query 的视图缺失分布。
影子支持可来自本臂自己的上一记忆和当前 TRAIN，不得只限当前 4~5 个新类。保留所有能合法支持的当前已见类别。
每根从当前已见且有合法查询的类别中按预注册顺序选两个作伪未来类；其余为前缀已见类 C0。不得从真正未来数据取这两个类。若无法满足至少三个可评估类别及两个伪未来类，记录并跳过学习，不补造类别。
前缀类分成至多五个伪到达阶段，使年龄真实来自模拟历史，而不是改 age 数值。16 个训练根交替 decreasing / mixed；支持 cap 按三档 32/8/2 设置，decreasing 按当时已知频次分档，mixed 预注册打乱；实际支持为 min(cap, S_available)，不复制。
前缀累计 64 次分类器更新，均分到其实际阶段，余数从最早阶段分配；历史压缩用固定 ZERO 构造器。仅最后一个前缀阶段的到达支持加前缀实际记忆形成修补池，不能回收更早被丢弃支持。
请求影子预算按根序号循环 64/128/256。实际记忆大小为 min(B_requested,|当前合法池|)，所有分支相同；不足产生替换时保留 no-op，不更改预算来救援。报告实际 B 与 C0、age、可访问支持量的联合分布，不声称已经与正式状态完全同分布。
每阶段每个可训练臂固定 16 个根，不按结果补采。根内分支共享初始分类器、optimizer、查询、伪未来到达、学习随机数与参照构造 RNG。相同随机数不保证不同记忆下抽到同一病例，必须准确表述。
候选 Mk 的后续学习：4 次 memory consolidation；12 次 future1+Mk；用固定 ZERO 在该分支自己的合法池压缩；4 次 consolidation；16 次 future2+memory。共 36 次分类器更新/候选。后续不再调用可训练修补器，保证本轮信用只来自当前选择。主方法学的是相对该固定后续策略的两阶段价值，不宣称评估了自身无限期执行的完整价值。
每阶段策略/监督网络可继续携带参数和 optimizer；所有带样本信息的根状态、未选择候选和影子支持/query 特征在阶段末清理，不跨阶段当额外 replay。

## 6. 统一回报与旧尾保护

在固定的 query 可评估类别 E_h 上计算：
  U_h=0.5*macro_BA_h+0.5*macro(1-Brier_h/2)
前缀修补时的旧稀缺集合 T0 固定为 C0 内实际影子支持计数的下三分位类（含并列）。它不是 HK 正式 tail。
先得到 no-op 分支在两个未来端点的 q0(c,h)=1-Brier0(c,h)/2。
  Hk=mean_h mean_{c in T0 intersect E_h} max(0,q0(c,h)-qk(c,h))
  Rk=(U1k+U2k)/2 - 0.25*Hk
空旧稀缺集合的有效阶段不计入 H 平均；全部为空时 H=0，明确标注不可评价旧稀缺保护。
参照是相同根状态固定 no-op 分支，不是被 actor 改写过的 birth 质量。最终公式对所有候选固定，候选不能改变 query 集合、分母或 tail 标签。
Q_fit 只生成训练回报；Q_audit 只作预注册诊断/准入，不能用于反向传播、候选筛选、checkpoint/温度/学习率选择。
减去 R0 本身并非创新；组内再中心化会抵消公共基线。收益应来自局部候选、集合状态、真实配对后续与可识别反馈。

## 7. A 阶段：小型准入，不先跑六臂全流

只使用两个流各自第一阶段已到达 TRAIN，seed=74002。每流固定24个根，无 actor训练，最多48*(64+9*36)=18624次影子分类更新。查询两份在同一组训练后的模型上评分，不因 Q_audit 再训练。
每根用 Q_fit 的 R 选 k*=argmax R_fit，平局优先 M0；只在 Q_audit 评估该固定选择相对 M0 的收益。报告全部根、并列、无替换、空尾组和原始正确数。
A 的继续条件（这是开发筛选线，不是统计显著性）：每流至少16个可评估且有非平凡候选的根、至少8个旧稀缺可评估根；两流各自 mean(R_audit[k*]-R_audit[0])>=0.005；mean(U_audit[k*]-U_audit[0])>=0；mean H_audit[k*]<=0.005。
0.005 是 U/R 的分数单位，不等价于0.5个BA百分点。根共享来源数据，不当独立患者重复。
可报告 Q_audit 自身最佳候选作为乐观描述上界，但绝不能用它做候选选择、训练或替代上述跨查询门。
A 不通过或证据不足就结案，不启动 B，不读取 VAL/TEST，不改变候选数、预算或门槛重试。通过后才进入已授权的 B；A 不产生可携入 B 的学习权重。

## 8. B 阶段：预注册六臂、三 seed、两流

全新初始化，seeds=[74002,74003,74004]，即使某 seed 结果不佳也完整保留。检查历史使用情况并披露，但不能看结果换 seed。
六臂：
1 BASE_ZERO：原合法 ZERO 记忆，不修补。
2 BALANCED：原固定均衡记忆，不修补。
3 RANDOM_REPAIR：同一候选生成器，永远按 pi0 抽样，不训练。
4 SUP_GREEDY：相同全反馈训练预算的监督收益预测。
5 REWARD_SHUFFLED：同容量同目标、全排列均匀错配。
6 CGMR_MAIN：固定主候选，正确回报训练。
每臂自己的旧记忆，独立分类器和 optimizer。三个学习臂各自每阶段16个合法影子根；共有 1440 根上限、558720 次影子分类器更新上限，跳过/空候选按实际计数，不伪造花费。
正式总计2*3*6*5=180个分类器端点；每端点300次更新，共54000次。连同 A，影子+正式分类器更新上限631344，另外单列策略/监督优化、编码、query前向、工程验收等。
分离 classifier、proposal、memory draw、query split、shadow episode 的 RNG 流并保存种子表；同一配对实验共享外生随机数，不能依赖某臂额外调用改变后续 RNG。
正式流程仍为：当前合法到达与本臂影子训练 → 正式200步 → 基准/候选选择 → 正式100步 → 封存。
全部180个分类器及相关策略端点封存后统一 VAL 一次，不选最佳阶段、seed、臂或温度。

## 9. 预注册解释边界

分别报告，不能合并成一个模糊“成功”：
A. 学习增量：两流各自三seed平均 mean-macro 与 final-macro 均比 BASE_ZERO 至少+0.5pp；两指标均超过 RANDOM_REPAIR 和 REWARD_SHUFFLED；每流至少2/3 seed的 mean-macro 配对差为正，且任何seed final-macro 不比 BASE_ZERO 低超过1pp。
B. 风险保护：相对 BASE_ZERO，两流各自平均 final-tail / final-old-tail 不下降超过0.5pp；所有tiny类逐seed正确数与原分母贡献单列。tiny正确数下降则风险保护标为未通过，不得删类翻案。
C. 方法竞争力：两流 mean/final macro 不比 BALANCED 低超过0.5pp。学习增量通过而此项失败，不能称总体方法成功。
D. RL必要性：两流 mean/final macro 都超过 SUP_GREEDY，且每流平均mean-macro至少+0.3pp，才能给出初步支持。若监督相当或更好，应保留“可学习的修补可能有效”，明确“不支持RL必要性”。
这些是建议冻结的效应量开发门，不是显著性或临床泛化证明。三seed不是三个独立患者样本，两个流也不是两个独立数据集。多轮开发 VAL 的限制必须写明。

## 10. 软件与机制测试，训练前必须通过

- 精确字节、父子依赖、类别覆盖、锚点保护、只来自本臂合法池；所有候选 global/local 总量与 M0 相同。
- 缺失旧视图不可恢复；无合法候选时概率1 no-op；重复最终集合不重复计权；不存在任意 ID 后缀掩码。
- 同一内容重命名后可行集合/概率等变，区分概率等变与同seed逐位相同。
- Q_fit/Q_audit 与支持的 identity-component 交集为零；query不进入shadow prototype/梯度/候选/teacher，未来类未到达统计空值。
- f=0时恢复pi0；all-equal奖励优势0；所有非no-op都更差时，未裁剪首步的no-op logit梯度方向应提高其概率。
- 完整候选枚举目标的autograd与有限差分一致；不能遗漏p_old；首步ratio=1；two-update行为快照不变。
- 对小组枚举所有排列，均匀reward shuffle的未裁剪首步期望梯度近0；对derangement验证反相关，防止误实现旧对照。
- 固定features时只有actor改变概率；不存在隐式更新的价值编码器。
- 至少记录真实输出KL、TV、候选熵、no-op率、修改token数与在同状态32个独立draw下的选择分布；这些forward/draw不冒充32次独立训练。
- 恢复模型/optimizer后前向一致、finite、RNG可复算。

## 11. 成本、准入、交付和停止

继承项目当前有效且不弱于R107的资源/数据隐私限制：monitor OFF、最多两个允许GPU、CPU并发4、原始FIRST起24h硬截止，不重置计时；不占用其他用户任务，不下载/复制基座、不量化、不swap、不root落盘、不删除历史。若仓库有效约束更严格，服从更严格约束。
训练前做真实最坏形状 native profiling：C23、B256、最大合法池、九候选、完整影子组与分类器恢复；只做合规合成最大形状的部分必须明确标注，不把复制数据当科学样本。
以实测成本加安全余量验证 A+B 完整计划及报告能否容纳。不能通过少跑seed/控制/阶段、减更新或提前看VAL来赶预算。不准入就交付实现与预算证据，正式科学运行不启动。
计算实际/逻辑成本分开；共享编码计物理一次；分别记录分类器、actor、监督、工程优化步，query前向、候选构造、CPU/wall/GPU峰内存/RSS/磁盘。GPU kernel时长未测就写未测。
公开安全源码、冻结协议与配置、FEATURE_SCHEMA、SEEDS、candidate/feedback/audit汇总、全部逐阶段/逐类/tiny/救回损害、成本、恢复核验、失败记录。禁止公开像素、样本/患者原始ID、特征、权重、optimizer、私有路径或逐样本概率日志。
推荐交付：PROTOCOL.md、CONFIG.json、FEATURE_SCHEMA.json、SEEDS.json、PILOT_GATE.json、QUERY_COVERAGE.csv、CANDIDATE_AUDIT.csv、FEEDBACK_TRANSFER.csv、POLICY_BEHAVIOR.csv、STAGE_METRICS.csv、CLASS_STAGE.csv、TINY_CONTRIBUTION.csv、COSTS.json、VERIFICATION.json、ANALYSIS_ZH.md。
先冻结协议commit再训练；结束独立核验后发布结果commit，确认远端SHA与公开报告可访问。私有信息仅留合规私有存储。
最终明确是哪一种：A反馈不可迁移/证据不足；B学习无增量；学习有增量但风险/竞争力失败；监督同样有效；或初步支持策略增量。不要只写“跑完了”。
无论结果如何，本轮结案即停止；不自动启动新方案或监控，不为失败追加调参，不读取TEST。
