# R122 条件信息干预：完整结果与决策

**结论：冻结诊断8/10门通过，整体未通过；关闭本次“提议裁剪替代全图识别”的固定干预，不启动其RL训练。**

这是175个唯一TRAIN组件上的一次信息干预诊断，所有模型参数冻结，0优化器、0VAL/TEST。识别器为按时间选择的最新pre-RL R121共同SFT，已拟合全部TRAIN。A/B仅是组件拆分，不是全链交叉拟合或独立患者验证。

|读出|macro %|head %|tail %|tiny %|micro %|正确/175|
|---|---:|---:|---:|---:|---:|---:|
|ROOT|48.9130|81.2500|13.6364|0.0000|51.4286|90/175|
|CENTER|22.2826|39.5833|3.4091|0.0000|23.4286|41/175|
|GLOBAL_REPEAT|18.4783|33.3333|2.2727|0.0000|19.4286|34/175|
|CENTER_DETAIL|23.3696|41.6667|3.4091|0.0000|24.5714|43/175|

CENTER_DETAIL相对GLOBAL_REPEAT：macro +4.8913pp，head +8.3333pp，tail +1.1364pp，nonTiny18 +6.2500pp；A/B的nonTiny18分别 +1.3889/+11.1111pp。微平均多正确9个组件（43 vs 34），尾类净增益仅oesophagitis b d的1/8，所有tiny仍0。每类最多8个组件，小分母不支持稳定泛化结论。

## 原冻结十门（不更改）

|门|通过|
|---|---|
|macro_increment_ge_1pp|True|
|tail_increment_ge_2pp|False|
|head_increment_ge_minus1pp|True|
|non_tiny_increment_positive|True|
|A_non_tiny_increment_positive|True|
|B_non_tiny_increment_positive|True|
|GLOBAL_valid_fraction_ge_90pct|False|
|CENTER_valid_fraction_ge_90pct|True|
|equal_visual_and_input_token_costs|True|
|CENTER_generated_tokens_not_larger|True|

两项失败：尾类增益1.1364pp低于预定2pp；GLOBAL_REPEAT提议合法率153/175=87.4286%，低于90%。CENTER_DETAIL为164/175=93.7143%。非法或截断提议均保留，固定全图STOP回退；没有重采、丢样本或修改parser。

## 实际费用与对照解释

|配对策略|视觉编码行|输入tokens|生成tokens|逐调用wall合计秒|
|---|---:|---:|---:|---:|
|GLOBAL_REPEAT|525|67900|4614|179.9176|
|CENTER_DETAIL|525|67900|4522|176.6536|

上述为正式175组件配对分支费用，不含ROOT/CENTER描述控制与native。两臂真实视觉行和输入tokens相等，CENTER生成少92tokens；wall只描述本次顺序运行，不作吞吐基准。ROOT/CENTER更便宜，不能说成同计算比较。重复全图是信息消融，不是最强应用baseline。坐标元数据和像素一并改变，不能归因为单独的pixel因素。

正常run总1062响应：354提议+708识别；1420TRAIN编码、0VAL编码、0优化器。加保留failed_native1的4次编码，总1424；失败0响应/0优化器。成功worker wall451.5289秒、CPU451.6506秒；包括首次准备/失败/重试的原始总历时1113.8393秒，未重置原4小时钟。CONTROL_EXIT=0；独立CPU saved-only验证175组件、88/87拆分、276类行、parser/fallback、十门、所有费用及仪器不可变均通过，指标重算最大误差0。

首次native失败和修复：原生全序列head与手动末token head的BF16矩阵形状不同，raw差.0625。修复对齐同[B,1,D] head后差0、hidden差0，原.05阈值未放宽，科学识别路径和所有协议不变。失败CPU7.0511/wall6.9074秒、childCPU9.9757秒与源码日志保留。不声称全logit或sampling bit-exact。

## 保存记录的机制审计（事后描述，不新增门）

直接全图ROOT正确90/175，GLOBAL_REPEAT正确34，CENTER_DETAIL正确43。两种裁剪策略相对ROOT分别救回1个/损失57个、救回1个/损失48个：这支持“当前裁剪替代读出丢失了有用信息”的描述，不能区分定位错误、全局上下文丢失与单图SFT的裁剪输入迁移，也不能证明更大输入或某保护器有效。

两个提议都合法的143个组件中，GLOBAL正确20、CENTER正确34；CENTER救回18、损失4。两臂一方无效的分层合计使净差减少5，所有组件合计净+9。因此整体改善不能仅解释成CENTER格式更好/回退更多；CENTER实际回退更少。合法性是干预后分层，这些不是新的随机对照或独立效应估计。

合法框面积中位数GLOBAL .075、CENTER .09；均值.0922/.2215，仅作描述，无定位真值，不称定位更准确。两臂174/175框不同，动作改变本身不证明动作更有用。

## 后继决策

1. 关闭当前固定干预：不改prompt/中心ROI/parser/seed救分，不在这种crop-only读出上训练GRPO。原十门失败保留。
2. 不把格式优化作为主线；即使格式全合法，现有读出仍远低于保留全图，且尾类收益稀少。也不把“使框有效”作为医学奖励或新方法贡献。
3. 唯一未被此诊断回答的前提是：获取的新证据在保留已见全图的状态中，能否产生可迁移的额外信息价值？下一项为有界的context保留与反馈可识别性审查，使用已冻结框而非重新搜索ROI；须明示两图仪器的输入迁移、全部实际费用和同反馈控制。当前没有该学习协议、没有新GPU运行，不把待审查候选写成已准入。
4. 若仍是单步固定视图选择，可廉价精确求解或监督拟合，则否决采样RL主线；不能因RL必须为主线就宣称RL必要。新路线须有真正观察依赖的序列状态、可用奖励、匹配预算的fixed/myopic/同反馈控制，并与已有工具视觉RL区别。没有这些条件不盲跑。

R122的负门不否定一般主动视觉取证或所有RL，也不改变R117正信号和R121复验失败。真正continual另需generic无未来类、每臂自有连续RL状态与有界旧证据；当前静态TRAIN仪器不具备这些条件。

源码、原协议、聚合结果、独立验证和保存审计公开；组件索引、逐例预测/提议、图像、特征、权重、私有路径与失败原日志不公开。
