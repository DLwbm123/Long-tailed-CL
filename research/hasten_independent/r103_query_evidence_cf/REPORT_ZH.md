# R103 查询条件证据选择与整链交叉拟合

教师条件、全已到达历史可用的机制实验，非严格有限记忆 LT-CIL。固定seed74002，仅Ordered与分层order0。无其他顺序/种子、无超参搜索、无独立患者或显著性主张。旧F已转为反馈训练，VAL有历史开发使用，TEST0。

|流|读出|mean macro|final macro|final tail|mean old-tail|有效old-tail阶段|
|---|---|---:|---:|---:|---:|---:|
|a|T|83.019027|60.599060|36.009421|56.451442|2|
|a|CENTER|82.509482|57.876022|30.810057|53.358290|2|
|a|MULTI|82.879235|59.026383|33.678022|54.737532|2|
|a|UNIFORM|83.359307|60.234200|35.680186|55.399107|2|
|a|SUP_CF|82.989524|60.599060|36.009421|56.451442|2|
|a|RAW_EXACT_CF|83.204685|60.599060|36.009421|56.451442|2|
|a|GROUP_EXACT_CF|83.172175|60.480162|36.009421|56.451442|2|
|a|GRPO_CF|83.170600|60.480162|36.009421|56.451442|2|
|a|GRPO_IN|82.268383|57.420772|30.106593|52.418423|2|
|b|T|75.454076|63.591566|43.265783|43.534157|4|
|b|CENTER|74.887606|62.112456|40.261948|41.496650|4|
|b|MULTI|76.925583|68.418117|54.520834|46.971579|4|
|b|UNIFORM|76.042973|66.825952|49.747483|45.622687|4|
|b|SUP_CF|75.348843|63.480150|43.044053|43.296338|4|
|b|RAW_EXACT_CF|74.680454|62.924393|42.596755|43.666432|4|
|b|GROUP_EXACT_CF|74.668333|62.924393|42.596755|43.666432|4|
|b|GRPO_CF|74.668333|62.924393|42.596755|43.666432|4|
|b|GRPO_IN|76.932929|67.689683|53.259160|45.190039|4|

## 预定配对差值（GRPO_CF−参照，pp）

|流|参照|mean macro|final macro|final tail|
|---|---|---:|---:|---:|
|a|T|+0.151573|-0.118898|+0.000000|
|a|UNIFORM|-0.188707|+0.245962|+0.329235|
|a|SUP_CF|+0.181076|-0.118898|+0.000000|
|a|RAW_EXACT_CF|-0.034085|-0.118898|+0.000000|
|a|GROUP_EXACT_CF|-0.001574|+0.000000|+0.000000|
|a|GRPO_IN|+0.902218|+3.059390|+5.902828|
|b|T|-0.785743|-0.667173|-0.669028|
|b|UNIFORM|-1.374640|-3.901559|-7.150728|
|b|SUP_CF|-0.680510|-0.555757|-0.447298|
|b|RAW_EXACT_CF|-0.012121|+0.000000|+0.000000|
|b|GROUP_EXACT_CF|+0.000000|+0.000000|+0.000000|
|b|GRPO_IN|-2.264596|-4.765290|-10.662405|

六视图始终全部编码；选择教师也不节省编码计算。固定融合UNIFORM为概率平均；学习臂为确定性argmax动作，不混合、不采样部署。候选Oracle仅封存后诊断，不能当作可部署方法。

归因按以下六项读取完整表：候选互补性（CENTER/MULTI与ORACLE）、固定融合解释量（UNIFORM）、学习选择对T/UNIFORM、GRPO对SUP与两种EXACT、CF对IN旧尾损害、OOF到FULL输入/动作变化。候选有效不等于GRPO有效；胜SUP而不胜GROUP_EXACT不支持采样必要性。CF与IN同时含上游拟合规模和置信度差异，不能只归因于泄漏。

Ordered前两阶段无tail，null；旧尾有效均值分母另列。tiny按原始身份绑定，CLASS_STAGE提供correct/count，重复阶段不当独立病例。F是训练反馈，未做旧F独立评估。

全体50router/60fusion/20新折teacher及10复用FULL教师封存后统一VAL，共90stage读出。新增更新/编码、复用历史教师成本和失败费用分别记账；worker子项不能重复相加。GPUkernel时间未采样。准备/文档/SSH/发布CPU非0未全采样。

本轮完成后停止，monitor OFF，不启动后继。私有身份/图像/逐例预测/特征/权重/优化器/原始日志仅留授权data盘。

## 结论：主线GRPO未超过强监督和精确控制

固定主候选GRPO_CF的Ordered五阶段平均macro对T为+0.151573pp，但final macro为−0.118898pp、final tail为0差；分层流mean macro−0.785743pp、final macro−0.667173pp、final tail−0.669028pp。因此本轮不支持GRPO稳定超过强监督。Ordered对SUP平均macro+0.181076pp但final−0.118898pp；分层对SUP平均−0.680510pp、final−0.555757pp。对RAW_EXACT平均macro均略负；对GROUP_EXACT，Ordered平均−0.001574pp、最终相同，分层所有阶段最终分类预测完全相同。未显示采样GRPO相对精确目标的独立收益。

### 六项预定归因问题

1. **局部证据有互补性，但不普遍且小分母主导。** Ordered最终CENTER/MULTI macro均低于T；后验候选Oracle macro62.116151，对T+1.517090pp，27张教师错可被某候选救回。分层最终MULTI68.418117/54.520834 macro/tail，对T+4.826552/+11.255051pp；Oracle75.383917，对T+11.792351pp，67张教师错存在正确候选。但MULTI的+4.347826个macro百分点来自legacyC1仅1张VAL由0/1到1/1，约占净macro增益90.1%；除去这项原分母贡献，其余类别合计+0.478726pp，不能将单例提升外推为广泛医学泛化。Oracle不可部署、不作为筛选门。

2. **固定融合已有部分收益。** Ordered UNIFORM平均macro83.359307高于GRPO83.170600，最终macro60.234200低于GRPO60.480162；分层UNIFORM最终66.825952/49.747483，明显高于GRPO62.924393/42.596755。不能把全部多视图收益归给学习选择或GRPO。

3. **主线选择器多数保留教师，未利用全部候选空间。** 最终GRPO_CF选择动作0比例：Ordered99.7055%，分层98.4099%。Ordered只选择5张局部候选，救回0、损害3；分层选择27张局部候选，救回8、损害15。不能只凭旧尾类保护或概率变化称整体成功。SUP/RAW在Ordered最终全部选教师，GRPO仍有3张非尾类损害。

4. **采样与精确优化的结论接近。** Ordered最终GRPO与GROUP_EXACT预测逐例完全相同；分层五个stage两者预测全部相同，虽然策略概率/训练轨迹不相同。Ordered中间stage1/2分别有6/26个预测不同，不能声称所有轨迹bitwise相同。GRPO_CF各stage全同奖励组约92.0%—99.3%，真实ratio clip几乎从未激活（分层stage0仅约0.0001907% step-query-action项，其他stage均0），grad norm clip均0。多次old-policy更新确实实现，但此训练条件下PPO裁剪并未成为实质影响项。不能据此判定所有GRPO无效。

5. **CF对IN的收益取决于任务流。** Ordered CF最终相对IN +3.059390macro/+5.902828tail，旧尾类新增损害从8张降到0（IN救回3，CF0）。分层CF最终相对IN −4.765290macro/−10.662405tail，旧尾类净BA略高+0.436185pp（IN救回12/损害14，CF救回1/损害0），但整体更差。分层IN最终macro67.689683超过T4.098117pp，其中同一个legacyC1单例贡献+4.347826pp，其余类别按原分母合计−0.249709pp；不能挑IN替换预先主候选或声称RL已稳胜。CF与IN还改变了反馈生成器数据规模、置信度等，并非纯粹隔离泄漏。

6. **存在半数据到FULL的反馈/部署差异，但不能确定唯一病因。** 最终GRPO_CF TRAIN OOF→FULL策略平均L1差Ordered0.059216/分层0.091580，argmax动作改变0.2346%/3.5918%；同表教师macro从OOF60.959160/59.766795到FULL86.992459/82.664990，候选在拟合内的优势明显大于OOF。OOF仍是选择器训练表，不是独立验证。具体是否因容量、保守正则、反馈稀疏或分布迁移导致保留教师，当前没有匹配干预，不能因果指定。

## 数值读出一致性限制

额外saved-only核对发现：R103冻结教师与R102的Ordered五stage预测完全相同；分层前四stage相同，最终1/1698张预测不同（new index5/legacy19/original13，类VAL160张，原正确变错误），对应final macro−0.027174pp。权重、原型和TRAIN原生容差一致；但未满足全部VAL逐例与R102严格相同，不能声称严格bitwise读出复现。不同冻结编码/批处理路径的浮点差异是可能解释，未用新forward证明。没有事后替换预测、重评或恢复训练。所有主比较使用同一次R103的T和共同FULL bank；此差异不改变主线未胜SUP/EXACT的结论。BASELINE_READOUT_AUDIT.json保留负证据。

## 完成、失败与费用闭合

正式两worker均exit0，总控exit0，20新折教师、60fusion、50router、90stage读出，2160/30720/25600正式更新；327680个G8组、2621440动作采样。10个R102 FULL教师按阶段复用，其历史CPU单列3317.629602s（不混入本次增量）。

attempt2 worker CPU合计17265.047484s，计入首次0科学import失败后全部worker CPU17268.850606s；controller CPU478.755644s，失败controller另见COSTS_COMPLETE。正式controller wall9001.146398s；原FIRST至科学完成9792.459321s（2.720小时），未重置。native工程176次优化与外部软件工程10次另列，不记正式；native/teacher/crop/fusion/router/eval CPU已含worker。

峰combined RSS14356021248B，science closeout allocated1125130240B，两worker GPUreserved各2126512128B。首次0科学依赖失败保留；总控和worker正常结案，没有科学失败。额外saved-only汇总CPU1.622513s/wall1.508480s，新增encoder/optimizer/VAL/TEST0。先前终端收据被误判为缺失，实为查看/落盘时序，已依据实际exit0纠正。

准备/绑定/软件测试/SSH/文档/发布CPU非0未完整采样；GPUkernel时间未采样，不以CPU或并行wall冒充GPU效率收益。详COSTS_COMPLETE/REUSED_TEACHER_COSTS。

全部预定阶段/对照已执行并校验，保留所有正负结果；当前没有独立多seed、额外order、患者独立或显著性证据。按授权完成后停止，monitor OFF；未开展任何后继调参、视图搜索或新实验。

独立saved-only复核全部90个stage读出之micro/macro/tail/old-tail、所有空组/均值分母、逐类计数/救援损害及Oracle，误差符合1e-10校验；新增encoder/optimizer/VAL/TEST为0。COST_PHASES.csv提供20折teacher、60fusion、50router的130条分段CPU/wall（均已含worker，不再加）；crop/加载/屏障等待等未独立计时的部分仍含worker总额，不能把差额全称crop成本。
