# R101 joint representation / instance policy

COMPLETE_JOINT_MECHANISM_NEGATIVE

|mode|mean macro|final macro|mean tail|final tail|
|---|---:|---:|---:|---:|
|T|66.840041|64.876782|50.729225|44.924938|
|A0|66.840041|64.876782|50.729225|44.924938|
|UNIFORM|64.971043|64.712544|47.521432|44.230954|
|CE_WEIGHT|65.688739|66.992720|48.769353|49.612333|
|RLPO_STYLE|64.641003|64.084601|47.061237|43.832489|
|PROTO_GRPO|65.026815|64.594770|47.295562|44.211636|

All four arms share the adapter, teacher anchor and fixed S evidence. Original T is a full-history strong teacher; A0 is a fresh identity/uniform module readout. Not strict bounded-memory CL, independent patients, multiple seeds or proven top-tier novelty. No posthoc arm/epoch selection. RLPO_STYLE is a transferred geometric auxiliary, not author code or standard GRPO. PROTO_GRPO is a one-update Gaussian group-relative policy gradient with leave-one-out advantages and no group-standard-deviation normalization; sampling uniqueness is a separate stronger claim.

Main comparisons, all class-stage/group signs, rescue/harm and original-denominator contributions are included. Continuous reward, concentrated weights or training accuracy do not establish classification benefit. Native/profile costs are included in worker, not added twice; borrowed original teacher/feature costs are historical, not free. Preparation/SSH/publication CPU nonzero and not fully sampled. No monitor.

## 完整结果解释与贡献分解

正式attempt1 exit0，21504/21504更新，24最终+24中间+24stage起点共72个保存状态。全部模型/optimizer有限，跨stage参数carry和参考EMA重建通过。非工程失败；主方法teacher门和两个matched门均失败，只有1/6stage对原教师macro/tail同时非负。

PROTO_GRPO相对原教师：平均macro/tail −1.813225/−3.433662pp，最终−0.282011/−0.713302pp。相对CE_WEIGHT：平均−0.661923/−1.473791pp，最终−2.397949/−5.400697pp。RLPO_STYLE也未超过UNIFORM或CE_WEIGHT，本轮没有建立奖励引导辅助接入强表示的优势。不能将这一个固定配置的负结果外推为所有RL无效。

CE_WEIGHT最终macro/tail对教师+2.115938/+4.687395pp，但平均仍−1.151302/−1.959871pp。最终class12从0/2到1/2，这一个正确病例贡献macro+2.173913pp、tail+4.545455pp；其余类别合计最终macro−0.057975pp、tail+0.141940pp。因此不能仅凭最终数字宣称稳定提升或选择它替换预注册主候选。class1/2最终仍0/1，PROTO_GRPO三tiny最终仍0/1、0/1、0/2。

原分母分解：CE_WEIGHT对T平均tiny贡献macro/tail+0.759144/+1.515152pp，其他类−1.910446/−3.475023pp；PROTO_GRPO对T平均tiny+0.396825/+0.757576pp，其他类−2.210051/−4.191238pp。主方法无新增T-correct tiny损害，保护门通过不等于修复了tiny。完整逐类正负、old/new/head/tail和救回/损害暴露见公开聚合CSV。重复stage不是独立病例。

## 训练信号与迁移

PROTO_GRPO在各stage最终B macro约96.19%—99.04%，tail95.69%—100%；这些是模块训练查询，不是验证成绩。F上对T平均macro/tail−3.867436/−5.110433pp；四个学习臂F均下降。这提示共同适配/支持构造与B反馈的泛化也存在问题，不是单纯没有学到B奖励。该观察不构成对某一个组件的因果定位。

3072个G8组均有数值回报差，但奖励含密集项，这不代表每组硬分类收益都不同。策略梯度裁剪比例stage0—5约8.40%、42.77%、73.44%、76.95%、83.98%、57.62%；这些是完整辅助梯度（含KL）口径，不是纯RL梯度方差。不能仅据裁剪比例归因或事后调权重补跑。组回报/硬收益/密集项/损害与权重、adapter变化均公开聚合。

## 实测费用与范围

worker wall583.281434s≈9.72分钟/CPU736.335072s；原FIRST至退出1259.286194s≈20.99分钟，含实现准备，24h未重置。controller CPU11.284875s另计。worker core CPU735.363066s/wall581.253211s，以及native/profile CPU3.554851s/wall3.520755s已含worker，不重复加。外部软件CPU0.583948s/wall0.551617s另计，0optimizer/encoder。saved日志聚合另有CPU/wall收据，0新score/encoder/optimizer/F/VAL/TEST。准备、导入、SSH、报告和发布CPU非0未全采样。

正式12288监督+9216策略更新；原生16监督+12策略=28工程更新单列，工程参数丢弃。监督B查询行15032320、支持变换行52258304、辅助B查询行11274240、采样查询打分行30064640。3072个G8组/24576完整rollouts，59723776高斯坐标实际抽取，其中59699200坐标对应多支持有效动作；单例坐标没有权重自由度。诊断、reference和prototype重复算子不是完整硬件FLOPs计数，不声称这些计数覆盖全部计算。

全部24端点封存前F/VAL/TEST0；封存后F43776/VAL44028分类行，VAL新图像编码7338行，TEST0。模型训练使用借用缓存，并不意味历史特征提取免费。worker peak RSS3804577792B；2秒采样group峰3180691456B不能当连续真实group上界。GPUreserved峰968884224B；退出实际分配磁盘116346880B，权重/optimizer/私有预测等全部留data盘。保守预算2.52h含2h结案余量不是实测耗时或效率胜利。

本轮按冻结负门结束，无调参救援、选择epoch/arm或新实验。monitor保持OFF。公开source/protocol、完整聚合结果、正负贡献、校验和费用；身份、逐例、Q逐类风险/特征、权重、optimizer和raw训练日志保留私有。
