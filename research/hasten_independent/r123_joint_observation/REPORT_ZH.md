# R123 固定框联合观察：完整结果与路线收敛

**结论：正常退出0，独立保存校验通过，但冻结诊断7/13门通过、整体失败。关闭当前冻结仪器的联合证据救援，不继续接口/顺序/prompt/ROI/seed变体，不启动其RL学习。**

本轮是175唯一TRAIN组件上的冻结读出资格诊断，非RL训练。全部框、原非法全图回退、A88/B87与原问题文本固定，无新提议或权重更新，无VAL/TEST。识别器R121 SEED_SFT已拟合全部TRAIN，组件拆分不是全链crossfit或患者独立。

|读出|macro %|head %|tail %|tiny %|micro %|正确/175|
|---|---:|---:|---:|---:|---:|---:|
|NATIVE_ROOT|47.8261|78.1250|14.7727|0.0000|50.2857|88/175|
|DUPLICATE|47.2826|76.0417|15.9091|0.0000|49.7143|87/175|
|FIXED_CENTER|45.1087|71.8750|15.9091|0.0000|47.4286|83/175|
|GLOBAL_BOX|43.4783|70.8333|13.6364|0.0000|45.7143|80/175|
|DETAIL_BOX|47.2826|75.0000|17.0455|0.0000|49.7143|87/175|

## 配对增量与原门

DETAIL_BOX对DUPLICATE：macro 0.0000pp、tail +1.1364pp、head −1.0417pp；三项均未过原+1/+2/−1门。对更便宜的NATIVE_ROOT：macro −.5435pp、tail +2.2727pp、head −3.1250pp，只尾类门通过。对FIXED_CENTER和GLOBAL_BOX的macro/tail增量均为正，但不能用较弱对照替代重复全图及原生单图。

A/B nonTiny18相对DUPLICATE为−2.7778/+2.7778pp，未在两组件拆分上同向。所有tiny仍0。主读出与DUPLICATE均87/175正确，NATIVE_ROOT88/175；不能把头尾之间交换或优于较弱框当成总体成功。

|原冻结门|通过|
|---|---|
|DETAIL_over_DUPLICATE_macro_threshold|False|
|DETAIL_over_DUPLICATE_tail_threshold|False|
|DETAIL_over_DUPLICATE_head_threshold|False|
|DETAIL_over_NATIVE_ROOT_macro_threshold|False|
|DETAIL_over_NATIVE_ROOT_tail_threshold|True|
|DETAIL_over_NATIVE_ROOT_head_threshold|False|
|DETAIL_over_FIXED_CENTER_macro_positive|True|
|DETAIL_over_FIXED_CENTER_tail_positive|True|
|DETAIL_over_GLOBAL_BOX_macro_positive|True|
|DETAIL_over_GLOBAL_BOX_tail_positive|True|
|A_non_tiny_over_DUPLICATE_positive|False|
|B_non_tiny_over_DUPLICATE_positive|True|
|four_joint_visual_and_input_costs_equal|True|

## 实际费用

|读出|视觉编码行|输入tokens|生成tokens|逐调用wall合计秒|
|---|---:|---:|---:|---:|
|NATIVE_ROOT|175|18725|1055|36.2895|
|DUPLICATE|350|30275|1031|38.2636|
|FIXED_CENTER|350|30275|1058|38.4734|
|GLOBAL_BOX|350|30275|1056|38.2905|
|DETAIL_BOX|350|30275|1054|38.3606|

四联合读出同350视觉编码行、30275实际输入tokens、max24greedy，上表仍披露不同生成tokens和实测时间。NATIVE_ROOT更便宜，只作原图损害控制，不假称同计算。顺序按rank旋转，wall只描述本次，非正式吞吐基准。旧R122框的提议和R121SFT均有历史训练/推理费用，复用不等于总体免费。

正常本run885响应/1597TRAIN编码/0新proposal/0VAL/0optimizer，全部费用核对通过。worker wall199.0652/CPU198.7762秒、原始首次准备至退出496.1183秒，原4小时未重置；失败attempt0。GPUpeakallocated7.7688GB/reserved8.0761GB，groupRSSpeak2.5990GB。没有新feature/checkpoint磁盘副本。

## 低成本工程与独立保存验证

native固定原TRAIN0,1的完整5读出矩阵，10响应/22编码；两图均原生HF，无手动multiimagecache。真实grid为每图1x16x16、64visualtokens，两图128；四联合输入tokens实测一致。单图native/manual hidden差0、同[B,1,D]head差0，rawfullsequencehead差.0625保留，.05阈值未放宽。全部adapter及generic参考不可变。

CONTROL_EXIT0后CPU-onlyfinalizer重新核175/88-87、原始框和回退、独立整数计数/指标与13门、345类行、native/费用和封存记录，全部通过，无新增模型调用。SINGLE_PATH_COMPARISON另外保存：R122旧manual ROOT与本轮native单图有12/175预测不一致，正确90→88。另经只读核对模型generation_config.json，native接口继承repetition_penalty=1.05，而旧manual greedy没有该处理；do_sample=False、max24、pad/eos在本轮显式覆盖。因此12个差异不能归为BF16单一原因。R123所有5臂都用同一native配置，结论应采用本轮NATIVE_ROOT，不能偷用旧90作同接口baseline，也未改成penalty1.0重跑救分。详见NATIVE_GENERATION_BINDING。

## 科学含义与停止规则

R122的crop-only DETAIL macro23.3696，本轮global+同框47.2826，说明保留全图与读出接口改变后恢复了大量可读信息；这是联合输入/推理路径差异，不能分离成纯上下文因果效应，也不是新学到的策略。关键控制DUPLICATE也达47.2826，所以没有证据把恢复归为额外取证的macro增量。尾类有相对原生单图的正信号，但头类损伤和缺乏宏平均增益仍违反联合目标。

本轮只排除了这个冻结SFT/固定框/固定读出条件，不否定通用多模态工具RL、MedVR作者结果或所有RL。它没有检验训练后可改变框/表征的策略，也没有证明学习能够补救，不能据此自动发起新SFT/新prompt/更多框。R122原十门失败与R117/R121原正负结果保留，不用新参考隐藏失败。

下一研究不再围绕同一冻结取证仪器做静态救援。尾类奖励可发现性、连续观察与同预算非myopic增量仍未同时建立，因此当前不准入采样RL学习；没有运行中GPU实验或等待用户审阅。持续自主授权保持，可审查明确不同的RL主线问题，但必须给出与闭合R67/R105/R113和本轮不同的机制与可获得反馈，先冻结/实现/真实准入后执行，不为GPU忙制造实验。

源码、原协议、完整聚合、验证与费用公开；组件索引/原框/逐例预测、图像、特征、权重、私有路径与原日志不公开。runtime与后处理交付SHA分开记录。

私有启动配置沿用R122的一些未使用描述字段（旧policies、normalcounts、proposalmax64），实际R123worker未读取它们；真实五臂/885响应/1597编码以运行前公共freeze/config、源码和保存验证为准。原私有文件保留、不追改以掩盖继承，绑定审计明确列出；未调用旧finalizer。
