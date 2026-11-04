# R102 HK新任务流：完整试跑结果

新任务流下的教师条件机制试跑，非严格有限记忆 LT-CIL。无新算法、超参搜索或旧教师复用。

## 任务分布

|stream|T1|T2|T3|T4|T5|
|---|---:|---:|---:|---:|---:|
|Ordered|3331|2400|892|165|33|
|Stratified order0|1563|1468|1505|1223|1062|

## 方法结果

|stream|mode|五阶段mean macro|final macro|final tail|mean tail有效阶段|
|---|---|---:|---:|---:|---:|
|HK_Ordered_NoLargeBase_5Tasks|T|83.019027|60.599060|36.009421|3|
|HK_Ordered_NoLargeBase_5Tasks|A0|83.019027|60.599060|36.009421|3|
|HK_Ordered_NoLargeBase_5Tasks|UNIFORM|81.512514|56.473342|28.433040|3|
|HK_Ordered_NoLargeBase_5Tasks|CE_WEIGHT|81.605845|59.191016|33.638667|3|
|HK_Ordered_NoLargeBase_5Tasks|RLPO_STYLE|81.632708|59.317742|34.726198|3|
|HK_Ordered_NoLargeBase_5Tasks|PROTO_GRPO|81.126344|57.450889|30.868105|3|
|HK_StratifiedShuffled_NoLargeBase_order0|T|75.459511|63.618739|43.265783|5|
|HK_StratifiedShuffled_NoLargeBase_order0|A0|75.459511|63.618739|43.265783|5|
|HK_StratifiedShuffled_NoLargeBase_order0|UNIFORM|74.400207|60.803897|38.519884|5|
|HK_StratifiedShuffled_NoLargeBase_order0|CE_WEIGHT|74.975776|62.402792|40.974266|5|
|HK_StratifiedShuffled_NoLargeBase_order0|RLPO_STYLE|74.588449|60.863353|38.480529|5|
|HK_StratifiedShuffled_NoLargeBase_order0|PROTO_GRPO|74.969388|61.112341|38.702259|5|

两流各5新教师和20最终学习端点，共10教师/40学习最终端点；40中间和40stage起点另计。所有teacher只用该流当前已到达TRAIN，从同一通用Biomed重新训练；LoRA携带，优化器每stage重置，10epochs配方。方法模块每stage512cycles且参数/优化器跨stage携带。全部40学习端点封存后才F/VAL，TEST0。

Ordered前两阶段没有已到达尾类，tail为null，均值分母3；混合流tail有效阶段5，不能把二者tail平均当作同口径顺序因果效应。tiny按原始标签绑定，逐stage correct/count和legacy/new索引见CLASS_STAGE.csv；单例重复stage不代表独立患者。新五阶段均值不可减旧六阶段66.84作为增益。

每条流的RESULT_SUMMARY包括PROTO_GRPO对同流T/UNIFORM/CE_WEIGHT/RLPO_STYLE两两差值；CLASS/GROUP_STAGE和RESCUE_HARM报告新增尾类学习、旧尾类保持、所有正负类别。F是历史开发且教师见过的TRAIN角色，非独立测试；训练B奖励提升不自动代表F/VAL改善。没有事后新成功门、最好epoch/arm/order选择、重复seed或显著性主张。

成本分为新通用前缀、教师训练/观察、新阶段特征提取、模块native/训练及评估。TEACHER_RECORDS逐stage列CPU/wall和更新/暴露；各stream COSTS含所有worker子项，不能重复加。root CONTROL_EXIT和COSTS_COMPLETE保留完整wait4 CPU/RSS/原clock/磁盘；GPU设备wall与kernel时间不同，未采样kernel时间不虚构GPU算时收益。准备/SSH/文档/发布CPU非0未全采样。大文件data盘保存，身份/图像/特征/逐例/weights/optim/rawlogs不公开。

本轮完整试跑后停止，不开新算法、其他order/seed/dataset。按用户最新要求本会话唯一30分钟监测只负责本轮故障/完成和交付，交付后停用。

## 结论及主方法配对差值

主方法PROTO_GRPO在两条新任务流的平均宏/尾与最终宏/尾均低于同流强监督教师T。四个学习臂最终宏/尾也均未超过T。因此本轮不支持“换任务流后GRPO超过强监督”；这是当前固定动作/奖励/实现的负结果，不能外推所有RL无效。

|流|参照|平均宏差pp|平均尾差pp|最终宏差pp|最终尾差pp|
|---|---|---:|---:|---:|---:|
|a|T|-1.892683|-7.850274|-3.148171|-5.141317|
|a|UNIFORM|-0.386170|-2.721109|+0.977547|+2.435065|
|a|CE_WEIGHT|-0.479501|-1.874796|-1.740127|-2.770563|
|a|RLPO_STYLE|-0.506364|-1.781214|-1.866853|-3.858093|
|b|T|-0.490123|-1.330314|-2.506399|-4.563524|
|b|UNIFORM|+0.569180|+1.162169|+0.308444|+0.182375|
|b|CE_WEIGHT|-0.006388|+0.116398|-1.290452|-2.272007|
|b|RLPO_STYLE|+0.380938|+0.667394|+0.248988|+0.221729|

a=Ordered；b=Stratified order0。分层流GRPO平均宏接近CE（−0.006388pp）、平均尾小正（+0.116398pp），但最终宏/尾均负；不能据此挑指标称成功。Ordered GRPO对UNIFORM最终宏/尾正，但平均宏/尾负；分层流对UNIFORM平均/最终均正。这些局部正值完整保留，不等于超过强监督。

tiny原legacy1/2/12在最终VAL：Ordered全部六种读出均为0/1、0/1、0/2；分层流均为0/1、0/1、1/2。不存在最终tiny净改善。分母很小，不能作普遍病例修复结论。

## 各阶段主方法相对同流教师

|流|stage|macro差pp|tail差pp|
|---|---:|---:|---:|
|a|0|+0.130719|N/A|
|a|1|-1.028676|N/A|
|a|2|-2.484170|-10.389610|
|a|3|-2.933116|-8.019895|
|a|4|-3.148171|-5.141317|
|b|0|+0.121212|+0.000000|
|b|1|+0.798732|-1.696429|
|b|2|+2.079947|+5.289116|
|b|3|-2.944110|-5.680733|
|b|4|-2.506399|-4.563524|

## 实测费用闭合

正式attempt2两worker CPU合计4863.432866s，含失败attempt1后五阶段计划全部worker CPU合计4887.636533s；attempt2 controller CPU 126.511132s，attempt1 controller另1.179189268s。两流并行正式controller wall 2394.648729s；从原FIRST到科学退出4100.496419s（约68.34分钟），不重置。首轮失败费用/导出修复见ENGINEERING_HISTORY。

教师正式更新2100次（Ordered1200/分层900），模块35840次；工程64次另记，不计正式科学步。峰采样combined RSS 14846033920B，科学结案allocated disk 8623788032B，两个worker峰GPUreserved各2126512128B。完整wait4/core/native/教师分段及暴露在COSTS_COMPLETE及a/b成本文件，不重复相加。未测GPUkernel时间；准备/绑定/SSH/文档/发布CPU非0未全采样，不声称这些免费。

完整逐stage/逐类/旧新头尾/救援与损害/贡献表均发布；私有样本身份、图像、逐例预测、特征、模型/优化器和原始日志留在授权data盘。单seed且只训练order0，任务流差异不能视为独立重复或单一顺序因果效应。

## TRAIN反馈与F/VAL迁移（固定最终cycle）

B reward为训练效用，非准确率；F为教师可见的TRAIN开发角色，不能当独立泛化证据。下表F/VAL差以各stage同流T为参照，均为百分点。

|流|stage|B deterministic reward|F macro差|F tail差|VAL macro差|VAL tail差|
|---|---:|---:|---:|---:|---:|---:|
|a|0|0.005270454319684082|+0.122699|N/A|+0.130719|N/A|
|a|1|0.028021413322845898|-1.093174|N/A|-1.028676|N/A|
|a|2|0.10544555052140078|-4.272891|-10.763889|-2.484170|-10.389610|
|a|3|0.15002814951118268|-5.966916|-10.119048|-2.933116|-8.019895|
|a|4|0.1281405409233108|-4.208807|-6.277056|-3.148171|-5.141317|
|b|0|0.0042860395168145705|-0.122699|+0.000000|+0.121212|+0.000000|
|b|1|0.14158276282794818|-5.103329|-13.541667|+0.798732|-1.696429|
|b|2|0.1330226281072752|+4.614880|+11.649660|+2.079947|+5.289116|
|b|3|0.12807667857543725|-6.778468|-12.764550|-2.944110|-5.680733|
|b|4|0.15300115370189063|-12.695912|-24.878247|-2.506399|-4.563524|
