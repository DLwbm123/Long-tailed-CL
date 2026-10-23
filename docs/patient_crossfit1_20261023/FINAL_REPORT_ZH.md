# NIH患者交叉拟合：完整独立验收通过，主方法性能门失败

状态 **COMPLETE_VERIFIED_GATE_FAILED_CLOSED**。主RIDGE_OUT仅 **1/17门** 通过（initial≥44），匹配机制OUT对IN四门 **0/4**。本固定配方关闭，不扫参数、不换主臂、不重复资格或评分。独立验收PASS证明本次规定范围内的计算与记录一致，不证明方法有效。

|臂|最终 /300|最终 %|六阶段均值 %|tail4 /60|tail7 /105|initial /150|later /150|基阶段 /150|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|FROZEN_NATIVE|75|25.000000|30.051146|20|28|34|41|53|
|CE_RC|75|25.000000|35.864638|10|15|47|28|73|
|RIDGE_IN|78|26.000000|30.169092|19|29|36|42|54|
|RIDGE_OUT（主）|71|23.666667|30.130071|9|14|44|27|57|

主对Native：final−4、mean+0.078924pp、tail4−11、later−14；主对CE_RC：final−4、mean−5.734568pp、tail4−1、later−1。**匹配OUT−IN**：final−7、mean−0.039021pp、tail4−10、later−15；只有initial+8，不能支持患者外拟合风险改善后来类迁移的假说。绝对门为final≥91、mean≥36.005291005291%、tail7≥25、initial≥44、later≥41，只有initial通过；相对三个控制的12门全部失败，详见[FINAL_DECISION.json](FINAL_DECISION.json)。不能将CE较高的基阶段73或均值35.864638%替换为主方法成功：它最终仍75、后来类28，弱于Native41。

四臂冻结同预训练ViT12，保留完整CLS+patchmean1536，三学习臂基类各9400次配对更新、36864提示参数、同.05原生响应约束；后期表示永久冻结，只累计真实到达患者/类别等质量充分矩并求全seen FP64解析头。IN拟合/评价同池，OUT双向跨患者池；训练query仍是监督训练数据，不是独立验证。CE与ridge还改变损失和临时头容量，只有IN/OUT是本机制匹配对照。实际执行遵循已发布[FULL_GRAPH_ADDENDUM_ZH.md](FULL_GRAPH_ADDENDUM_ZH.md)：完整耦合计算图、原attention、FP32/micro8，0cache/0MLP重算；原PLAN中早期cache/VJP描述作为历史保留，不改写。

## 独立验收与边界

NumPy独立完整12层：1200 query响应行＋160到达类proof行，24 normal-system/score组；score误差2.72212808e-14、feature误差2.59861577e-14、FP32 proof误差1.06367628e-05、normal残差5.89805982e-16，预测差0。患者交集0、全配对draw独立重建、到达时完整当前患者矩在线独立复算、24头val前封存和计数核对通过。验收0新图像编码、0优化；其NumPy响应计算仍计费，不是0计算、全优化轨迹或历史原图重放。[FINAL_VERIFICATION.json](FINAL_VERIFICATION.json)

worker/verifier均return0，guard终态已落盘，现场核对两个进程组和守卫均无遗留。存储I/O等待期间SSH可达，守卫RPC等待、verifier页等待曾直接观察；现已自行恢复并完成，未重启或重复验收。没有独立等待时长分段，不精确归因全部耗时给NFS或GPU。

## 全部成本与原钟

原first **2026-10-23T11:56:58.659005+08:00**；全组退出 **2026-10-23T22:53:38.970814+08:00**，总 **39400.311810s=10.944531h**，原16h截止前尚余 **18199.688190s**。资格失败、修复、等待、正式训练、验收共用此钟，迁移未重置。正式worker驻留29588.182884s，verifier驻留6462.965653s含内部6453.820387s，均嵌套于总钟，不能再相加。

总物理prefix调用 **68670**＝科学68058＋所有资格312＋val300；prefix不是完整12层body或独立患者。三学习臂正式共28200更新，每臂实际586610行，共1759830完整body前反向行；IN/OUT各18800内dual、合计37600，另24正式解析解。原生teacher60129完整body、四臂映射272232完整body行、FP64部署1200响应行、六阶段score共5400行另计。原生控制0优化是设计，不是0编码。

保留首次梯度断言及修复全链/预算FAIL、checkpoint执行预算FAIL和完整图资格PASS：资格prefix156＋78＋78；所有丢弃优化 **64**＝原12CPU＋10GPU、checkpoint18、完整图资格18、正式worker数学6；CPU合计30/GPU34，不与28200科学更新混计。此前GPU完整峰未保存，仍为未知。各阶段parity/teacher/timed/FP64/NumPy及独立proof开销完整沿用[PREFLIGHT_COSTS.json](PREFLIGHT_COSTS.json)、[EXECUTION_RESULTS.json](EXECUTION_RESULTS.json)、[FULL_GRAPH_QUALIFICATION.json](FULL_GRAPH_QUALIFICATION.json)；不重复全class资格，组合证据范围保持原说明。

正式teacher323.913639s、基阶段三臂训练驻留21219.431516s、部署584.689470s。prefix内部科学105.883764s/val0.470948s不含完整读图解码等待，不能当完整encoder时间。逐stage映射/统计/solve/save及全部计数在[FINAL_COSTS.json](FINAL_COSTS.json)，合并账本[FINAL_COMBINED_COSTS.json](FINAL_COMBINED_COSTS.json)。未知准备/分段不补0。

全实验根＋五入口guard采样峰 **4884378629B**，低于5.4GB预停/6GB硬限；窗口全部根remote末次采样 **11511483152B**，沿原23:15基线，低于27GB预停/30GB累计限，本地此前约32MB加本次小报告/回执未接近100MB余量。RSS采样峰 **26976329728B**<40GiB；正式GPUallocator峰 **6788481024B**<8GiB；资格峰6796869632B另列，不能相加为同时峰。采样不保证连续峰。没有删除历史证据腾额。

## 研究决策

患者交叉拟合在这轮固定配置下未带来联合正向收益，不能继续以同配方调参、加seed或换控制救分，也不声称已定位唯一失败原因。本轮到此关闭；持续NIH主线授权仍有效，后继须围绕方法主线另立直接假说、权限、对照和预算，先工程准入再科学启动，不能给本轮续钟或盲开并行。本次尚无已冻结/已启动的新后继实验。现有唯一半小时入口继续负责后续研究与交付，不新增监测。

300val为反复开发筛查，非独立临床test或同协议SOTA；未读test/Etest、future提前初始化或用val调参。公开范围仅源码与聚合结果/门/成本；身份、图像、token、feature、权重、矩、heads、逐图score、原log与全文保持私有。此前12:07窗口结束和10:07滚动汇总不重复或追改。
