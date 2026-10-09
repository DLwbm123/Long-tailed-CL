# EDGE-BUDGET1 启动回执（历史）

更新：12/12条已于2026-10-09 13:32:21全部完成。见[结果表](results/REPORT_ZH.md)、[完成分析](results/ANALYSIS_ZH.md)与[审计](results/AUDIT.json)。两个医学初筛均未通过，本配置停止新增训练；以下保留原启动记录。

正式启动：2026-10-09 10:55:53，最近启动检查：2026-10-09 10:57:58，均为北京时间。状态为RUNNING_NOT_RESULTS，尚无正式效果结论。

|GPU|数据|当前臂|后续固定队列|
|---|---|---|---|
|0|ISIC|static_pc|coarse_rl → edge_rl → edge_search|
|1|Hyper-Kvasir|static_pc|coarse_rl → edge_rl → edge_search|
|2|CIFAR-100-LT|static_pc|coarse_rl → edge_rl → edge_search|

三条协调器及worker均存在且命令行中性；三个worker进入首任务首epoch训练阶段，未见启动错误。STATUS只在阶段边界更新，steps0为进入阶段前的快照，不代表已卡死；没有声称完成任何正式轨迹。

三个原生预检均PASS，各8次策略更新、0次适配器更新，峰值分配约11.86/12.07/13.97GiB。预检提议均未通过正收益执行门，执行static_pc；这是保留的激活诊断，不是方法成功或科学失败重试的依据。预检总suite驻留353.1475372314453秒，正式成本待各suite封口后计入。

科学源冻结于b57270fc49292245de05548e4cc916b1c6076f24。新增结果汇总器仅做读出，未修改已启动训练源码。每小时监测已继续绑定本聊天并指向新轮次；原截止2026-10-11 00:14:23不变。全部12格完成后汇总交付，不根据中间结果改协议。

之前两条proposal_execution诊断均自然完成，未被终止或覆盖；其结果与中文分析已交付于a23e5f87acbbdfe6e545c5c6a3648756af0ca2b3。首两轮驻留50760.155485622585秒持续累计。
