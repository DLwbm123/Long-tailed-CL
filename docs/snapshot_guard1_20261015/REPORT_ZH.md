# 快照守卫正式轮已启动（非完成结果）

部署科学源 `f4b272de092a2db492126654a8bcceef071fe8e1`。用户“请立即开始实验”授权后新建正式目录，原生产数学/训练/eval/scope与16ad480预检源完全相同。四条真实零更新预检复用已交付df79726d，不重跑或重复收费。生产8项CPU复用aaefd9cc。

北京时间 2026-10-15T21:25:31.720044+08:00 正式launch；截止 2026-10-16T21:25:31.720044+08:00，实际launch起24h，含等待/train/eval不重置。

|GPU|启动轨迹|状态|阶段|已报告更新|
|---|---|---|---|---:|
|0|ISIC Core+PMT+guard|RUNNING|task1 task_start_features|0|
|1|ISIC LT+PMT+guard|RUNNING|task1 task_start_features|0|

HK两条按原队列等待。上表仅一次启动快照，不能当实时进度或训练完成。当前无立即失败，0已闭合GPU operation，两活跃费用未闭合不预估。启动显存free GPU0=40445MiB、GPU1=37250MiB；均超过24576MiB准入。NAS nfs/free/小probe及中性父子ps/NVML已核实。初始NVML显示0MiB不能当峰值或进程故障。

正式四轨迹18stage/1288更新：ISIC每臂468，HK每臂176。全部原数据/seed/order/每臂参数/预算不变；主四格和无约束PMT贡献对照按PROTOCOL冻结。全4条TRAINED及所有stage source/version/order/steps/seal一致后统一development val；不提前部分val、不做val调参、test/reserved封存、RL暂停。原控制只复用，明确配对推理在eval中计费。

新增CPU调度验收PASS：12项非法配置/封存检查，mock8次train/eval、失败阻断后续依赖、无预检重派发。0GPU/0医学图像/ckpt/0optimizer/0valtest；CPU子进程驻留9.429333493秒、user4.972727、system1.196175、peakRSS422.246094MiB。mock不能算科学训练或真实GPU预检，原生产CPU未重跑。

本轮正常仅4train+4eval=8新增GPU驻留。历史闭合393758.130153168秒含已完成预检，后续只加本轮新增一次；CPU守卫和传输嵌于GPU驻留，不叠加。所有失败保留、不重试同轮、不改健康source。关闭全模型精确和全fit最高bank已验收，不代表实际训练医学收益或完整多task成本满足24h；尚无本轮结果。

沿用同一lca半小时监测，正常健康安静。终态active空且父worker退出后一次匿名收集、全正负/贡献/类别/新旧/成本和精度差异报告，并核实GitHub+NAS完整交付。当前只交付启动与协议，不是最终结果。
