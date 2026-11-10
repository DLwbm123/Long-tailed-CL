# 工程恢复记录：跨设备断言

第一次真实native通过：五条优化路径非零且全部丢弃还原，匹配hidden/head差0；raw fullhead .0625及采样TF .0759369保留。完整冻结forecast11433.555秒含1.5margin和1800reserve，小于当时78127.786秒，未放宽阈值。

随后固定探针在actor已恢复后，anchor恢复断言将CUDA张量与CPU保存快照直接torch.equal，触发RuntimeError。不是数值坏、GPU坏或机制负结果。第一次退出1，0科学训练更新/0科学VAL；实际native96+探针64=160响应，4TRAIN工程编码、5native optimizer、1探针counterfactual optimizer（已丢弃）全部计费。wholeworkerCPU73.672867秒、原含准备4715.715025秒。失败源码、配置、日志、COSTS与所有产物封存，不覆盖、删除或从账本扣除。失败探针尚未保存token级收据，不能事后补造该次原始响应或梯度结论。

只修恢复检查统一比较设备，并将重试探针原始收据提前保存；不改数据、随机流、loss、系数、更新数、五臂或门。新attempt沿用同root/原prep/23h截止/完整累计guard，不重跑通过且不受断言修复影响的native；新进程重新加载相同prefix并核actor/anchor，复核原完整forecast仍小于当前remaining，再从固定探针开始。失败64探针响应及counterfactual步不免费，成功计划另生成18584响应，全attempt预期18744响应、160科学+5native+2probe=167 optimizer尝试。不能把重复探针当独立确认或挑采样结果；仍原seed/首batch。

此记录只解释必要工程修复，不称科学启动或效能已知。runtime修复提交与原native提交分别记录。首次CPU软件5.224675秒仍只计一次；shell/SSH/Git准备CPU未知非0。
