# OPSA 内部适配实验：训练前回放检查未通过

结论：UNRESOLVED_OPSA_INPUT_REPLAY，scientific_complete=false，method_success=false。这次尚未训练 OPSA，不能解释为方法性能负结果。

协议/源码 6d4ce783302137c96b71400c57e75ae0ee619c95 在首次实际执行前已推送并核对远端及匿名协议访问。模型使用既有预训练骨干；固定最后一个 Transformer block 输入的内部适配器和 F/H/L/N/M 对照，具体训练与风险门见 PROTOCOL.md。

首个 ISIC producer 的32张支持图，未安装任何新适配器时，回放特征相对原缓存 maxabs=0.00024188394763818843（门1e-6）；原F分数 maxabs=0.0005049140215749037（门1e-5）。断言 F_SUPPORT_REPLAY_FAILED 后退出。两个观察值已在断言前落盘。

只执行1次逻辑 encoder wrapper 调用、32逻辑图次；wrapper含原有双流，不把它解释为仅1个内部算子调用。0训练backward、0optimizer step、0ridge solve、0mapper fit，0query图像读取。合成CPU内部hook、冻结骨干、组权重、零增益门和调度检查通过；训练/全模型封存/回读/query风险检查均未到达，不能称为通过。启动阶段的metadata和原F封存文件含已知query信息，未用于训练或评分。

当前未定位唯一原因。新入口从已授权原图执行canonical变换，显式关闭TF32；旧缓存生产路径和数值上下文可能不同，但这些只是待区分因素，不能断言缓存损坏或某一因素已解释误差。保留原门，不改源放宽容差、不自动重跑或更名续跑。既有GGA失败及其他负端点不受本轮改变。

未触发时间/CPU/RSS/存储守卫。预算为21600s、CPU36000s、RSS8GiB、file64MiB、新增512MiB含32MiB余量；没有取消窗口17/20GB或用删除历史腾空间。父子退出并退休，运行失败已交付不等于科学完成。

成本（一次计入）：
{
  "activity_wall_seconds": 36.296040534973145,
  "known_worker_wait4_CPU_seconds": 34.011549,
  "coordinator_CPU_seconds": 2.641336,
  "peak_RSS_bytes": 1567694848,
  "round_remote_bytes": 225356,
  "window_remote_bytes": 14169777166,
  "unmetered": "preparation/reading/network/report CPU unknown; worker body not counted twice"
}

累计：
{
  "activity_wall_seconds": 32338.309566081356,
  "known_worker_wait4_CPU_seconds": 40698.690961999986,
  "coordinator_CPU_seconds": 1271.0587139999998
}

源码、协议、聚合工程观测、成本和失败公开。原图、身份、采样逐例、私有日志、模型和第三方材料未公开。
