# D34：真实TRAIN分布与均值诊断的差别

D33的NIH类均值在旧类集合内命中率随增量提高，新类竞争只1/3seed达到可见判据；不能把此当完整TRAIN分布。D34用同一批初始任务全部真实TRAIN缓存行，而不是均值或挑选例子，检查初始模型与最终模型在旧类集合和全已见类集合的原hyper命中率，同时列原raw NCM命中率。仅用于定位下一主线需要改善真实类内分布、旧类映射还是竞争；不改变任何训练或候选推理，不据此扫描融合/层级leaf/teacher系数。

只读R28linear64三域3seed的stage0/final共18封存网络、各域stage0原TRAIN缓存（NIH按原H-unit ensure路径L2归一化；ISIC/HK缓存已单位化）、当前模型所在stage文本bank和对应train.csv标签；不读VAL清单/元数据/特征/响应/TEST，不读未来stage缓存，不编码/下载图像或骨干，不拟合统计或优化。逐类重新求本固定初始TRAIN均值并与checkpoint旧fc行核对预处理一致；不作为新拟合或训练输入。原图文线性、FP64几何和scaled负双曲距离原公式，NCM为原unit cosine。保留每类全部TRAIN分母及所有seed的初始/最终正确数，无逐例输出。初始类数以实际冻结task_sizes为准，不预设这些行独立患者。原模型构造后strictload；前后参数exact不变，allfinite/GPU限制核验。

这只是训练内诊断，已被训练使用的数据不能当独立泛化/测试或因果百分比分解；均值正确与真实行正确分开。输出全部正负证据，不用部分结果选择图像或更改协议。若与D33均值结论不同，只说明均值探针覆盖不足，不自动证明唯一失败原因。

独立600秒从preflight起，retry共享原钟；300MB含16MiB余量，CPU2400继承/RSS8GiB/file64MiB/GPUreserved<8GiB CPU4，预计新增<5MB。pro/data ext4容量/write-readprobe/GPUfree前置核查，只读共享历史，不复制checkpoint/cache/模型，不清理依赖。中性主子argv、独立guard。公开源码/协议/安全全类计数与中文分析，不公开身份/features/权重/逐例响应/原日志/私有路径。
