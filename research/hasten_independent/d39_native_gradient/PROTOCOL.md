# D39：原配对识别与原几何项的投影后梯度关系

R38文本正集合目标完整失败，不扫描其温度/正例权重/子集/对称组合。下一决策先查原HASTEN内部目标是否冲突，避免再凭猜测增加目标。只读R28linear64三域初始/最终各3seed共18封存模型，0优化/图像/骨干编码/VAL清单、特征、响应/TEST。

每个stage固定CPU generator seed=20261024+100*domain_index+stage，从当前stage TRAIN manifest自然行顺序随机置换取前min(64,n)真实特征；NIH按原H-unit单位化，其余缓存原unit。每旧类按封存原mean/cov（已有1e-5I）与独立generator的固定Gaussian一行，不归一化/纠正/重抽样。相同probe供3训练seed，但各seed本身已封存统计可不同。只读当前cache、当前checkpoint和train.csv，不能读未来stage。模型构造的随机初始化随即strictload，不能称整个脚本无RNG；probe显式generator可复现。

按原公式重建paired contrast(.5双向单对角CE)、.1图文entail、.5层级entail/distance/root、.1mapreg。层级real leaf保持原text、旧leaf与parent冻结，更新mutable buffer仅在RAM probe中执行，随后恢复；原网络参数及层级缓冲不变证明。不是原自然batch实际轨迹，也非已发生optimizer更新/因果分解。

比较contrast梯度g_c与其余加权几何项g_g=.1anchor+.5hierarchy+.1mapreg，参数只取原requires_grad项；stage>0按原算法对shared mapping权重梯度投影，其余参数不额外约束。记录投影前后范数、cos、各损失/类分母与finite；不改变任何projection空间/维数，D13 defer保持。

事前可行性：NIH至少2/3seed在初始或最终probe有postprojected cosine<-0.01且||g_g||>=.01||g_c||、两梯度非零且全finite，并且网络/层级恢复/18cell覆盖全部通过，才支持另立固定“原几何梯度去除与原contrast相反分量”的主线实验。它只证明该probe可见冲突，不证明冲突导致VAL错误或过滤一定有效；若不支持，不盲启此训练。不搜索这些阈值/选择probe。

独立600秒原总钟含修复重试，300MB含16MiB余量，CPU2400继承/RSS8GiB childmonitor/file64MiB/GPUreserved<8GiB CPU4；预计<5MB。pro/data ext4容量/写读probe/GPU余量，全部旧资源只读、不复制checkpoint/cache/骨干、不清理历史。中性主子入口、独立guard。公开自有code/protocol/安全聚合与所有正负probe，不公开feature/权重/逐例响应/身份/原日志/私有路径；全部历史开发协议，无TEST/独立确认/临床证据。

元信息澄清：训练配置文件仍保留既有VAL路径/总数等字段，但本worker未访问VAL清单/特征/响应或用这些字段构造probe；不宣称配置文件不含VAL元信息。
