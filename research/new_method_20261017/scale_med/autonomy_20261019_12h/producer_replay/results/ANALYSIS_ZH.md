# 原支持缓存生产者与同像素运行时复演

## 结论

固定64训练图的原APART重放通过，最大误差1.39e-16；同一像素、相同batch32时，Torch2.1与2.6裸ViT的单位方向exact相同。**没有证据将先前norm_geometry的最大.0736方向差归因于缓存损坏或单独的Torch版本差异。先前范数归因门不改，仍未决。**

本轮不是分类性能实验：拟合/分类评分/验证/测试均0，不修改历史cache、模型或科学阈值，也不启动范数方法。

## 原生产链实际绑定

已读取原CONFIG、PROTOCOL_LOCK、extract_fit回执与run_real_query_node.extract代码。原流程ApartFeatures→锁定预训练权重，原seed、transform(False)、batch32、num_workers0、FP32；提取S_old3236、S_new2290，随后别的“teacher”阶段有48步读出/记忆更新，不能把它误认为此前1536特征的编码器训练。extract_fit明确encoder_updates=0、seeded_adapter_pools_untrained=true。原图像根与当前development_images解析相同；当前stage0 bundle由S_old按cache.y/components核对并保持行绑定。

旧环境当前版本Python3.10.4/Torch2.1.2+cu121/torchvision0.16.2，新环境Python3.12.7/Torch2.6.0+cu124/torchvision0.21.0；timm1.0.15和Pillow10.4.0相同，matmulTF32 false、cudnnTF32 true。这是当前可运行环境的实测，不认证过去至今安装文件从未改变。

## 三路径结果

原运行时按固定64条原训练索引解码一次，私有保存像素张量。原APART及旧裸ViT使用它，新裸ViT也读取同一张量；三条路径各64图次，192逻辑特征路径图次。APART内部可能有多个编码器分支，192不能冒称192次单个ViT前向或精确GPU计费。

|比较|最大绝对差|平均单位方向L2|最大单位方向L2|
|---|---:|---:|---:|
|original_APART_vs_cached|1.38777878e-16|3.43606277e-16|6.70559823e-16|
|original_bare_vs_cached_direction|1.08834e-06|1.94885392e-06|5.03883458e-06|
|modern_bare_vs_cached_direction|1.08834e-06|1.94885392e-06|5.03883458e-06|
|modern_vs_original_bare_direction|0|0|0|
|original_APART_vs_original_bare_direction|1.08834e-06|1.94885392e-06|5.03883458e-06|

第一行是1536全向量差，其余为768单位方向；不能混同单位。原缓存接受线max_abs<=1e-5，当前通过；裸ViT两个版本单位方向exact只是这个64图/batch32/相同像素配置，不推广到所有输入或设置。

## 与前一轮矛盾如何保留

norm_geometry用batch16并自行从run.Images读取图像；本轮用原Images解码、保存相同像素、batch32。当前复演控制了两条运行时内部的像素与batch，但尚未控制“前轮独立解码张量”和batch16/32两项差别。虽两份transform源码相同，不能以源码相同替代逐张量绑定。前轮没有保存逐图特征/像素，无法直接原地追加完整对照。

因此本轮决策HISTORICAL_CACHE_REPLAY_PASS_PRIOR_DIFFERENCE_STILL_UNRESOLVED。不覆盖714ccd2，不把接近方向或非常数范数升级为新增语义/质量标签。也不把新运行时的本轮匹配归为版本修复成功，因为新旧都匹配。

## 后续最小工程问题

唯一直接有区分力的下一项是：复用本轮私有同像素和现代batch32输出，在相同现代运行时、同一FP32模型下固定batch16，先确认是否复现原方向异常；另核对run.Images的输入Tensor与已封存像素是否逐元素相等。两项须另封存，不做精度/图像/阈值搜索，不读取旧测试。若仍无法定位，就保持未决，不能靠拟合分类器绕过结构问题。

这仍是工程隔离，不是独立性能确认，也不重启已停止科学路线。新的性能方案必须有独立机制和G/P/C强控制；新旧结果、旧CPU缺失与截止条件不变。

## 完成和成本

三阶段退出0，退休ps/GPU无本次PID已核。原始/现代主子完整中性argv现场已核；聚合阶段太短，现场ps时已退出。活动GPU进程名未在短计算区间捕获，已有LAUNCH/source中性入口，公开这一限制，不为补回执重跑。

串行协调墙钟84.635031秒；workerCPU 66.503144秒，协调CPU 1.286721秒。入口和正文包含于协调，不能再次相加。详costs.json；旧失败CPU缺失，因此总研究CPU仍未知。

单阶段GPU allocated峰值原运行时1017668608、现代585022464bytes，非GPU计费驻留。仅新增私有64图像素与少量特征输出、日志和报告，未复制旧环境/权重/缓存；公开只聚合。全窗保守计费4922183729bytes/20,000,000,000（包含默认cache全额与本地/Git1GB预留），未删除历史资产。
