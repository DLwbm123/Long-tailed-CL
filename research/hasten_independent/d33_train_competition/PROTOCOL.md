# D33：旧类TRAIN均值漂移与新类竞争诊断

R32视觉层级leaf未通过两套门，NIH最终下降而ISIC/HK有局部收益；拒绝固定visual-leaf机制，不扫其均值归一化、混合、parent或loss系数。下一步先诊断增量识别变化发生在哪里，避免盲加约束。此诊断不是R32救援，不改变任何已完成实验。

只读R28linear64封存的三域全部3seed、36个增量边界。每个边界用当时旧类原始TRAIN均值，分别在前一阶段模型的old-only文本、当前模型的old-only文本、当前模型的all-seen文本上计算原scaled负双曲距离。保留逐类正确/错误与own-minus-best-other margin；第三组对第二组只改变候选类别集合。前后模型可能同时改变多个训练因素，不能将它们差异宣称因果分解。该集合限制也不是候选推理方案，不用于VAL/融合。

每次仅读取当前与上一stage checkpoint及当前stage TRAIN缓存的文本bank；不读未来stage特征、图像、VAL清单/特征/响应或TEST，无优化/骨干编码/新统计拟合/随机Gaussian采样。所有原始类均值及网络权重不变，使用保存的raw fc均值不归一化；原模型Image/Text线性、原公式FP64几何。当前checkpoint中的旧fc行必须与上一阶段exact相同；strict网络恢复，无新learned initial。模型构造会使用随机初始化但随即strictload，诊断输出不参与优化，不能称RNG完全不调用。

统计每个域/seed跨所有边界旧类行等权的before、after-old-only、after-all-seen TRAIN mean命中率、两个差值及新类取代旧正确行的计数，保留所有class/stage分母。类均值可能掩盖真实TRAIN多峰、极少样本噪声，不能当真实图像准确率或泛化。事前后继判据：只有NIH至少2/3seed在after-old-only相对before下降>=1pp时才把映射漂移作为后续优先诊断；至少2/3seed在after-all相对after-old-only下降>=1pp时才把新类竞争作为后续优先诊断。两者可并存或均不支持；判据仅诊断可见性，不是科学效果门，也不自动授权扫描参数。若不支持，明确记录未知，不盲启动训练。

独立600秒总原钟从preflight起，所有retry共享；300MB含16MiB余量，CPU2400s继承/RSS8GiB/file64MiB/GPUreserved<8GiB，CPU4。pro规定/data本地ext4与写读probe/GPU余量核验；只读共享旧资源，不复制checkpoint/cache/模型、不清理历史。中性control/worker入口，独立guard及ps/GPUprocess检查。公开自有源码、协议、安全class计数/数值摘要及正负结果，私有feature/权重/原始日志不公开。所有历史开发集，无TEST或独立确认。
