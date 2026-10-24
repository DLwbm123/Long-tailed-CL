# D41 初始适配器对齐可行性（冻结）
目的：R06只改变增量新adapter，初始任务保持原随机初始化；它不能回答初始独立随机Image/Text仿射变换是否削弱预训练对齐。本诊断不救援R06/R28，不扫描初始化幅度。

三域只用stage0已到达TRAIN缓存；按manifest顺序每初始类前min(64,n)行、全部3seed74002/3/4，NIH遵守H-unit原单位化。每seed全新创建原模型，原随机初始化后，在同一未训练模型上分别计算原配置和固定Image/Text首adapter fc=I,bias=0（原alpha/共享mapper/其余参数保留）。identity赋值不消耗RNG，所有非fc参数exact保持。不使用任何已训练checkpoint、VAL清单/特征/响应或TEST，不更新参数、不做新统计拟合，不把零步结果当训练效果。

每域3seed×2配置=6配置评测，共9配对cell/18配置评测；同时记Euclidean cosine与原负双曲距离逐类命中/完整分母，仅hyper为事前可行性主指标。尾类仅初始任务TRAIN<250，不能推到后续尾类。原pairwise对齐与分类结果不是训练损失轨迹。

可行性固定：NIH hyper宏平均均值至少+1pp，初始尾宏均值至少+1pp，至少2/3seed两项同时改善；ISIC/HK宏平均均值各最多下降1pp。无初始尾类的域只报None，不造分母。只有通过才考虑另立固定初始identity训练协议；不通过停止，不扫Image/Text单侧、alpha、初始矩阵、seed或分类读出。该门不是采用门，训练仍须完整三域原BOTH门。

0优化，预计<10MB新文件；独立600s/300000000bytes含16MiB余量、CPU2400s/RSS8GiB/GPUreserved<8GiB/CPU4。原preflight起点含全部重试；独立guard、中性入口全argv、pro/data ext4写读probe、GPU余量检查、旧资源只读共享。所有失败及完整类/seed正负结果保留，结束按项目交付。无原图/骨干复制或下载，原Spatial暂停/D13暂缓不变。
