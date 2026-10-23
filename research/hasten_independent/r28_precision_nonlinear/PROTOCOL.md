# R28：统一蕴含精度下的线性/GELU配对主线实验

## 决策依据与新实验边界

R25原固定GELU配置在NIH seed74002 stage0完成4751步后，总loss有限但图文蕴含anchor参数梯度非有限；contrast及三项hierarchy梯度有限。原公式在该实际64行失败batch的原FP32输入下梯度非有限，改为FP64输入后有限（loss约.021043578→.021049785）。10行hierarchy调用两精度均有限。该证据支持优先修正完整蕴含计算精度；尚未确定函数内第一个源NaN算子，不把该局部结果说成全训练已稳定。

D27两个解析半径全部finite，不能取代实际batch证据。R25始终保留NUMERICAL_FAILURE/INCOMPLETE，无checkpoint或VAL效果，不用此实验改写R25、不重置其原钟。R28是另立的配对精度/结构实验，不继续扫描R25超参数。R19–R24辅助目标与线性读出失败、D26固定方差度量失败均保留。

## 固定两臂

linear64：原H0线性Image/Text adapters；gelu64：只在每个Image_Adapter原Linear后追加exact nn.GELU，无新增参数、宽度或构造RNG，Text保持原Linear。两臂均把原utils.ort.compute_stable_entailment_loss的parent/child/curvature浮点张量提升FP64，调用同一官方函数，不改公式、epsilon、min_radius、softplus beta或loss系数；输出保留FP64，参数与梯度存储仍原dtype。此修复覆盖原真实图文anchor及调用同函数的hierarchy路径，两臂一致，原lorentz FP64兼容修复保留。

两臂分别从各seed初始任务全新训练，不用旧学习初始权重；同一官方pin、相同随机seed74002/3/4、三域exactsplit/order/stages6/3/6、原Biomed单位图文、10epoch batch64 AdamW.001、自然shuffle和原Gaussian统计/每旧类1样本、原pairedcontrast/entail/hierarchy/mapreg全部loss/系数/backward/树/模板/冻结/alpha初始化、原TRAIN_cov nullspace与postAdam实际delta保护完全保持。没有teacher/ridge/head/centroid/10%辅助梯度，没有子空间扩维/学习。原native NCM+hyper L2→softmax→.5/.5推理保持，无融合参数搜索。GELU也改变虚拟parent既有Image_encode初始化路径，不能称只改真实图像或parent初始化完全相同。两臂结构参数key/shape一致，非线性无新参数，但训练轨迹可不同。

90个新stage/seed（每臂45），88,040优化步（每臂44,070），总计算预算约H0两倍，明确披露。两臂由同一campaign按linear64三域→gelu64三域执行，不读取部分结果做选择，所有90stage结束exit0后统一分析。每臂每dataset全部TRAIN头封存/恢复通过后才用已有sameencoder R16 unit VALcache，exactIDs/unitnorm，不二次normalize。共享只读TRAINcache每stage到达才读，不提前未来feature；禁止读取原图/骨干或下载。历史开发集反复使用，无TEST、独立确认或临床证据。

## 校验与门

原finite/replay/oldadapter/fixedparents/actualdelta/checkpoint strictrestore三分支预测exact/zeroVALseal保持。每stage新增audit：entailment_compute_dtype=float64、entailment_fp64_batches==原optimizersteps；image_activation与配置相符，GELU个数为stage+1或0，extra_parameters_from_activation=0。原NCM每seed/stage逐类correct/count与R24 prototype精确一致；每stage步数与H0一致。两臂合计270分支逐类计数检查，全部类/seed和救回损害/尾轨迹保留。

GELU候选必须同时对历史H0和本次同精度linear64满足主门：NIH阶段平均和最终各至少+1pp、TRAIN<250尾至少+2.857142857pp、至少2/3seed最终/尾同时改善；ISIC/HK最终总体与尾均值各最多降1pp。联合改善用原类级计数Fraction精确比较，浮点平手不算改善。两门必须均通过，不选择有利参照或域。linear64为必要精度匹配对照，不主动优化；其相对历史H0变化另列为精度修复效应证据，不冒称非线性收益。失败只拒绝该固定配置，不扫描activation/宽度/初始化/学习率/epoch/seed/精度常数或融合，不事后改门。数值失败保留，不伪装科学gatefalse。

## 资源与交付

pro5000规定/data本地ext4，共享R23assets/官方/TRAIN/VALcache只读不复制，R24安全参考聚合小副本计入预算。每臂目录独立，原数据与旧结果不修改。独立4h从首次preflight执行开始，所有修复/重试共享原钟；8,000,000,000bytes含256MiB余量，预计新增<5GB，CPU72000s继承/RSS8GiB childmonitor/单文件1GiB/GPU0reserved<8GiB/CPU4。独立控制器guard与中性主子argv、挂载容量write-readprobe和显存检查保持。原阶段恢复只跳完整stage，campaign跳完整域；不重训已完成stage，不改原钟。结束后只保存结果上分析，公开自有代码/协议/安全类级全部正负结果，保留私有图像/身份/逐例响应/特征/权重/原始日志。
