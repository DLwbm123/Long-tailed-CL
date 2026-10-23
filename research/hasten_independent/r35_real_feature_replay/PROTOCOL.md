# R35：每类至多8个真实TRAIN特征的增量回放

R32视觉层级叶节点两门失败，停止该固定配置。D33均值探针未显示NIH稳定旧类识别退化/新类竞争；D34同批完整初始TRAIN真实行却显示，NIH最终old-only比初始提升9.13–9.63pp，加入新类候选后下降10.11–10.88pp，ISIC/HK也有候选竞争。这不证明Gaussian回放是唯一原因，但说明均值正确不覆盖真实类内分布。R35测试有限真实特征记忆是否比单高斯回放更有助于主线增量判别；不是R30边界强度/均值替代扫描，不修改融合。

唯一训练机制：每个类第一次到达TRAIN时，在原manifest次序的该类特征行中，取k=min(8,n)个索引floor(j*(n-1)/(k-1))（k=1取0）；不按loss/VAL/图像外观挑选，不新增RNG，不能保证独立患者或覆盖多峰。保存原H-unit训练输入的512维单位特征，旧类每个原Gaussian抽样位置改成该类memory按每stage/每class计数循环的一行。原Gaussian分布建立与原抽样仍执行并丢弃，以保持其原随机数消耗；替换本身不调用RNG。原每旧类每batch1样本、回放总量和naturalshuffle不变，原真实新类输入不变。保留原mean/cov+1e-5I统计供NCM/投影/对照，不做边界纠正/归一化Gaussian/重抽样。

这是明确改变记忆表示与存储口径：额外至多8*C*512*4 bytes真实特征（另有字典/快照开销），最大23类约0.38MB/域，不是原HASTEN等记忆成本，也不能把已共享全部TRAIN缓存的存在冒充免费候选记忆。运行只在当前stage ensure时登记其TRAIN exactIDs，旧memory保留；恢复时按原runner顺序重建已到达stage的memory，不读取未来stage。features私有，不公开。NIH按原H-unit ensure归一化，ISIC/HK原unit缓存，选择后不再变换。

事前共享R28linear64三域3seed共9份stage0 checkpoint及TRAIN回执，旧VAL指标移除/不搬响应，不重训初始任务；31950旧初始步只计共享状态。36新增量stage、12120新优化步，45评测/135分支检查。三域exactsplit/order/stages6/3/6、seeds74002/3/4、Biomed单位图文、原线性Image/Text、10epochbatch64AdamW.001、原contrast/entail/hierarchy/mapreg全部loss/系数/backward/optimizer、原树/提示/parent/alpha/冻结、TRAIN_cov nullspace及postAdam实际delta全部保持。原公式FP64entail修复保留。无GELU/teacher/ridge/head/centroid/10%auxguard/visualleaf，D13 projection扩维/学习继续defer。原nativeNCM+hyper L2softmax.5推理不变。

每dataset所有候选头TRAIN封存strictrestore后才读sameencoderR16unit历史VAL缓存，exactIDs/unitnorm/no二次normalize；末尾stage0恢复重评三分支counts必须与共享R28linear64 exact。NCM每stage/seedcounts exact控制，所有有限性/旧adapter/固定parent/回放分母/投影/actualdelta检查沿用。无原图/骨干复制/下载/TEST/中间选参。

新增audit real_feature_replay：calls==examples==discarded_gaussian_rows==old*steps，by_class[c]==steps，bank_size_by_class每类1..8且包含全部当前已到达类、max_per_class=8，首调用CPU/CUDA RNG exact不变；真实memory行exact选取/不含未来类在synthetic检查。原optimizersteps与R28/H0逐stage相同，原replay计数分母不丢失。每stage保存checkpoint和3分支restore预测exact。

采用必须BOTH：对历史H0和R28linear64均NIH阶段平均/最终各至少+1pp、TRAIN<250尾至少+2.857142857pp、至少2/3seed最终与尾联合改善；ISIC/HK最终总体与尾均值各最多降1pp，joint用Fraction类级计数。失败拒绝固定8-memory循环回放，不扫memory数量/选择规则/抽样数/混合Gaussian比例/seed/融合或放宽门。不选择有利域/参照。

独立2h从首次preflight起含修复/恢复，4GB含128MiB余量，预计新增<2.5GB含9份共享初始副本；CPU36000继承/RSS8GiB childmonitor/file1GiB/GPU0reserved<8GiB CPU4。pro/data本地ext4容量write-readprobe/GPU余量前置，旧assets只读共享、不删除历史。中性主子全argv与GPUprocess核验、独立guard不依赖SSH/心跳。完成exit0三域齐才在savedresponses统一分析，发布所有类/seed/正负安全聚合及自有代码/协议；features/权重/身份/逐例/原日志/私有路径不公开。全部反复历史开发集，无TEST/独立确认/临床证据，不能把TRAIN命中率或生成类一致性当泛化。
