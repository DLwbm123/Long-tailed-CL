# CT8-A：既有训练梯度审计

独立分支analysis/ct8a-recorded-gradient-balance，参考CT6交付344d87d及CT7交付f83bf4d。CT3–7全部保持STOP。本轮只分析已公开的CT6聚合训练记录，不读取图像、样本级预测、模型或其他私有资产；新增GPU/模型forward/训练/fit/test访问均0。

问题：CT7中FD与冻结F1约97%预测一致，是否伴随FD在adapter参数组的梯度范数占优，以及真实目标梯度在任务后期下降？不能从一致率直接得出因果结论。定范围前已看过HK1993 Task4的前几条梯度记录，发现比例可能较大，因此本分析是事后探索性审计，不是盲确认或新显著性检验。

固定输入：CT6的FEATURE_DISTILLATION_AUDIT.jsonl与TRAIN_EPOCH_METRICS.jsonl，以及run_ct3p.py/run_ct1.py/APART训练源码用于核对定义。全部HK三个seed Task4–11、ISIC三个seed Task4，每任务10epoch，共270epoch、main/few两组540记录。不选seed/任务/epoch。工程门通过后一次完成CPU分析。

原梯度记录每epoch只测第一个batch。real_norm是原完整真实目标（CE、assignment、pull等）在该adapter组的梯度范数；FD_norm为未乘10的FD梯度，weighted_ratio=10*FD_norm/real_norm。它不是CE梯度比例，也不是所有参数的梯度。审计每项公式与已有ratio/cosine，自检二维解析向量的范数、抵消、零分母。零分母保留null并计数，不用epsilon制造有限比值。

固定描述指标：real/加权FD范数、比值、cosine、向量和范数及其与real方向cosine；FD范数大于real的比例、两梯度反向比例、合成梯度与real方向相反比例、空/零计数。先按epoch1与epoch2–10分组，输出8个dataset×main/few×phase总体汇总、24个seed汇总、108个task汇总、540原记录派生行。每格等权汇总中位数/min/max；不把重复batch视为独立样本做显著性检验。额外保留270行分项loss派生值，但不把loss数值比例当作梯度比例。

首epoch的FD=0可能来自师生初始化一致，必须单独报告，不能拿它稀释后续梯度占优。方向以原记录cosine为准；参数梯度不等于含优化器状态的实际更新方向。小real梯度可造成巨大比值，须同时报告绝对范数。记录仅一个batch/epoch，不能外推全部batch、临床泛化或beta最优值。

先锁计划、两份聚合输入hash和源码；检测270唯一完整epoch、5570累计steps、有限loss、固定教师、ratio公式与矩阵；工程CPU自检。单CPU worker≤30秒、输出≤1MiB，GPU0；不下载资产、不删除历史数据、不启动旧worker，不调beta，不扩大训练。不读任何test/reserved或私有路径。

交付锁、工程/完整性检查、预定540/8/24/108/270行、资源和中文结论，公开独立分支并匿名核验。NEXT_DECISION=STOP。后续若设计因果干预必须另立有限固定计划，不因本分析自动搜索或宣称蒸馏过强。
