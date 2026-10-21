# Fibrosis 首阶段多一张：原因取证结论

结论：**F1驱动额外关闭cuDNN TF32，偏离原Task1评价的运行精度设置，导致一张Fibrosis预测翻转。** 这是F1复放实现中的配置错误；不是可以忽略或放宽的复现误差。本次未训练、未拟合头、未改变门槛，也未重跑F1统计或改写原指标。

## 1. 源码、配置与入口绑定

- 原启动脚本/运行回执绑定41e6545c514b90b2b771d3eb7661cc9042a7977d，q145入口用runpy执行run_module_additions，coordinator通过原q107v解释器启动评价进程，评价日志与此依赖位置相符。
- 原evaluate第514行固定seed和4线程，第522行strict恢复封存model，按seen筛选150张，batch64/worker0/canonical Resize224-BICUBIC-antialias、RGB转换与Normalize(.5,.5)，第548行用原float32头和NumPy读出。该执行路径没有覆盖TF32开关。
- 当前仍可调用的原解释器入口，导入原依赖后默认matmul TF32=false、cuDNN TF32=true；模型初始化后flags仍为false/true。
- F1科学源714684f的run.py第24行、恢复resume与第一次batch定位均显式设置matmul和cuDNN TF32=false。将冻结CLS/patch探针的精度约定误套到了原可训练A的复放路径，strict模型加载和参数不变并不能抵消算子运行设置改变。
- 原checkpoint/seed/order/1536特征、head、标签与val清单绑定相同；原始6/15来自原A Task1报告，不是把另一数据集/CLIP或B的结果当父参照。

## 2. 环境核对

原入口与F1中性venv入口当前观察到同一组版本：Python3.10.4、Torch2.1.2+cu121、CUDA12.1、cuDNN8902、torchvision0.16.2+cu121、timm1.0.15、NumPy1.26.4、Pillow10.4.0、SciPy1.15.3。中性venv路径不同不等于库版本不同；原日志的导入位置也指向相同共享环境。

历史没有完整的进程环境/pip-freeze逐项封存，故不声称所有历史依赖逐字节未变。原因判断依靠下面固定设置复放的精确计数和成对输出证据，而不是仅凭当前默认值推断历史。

## 3. 固定原设置复放：300次诊断编码，无搜索

按预先写入的PRECISION_AUDIT_PLAN，同一原Task1 checkpoint和头、原150张、原batch64/worker0，先原cuDNN TF32=true，再F1覆盖false；matmul TF32始终false。没有新增神经训练、头拟合或其他参数变化。

| 运行设置 | Fibrosis正确数 | 首阶段全部正确数 | 与对应封存回执 |
|---|---:|---:|---|
| 原cuDNN TF32=true | 6/15 | 52/150 | 十类计数全部精确一致 |
| F1覆盖cuDNN TF32=false | 7/15 | 53/150 | 十类计数全部精确一致 |

两次仅一张预测不同，真实类别为Fibrosis。False条件全部150张特征与F1保存query的对应前缀**逐项相同，最大差0**；因此此次切换重现了F1实际使用的特征，而不是只碰巧得到相同正确数。True/false全150张特征最大差约0.002761。具体样本标识、向量、原始分数与margin保留私有，公开聚合不提供逐样本分数。

此前固定batch上下文定位两边都显式cuDNN TF32=false，所以只能说明选定样本在那两个上下文中相同，未触及此次已定位的设置差异；不能把此前结果解释为排除运行精度问题。

## 4. 对原F1结论的处理

原F1状态仍为REPLAY_FAILED_NOT_ACCEPTED，原回执/失败/成本/未验收74/300、mean27.9784%、tail4=13/60均保留，不追认PASS，不改成原设置后的分数。数值头重解PASS与预测管线同协议是不同要求。

由于F1**正式7929张当前类统计的编码也在cuDNN TF32=false下进行**，只把验证开关改回true还不足以修复对照；它会再次混合两种算子设置下的旧/新统计。若后续决定继续F1，须在明确继承原运行设置的新修复记录下重建受影响的后来类统计并完整验收，计入重复编码和原失败成本；无需也不应重训Task1。本任务没有执行该后继动作，也没有选新机制或调参。

## 5. 费用与资源

本次固定取证新验证编码300、训练编码0、神经更新0、头拟合0。加之前正式300和工程128，F1系列总验证编码728，正式与工程分别列；训练编码仍7993。环境只读版本探针耗时另列，所有动作共享原起点/8h截止，未以新名重置。

固定取证worker44.436秒，原起点至全部退出2611.855秒（含前序失败、等待、修复和诊断），退出实际计费767349769字节/3GB，原空间采样峰1428284068字节、进程组RSS采样峰4484501504字节。本次GPU reserved峰1621098496字节；原首次失败未知GPU峰继续保留。所有取证进程已退出，无新训练或后继实验运行。

证据：PRECISION_AUDIT_PLAN.md、precision.py、PRECISION_AUDIT.json、ENVIRONMENT_AUDIT.json、PRECISION_COSTS.json；原F1失败报告REPORT_ZH.md/DECISION.json不被替换。
