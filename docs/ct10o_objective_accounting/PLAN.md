# CT10-O：已记录目标函数与解析读出的会计审计

CT9-P已STOP之后独立实施。此为结果已知后的描述性机制审计，不是新的确认性性能实验，不复做CT8梯度比例或CT7预测分歧分析。

问题：训练总损失中CE、pool加权few、assignment、pull和FD的实际数值组成是什么？assignment项是否直接约束被评价的adapter特征？训练神经头与解析分类器W是否共享目标？这些检查帮助区分数值损失下降与最终解析读出改善，不证明性能因果关系。

固定范围：CT6 beta10与CT9 beta1，HK/ISIC×原seed1993/1994/1995，仅Task4×10epoch；120条已有epoch记录、12条轨迹汇总。只读公开聚合日志及绑定源码。不访问图像、fit/val/test/reserved、模型checkpoint、特征或逐样本分数；新增训练、optimizer step、forward、预测、GPU均0。

按原代码逐epoch重建 L=(main_CE+sum_CE+pool_weighted_few)/3+assignment+pull+beta*FD_mean，与记录loss对照。残差容忍1e-4*(1+abs(loss))，超阈值保留BLOCKED及原输入，不试调容忍。保持原每batch等权聚合，不能重解释为样本等权。轨迹汇总epoch1、epoch10及五个损失组成，不进行性能选择、相关性显著性检验、超参搜索或新的bootstrap。检查每个dataset/seed/beta恰10epoch及Task4步数2750/arm。

静态调用路径核对original_backbone的no_grad、PoolAssigner输入、实际优化器参数组、train head输出、FD pointwise normalize、eval probe布局、joint normalization、class-balanced ridge solve。源码事实与运行时未测问题分开；不改loss、学习率、路由、权重、数据或历史结论。

工程先用合成标量记录验证会计公式与故意损坏记录拒绝，再处理锁定输入。新独立分支analysis/ct10o-objective-accounting；代码及输入SHA在执行前锁定。单CPU进程≤300秒、新盘≤1MiB，失败不盲重跑；不新建远程worker。输出120行损失分解、12行汇总、工程/资源审计、源码路径说明、中文报告和NEXT_DECISION=STOP，公开聚合及全部负结果。本研究不启动神经头评价、训练或其他新实验。
