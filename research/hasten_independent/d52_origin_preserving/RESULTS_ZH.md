# D52：原点保持适配器的真实TRAIN学习pilot未通过

固定结构删除Image与Text适配器的全部bias参数，保留输入相关线性变换、alpha、全部原生损失与优化/投影/融合规则。原native模块完整初始化后删除bias，所有剩余初始核心tensor与control exact且无额外RNG，zero input→zero output exact，每步bias缺失检查通过；每模型减少1024参数。候选初始输出与control有意不同，不冒称输出parity。不同于R42一次性初始化、D43模块数平均、R17读出中心化、D51新增类向量。

机制依据有限：D13 TRAIN未中心化矩阵有强公共成分，不能证明来自bias；新pilot直接测试去掉可学习公共平移是否有益。几何原点与半径可能承载有效层级，删bias也可能伤害，不能以谱现象作因果结论。

|域|seed|两fold平均最终宏BA变化pp|两fold平均尾BA变化pp|
|---|---:|---:|---:|
|NIH|74002|+0.260955|+1.695402|
|NIH|74003|+2.079741|+2.586207|
|NIH|74004|-0.916128|-4.971264|
|ISIC|74002|-0.023609|-11.813187|
|ISIC|74003|+0.907881|-0.274725|
|ISIC|74004|+3.642256|-0.274725|
|HK|74002|+1.080999|-0.314403|
|HK|74003|+0.753365|+1.489999|
|HK|74004|+1.502404|+0.156250|

NIH仅74003满足事前宏/尾各+1pp，1/3不足2/3；ISIC尾三seed均下降，平均约−4.12pp，跨域保护门false。HK有局部正向变化，不能择域采用。拒绝固定原点保持结构，不启动完整训练，不扫单侧bias/幅度/权重或alpha初始化/epoch/seed/融合，不放宽原门。

三域3seed双fold双arm共36个全新fit-only模型、552实际optimizer steps，原stage0每类奇偶互换fit/holdout各最多64，2epoch同batch/optimizer/native全部目标，未加载learned checkpoint，所有类统计与父层级初始化仅fit行。候选与control原NCM全部counts exact；原型/hyper/.5final完整计数、安全审计均保留。比较对象是同次小样本native control，不能当作H0/R28完整增量效果。

600s/300MB预算内attempt1 exit0，总44.449001s/544768bytes，峰值RSS2555822080bytes；没有重置钟或删除历史。已捕获中性完整controller/campaign/HK worker ps及GPU中性python，非所有域worker完整liveps。全部PID退休。

仅初始阶段、小fit和2epoch有范围限制，不否定所有受约束适配器。无VAL/TEST/独立患者确认/临床结论。全部源代码、事前协议、全类/seed三分支正负结果公开；模型、图像/身份、特征、逐例响应、私有配置与原始日志不公开。
