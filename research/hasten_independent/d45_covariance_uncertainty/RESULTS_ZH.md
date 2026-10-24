# D45：少样本协方差收缩的TRAIN密度诊断支持训练检验

三域初始任务全体TRAIN按各类manifest奇偶两fold互换fit/holdout，54个类×fold统计cell。0优化/神经模型前向/图像编码/VAL/TEST/新随机数；存在TRAIN统计估计，不称0拟合或患者独立。两种协方差使用相同fit类均值，固定lambda=512/(512+max(nfit−1,0))，原1e-5I保持。

|域|fold|全类宏ΔNLL/维|初始尾类ΔNLL/维|
|---|---:|---:|---:|
|NIH|0|-1.903220|-9.442116|
|NIH|1|-2.198640|-10.910775|
|ISIC|0|-2.393175|-9.256484|
|ISIC|1|-1.847531|-7.147953|
|HK|0|-4.513166|-11.582377|
|HK|1|-5.435064|-13.981414|

预设密度门通过：NIH两fold初始尾类每维NLL均至少下降.1nat，三域两fold全类宏NLL均未上升超.01nat。原始高维小样本协方差对未观察方向可能过窄；候选更好的密度预测支持R46实验，但不证明判别准确率提升或Gaussian唯一瓶颈。均值估计误差未解决，等类pooled可能抹平真实异方差。

exit0总4.499009s/86016bytes，原600s/300MB合规，无重试。全部positive/negative类级统计和Mahalanobis/logdet/分母记录保留。短run首次ps时已退出，未捕获liveps，不冒称通过；neutral入口可追溯。

后继R46仅收缩旧类Gaussian回放协方差，原原始mean/cov保存不变、只派生RAM sampler covariance，每stage用当时候选已到达旧类raw cov等类pooled；不纳入未形成统计的新类、不读未来feature。lambda由n,d固定，无强度扫描；不改变均值/样本数/随机消耗shape/损失/保护空间/NCM与融合。共享9初始状态，36新增量stage/12120新步，完整三域BOTH采用门，0步密度收益不当最终训练结论。

全部历史开发集反复使用，无TEST/独立确认/临床证据。安全类级统计/source/protocol/report公开，特征、权重、私有路径、逐例响应和原始日志不公开。
