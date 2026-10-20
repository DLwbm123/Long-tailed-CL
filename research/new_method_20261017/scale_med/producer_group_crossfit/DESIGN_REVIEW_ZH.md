# 整条监督链的query排除核查

本轮诊断的最小单位是已有已知group，不重新从图像名推断患者身份。只用initial原训练2555，原681开发不入任何本轮评分/选择。采用互补两半：每行只有一个seen producer、一个out producer；没有用同一个已见全训练集的encoder假装OOF。

|环节|是否允许query信息|实现边界|
|---|---|---|
|组划分|标签只用于一次按类分层|geometry.split，固定74290，图随组；不按结果换划分|
|原始编码器起点|无本任务监督更新的共同预训练权重|setup只加载原权重；旧learned checkpoints/全2555 teacher被访问守卫拒绝|
|support组权、类别计数|不允许|每个producer的support子集独立weights|
|mapper均值/全局尺度|不允许|teacher只接收g[support]，两级norm均在该子集拟合|
|初始ridge头/截距|不允许|support-only固定alpha.01，两解均封存，无开发网格|
|训练batch、CE、optimizer|不允许|sampler仅接收support局部索引，并断言映射到全索引后与query互斥|
|方向/类别支持或正则|无此学习环节|不估保护方向、不引入SADP权重|
|更新结束/checkpoint选择|不允许|固定800终点；中间仅恢复点，无query早停或模型选择|
|query前向|仅全部encoder封存后的评估|EVALUATE一次编码每producer的2555行，包含seen及out；无梯度|
|风险汇总与bootstrap|允许且仅作诊断|全部MODEL_SEAL后计算，两个预设contrast，条件于两个producer|
|未来/test/原始图像|始终禁止|只复用canonical pixels0；runtime审计钩子拒绝其他资产|

物理容器边界：共享inputs0及像素chunk可能含681开发行，不能声称文件级完全隔离。本轮加载立即过滤valid；过滤后的y/g及图像才传给划分/训练/编码/评分，不计算任何681结果。两个producer均可读取完整训练metadata用于已封存split，但teacher及更新的数据流仅support。synthetic query标签/特征置毒检查的是划分封存后support输入不变，不把其泛称对整个网络的因果泄漏证明。

基准依据是原特征生产链审计的零本任务encoder更新和现有batch32缓存回放；不认证预训练语料无重叠。两个新producer会从相同原初始化各自重置；不能用旧C/P、旧D或全2555 teacher热启动。历史研究已看过这些组，因此新内部query仅是本次producer未用组，不是从未被研究使用的独立临床测试。

强对照F也可能有读出层的in-sample偏差，所以主contrast要从T的seen-out差减去F的seen-out差。两模型来自互补训练组，样本、头/mapper与训练质量差异仍可能影响响应；该差分不能直接命名为encoder纯因果效应。raw logit margin不跨头尺度推断表示优劣。

这落实Pro的“同头/refit/固定头必要性要分开”和cross-fitted remapping论文§3.3/附录C的整个训练算法排除fold原则；不是论文的GLMC或remapper复现。SADP的中间层AGOP实现未唯一定位，因此本轮不引入其保护项。原文来源及阅读范围见PROTOCOL.md。
