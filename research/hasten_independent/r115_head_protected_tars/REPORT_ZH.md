# R115 头类风险约束 TARS-RL：最终诊断

**冻结成功门：未通过。**

|方法|macro %|head %|tail %|tiny %|micro %|
|---|---:|---:|---:|---:|---:|
|SFT|47.0895|81.0961|9.9913|0.0000|80.3298|
|CE_CONT|47.1721|81.1367|10.1199|0.0000|79.4464|
|TARS_CE|48.0935|78.6563|14.7522|0.0000|78.7986|
|GRPO|47.5682|78.5416|13.7789|0.0000|79.8587|
|CONST_GRPO|47.1206|77.6742|13.7895|0.0000|79.3875|
|TARS_GRPO|48.3287|76.9005|17.1596|0.0000|79.4464|
|PROJECTED_TARS|48.9301|77.8498|17.3813|0.0000|79.7998|
|SHRINK_TARS|48.5840|75.7830|18.9123|0.0000|79.1519|

R114六臂是已有固定锚点，非本轮新独立重复。新两臂共同使用额外TRAIN头类反馈，方向修正与仅缩步的反馈/调用次数匹配；生成token长度和最终应用步长可不同。

逐项门：macro_over_SFT_ge_1pp=True；tail_over_SFT_ge_2pp=True；head_over_SFT_ge_minus1pp=False；macro_over_SHRINK_nonnegative=True；tail_over_SHRINK_positive=False。

每臂512次AdamW尝试，实际非零参数更新：PROJECTED_TARS=469；SHRINK_TARS=419。

TRAIN minibatch NLL守卫不等于未见样本准确率或每类保护保证；约束真实Adam位移，保留未投影梯度的优化器矩状态。几何约束借鉴A-GEM，非新颖性或完整持续学习方法声明。

单开发seed74005，静态HK23，VAL反复开发使用，TEST0，患者独立性未知。最小类分母仍很小；全23类整数计数、零步和守卫记录完整公开，不挑epoch/seed。

复用R114基础模型、SFT及缓存，无新bootstrap/VALencoder；复用成本另引R114，不当作本轮新执行。权重/图像/特征/逐例预测/私有路径不公开。
