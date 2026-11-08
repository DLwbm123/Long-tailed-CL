# R114 生成式识别 TARS-RL：最终诊断

**已完成。冻结成功门：未通过。**

单个开发种子74005，静态HK23，TRAIN6821/VAL1698/TEST0；不是持续学习结果或独立患者确认。

|方法|micro %|macro %|head %|tail %|tiny %|无效预测|
|---|---:|---:|---:|---:|---:|---:|
|SFT|80.3298|47.0895|81.0961|9.9913|0.0000|0|
|CE_CONT|79.4464|47.1721|81.1367|10.1199|0.0000|0|
|TARS_CE|78.7986|48.0935|78.6563|14.7522|0.0000|0|
|GRPO|79.8587|47.5682|78.5416|13.7789|0.0000|0|
|CONST_GRPO|79.3875|47.1206|77.6742|13.7895|0.0000|0|
|TARS_GRPO|79.4464|48.3287|76.9005|17.1596|0.0000|0|

TARS_GRPO相对公共SFT：macro +1.2393pp；tail +7.1683pp；head -4.1957pp。

各冻结门逐项：macro_over_SFT_ge_1pp=True；tail_over_SFT_ge_2pp=True；head_over_SFT_ge_minus1pp=False；macro_over_GRPO_positive=True；tail_over_GRPO_positive=True；macro_over_CONST_GRPO_positive=True；tail_over_CONST_GRPO_positive=True；macro_over_TARS_CE_nonnegative=True；tail_over_TARS_CE_positive=True。

同一SFT起点、相同自然查询序列与更新次数。CE控制的生成和反向次数更少，不称计算量匹配。原始TARS、常数强度RL和加权CE分别检查类别分配、总放大尺度与RL增量。

TARS仅放大已有优势，不制造奖励多样性。逐臂常量组、奖励、KL和梯度见FINAL_VERIFICATION.json；权重来自固定SFT的TRAIN一次贪心识别，可能含拟合偏差。

共享Qwen2.5-VL-3B，冻结视觉/基础参数，语言q/v LoRA rank8 alpha16，224分辨率。与原文全语言参数/四H100/三seed不同，不是原论文精确复现。

BF16缓存生成与训练前向存在数值差异；行为概率在相同训练微批路径重新冻结，未更新策略概率比严格1，原始缓存差异保留。称数值近似on-policy，不称比特精确采样概率。

首次实现于正式RL前修复，失败11–12次SFT更新和工程成本保留，原时钟不重置。完整新增成本还须合并attempt1、两图概率诊断及控制器账；未知CPU/GPU kernel费用不补0。

VAL已用于历史开发；tiny分母小，所有类整数计数公开。没有选择最好epoch/seed，不据结果改旧协议。图片、身份、特征、适配器、逐例预测和私有运行路径不公开。

本报告是科学证据收口；自主研究须根据全部对照与奖励多样性选择新假设，不能仅重复监测、改名重跑或把静态收益包装成持续学习成功。

## 增益集中性与头尾竞争

尾类平均增益+7.1683pp集中在两类：ulcerative colitis grade3由2/21提高到12/21，oesophagitis b d由14/41提高到30/41。其余尾类没有正的逐类准确率增量；极少类(tiny,TRAIN<25)在全部六臂均为0。故不能说所有尾类普遍改善或最稀缺类别已被解决。

头类损害主要包括ulcerative colitis grade2由43/66降到28/66、oesophagitis a由15/61降到6/61、dyed resection margins由144/165降到127/165。相邻诊断级别一升一降，与共享判别边界重新分配相容，但仅靠这些计数不能确定唯一机制。逐类贡献见CLASS_ATTRIBUTION.csv；整体macro+1.2393pp不掩盖头类−4.1957pp或tiny零分。

依据这组完整结果，独立R115检验头类TRAIN风险约束的实际Adam更新，比较方向修正与仅缩步；本轮冻结门保持失败，不改阈值、不挑seed/epoch。R115仍是诊断，尚非完整持续学习验证或最终创新方法。
