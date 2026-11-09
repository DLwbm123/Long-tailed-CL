# R116 固定参考分布约束 TARS-RL：最终诊断

**冻结成功门：未通过。**

|方法|macro %|head %|tail %|tiny %|micro %|
|---|---:|---:|---:|---:|---:|
|SFT|47.0895|81.0961|9.9913|0.0000|80.3298|
|KL_PROJECTED_TARS|46.6778|81.0717|9.1571|0.0000|80.6243|
|KL_SHRINK_TARS|47.5984|81.0061|11.1537|0.0000|80.6243|
|PLAIN_TARS|47.3044|76.0889|15.9031|0.0000|78.6219|

逐项门：macro_over_SFT_ge_1pp=False；tail_over_SFT_ge_2pp=False；head_over_SFT_ge_minus1pp=True；macro_over_KL_SHRINK_nonnegative=False；tail_over_KL_SHRINK_positive=False；head_over_PLAIN_positive=True；tail_over_PLAIN_nonnegative=False。

两个约束臂每步在相同48个固定TRAIN头类锚点上检查12类相对固定SFT的条件分布KL，预算.001nats；普通TARS控制同为256更新，但没有这部分保护反馈，前向成本更少。

完整词表KL只在参考贪心响应前缀处计算，不是全序列KL或临床/人群准确率保护证明；锚点见于原SFT训练，不是独立holdout。

实际非零参数更新：KL_PROJECTED_TARS=29/256；KL_SHRINK_TARS=13/256；PLAIN_TARS=256/256。零接受步、奖励常量组与最终逐类参考KL见FINAL_VERIFICATION.json。

复用SFT/缓存，不重复计作新训练；本轮256更新是运行前确定的预算，R114/R115的512更新仅作历史背景，不声称同预算比较。

单开发seed74005、静态HK23、反复开发VAL1698、TEST0；全23类整数计数公开，不挑最好epoch/seed，不宣称完整持续学习或最终创新。图像/身份/特征/权重/逐例预测/私有路径不公开。

辅助起点诊断另计56次既有SFT的TRAIN教师强制查询、0图像编码/0优化/0VAL；结果与费用见CANONICAL_DISCOVERABILITY.json，不混入本轮RL效果或主worker计数。
