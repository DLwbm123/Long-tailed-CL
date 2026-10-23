# 完整类别分数增长：真实GPU资格失败关闭

科学未启动。本轮worker return1，完整自有组已退出，未启动独立科学验收。实际部署源8d39bd11e99095c4dcd06798a93b94d76447e57b。三臂冻结协议保留，但工程准入失败不能解释为方法负性能；不重跑、换参、放宽门或改名复活。

## 已到达证据与缺失

独立24维小模型NumPy目标误差7.21645e-16、三个导数finite-difference最大1.29406e-11。真实20合法基类图完成embedding→FP32 lower10→FP16 token；两增长臂各8+2类两阶段、各两步canonical micro8 AdamW。两臂FP64读出对独立NumPy误差均1.42109e-14；GPU FP32误差2.42394e-6/2.44706e-6，缓存/直接路径误差5.72205e-6。主臂旧upper/head/旧两probe分数差均0，控制旧upper差0、旧head最大变化0.000300020、旧probe分数变化0.145960；两臂新upper变化0.000301361。同保存梯度/同初态独立AdamW逐坐标舍入门四次通过，最大归一化误差0.139026≤1，绝对误差最大1.91009e-6（本轮门为3e-7+1e-6×|预期参数|，不等于旧MIL的跨batchshape实际更新门）。这些均是局部工程证据。

随后完整原生链函数完成20份保存embedding/lower接口的独立运算，在最终`lower10 FP32/NumPy≤5e-5 AND upper2 FP64/NumPy≤2e-6`联合断言失败。**两个误差值均未在assert前落盘，失败分项和根因未知**。这里未兑现本轮“所有断言前落盘”要求，不能用已通过的读出检查推定整链正确，也不能将缺失数值补0。仅源码补写该断言前立即保存两个误差和行数，未再部署或运行，不能恢复本次结果。原FAILURE/log和资格权重私有保留。没有达到预算forecast、bulk或val。

| 臂 | 科学更新 | final/300 | mean | tail4/60 | tail7/105 | initial/150 | later/150 |
|---|---:|---|---|---|---|---|---|
| NATIVE_RIDGE | 0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| FEATURE_FROZEN | 0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 主FULL_SCORE_FROZEN | 0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

主13门、机制4门均NOT_RUN，非0/13性能失败，不用历史Native75代填本轮空白。

## 完整费用、原钟与停止

原first1792778979.601439、deadline1792822179.601439（12h）不变；全组退出1792779142.5113022，实际162.909865s，包括部署/等待/资格，不续钟。资格20次图编码、4次丢弃优化、256模型forward/256backward行；初始原生upper20行、旧upper缓存40行，CPU FP64/NumPy真实读出各4模型行（各含两个expert），完整链保存接口20行。小维Torch forward/back各2行、NumPy目标14行、0优化另计。正式train/val/优化/head均0。读取/缓存和完整链行数中按已到达源码控制流推导的口径见QUALIFICATION_COSTS，不冒称全部逐行落盘。原始COSTS.json是资格前快照，其中丢弃更新0已过时；本报告按之后落盘QUALIFICATION_PROGRESS的4次更新对账，不改写原文件。所有旧失败与源成本仍累计；没有记录的准备费用未知，不补0。

失败退出记录GPU reserved峰996147200B<8GiB，仅证明这次已执行部分；RSS采样峰3419336704B<32GiB。新根+中性入口退役实测569206948B，采样峰569204900B；后者不是最终占用上界。大权重文件实际`max(size,blocks×512)`约为逻辑字节两倍，原bulk物理预计尚未验证，不能声称科学存储资格通过。初始累计窗口11742105959B，末采样12311241063B，不重置30GB/27GB预停/本地100MB约束、不删除任何证据。后续小报告增量计入。

本轮资格关闭，当前没有活动科学实验或已冻结后继。持续NIH主线授权仍有效；后继必须直接服务方法主线、有独立科学依据，不能以反复工程失败为由循环旧诊断/复活本配方，也不能盲目开始新GPU批次。公开仅源码和上述聚合；图像、患者身份、tokens、权重、逐图分数、原log保留私有。源码改进不等于准入已通过。
