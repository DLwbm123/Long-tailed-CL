# 实现核验与设计局限

- IMPLEMENTATION_CONFIRMED：原 270 行指标、2,754 行逐类正确数与分母独立复现；来源、映射、最小 original_label tie、seen 候选与任务步数通过。
- IMPLEMENTATION_CONFIRMED：90 个序列化状态逐名核对 core 引用与全部非共享参数；12 个最终和 4 个转移父模型严格恢复。原 .pool. 命名检查包含两个池的嵌套 adapter，但不覆盖 keys/assigner/heads；本轮另外记录这些组。
- IMPLEMENTATION_CONFIRMED：identity、仿射 full Gram/交叉项、类均衡目标、PSD factor 方向与 eta=0 合成验证通过。源码每个 epoch 从 immutable pre-task anchor 映射，任务末提交一次；4 个实际末任务映射另行复算。未发现重复 epoch 输运的证据。
- SPECIFICATION_WEAKNESS：U 没有直接保持旧图像表征的 loss；Risk 是不反传 adapter 的任务末读出。synthetic CE 直接作用于 heads，上游 adapter 的直接梯度为零，这符合原定义。
- SPECIFICATION_WEAKNESS：真实 main/sum CE 为 MEAN，few 与 assignment 为 SUM；保留原 /3、pull 和系数。梯度尺度探针说明实际差异，不把协议要求的 reduction 写成软件 bug。
- SPECIFICATION_WEAKNESS：训练 batchwise 与提取 pointwise 的路由差异已在 CT1 声明；本轮记录 pointwise pre/post 切换，不声称二者同一映射。Risk slack 允许牺牲 margin，optimal 不保证各类识别。
- SUPPORTED_MECHANISM：只能按 Q00/Q10/Q01/Q11、真实/合成与固定转移对照解释，结果见最终报告及全部逐配对表。
- UNRESOLVED：有限末任务转移不等于所有 seed/任务因果确认；不能据结果证明信息不可逆丢失，也不能把离线旧图像重编码当合法无回放方法。
