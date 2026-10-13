# MPC / DSM 最小同状态机制诊断（前瞻冻结）

只读 base、local_transport 在 ISIC/HK 的四个最终 checkpoint，复用私有训练聚合统计。主方法仍固定 local_transport。模型不变；不重训、不调参、不反馈下一阶段，不读训练影像或 test/reserved，只用已有 development val。全部四个 CPU 准备与真实 GPU 零更新预检通过后统一评价，GPU0/1，失败停止依赖，不自动重跑。4小时从实际启动计算。

固定六个头：MPC（原公式）；REG_ZERO（原 DSM γ，target=0）；IDENTITY（同 γ，target=原解析最优头）；ETF_FULL（原 DSM）；ETF_DISC（同 γ，将同一个 ETF 缩放至原 WC 的范数）；MPC_DSM（原组合）。原 R、competition pairs、η=.4*16/(n+16)、温度16、γ=.1 trace(mean Q+.001R)/1536 全部固定。IDENTITY 是求解一致性检查，不是新增候选；ETF_DISC 是预先指定的尺度诊断，不按 val 选择。只诊断最终状态，不报告新平均增量/遗忘结果，也不把六头当独立训练重复。

拆解 ||W||²=||WC||²+||W11ᵀ/K||²；报告全头、判别部分与公共部分范数及能量占比。MPC 公布类级供体权重、cosine、位移沿供体方向投影，与同一状态的基线混淆及纠正/破坏计数对照。不能从支持量、方向对齐或 val 标签认定真实中心恢复；本轮没有独立留出训练中心估计。供体行只是类对聚合数值，绝不导出均值向量/矩阵。

同一批冻结 encoder 特征同时评价全部头，OFF 必须与已公开控制最终 Acc、全部类 n/recall 精确匹配。原 MPC/DSM/组合 ON 还需与上一轮最终指标精确匹配。GPU预检真实前64张 val，只验有限、所有头 CPU/GPU 分数容差及全模型 state 不变，0 optimizer/0 policy，不输出分类指标。CPU 检查显式稠密二次目标 oracle、同 γ 和 identity 不动、范数正交分解、MPC moment 一致性、封存拒绝与纠正/破坏恒等式。

报告全部总体/旧类/新类/31类 recall、Acc/BA/tail/F1、新类纠正/破坏、旧旧/旧新/新旧/新新混淆。固定 OFF 最强竞争类的几何归一化 pair margin：xᵀ(w_y-w_j)/(||x|| ||w_y-w_j||)，同时保留真实动态 margin 与零点穿越计数；不逐样本导出、不用 GT 路由或 fallback。REG_ZERO→ETF_FULL 区分加正则和目标方向的条件贡献；ETF_FULL→ETF_DISC 是尺度差异，不声称单因素效用因果或优化建议已经有效。

正常4 CPU prepare+4真实GPU预检+4评价，每次进程驻留独立记录；CPU合成、CPU准备、GPU、并行墙钟分开，复用旧控制/AIR费用0。公开匿名数值、源码/协议/正负报告/费用到原 GitHub 分支（显式代理核验），然后 NAS 独立 public_delivery/DELIVERY；模型/影像/逐样本分数/IDs/features/均值/Q/权重供体矩阵/private配置/raw日志均私有。极少 n 类保留计数但不导出连续 margin。单 seed、重复 development val，仅机制诊断，不是官方 ConCM、独立患者或临床/test 证据。
