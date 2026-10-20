# 四格前向数值研究：batch大小与梯度上下文

独立工程研究，不是GGA梯度诊断的重试。旧GGA仍为UNRESOLVED_SINGLE_IMAGE_FEATURE_PARITY，其缺失误差仍是NA；本研究不认证其旧阈值，不计算组梯度，不启动保护训练或性能实验。method_success=false。

样本为固定seed74320的32个新训练组，每组一张原PLAN首行图，排除了旧GGA全部64组。类计数6/0/2/24，fold计数15/17。因此没有类1，不能推类1机制或总体风险。没有开发/future/test评分。整个研究只做128逻辑图次、66次前向：同一序列在batch32/batch1 × no_grad/enable_grad四格。102912原bias在四格均requires_grad，其他参数冻结，无backward、autograd.grad、CE、teacher、mapperfit或ridge。

四格先全部FEATURE_SEAL，再计算下列差异；所有数值为描述性工程量，没有事后容差选择，也不等于误差的唯一根因。完整指标、interaction、状态exact检查见aggregate.json。

| 比较 | maxabs | RMS |
|---|---:|---:|
| B32_NG_vs_B32_GRAD | 0 | 0 |
| B32_NG_vs_B1_NG | 0.000844457572675 | 3.11004258809e-05 |
| B32_NG_vs_B1_GRAD | 0.000844457572675 | 3.11004258809e-05 |
| B32_GRAD_vs_B1_NG | 0.000844457572675 | 3.11004258809e-05 |
| B32_GRAD_vs_B1_GRAD | 0.000844457572675 | 3.11004258809e-05 |
| B1_NG_vs_B1_GRAD | 0 | 0 |

| 对原cache | maxabs | RMS |
|---|---:|---:|
| B32_NG | 8.32667268469e-17 | 4.72548096696e-18 |
| B32_GRAD | 8.32667268469e-17 | 4.72548096696e-18 |
| B1_NG | 0.000844457572675 | 3.11004258809e-05 |
| B1_GRAD | 0.000844457572675 | 3.11004258809e-05 |

差分交互残差：maxabs=0.0，RMS=0.0。参数版本、bias及nativeheads均exact不变；四格finite和计数验收通过。

在这32个固定新组和本运行上下文中，梯度记录开关没有可观测特征差异，batch大小变化存在可观测差异；该受控比较未定位内部算子或唯一底层原因，不能概括所有图像、batch布局或FP32实现。batch32与原cache接近不构成所有逐图计算的认证。旧GGA精确失败值仍NA，没有新的类1梯度结果或风险证据。

活动wall44.145038s，wait4 workerCPU43.858089s，协调ownCPU5.275543s，一次并账；worker正文不重复加。RSS3550490624bytes。remote round57983bytes+16MiB<134217728bytes。window含default/localGit外部预留14661927231bytes，另16MiB报告余量<17GB。所有历史资产保留，无像素/模型/环境复制。准备、阅读、网络与报告CPU缺口未量化。

启动父子完整argv中性已核；活动GPU进程名未捕获，完成后不为补该观察重跑。GPU peak allocated见WORKER_RESOURCE_RECEIPT；原始NFS挂载/容量/写读probe及GPU0 admission已核。

四格特征仅worker RAM，没有特征序列化或下载；私人组/行清单仅NFS，公开聚合、源码、协议和成本/退出回执。父子exit、退役和资源见VERIFICATION。整体研究继续，但本结果不提供新的性能候选，也不支持复活旧CE-bias或GGA。
