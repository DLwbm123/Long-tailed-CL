# 稳定接口下的序列谱球面记忆：前瞻CPU协议

前轮6f92929通过冻结两高维8批主门，但Gaussian控制同轮也两高维8/8，small主候选只有5/8；不能把跨轮改善归因固定半径。下一阶段验证类增长时接口和各stage近似误差，不改分布强度、容量、步数或门。

主候选block_spherical，保留random、原moment48、Gaussian同谱对照。每类16d+16 float64，15谱槽和尾方差，每次16点8对，不增加总采样数。当前任务内baseline48 Adam lr.02；current谱拟合与Gaussian/candidate共享。初始化按seed+1000+class，顺序压缩改变原pooled目标聚合，不声称复现静态轨迹。样本全为原seed263/264/265非高斯unit合成fixture，公开比例非医学fit。

small序列2+1+1，ISIC4+2+2，HK13+2+2+2+2+2。构造函数只收当前任务支持与标签范围，不收旧memory/oracle，范围非法拒绝。旧memory每张量append前后精确；每类噪声seed=base+100000*class，添加新类不改变旧生成点。合成旧支持仅由独立evaluation oracle访问，不授权未来读取医学旧图。部署不保留support/IDs，但低样本统计可包含原支持信息，当前仅合成。

每stage检查四个固定随机query的旧loss精确不变、旧生成点精确、固定输入缩放ETF旧旧logit差恒等、严格schema/frame/source/task/seen/dimension和private checkpoint往返精确。ETF容量按公开总类数固定，无未来支持。恒等不保证encoder更新/新类竞争；没有分类训练。旧记忆append不重压缩，不称验证有损重压缩；总存储随seen增长，各控制等量，不称固定全局容量。

每stage48queries=4固定随机+2四方法共同stress各8，rank8/beta8/Fro2/vnorm2/stress12Adam .01，仅评价不更新memory。stress随stage变化不能全部解释为memory漂移。query冻结后8新16点批，两高维每stage每批主候选max<=.9各moment/random且mean不增，加全部接口/算术PASS才有限序列信号；8批平均仅诊断，不挑stage/批/类。所有old/new/class误差和Gaussian正负必须公开。

四独立CPUcase：metadata非法拒绝/旧支持拒绝/类别固定种子前缀；三完整序列。原d32/1536/1536和N512/9434/1038，不读取医学images/features/checkpoint/val/test，不使用GPU。case1800/总体5500秒资源上限不变，超时/失败不重试，已完成stage匿名结果保留。全部构造、checkpoint、评估包含逐次wall/user/system/RSS，内部operation不重复加。未来仍需完整分类训练同架构控制/生产预算/新正式协议源公开冻结/真零预检，当前非GPU-ready。
