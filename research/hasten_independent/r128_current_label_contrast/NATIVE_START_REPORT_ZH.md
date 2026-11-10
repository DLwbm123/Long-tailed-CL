# R128 原生准入与真实科学启动

已按冻结current-label contrast四臂协议完成CPU fixtures与真实GPU1 native，科学训练已经进入stage1 BASE_TARS_RAPO；这不是主臂或任何方法的效能结论，不依据部分输出改协议。

原生matchedhidden/同形head均0，rawfullhead差0.0625、sampling/TF最大差0.0759369如实保留，0.05阈值未放宽，同优化TF未更新差0。四条真实RL/CEaux/KD/contrast路径加一条SFT工程路径均有非零参数步，全部丢弃还原合法历史prefix/anchor；不能记科学学习。最坏teacher profile额外32个16x24 TF行、0新生成，只用已见首task词及neutral padding，student最长257/teacher272 tokens，不送未来词。

完整矩阵按冻结分臂native非teacher路径+所有行有效的worst teacher envelope，乘1.5再加1800reserve，实测74657.825秒<原remaining80674.621秒才科学启动；没有放宽R126的旧20h或1.5margin，其旧反馈实验仍未启动。R128是无旧replay的当前完整标签对比新问题，原first preparation1794277179.063、独立23h截止1794359979.063，全部准备/CPU/native/失败/子进程共钟不重置。

四臂为相同TARS_RaPO骨干、不加aux；加全强度CE1；加正教师sampled-KD1；加正负对比主κ.2δ.2并作显式符号投影。teacher用完整当前label，明确计为辅助监督；强同标签CE/KD控制不可省略，不把辅助监督提升当纯RL或临床新信息。teacherK10全局跨任务持久，当前TRAIN only/seenVAL全臂seal后/无buffer，不改变学生seen菜单/输出长度/提示/seed。

复用唯一已验证generic首task SFT前缀：306/4868queries/624VAL是历史成本，0本轮额外SFT，绝非allTRAIN SFT或挑VAL最佳。正常本轮2048新optimizer曲线/171512生成/4原生TRAIN编码/0新VAL编码；加历史306才是完整链2354步。所有原R125失败门、R127条件归因保持，原九严格效能门以本轮BASE/CEaux/POSKD新控制预先冻结。

生成prefill/active/含pad cells与context-teacher TF input cells已精确计数；base actor/ref/optimization TF的每kind padded长度没有独立日志，saved-only会按实际总decode cells和cap24给出input-token区间，不伪称每个forward token精确。其物理response行数可精确推导，所有这些算子的真实CPU/wall/native外推/硬钟/存储预算均已计入，不为补记录重启健康worker或重生成。

只GPU1/pro5000，主/子中性argv及物理绑定已实际核对，guard无触发；整个窗口旧7085854720B+本机200MB+本root预计<650MB<10GB，2s9GB/deadline−120/RSS48GB/free1GB硬guard。源码/聚合公开，native公共记录去除class_groups，私有rollouts/预测/索引/权重/ACCESS/路径不fetch或公开。实际CONTROL_EXIT0后才运行本R128 CPU-only finalize_saved。
