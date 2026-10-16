# 完整旧类支持回放参照：冻结协议

冻结后只运行四个已有场景（ISIC 100264/200264、HK 100265/200265），每个从已保存 teacher 出发，使用原有限训练支持的全部旧类样本、原新类样本训练 48 步；共 192 次 adapter 更新、0 次 memory/teacher 更新。复用原 ETF、非线性 adapter、float64 CPU 两线程、学习率 1/16、类别均衡 CE + 系数 1 的 teacher-correct 旧类中心化保持项。每个旧类内部均匀权重；CE 每类 1/K_seen，保持项每旧类 1/K_old。调用既有训练函数，不改训练目标、步数、构造、seed 或门槛。

这是有限未压缩训练支持的机制参照，存储和计算预算不与压缩方法公平；不是性能上界、真实医疗图像结论或独立确认。原支持由同一确定性生成器和配置恢复，不增加训练样本。原 moment、selected、candidate 三种已封存终点直接复用，不重训、不重选门。

四个新模型与训练费用先分别保存，再统一评估。复用每类 1000 样本、评估 seed 310264/310265 的既有平衡 bank。原三个模型逐类正确数须与公开历史结果一致。主比较 full_support−moment，辅助比较 −selected/−candidate；沿用整体 BA 正收益、旧/新组各至多下降 1pp、条件区间判定。每类四格 paired multinomial bootstrap 10000 次；每场景基准 seed 954000+3i，依次比较 moment/selected/candidate，线性分位数双侧 pointwise 95%。这些区间仅反映给定训练状态的评估抽样不确定性，非跨训练 seed 或联合覆盖；四场景全部 SUPPORTED 才支持共同通过。失败、不完整或缺失为 NA，不补种子、调参或科研重跑。

为四种终点新增诊断：逐类和 all/old/new/原 accepted_old/原 fallback_old 组 BA、类别均衡 CE、真类减最大其他 seen 类 margin、真类减最大相反组 margin、max_old−max_new 分数差；逐类 margin 分位数 [0,.1,.25,.5,.75,.9,1]、负值/零值/绝对值≤.25 数量。门分组沿用旧选择结果，仅解释，不重选。argmax 决定正确性，零 margin 不直接判错。CE/边界指标不替代 BA 判据、不增加模型选择。

对 full_support 与三基线逐类记录 both_correct/rescued/harmed/both_wrong、wrong→different_wrong、错误目的组 old/new，以及正确→错误 old/new、错误 old/new→正确的真实配对转移；边际错误数量不能冒充样本翻转。所有 seen 类同属一个分类头。不给 old/new 各半权重；宏 CE 与宏 BA 均对组内类别等权。

零训练更新算术检查覆盖 CE/margin/tie/类别权重/配对分解。独立 worker 费用由 supervisor 全量记录，训练费用随封存模型及时落盘。单 case 上限 900 秒，总上限 1800 秒，禁止 GPU。历史 CPU 8172.503710279241 秒、历史 GPU 400036.74851310626 秒；新增费用和父进程墙钟分列，前次 ISIC 内部分项缺失仍为 NA。源代码及本协议先公开，再运行；完成后公开结果、诊断、完整费用和分析，保留 NAS 回执，停止。无自动下一轮。
