# 三来源入口已完成：R129初稿未运行，现已否决直接启动

后续目标核对发现：R129旧类GKD与分任务TKD比较，不能单独识别RaPO保留优势与CTAN尺度；其CE缺少同rollout KL。该未部署草稿保留为0新modelcall/0optimizer证据，不称科学失败或已准入。完整修订分析见PLAN_RECONCILIATION_ZH.md。

# 原入口记录（已被上方未运行结论覆盖）

[LwF §III/V](https://arxiv.org/html/1606.09282v3)在新任务输入上保存旧输出并蒸馏，不要求旧图；作者明确指出新输入可能不能覆盖旧分布。其任务输出头和warm-up等条件不能直接当本生成式接口或旧类准确率保证。

[SS-IL §3.2/4/5](https://arxiv.org/html/2003.13947v3)区分全部旧类联合GKD与各旧任务分别归一化TKD；整体SS-IL还用分离CE及旧exemplar replay。只借归一化机制不等于复现整篇、无回放原法或参数梯度正交保证。

[COPR §2.2/3/E.2](https://arxiv.org/html/2402.14228v2)对有限回答集合归一化策略概率并拟合目标分布；旧约束用replay buffer，阈值在旧buffer上计算。本文未审代码。不能把历史输入换成新图后仍声称旧约束成立，也不把当前标签的exact reward当人类排序数据。

本地推断：可定义生成式canonical类别+EOS的完整序列概率，再在旧类集合条件化；这不改变部署生成、prompt或硬mask。全旧类条件分布与按旧任务分块条件分布是同教师/同答案/同评分预算下可区分的目标。分块softmax对每块加同一常数不变，所以它不识别任务间总质量；全旧类条件化仍不识别旧/新及canonical集合外质量。这是明确风险，非已解决菜单漂移。

选择一次R129机制资格实验：BASE TARS_RaPO、全旧类语义GKD、分任务语义TKD主、同TKD强TARS_CE。所有anchor仅自己的上一task policy，同当前菜单；当前TRAIN only，无旧图/旧buffer/额外SFT；teacher没有完整当前答案前缀。GKD与TKD首个post只有一个旧任务，数学完全同目标，必须实际相同，作为内置核验。R128 sampled-token完整答案教师、R126真实旧图奖励与本实验不同。

这仍是已知机制的生成式适配，不自动新颖或最终方法。原九门完整保持，原头类门不因R127换基线；强CE同旧teacher和当前gold，少G8/KL成本明示。若不胜全旧类GKD/BASE/强CE或旧头尾失败则关闭该冻结语义KD配方，不扫温度/λ/seed/task分组/长度/提示补分。须真实native全路径与完整forecast通过才训练；当前只冻结，未伪造启动。
