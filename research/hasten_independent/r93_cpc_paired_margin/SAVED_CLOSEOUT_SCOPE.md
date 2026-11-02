# R93结束后saved-only结案范围

正式attempt4已exit0，6144steps、24builder+optim、120Fcontexts/arm及480主/held行完成，门false。原模型/优化器不变，不恢复训练、不改门，不读VAL。按原协议补齐old/new/head/tail与class-stage描述及结果校验。

仅加载封存24模型、原固定120情境和既有TRAIN F特征，最多960次builder前向，960个候选score-bank（每bank三动作），评分行以实际计数为准；0optimizer/encoder/actor/VAL。读取saved optimizer只做finite检查不step，不构造新episode或择优。复算每行原NOOP/HALF/FULL宏尾最大误差<=1e−8pp，重算原gate各项和tiny保护，不追加判门条件。新old/new/head/tail与per-class-stage为描述，固定等权全部context，不按结果挑选。

输出stage/class/分组准确率聚合，tiny/rescueharm为重复情境暴露计数非独立病例；不公开逐例/ID/features/Q风险。原RESULTS及snapshot不覆盖。费用单独worker收取，不重复加core，仍原deadline、CPU4、共享memorylock、RAM/磁盘硬限和remote-home。预留600秒科学计算+3600秒分析交付，小于原剩余；仅saved-only检查失败才修复结案，不重fit。
