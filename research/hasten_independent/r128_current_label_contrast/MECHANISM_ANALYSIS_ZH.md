# R128 完整证据与机制边界



原九门3/9；主对BASE四post平均macro -0.3489pp/tail -1.0146pp；classwise遗忘主 16.7067pp、BASE 16.0038pp。原firsttask head损 43.1935pp，仍含菜单变化，不归为纯参数遗忘或换门。

主对CE_AUX宏 +6.7014pp/尾 -1.2421pp；对POS_KD宏 +7.2603pp/尾 -0.0216pp。宏优势不能替代尾/保留联合门。

4臂各512实际非零更新。主科学positive/negative teacher各26112行，POS_KD正教师32768行；有效teacher机会不是缺少分支执行。但未保存原始teacherlogp/逐类梯度，不能证明符号投影压制、错误强化或方差是失败原因。主相对BASE正确TRAIN响应多335也不建立VAL增益。

最终主tiny0，BASE tiny5%仅UC grade2-3的1/4，其余四tiny0；CE_AUX和POS_KD最终tiny均0。阶段3 POS_KD的20%来自UC grade0-1的3/5且最终该类0/5，不能把阶段峰值当稳定保留。

PAIRED_CLASS_CONTRIBUTIONS按全部到达类和全部四post完整分解，不据此删类/改门/定向调权；重复查询不是独立患者，单顺序旧seed共享前缀是开发条件。

只确认当前完整标签对比辅助监督配方未建立对骨干的增益；不否定全部RL、原SIPO或所有上下文方法。关闭本配方参数/teacher周期/seed/提示网格，不保留无效模块包装创新。
