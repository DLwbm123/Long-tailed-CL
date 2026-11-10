# R128首个新增任务：保存结果中间核验

仅第一个posttask，非最终R128结论；不改原四post平均/最终head/遗忘九门，不因中间reward/VAL调整训练。

|arm|seenmacro %|tail %|old5macro %|new5macro %|oldhead4 %|
|---|---:|---:|---:|---:|---:|
|BASE_TARS_RAPO|34.9744|9.5238|42.1014|27.8474|49.0553|
|CE_AUX_TARS_RAPO|30.2559|5.9524|21.7533|38.7584|27.1917|
|POS_KD_TARS_RAPO|27.9645|2.3810|21.2376|34.6913|26.5470|
|CONTRAST_TARS_RAPO|34.7141|8.3333|41.4962|27.9320|48.2988|

主相对各控制的macro/tail/old5/new5增量pp：
BASE_TARS_RAPO：macro -0.2603, tail -1.1905, old5_macro -0.6052, new5_macro +0.0846
CE_AUX_TARS_RAPO：macro +4.4582, tail +2.3810, old5_macro +19.7428, new5_macro -10.8264
POS_KD_TARS_RAPO：macro +6.7496, tail +5.9524, old5_macro +20.2586, new5_macro -6.7594

全部4臂各128真实参数步/8192 TRAIN回答；同current query tape、连续Adam step128、sigma逐式、POS_KD和主臂teacher最后同步120、负例资格/众数平票、所有964组件/类整数分母与seal先后均CPU独立核对。
teacher正负上下文含完整当前类别答案，收益是辅助监督构建，不当纯RL必要性或临床证据。CE/KD同标签且实际计算不同，不能只比较正确TRAIN回答数/teacher行数宣称效果。
old5 macro不等firsttask head4；当前全10类tail/tiny小分母完整披露，原firsttaskhead门要到最终task判定。组份额或teacher资格只说明可用机会，不是梯度/泛化效果。
仅用远端保存的VAL预测、normalized rollout/done/parsed/negative与各stage权重optimizer状态核验，无新生成/模型前向/optimizer；私有逐例输出与索引不fetch或公开，全部CPU费用另记。
后续真实任务与全部四臂继续原冻结协议，原23h首次准备钟、累计10GB与独立hardguard不变。

本次CPU审计完整子进程计时（含torch导入）2.062131秒，wall 2.093726634979248秒；0新模型调用，成本保留INTEGRITY_PULSES。

主对BASE的macro/tail均未增益；对CE_AUX和POS_KD的macro/tail更高伴随new5macro更低，不能称全面获益。全部臂tiny为0，新增两个tiny类VAL各1，完整类表保留分母。
本核验不独立重评分未保存的校准逐例correct、teacher原始logp或每类梯度；实际L2只证明参数确实更新，组份额/资格不证明teacher贡献。
