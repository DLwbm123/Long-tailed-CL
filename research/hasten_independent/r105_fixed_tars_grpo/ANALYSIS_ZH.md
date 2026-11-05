# R105 结果解释：TARS改变了当前GRPO结果，但采样优势未建立

本轮按用户要求实际完成：固定R104 FULL教师及CENTER/MULTI，所有反馈和部署采用同一构建器；没有再训练教师、融合或视觉表示。8学习臂均512更新，完整4096更新后封存并统一VAL。worker/controller exit0、12读出逐例保存重算校验通过，TEST0。

## 完整配对结果

|读出|micro %|macro %|tail %|
|---|---:|---:|---:|
|T|82.567727|59.799132|38.103445|
|CENTER|83.804476|67.829601|54.055202|
|MULTI|83.686690|64.674698|47.622335|
|UNIFORM|84.393404|65.556404|48.395029|
|SUP|84.393404|60.801629|37.892274|
|RAW_EXACT|83.686690|61.381548|40.354527|
|GROUP_EXACT|83.510012|68.163992|55.127254|
|GRPO|83.510012|61.698412|41.490891|
|TARS_SUP|83.627797|67.749770|54.055202|
|TARS_RAW_EXACT|83.333333|67.951945|55.127254|
|TARS_GROUP_EXACT|83.274441|67.924771|55.127254|
|TARS_GRPO|83.333333|67.951945|55.127254|

## 能支持和不能支持的结论

1. 同条件TARS_GRPO相对普通GRPO：macro+6.253533pp、tail+13.636364pp，micro−0.176678pp。这是本次固定查询/种子下的加权结果差，不应抹去。
2. 相对固定强教师：TARS_GRPO macro+8.152814pp、tail+17.023810pp；但固定CENTER本身已是67.829601/54.055202，相对CENTER只+0.122344/+1.072052pp，micro−0.471143pp。大部分对教师收益已由固定融合解释。
3. 相对TARS_SUP：macro+0.202175pp、tail+1.072052pp；相对TARS_RAW_EXACT三项指标全相同，并进一步核对全部1698个预测标签完全一致，不只是准确率偶然相等。相对TARS_GROUP_EXACT只有macro+0.027174pp、tail0pp。未加权GROUP_EXACT macro68.163992反而高于TARS_GRPO0.212047pp，tail相同。
4. 因此当前证据支持TARS在本配置改变普通GRPO行为，尚不支持GRPO优于同信息预算的精确优化或采样必要性。TARS也不是对所有优化器一致正向：加权组精确的macro低于普通组精确。

## 极少样本和权重必须一起看

TARS_GRPO相对普通GRPO，在tiny原legacy1有0/1→1/1、legacy12有0/2→1/2两次新增正确；legacy2都0/1。其tiny macro贡献+6.521739pp，超过总净增+6.253533pp，说明其他类合计贡献为负。去掉全部三个tiny后的20类macro，TARS_GRPO反而比普通GRPO低0.308437pp；尾类总增+13.636364pp恰为这两次新增正确贡献。此为事后敏感性描述，不改主指标或挑类重定义成功。

CENTER、普通GROUP_EXACT和TARS_RAW_EXACT在三个tiny上的correct/count也与TARS_GRPO一致。这些不是TARS-GRPO独占的救回，也不证明所有极少样本类被解决。对强教师的macro净增约80%同样来自这两例；余下20类对教师仍+1.875736pp，但不应把这部分固定融合贡献当成采样优化胜利。

最高TARS权重68.141253出现在原legacy12(TRAIN6、教师正确3)，而原legacy1(TRAIN3、教师正确3)和legacy2(TRAIN6、教师正确6)权重都1。这反映本轮严格采用拟合内TRAIN错误率，稀缺但TRAIN已全对的类不被放大。不能据此用VAL错误替换TRAIN误差来救援。固定权重来自23类完整表，未挑类或调gamma。

## 反馈和更新诊断

固定生产函数消除了OOF教师→FULL教师的构建器替换，不保证TRAIN与VAL分布相同；本轮是拟合内反馈，不能称全链交叉拟合、独立奖励holdout或患者泛化。自然查询分布是相对R104类别均匀采样的明示变化；本轮内部8臂严格匹配，跨run绝对差不能只归TARS。

共享自然查询每臂16384抽取，重复4次更新；8臂总524288个query-update暴露不是独立查询案例。两GRPO臂实际32768组/262144动作；131072组-update/1048576动作-update暴露含4次复用，不当新采样。所有old概率和动作已私存，支持以后审计，不公开逐例。

普通GRPO与TARS_GRPO的采样零优势组分别95.111084%和96.154785%；共同自然查询结构零组91.973877%。TARS并未消除零组。平均KL分别0.276463和0.492266，平均梯度norm0.024945和0.131225；加权不仅改变类别分配，也改变总梯度尺度及相对KL约束。没有全局常数放大/权重错配控制，不能分离这些成分；不会追加搜索把结果调成正。ratio clip频率均0，梯度clip为另一个量；TARS_SUP有约1.953125%梯度clip，其他科学臂0，详REWARD_DIAGNOSTICS。

## 完成性、复用与成本

新后台流程controller wall 148.299280s（2.47min），原FIRST到退出1040.766227s（17.35min，含准备）。worker wait4 CPU178.450853s，controller CPU6.343895s；native/profile/train/eval已含worker，不重复加。workerpeakRSS2621726720B，sampled grouppeak2579251200B，退出allocated disk8499200B，GPUpeakreserved878706688B。

软件测试CPU.376398s/wall.556020s另计，导入准备开销非0未全采样。新native64工程router更新、24generic+24teacher图像行(两张TRAIN图、六视图各两pass)，另有整表缓存一致性重建的20463个TRAIN候选评分行(6821×3)，不是新编码或额外训练；正式teacher/fusion更新0，4096router；VAL编码10188generic+10188teacher行。实际distinct VAL1698，12读出非12份独立样本。

明确复用历史：R104 FULL教师280更新，fit CPU473.737995s/wall380.788121s；两fusion各512更新，fit CPU合268.372049s/wall268.286758s。这些历史子项不覆盖全部R104编码/工程/准备成本，不作精确端到端分摊，不与本轮CPU相加伪造新消耗或宣称端到端只需2分钟。

完整准入预算7978.37s含safety2和7200s结案余量，是预算而非实耗，更非科学效率收益。原24h未重置。GPU kernel时间未采样，文档/SSH/发布CPU非0未完整采样。结案仅保存预测标签等同性检查、公开聚合计数重建micro/macro/tail，误差1.4211e−14pp，0新forward/optimizer/VAL评分/TEST。无正式失败或重启；已完成的R104保持原样。

## 停止与公开边界

所有12模式/23类/原tiny分母/收益损害/配对差/权重/费用均保留。仅单seed、同一反复开发VAL，没有显著性/患者独立证据。最终结论是加权的局部正信号与明显tiny敏感性，采样GRPO独有优势未建立；不外推所有RL无效或预设方法创新。完成交付后停止，monitor OFF，不自动新增实验、消融、seed或调参。私有图像/身份/逐例预测/特征/权重/optim/rawlogs/old-policy traces留在授权data盘。
