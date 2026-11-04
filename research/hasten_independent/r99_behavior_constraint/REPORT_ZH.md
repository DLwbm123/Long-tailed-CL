# R99 动作KL与分类行为保持：完整匹配结果

COMPLETE_PRIMARY_MECHANISM_NEGATIVE

主候选CF_BHV。A=False；B={'SUP0': False, 'SC_BHV': False}；约束机制C=False。

|arm|平均macro|最终macro|平均tail|最终tail|
|---|---:|---:|---:|---:|
|T|66.840041|64.876782|50.729225|44.924938|
|SUP0|66.757686|64.754581|51.227872|45.200420|
|SC_ACT|66.757686|64.754581|51.227872|45.200420|
|SC_BHV|66.762105|64.781092|51.227872|45.200420|
|SC_NONE|66.131113|64.808785|49.561206|45.200420|
|CF_ACT|66.745423|64.783566|51.227872|45.200420|
|CF_BHV|66.753558|64.781092|51.227872|45.200420|
|CF_NONE|66.693661|64.823171|51.069142|45.200420|
|PATH_ACT|66.745423|64.783566|51.227872|45.200420|
|PATH_BHV|66.753558|64.781092|51.227872|45.200420|
|PATH_NONE|66.656147|64.906361|50.910412|45.200420|

九臂从同一R98 SUP0固定参照重新开始fresh optimizer。本轮新增13824更新，R98 warmup3072为借用历史成本，不是本轮新增或免费全生命周期。强R67表示冻结，全历史TRAIN预算，非严格小记忆CL或患者独立确认。

BHV尺度由固定SUP0/S/B输出小扰动校准，不看F/VAL；仅匹配起点局部尺度，不保证全程等强。NONE用于识别保持项整体作用；不事后挑NONE为主候选。PATH若相当/更强，不支持采样RL独立优势；SC同样改善不能归给RL。差分中的差分及全部对照正负见RESULT_SUMMARY。

全部54最终和54中间端点先封存，再S-full→F读出，再全TRAIN部署并一次VAL，TEST0。tiny最终VAL分母1/1/2；正确保护不等于已修好。逐类、原分母贡献、T/SUP0救回损害及component影响均保存，不发布身份或逐例。

gradient诊断拆分CE/纯RL/加权保持，不将辅助+KL合并当纯RL。实际分类翻转与utility并列报告，分数变化或奖励上涨不当分类成功。

完整成本见COSTS；native/profile已含worker，不重复相加。历史借用教师/特征和SUP0的费用不算本轮新增但必须披露。准备/SSH/发布CPU非0未全采样。监测关闭，完成后不自动追加实验。

## 结论与匹配对照

更换保持约束未达到预设目标。CF_BHV相对CF_ACT平均macro +0.008135pp、最终macro −0.002474pp，平均和最终tail均无变化，约束机制门C失败。相对保留同样CE与BHV的SC_BHV，CF_BHV平均macro −0.008547pp，最终macro/tail完全相同；PATH_BHV与CF_BHV在六stage的micro/macro/tail指标完全相同。这些是指标相同，不声称参数或逐例预测完全相同。当前不支持采样RL带来独立分类优势。

CF_BHV相对SUP0在六stage合计救回1次、损害1次；CF_ACT救回1次、损害2次；SC_BHV救回1次、损害0次。这里是重复stage的病例暴露数，非独立病例数，也不能仅以净次数推出macro改善。CF_BHV相对原强教师平均macro −0.086483pp、最终macro −0.095690pp；tail +0.498648/+0.275482pp不能归为本轮RL收益，因为SUP0已有同样tail水平。tiny1/2/12无新增损害，但没有得到修复证据。

无保持项CF_NONE相对SUP0平均macro −0.064025pp、平均tail −0.158730pp；最终macro +0.068590pp而tail不变。移除保持项未形成稳定收益。SC_NONE平均macro/tail下降更明显。由此不能把失败单独归咎于动作KL，也不能证明所有行为保持约束无效；本轮只有固定局部尺度、单seed、固定动作/奖励/步数的证据。后续应重新审视动作可改变的决策空间与部署迁移，而非据VAL调KL系数或挑臂。此轮不自动启动后继实验。

## 结束状态与费用

正式attempt1 exit0，13824/13824新更新，54最终+54中间模型/optimizer状态全部保留；saved-only校验通过，未新增encoder/optimizer/F/VAL/TEST用于校验。训练封存前F/VAL/TEST均0，封存后按协议一次读出，TEST仍0。非工程失败，状态COMPLETE_PRIMARY_MECHANISM_NEGATIVE。

worker wall3868.538716s（约64.48分钟），原clock至退出4948.692322s（约82.48分钟，含准备），未重置24h。worker CPU4037.315530s，controller CPU84.262247s；worker core CPU4036.360369s/wall3866.307844s已含于worker，不重复加。native/profile CPU24.109019s/wall23.986390s与calibration CPU9.281866s/wall8.729152s也已含worker。外部软件检查CPU1.334464s/wall1.105662s另计、LR0工程optimizer1次；worker工程optimizer36次另于13824正式步数。准备/导入/SSH/文档/发布CPU非0未全采样，不声称完整全生命周期总CPU。

峰worker RSS3861504000B、2秒采样group RSS3881230336B、GPU reserved973078528B；退出分配磁盘113836032B。大文件继续保留data盘。借用R67教师、R97特征与R98 warmup历史成本另属先前实验，不能把本轮缓存训练时间作端到端效率胜利。CPU/GPU预估预算不是实测耗时或科学效率收益。
