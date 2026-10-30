# R86：固定 patch 信息与双份 CLS 对照

同一冻结R81表示状态，原S/Q，全六阶段24个监督CPU头；0视觉optimizer/controller/VALTEST。新通道为CLS+固定patch均值，对照双份CLS，同1024维和同分类头；全部头先seal后Q。

事前必要门 **False**，全四项同时正阶段 **4/6**，极少类门 **False**。

| patch 减参照 | 宏 pp | 尾 pp |
|---|---:|---:|
| duplicate | +0.134297 | +0.334994 |
| CLS | +0.134297 | +0.334994 |
| NCM | +0.438050 | +0.572630 |

| stage | 宏 vs duplicate | 尾 vs duplicate | 宏 vs CLS | 尾 vs CLS |
|---|---:|---:|---:|---:|
| 0 | -1.446668 | -3.218389 | -1.446668 | -3.218389 |
| 1 | -0.244828 | -0.759257 | -0.244828 | -0.759257 |
| 2 | +0.609532 | +1.732333 | +0.609532 | +1.732333 |
| 3 | +0.812225 | +1.707837 | +0.812225 | +1.707837 |
| 4 | +0.454229 | +1.215443 | +0.454229 | +1.215443 |
| 5 | +0.621294 | +1.331995 | +0.621294 | +1.331995 |

固定全部196非CLS token均值，无patch选择/混合比例/温度搜索；双份CLS数学等距控制由软件检验，实际数值差另行披露。原CLS/NCM/natural参照只读缓存，0重训0重评分。完整23类及Qa/Qb、旧头旧尾新头新尾均保留。class1fold1仍无双component查询覆盖，不补可靠性证据；单HK74002反复TRAIN开发，不是独立患者/新seed/VAL确认。监督通道信息即使有效也不能归为GRPO。保留R80-R85原失败；本次授权只解除固定诊断，其他分支不恢复。

## 实际结论与完整限制
全局四项增量为正，最终阶段对原NCM宏/尾分别 +0.536975612/+0.522837767 pp；但事前要求至少5/6阶段四项同时正，实际只有4/6。stage0和stage1对两个CLS对照的宏/尾均负，不可按后四阶段选择性报告。极少类门要求class1/2/12相对NCM各非负、至少两类严格正；实际只有class12严格正，所以总门失败。

class1相对NCM与两个CLS对照均为0，12cell均无净正确数变化；class2相对NCM为0，但相对两个CLS对照均为-2.777777778pp（1cell受损、11相同）；class12相对三个参照均为+2.777777778pp（1cell改善、11相同）。这不是新独立患者数，也不证明净计数不变时逐例预测不变。全部23类完整方向、全阶段full/Qa/Qb及旧头旧尾新头新尾见results，含natural完整缓存对照。

有据的增量是固定patch通道在后四阶段及整体平均上提供了一些有用信息；不足以声称稳定极少类收益、CLS表示不足是唯一原因、GRPO成功或已通过科学门。相对NCM的总增量含R84已有监督头收益，不可全部归因patch。双份CLS对照与原CLS-balanced全部保存预测相同，保存logits最大差1.48335635e-5，数值差存在但本次计数结果相同。

## 成本、工程与保存
R86完整6stage×2fold×2arm=24监督CPU头，251个LBFGS iteration、302个closure，最大最终梯度4.93090775e-7；29217科学TRAIN编码行、58434新classifier查询行。6821唯一FP32 prefix占RAM4127960064B，prefix最大参考误差1.65775418e-7；新CLS与R84同源保存CLS逐行最大差0。全部24头和6套feature先seal后Q，saved verifier通过；结案及数值对照审查0新模型评分/优化/VAL/TEST。

native使用2张真实TRAIN图，prefix参考完整forward2行；随后4次两行tail公式评估共8行（base一次、source pool一次、source original-forward一次、独立patch公式一次），不能把2张图当作仅2次全部模型forward。科学prefix构造另做6821行原冻结参考forward，不与29217源tail行或58434classifier行混计。0视觉optimizer、0controller、0VAL/TEST。Mac与服务器软件各2toyCPU拟合共4，不算科学24头或视觉updates。

本次仅一次正式attempt，退出码0。guard worker wall356.799793秒、CPU530.997817秒；原始clock从准备/排队起至退出2813.085983秒。worker峰RSS5980635136B；2秒抽样group峰5936906240B（采样不同，不要求数值大于worker峰）。磁盘峰436027826B，硬cap2GB；root/remote prefix spill均0。Mac软件CPU2.352893秒另计；saved数值核查CPU1.524840秒/wall0.610097秒/RSS389419008B。未独立采样的远端准备及文档交付CPU不是0；服务器软件/native已含guard worker CPU。所有费用口径见COST.json。

按用户明确指令，启动前把RAM准入由128GiB改为64GiB，再由GPU0/1双卡改为仅物理GPU0同时承载冻结prefix与tail。实际准入MemAvailable122160328704B/GPU0free26780MiB，保留30GB总硬上限、worker28/group29GB每2秒检查、GPU0合计reserved<12GiB、原20h clock和2GB新存储cap；不请求GPU1。native/software通过才全六阶段，无效益短pilot。全部科学权重/优化器/pooled features/源状态/私有分数保留，仅清再生prefix RAM/index；每token冻结激活瞬时计算不落盘。无工程失败/科学重训/时钟重置。

## 后续边界
R86固定诊断已完成。当前证据不支持升级controller或开patch选择/融合权重/温度/层数搜索；不挑阶段、不改必要门。仅本诊断曾被解除暂停，其他分支仍关闭。保持已有R80-R85负证据和当前正负结果，不把受限诊断完成解释为所有主线工作暂停；后续只推进具有区别性依据且在授权范围内的工作。

公开仅自有源码、协议、宏及类别方向聚合和费用；私有图像身份、逐例数组、Q逐类risk/margin、feature/权重/optimizer、原日志与配置路径不公开。公开源码的中性部署路径占位替代私有部署位置，运行时配置需由私有环境提供。
