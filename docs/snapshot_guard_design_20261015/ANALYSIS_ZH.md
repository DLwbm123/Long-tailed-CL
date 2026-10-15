# 全约束快照守卫：分析与下一步边界

## 1. ISIC 最大阶段的 cap 问题确实被本次算法解决

原同一N9434/K8/1536维 batch-native fixture 的OFF及无约束目标与此前公开值误差<1e-8，旧64cap时最小slack约−.00261；本次仍全部66038约束，Core/LT正dual均109，slack−2.264386449e-9/−2.152222685e-9，KKT与gap全过。未筛支持或放宽margin，也没有把cap扩大到109：算法不保存逐条逆方向，全部dual起始在场，每次对聚合力求一次native逆。

旧fixture沿用minibatch proximal .01，不能拿它替代epoch/boundary full-bank目标。四条新的ISIC快照用原生full-bank proximal0，各自OFF/ON competition及metric；正dual512–515也成功，说明本次规模下不会被64条活跃缓存锁死。

## 2. 存在合成可行非identity更新，但没有分类纠错证据

八条均满足预声明合成有用门：数值可行、去公共logit后范数>1e-10、OFF相对二次目标收益>1e-12且保留可用收益>1%。旧ISIC两条保留99.969123/99.969133%；新ISIC epoch99.980036/99.980035%，boundary99.979941/99.979941%；HK99.987993/99.988032%。没有identity或未解，不说明所有数据分布都不会锁死。

负结果同样明确：八条合成救回/破坏全部0/0。当前输出未单列OFF错误储量，不能分辨是否已饱和或只是没有跨越预测边界；不为补这个细节重跑。非公共head变化约.895–.968和二次收益不能替代分类贡献。已知合成affine、强类信号样本只检查算子/约束，不检验真实旧域漂移外推、稀少旧类混淆、新类val泛化。

## 3. 资源可执行，但不是免费，也不能外推GPU时间

九个独立子进程驻留1166.576650228秒（19分26.577秒），父wall1166.709580658秒；user2365.323555/system53.651161秒独立列，不相加成wall。最大RSS3733.51953125MiB（约3.646GiB），发生HK Core。各case完整驻留包含import、合成构造和求解；同机两线程条件下ISIC快照约96–144秒、HK约194–217秒。

单次solver（含最终全点审计）旧ISIC132.49/133.72秒，epoch76.96/121.30，boundary83.86/77.19，HK169.57/195.86；这些包含于case驻留不能重复收费。每次ISIC66,038个double dual仅528304bytes，但L-BFGS历史/工作区、全支持、native类二阶矩等仍占内存，实际peak比单vector可靠。H逆133–151次（ISIC快照）或100/151次（HK），旧fixture297/308次，因此避免全fit重提不代表没有额外解析求解成本。未测真实ViT/GPU/完整多task总开销，不宣称满足未来GPU预算。

## 4. 快照契约可落地范围与解析头兼容性

既有canonical/captured同图结果可供epoch初始化和最终boundary使用，不新增image提取。当前fit实际统计不运输，PMT只旧统计，LT保留区域prior。因此运输负责旧坐标对齐，守卫只在观察支持的切换时点约束其副作用；本次不证明它修复了HK旧旧混淆。参考头是同状态OFF，非独立训练Core/LT轨迹，不能承诺独立控制new Acc无损。

epoch头在下一encoder更新后就失去支持凭据；中间minibatch native .01/proximal/梯度照旧。本次投影detach，不利用错误的固定头envelope梯度，也不声称守卫持续贯穿训练。boundary必须放在最后head重建之后，否则被原head重建覆盖。已有PMT evaluator的无约束重建相等断言不可复用；未来需独立guard-aware checkpoint/eval版本。private aggregate force仅能重验stationarity；没有全支持不能由它独立复验所有约束，需明确附带当时全支持检查的标量凭据。

小Core/LT枚举oracle误差7.250006207e-11/4.717999912e-11，正dual3，关闭精确、adjoint、类别排列、公共logit、旧frame/时点拒绝和合成checkpoint通过；零漂移精确identity通过。该合成证书回归不是完整医学worker参数/bank/loader/早val/封存回归。

## 5. 结论与最小下一步

当前阻碍从“ISIC全fit约束在64cap下解不出”转为“快照守卫能否在原生产调用链产生医学净收益、成本是否可接受”。本轮已经完成用户要求的冻结/CPU/资源与全部正负交付，不自动启动新CPU/GPU。

如后续继续，先做独立生产接线方案：关闭时Core/LT参数、统计、head精确回归；epoch和boundary注入点与失效版本检查；guarded checkpoint的重建与公平same-state OFF贡献对照；共享现有提取而不增加曝光的完整预算账。只有另行冻结这些检查并得到GPU授权、全生产CPU与真实零更新预检通过，才讨论最小主四格与必要无约束运输对照。不能用本次合成通过跳过这些步骤，也不调val强度/选点或降低Core。
