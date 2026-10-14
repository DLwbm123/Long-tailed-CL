# 动态锚定医学主四格与同强度静态对照：冻结生产协议

用户授权在ISIC/HK验证动态锚定，并加入同强度静态锚定对照；Core/base和local_transport严格复用，不重训控制、不调参，test/reserved封存。源码冻结提交由启动HANDOFF/PROTOCOL_FREEZE独立记录；后续启动/报告提交不是科学训练源。

## 科学矩阵与原预算

主四格：Core、Core+local_transport、Core+dynamic_anchor、Core+local_transport+dynamic_anchor。静态贡献对照：Core+static_anchor、Core+local_transport+static_anchor。新增4配置×2数据集共8轨迹。原控制4轨迹训练复用、0新控制更新。

每条ISIC新轨迹固定100/72/296=468更新；HK固定98/4/2/20/18/34=176更新。全部新轨迹共2576更新。原2epoch、batch64、workers4、AdamW .0003/WD.01、每task cosine eta_min1e-5、clip5、FD10、ridge.001、proximal.01、competition beta.5、seed/split74002、原order/tasks/fit-meta/图像变换均不改。train仅current fit，meta按原协议加入task-end原生全current统计，不作为独立校准。

只读复用18718/295的ISIC train/val，6821/1698的HK，权重为同缓存ViT IN21k→IN1k。训练变换Resize256/RandomCrop224/水平翻转.5/Normalize.5，canonical/eval Resize224/Normalize.5。不得下载/复制影像、读test、旧图像回放、调val超参、加epoch、seed或预算臂。GPU仅0/1，2/3禁止，不干预其他任务。

固定队列：ISIC动态单独→ISIC动态组合→HK动态单独→HK动态组合→ISIC静态单独→ISIC静态组合→HK静态单独→HK静态组合。两卡slots空后按队列接下一条。24h期限从实际launch起，包含资源等待/预检/训练/评价，不重置。

## 冻结机制与静态定义

沿用dynamic_anchor_design_20261014的数学设计：任务起点原生完整W0和G0=ΣQ0/K+.001R0冻结，当前新类原生bootstrap仅fit统计。ρ=n/(n+16)/(1+age)，λ=.1 trace(G0)/1536，Π=I−11ᵀ/K，H=Πdiag(ρ)Π。当前支持度计数n取起点实际candidate计数：旧类full train、新类fit；任务内保持不变，不在边界用meta抬高ρ。age为引入后任务数，不声称旧类漂移置信度。

动态目标按同类中心化G0逆方向的最小平面旋转更新，角度乘ρ；无方向或反平行未识别时精确保留W0并按设计置对应ρ为0。静态臂目标固定A=W0，其余强度公式、ρ规则、计数/年龄、刷新和目标范围相同。后续训练轨迹分化后，各自G0不同，绝对数值λ不必相同；“同强度”指相同relative_curvature和计算定义，不假称不同模型状态下绝对曲率完全一致。

每epoch开始用上个边界已提取的fit统计刷新A，A/ρ/λ在块内detach；最后epoch后再次用fit-only候选刷新，然后meta仅加入原生统计进行封存解析求解。原有common/global与区域local transport保留，不重复添加局部中心或运输；旧类方向仍是运输估计，不额外用旧图像估计。

目标为原生fit/pair/old/ridge/proximal/FD10加 λ/2 tr[(W−A)H(W−A)ᵀ]。解析方程相应加λWH及λAH，保留原Woodbury预条件和PCG；梯度为固定目标块的envelope gradient，不宣称对目标生成全过程求导。新类使用自身身份参考、保留类别均衡竞争，公共logit分量不受锚定；不保证新类间隔/recall无损。

每epoch首batch观察同状态关闭锚定的头和特征梯度差，不更新shadow模型，不增加optimizer预算。记录strength/proximal比例、目标差距、旋转角、支持度、头/梯度实际变化、batch force及求解残差；激活或CPU数学PASS不是科学收益。

## 生产屏障与checkpoint

完整高维CPU数学、原控制disabled参数/head/bank精确回归、动态/静态完整call-chain、Woodbury/proximal、checkpoint roundtrip、早val拒绝、非法scope/budget拒绝、报告分母及救回破坏检查PASS后冻结公开科学源码。

8条各自真实第一batch backward预检，0optimizer/0policy、全部模型参数与buffers精确不变、非零有限adapter梯度、finite head/target、solver≤1e−8。预检独立输出，不计正式统计/预算。8/8成功receipt后才TRAIN；任一失败停止依赖并保留费用，修复须按最新授权新源新轮重新冻结，不能覆盖失败或修改健康源。

全部8条STATUS精确TRAINED，各steps/stages/order/seed/source/method/version/target范围核对后协调器统一EVALUATE。worker在读val.csv前再次核对phase/training_sealed_at与8条训练状态。checkpoint保存full model、bank/region/竞争权重、锚定A/ρ/λ、双精度head及精确metadata；全部NAS私有不外传，评价重建head≤1e−8并检查float readout。

## 完整评价、资源和交付

已有development val统一评价全部36个新阶段（8臂×3/6stage）。主指标完整finalBA/avgBA/tail/forget/old/new，另final/avgAcc、已有F1；Core/LT旧F1未提供留null。完整每stage、31类n/recall、原频数组、新旧类混淆、救回/破坏/间隔均保留，重点HK n=3/6/6/18和ISIC0/3、HK8/3但不选择性报告。

控制/AIR指标复用本地已公开RESULTS，不按val挑控制。为了同批次救回/破坏计数，评价读取原控制checkpoint做新的配对推理；其阶段类recall必须与公开历史结果一致。该新推理计入本轮eval驻留，控制原训练费用不重计。特征/逐样本预测只在私有进程内存中临时使用，不写公共文件或收集导出。

报告动态对各自控制、静态对各自控制、动态−静态、组合−单独、交互及AIR差距全部六指标正/零/负；不能把静态稳定正则收益叫动态几何贡献。相对各自控制开发门：BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp，两数据集逐门核对；门内仍可能有逐类损伤，全部保留。单seed/order、反复development val，非独立患者、临床/test/统计显著或等资源SOTA证据。

正常8preflight+8train+8eval=24成功GPU进程驻留记录，每次出现计一次；失败额外完整计入，不按job名去重，不加cumulative重收费。CPU/客户端wall/并行wall单列，pipeline在operation wall内，未独测process residence留null。

结束时先查results/DELIVERY防重复，local_runtime/collect.py仅匿名聚合与标量，INCOMPLETE保留部分、全部失败cost及去标识原因，missing不填0。源码/协议/结果/REPORT/CONTRIBUTIONS/AUDIT/cost经显式代理push同GitHub仓库branch，remoteSHA及精确报告匿名HTTP200后再NAS public_delivery/DELIVERY/HANDOFF。禁止发布影像、权重、矩阵、身份、逐样本结果、私有配置、rawlogs/tracebacks。聊天直接给完整主表、贡献、类别得失、AIR差距和费用。

用户最新追加每半小时监测及自主诊断/改进授权：复用同一自动化lca，不建立重复。健康保持安静；问题保留证据后解决，当前冻结矩阵完整交付后再依据完整结果分析并前瞻冻结下一版，不能改正在运行的预算/源或进行val网格。方法贡献与竞争力以机制对照、正负结果和资源差异支持，不承诺顶会顶刊录用。
