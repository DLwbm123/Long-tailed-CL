# 前瞻贡献协议草案：不是生产启动许可

矩阵和机制在任何新医学指标读取前声明。当前只有CPU helper，没有正式训练配置、worker、LAUNCH或GPU准入。未来生产冻结必须另外记录实际source、完整CPU检查及启动屏障；本文件不冒充这些凭据。

## 主四格与最小机制对照

主四格固定：Core、Core+LT、Core+PMT、Core+LT+PMT（完整global+独立regional配对映射）。严格复用前两条现有控制，不降低Core、不重新挑主方法，主方法仍固定LT背景。

为分离全局统计映射与区域映射，增加一个最小贡献配置Core+LT+PMT_global_only：global采用配对映射，区域保留原LT共同平移。比较它与LT可看global映射贡献，完整组合减它可看regional贡献。Core+PMT与Core本身分离非等方映射与原共同平移；CPU还需证明纯平移路由精确退化原控制。不得事后用最好的一条替换主四格结果。

因此若进入正式生产，共3个新配置×ISIC/HK=6新轨迹，严格沿用每条ISIC468（100/72/296）、HK176（98/4/2/20/18/34）更新，总1932；这是机制对照，不增加epoch、seed或同方法预算档位。6条真实零optimizer预检全PASS后才TRAIN，6条全部精确TRAINED后统一val，正常6preflight+6train+6eval=18驻留记录逐项计费。未满足这些条件不得启动或评价。

## 不变项与统计时序

原seed/split74002、order/tasks、train/val、cached ViT、2epoch/batch64/workers4/AdamW.0003/WD.01/clip5/FD10/ridge.001/proximal.01/cosine及原fit/meta和变换全部保持。仅GPU0/1，test/reserved封存，RL暂停，无旧图像/特征回放，不增加旧训练图像推理校准。不通过val选择秩、强度、ridge或步长。

输入时间点严格匹配原common_shift：epoch map只用current fit配对；最终boundary map与原shift一样使用全部current train，meta只在这个原生封存阶段加入，不作为独立校准或调参集。当前新类统计按原sample_weights和实际数据重建，不由旧映射生成。训练内before/current按私有行索引配对，不能仅检查同类标签。

每个epoch从不可变task-start bank估计teacher-start→当前epoch的map，绝不将上epoch候选再运输一次。boundary后封存新frame，下一task才以此为新起点。所有map/行索引/模型/矩阵/逐样本特征保持NAS私有；公开仅聚合标量、类别计数和混淆。

## 生产CPU与真实预检需补齐的内容

1. disabled完整Core/LT参数、head和bank精确回归；纯平移global/region退化原路径；metadata/version/frame/配对索引/ckpt roundtrip；非法scope/budget/早val拒绝。
2. 非零映射下1536×23完整调用链、old/current混合、组件半径unknown schema、R/competition/proximal/Woodbury一致，原生envelope梯度适用范围。map及old统计块内detach，不偷偷对每batch重新估计映射。
3. 每配置实际第一训练batch backward但0optimizer，参数/buffers精确不变、必需adapter梯度非零有限。首task没有旧bank，不能因此声称已覆盖最大旧bank映射：每预检另做明确标为shadow的最高旧类数GPU矩运输与内存检查，0optimizer、不写正式统计，并披露其合成映射性质，不把shadow当真实漂移或效果。
4. 不借GPU2/3、不改预算压batch；失败保留全部驻留和错误，按新源新轮处理，健康源不改。

## 完整报告与科学解释

六指标、Acc/已有F1、每stage/31类固定n/recall、old/new混淆、配对控制救回/破坏与真实间隔；主四格、global贡献、regional贡献、交互及AIR差距全部正零负。控制指标复用本地公开结果，必要配对控制推理计入本轮eval，不重计旧控制训练。

增加训练侧map秩/有效秩/ridge/配对残差、global与四regional分开、old类均值差与trace(Q)变化、component半径界宽度、frame应用次数、最高内存等审计。它们不能称为旧类真漂移概率、独立校准或收益证明。模型可能在未观察旧类方向上错误推广，可能伤害新类，必须由冻结完整结果检验。

开发门仍为相对各自控制BA≥+1pp、tail≥−.5pp、forget≤+1pp、old/new≥−1pp，在两数据集分别报告。门不是逐类无损保证，也不是统计显著性或临床证据。任何失败/identity/负结果均完整GitHub+NAS交付，不据val自动调整映射参数或追加预算。
