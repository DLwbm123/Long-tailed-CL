# 训练支持逐样本边界约束：独立CPU证伪设计

本设计在PMT六轨迹完整交付后开展；来源是已公开分析中的通用失配问题，不按val挑类、阈值、rank或强度。没有GPU授权入口、医学loader、医学checkpoint或val/test读取。本阶段仅合成CPU数学检查，不能称完整Core/LT兼容或GPU-ready。

对固定旧矩运输提案保留原解析二次目标E(W)。以current-fit真标签和块内冻结的同状态OFF头Wref定义所有已观察训练样本/其他类别约束：x_iᵀ[(w_y−w_j)−(wref_y−wref_j)]≥0。Wref总是可行，故不能声称约束能保证同时救回旧类或泛化到val；不新增旧图像。当前CPU仅用二类二维SPD ridge特例检查数学，不混同完整native competition算子。

若约束活跃，优化值对特征x的正确envelope来自Lagrangian：E_x−Σλ_i g_i,x，不能只把W当detach后微分E而漏掉约束导数。Wref块内固定；若未来改成每batch重新求OFF，须重做隐式导数而不能沿用本检查。

固定合成设定：old=[.5,.02]；raw old映射M=diag(2.8,1)；new=[1,0]，后续导数沿new的第二坐标扰动。ridge=.001、类质量各.5。另测两点约束、正交坐标变换/类别排列/公共logit平移、关闭精确回归。若只能判别identity或漏掉Lagrangian项，则如实拒绝继续GPU；不能因为全头范数非零就称机制作用有效。

生产前仍缺：1536×23完整native竞争/proximal/Woodbury的约束算子与dual/KKT证书、全部current-fit约束的有效分离/求解与资源边界、失效处理、原Core/LT关闭精确、训练梯度/metadata/checkpoint/封存入口。所有当前支持约束都不保证旧−旧风险或未观察新类；必须先完成这些CPU证伪，再前瞻冻结最小贡献协议，不能立即再跑一轮。
