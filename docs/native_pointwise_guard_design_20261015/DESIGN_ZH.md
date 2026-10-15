# 完整原生解析头的逐点边界约束：独立 CPU 证伪

仅独立合成 CPU 设计，不是生产实现或新医学实验。保留 Core/LT 原 competition、完整二阶矩、metric、proximal 和 Woodbury；不修改这些原函数。既有 PMT 交付 080c2a33 的新类损害是动机，不能据已见 val 挑类、点、阈值。

目标为原 SPD 二次型 E(W)=1/2<W,H(W)>−<B,W>，约束所有已观察 current-fit 点的真类−每个竞争类 margin 不低于块内冻结 Wref。H(W)=G W+.5 Σc Qc W Lc；G=ΣQc+.001R+.01I。原参考、competition、metric、旧矩、previous 均在求导块内固定。当前点 x 同时影响 H/B 和约束/目标；正确 envelope 为 E_x−Σλ g_x。若 Wref 随 x 重算，本设计的梯度不适用，必须另推隐式导数。

matrix-free 分离仅扫描 N×K scores，按最违约边加入约束；H逆调用原 PCG；小活跃集非负 dual 有阻塞移除步骤。64 atom 上限和精度失败必须显式报错，不能静默放松或返回 identity。仅缓存活跃 atom 的 d×K 逆方向与小 Gram，不建 dK×dK Hessian 或 N(K−1)×dK 约束张量。该上限是 CPU 设计资源边界，未批准作为生产策略。

预先固定三类证据：6×3 Core/LT 显式 native Hessian 与完整小约束枚举 oracle、移动支持 Lagrangian 有限差分；4×3 正负坐标各类别全支持强制判别 identity；1536×23 Core/LT（21旧类）双点秩二支持 KKT/资源检查。高维 stress Wref=无约束解+.02边方向（小例 .3），用于主动约束算法检查，不能冒充真实同状态 OFF。另单独报告由未运输旧 bank 得到同状态 OFF 的固定 raw affine(2.8,.65,1,...)案例，所有正负均保留。不扫变换/强度挑优。

高维仅2点，不能说明高秩真实 fit 约束的可行非 identity 空间或运行代价。另4维全支持 identity 反例不能推广为所有高维必 identity。支持外不提供旧类或 val 保护保证。训练内守卫不等于 val 无损；旧域漂移外推不可识别性仍存在。未接医学 loader/worker/封存/隐式刷新，不称 GPU-ready，无 GPU/医学图像/医学 checkpoint/val/test/optimizer。
