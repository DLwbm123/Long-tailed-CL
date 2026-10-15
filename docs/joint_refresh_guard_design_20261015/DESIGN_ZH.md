# 全fit支持移动与OFF参考刷新：联合CPU有限差分

前一19项CPU检查先完整交付后，本补充仅核对用户要求的联合梯度缺口。不重跑前一已交付检查，不读医学数据/checkpoint/val/test，不启动GPU。不增加正式seed、参数网格或预算臂。

固定复用同一合成fixture定义：ISIC T2 N2292/K6/rank1536/61active，HK T6 N1038/K23/21旧类/32active，两背景Core/LT，共4项CPU。选择依据是CPU梯度覆盖：一个已解满rank支持、一个最多旧bank；不是val类/阈值选择。最大ISIC T3的64atom未解完整保留，不扩大cap或以这四例覆盖它。

样本、alpha、old/moved_old、a、R0/R、previous、固定affine和随机流与上一协议一致。全fit每个点均按同一个0/1坐标平面的正交旋转移动 x(theta)=x R(theta)，保持单位范数；原batch64目标同步取相同indexed batch，Wref(theta)是未运输旧bank的原生OFF重解。a、两个R、previous和旧矩在块内固定；当前特征变化、约束支持与参考重解共同进入theta导数。不是只动batch或把Wref继续detach。

E=.5<W,H(theta)W>−<B(theta),W>；g_ij=x_i(theta)^T((w_y−w_j)−(Wref_y(theta)−Wref_j(theta)))。

由当前batch矩显式求Hdot/Bdot。全包络导数为 .5<W,Hdot W>−<Bdot,W>−Σλ xdot_i^T(Δw−ΔWref)+<Z,Bdot0−Hdot0 Wref>，其中Z=H0^-1 A*λ。额外解reference切向量作adjoint互检。不构建dK平方H，不对PCG迭代图反传。只作theta=0,±1e-5三次完整全fit守卫重解；前瞻误差门2e-6+2e-5|FD|。任何base/扰动求解cap/超时则联合梯度missing，不能补0或归为identity。

原64atom/精度不变，每case360秒硬CPU cap、每guard240秒软cap、整体1800秒停止派发。所有wall/user/system/峰值RSS逐进程保留；与上一轮707.928139761秒分别记录再按独立操作相加，pipeline/operation包含驻留不额外加。该补充不证明真实encoder或医学性能，也不覆盖a/R/previous/旧bank随theta刷新、FD等独立encoder损失；这些原有可微项不因解析头核对而改变。即使4例通过，T3未解、生产时序与真实零更新预检仍缺失，不启动GPU。
