# 复用提取快照的全 fit 守卫：前瞻 CPU 协议

本轮仅合成 CPU 设计，不启动 GPU、不读取医学数据/checkpoint/val/test，不接正式 worker。固定原 Core/LT 算子、数据设定与已公开规模。旧轮不重跑。

## 保护时点与范围

1. epoch 初始化：复用现有 canonical 提取结果，原 static_PC choose 后、下一次 minibatch 之前，投影头初始化。参考头为同 encoder/同统计 frame 的原 OFF/common_translation 原生头，块内冻结。
2. boundary：复用最后一次 epoch captured；保留原 all-current（含原 meta）统计加入时序，最后一次 bank_head 重建后、写 checkpoint 前投影。约束支持仅原 fit，meta 不当独立校准。
3. 每个 fit 点对全部竞争类，要求 true-class logit 差不低于该快照 OFF。固定 encoder 上有数值可行性凭据；下一 encoder 更新后凭据失效。中间 minibatch 沿用原 proximal .01 与原梯度，不偷偷宣称连续保护，也不把旧刷新梯度检查迁移成新生产已通过。
4. head-only 投影 detach；无新增图像提取、无额外医学支持访问。旧类域外推广、旧旧风险、val 新类无损均不保证；同状态 OFF 不是独立训练 Core/LT 轨迹控制。

## 不靠扩大 cap 的全约束求解

对 native SPD 二次目标 E(W)，令 Wu 为无约束解，A 为所有 N(K−1) 点边界算子，统一乘 1/sqrt(N) 仅改善数值尺度。解非负全 dual：min .5<Atλ,H^-1 Atλ>−<A(Wref−Wu),λ>。每次仅一次聚合 native inverse；不缓存每约束逆方向、不构造 dK 平方 Hessian 或巨大三阶约束张量。所有 dual 变量从开始存在，active cap 为 null；L-BFGS 历史10对不是活跃约束上限。

采用环境已安装 SciPy，不安装依赖。maxiter2000/maxfun10000/maxls40/ftol1e-20/gtol1e-12 固定。每 case guard 软840秒、子进程硬900秒、总8100秒，仅资源停止标准。停止不等于收敛。独立全约束 slack≥−2e-8、stationarity≤1e-8、互补≤1e-7、绝对 gap≤1e-7(1+|E|) 全满足才 CERTIFIED。超时/非收敛/约束失败保留 UNRESOLVED 或缺失，不降约束、不补 identity/0、不调参数重试。

## 九个独立 CPU 子进程

小 native Core/LT 枚举 oracle、关闭精确、adjoint、排列/公共 logit、时点/frame 拒绝与 checkpoint 回归；两条精确旧 ISIC T3 capped fixture 用新 solver 求解（不是重跑旧 capped solver）；ISIC 最大阶段 epoch/boundary × Core/LT 四条；HK 最终 boundary × Core/LT 两条。全部大问题1536维，ISIC Nfit9434/K8，boundary原统计N11835；HK Nfit1038/K23，boundary统计N1298。按公开 train counts 合成单位样本 .8signal/.6noise，旧 affine(2.8,.65,1...) 固定；不冒充真实医学 fit 类分配。新快照 seed241+K，legacy 保留旧规则并断言原目标一致。

记录未加 ridge 的支持秩、全部 dual 规模、活跃增长、逆调用、slack/KKT/gap、相对 OFF 的判别变化和二次收益、合成救回/破坏、每独立进程驻留/user/system/peakRSS。合成收益不称医学贡献。失败和 timeout 全保留，pipeline/operation 包含于进程驻留不相加；旧 GPU 历史391490.24710153416秒不变。

## 解析头与 checkpoint 边界

full-bank native 的 proximal=0；旧 minibatch 对照保留 .01，不能混淆。全 bank competition/metric 使用各自 OFF/ON 原生统计，静态 PC 系数为0，保持原接口。原 PMT evaluator 强制无约束头重建相等，不能用于 guarded head。将来必须独立版本支持 force=Atλ/head/ref/frame/encoder 证书；重建验证 stationarity，不声称在不保存全支持时可独立重验 feasibility。当前只合成证书回归，无医学生产 checkpoint 接线，不是 GPU-ready。

本轮无正式训练预算/新 GPU 轨迹冻结。CPU 结果齐全后完整报告可行/identity/非 identity/未解/成本，再判断是否值得生产接线。
