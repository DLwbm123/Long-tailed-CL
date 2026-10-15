# 全 fit 规模守卫与参考头刷新：前瞻 CPU 协议

只做独立合成 CPU，不启动 GPU，不读医学图像、医学 checkpoint、val/test，也不新增旧训练图像访问。旧已交付轮不 poll/collect/重评/重交付。复用 native pointwise 设计及原 Core/LT full-moment competition/proximal/metric/Woodbury，不改原生产调用。

## 固定覆盖

按公开 PMT PROTOCOL 的全部阶段，ISIC fit N=3177/2292/9434，classes=4/6/8；HK N=3133/93/38/610/544/1038，classes=13/15/17/19/21/23。每阶段 Core/LT 各一，共18个独立规模 fixture，维数1536；旧类数按原 order/task_sizes，旧合成样本计数取公开 train_count。当前类的每类 fit 个数按公开 train_count 比例最大余数分配，不冒充真实医学类级 fit 划分。只使用公开 train 聚合，禁止用 val 混淆/指标选点或类。

样本生成=normalize(.8*固定类别坐标+.6*单位高斯噪声)，合成 seed=227+100K+task。新旧原始样本单位归一化，旧统计按固定 diagonal(2.8,.65,1,...) raw affine 更新，当前新类不运输。a/previous 与各自 R 在求导块内固定。旧 bank 的 Q 从该类完整合成训练数量构造，不以每类4点假装常见旧类满矩；保留 HK 3/6等稀少类的低秩事实。全 fit 约束扫描全部点，不以 batch64 替代。目标仍是原 batch64/min(N,64) native 目标，alpha=N/(K*B*n_fit_class)。参考为同状态、相同 batch 的未运输旧统计 OFF native 头；首 task 无旧类时是零运输 identity 控制。

通过未加 ridge 的小 Gram 特征值检查全支持秩；rank阈值仅机器精度。执行完整约束分离，记录每个 atom 的违约量/活跃数/二次目标/驻留。沿用64 atom cap，不为通过而放宽约束或调强度。只优化缓存与 Gram 构造，不改数学准入；额外的有效目标收益需高于1e-12、保留>1%无约束收益且非公共变化>1e-10，该合成门不是医学方法效果门。

## 失败与界

每 case 硬墙钟360秒、guard软240秒、整体7200秒，2 CPU threads；这些是本次 CPU 探针的资源停止条件，不是正式训练预算或 GPU 速度结论。cap/超时/数值失败不填 identity 或0收益。若64条仅能得到松弛问题的最优解，其剩余目标收益是完整问题可达收益的上界；未满足全点约束的 candidate 禁止当成守卫输出。stationarity/dual/互补失败则不能使用该界。

报告实际每子进程 wall/user/system CPU/ru_maxrss（Linux KiB转MiB）、pipeline及全部工程失败；区间包含关系不重复相加。合成数据构造、rank、reference、guard等CPU时间与医学 encoder/GPU开销不同，不能外推训练 ETA。没有医学效果证据。

## 刷新梯度

Wref=H0(x)^−1B0(x) 随当前特征变化，a/R/previous/旧矩块内固定。完整 envelope=E_x−λ g_x，此外必须加 reference VJP；S=A*λ，Z=H0^−1S，增补 <Z,dB0−dH0 Wref>，不对 PCG 迭代图反传。对照 naive、只包含移动约束却 detach Wref、以及正确 implicit VJP。

Core/LT 的6×3活跃约束例重解最优值有限差分；固定 .3 stress offset只保证活跃面，导数0，非医学OFF证据。另1536×23原生 reference VJP 独立有限差分。若全规模约束求解失败，不得把独立reference VJP PASS写成完整全fit constrained刷新梯度PASS。若未来同时刷新 a/R/previous 或支持提取时序改变，本检查不覆盖，需重验，不偷换全fit为minibatch守卫。

## 后续准入

仅根据前述固定检查报告可行非identity空间、原目标保留、支持边界与资源，不靠全头范数或单例新类正确数宣称贡献。全部正负/identity/超限保留，未观察旧域外推与旧旧错误风险仍不可识别。医学生产/封存/全参数回归/实际预检未接，不冻结新训练轨迹数、不启动GPU。后续主四格和唯一必要无守卫运输对照仍须独立正式协议、公开源码、完整CPU和真实零更新预检。
