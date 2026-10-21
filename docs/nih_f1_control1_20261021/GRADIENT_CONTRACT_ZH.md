# 原 A 的实际 adapter 梯度契约

仅静态追踪封存41e6545源码；没有导入运行模型、合成autograd试验、checkpoint/图像/特征读取、拟合、矩解、优化或评分，无新实验时钟。继承既有调用链，不重做论文/兼容性/对照审查。适用于NIH base的实际分支，不将第三方源码其他分支当已运行机制。

## 固定状态与活动计算图

仅两adapter池 `.pool.pool.` / `.pool_few.pool.` 可训练。`encoder.eval()` 不关闭autograd；当前batch的 `z=encoder(x)` 仍可反传。特征为两支pre_logits拼接后的整体L2归一，梯度通过归一化回到adapter。

`module_additions.extract`有`@torch.no_grad()`，且经CPU NumPy重建float64 tensor，因此canonical before/current、基于其得到的共同shift、old bank、候选统计、R与pair权重均不连接当前参数计算图。teacher在任务开始复制并冻结；当前增强图的teacher输出在no_grad中计算。

每batch以`z.detach()`建立矩并在no_grad内求解析head，previous也来自已脱离计算图的head。head不是独立神经优化器参数；其数值会随当前特征、旧矩、R、竞争和proximal改变，但反向传播不经过求解器。

## 各项如何影响当前adapter

|项|本batch直接adapter梯度|实际作用|
|---|---|---|
|fit：当前加权平方误差|有，经z|使用当前解析head拟合seen空间one-hot|
|pair：当前样本平方竞争|有，经z|当前真类与其他seen列之差；使用固定pair权重|
|pair：历史矩部分|无|仍参与head求解，因此间接改变当前梯度方向|
|old_loss：旧类平方风险|无|对已求出的head计算标量；旧矩也进入head求解|
|ridge：R正则|无|R与head固定；正则在求解阶段影响head|
|proximal：head与previous差|无|求解器约束head变化；不是直接adapter参数距离惩罚|
|FD：当前z与任务开始teacher|有，经z|系数10，作用在当前训练图；不是旧图的经验FD|

AdamW的weight_decay=.01和梯度clip5还会影响参数更新，不能把损失导数直接当完整optimizer步。不存在base额外localCE或Specific分支的直接梯度。

## 精确到源码系数的条件表达

记当前batch归一化特征z_i，固定解析头列w_j，s_ij=z_i^T w_j，目标t_ij，权重alpha_i；令d_ij=w_yi−w_j、m_ij=s_i,yi−s_ij−1，pair矩阵a。

在本batch把head/历史状态视为常数，直接依赖z的目标是：

`L_direct = 1/2 sum_i alpha_i ||s_i−t_i||² + 1/4 sum_i alpha_i sum_j a_yi,j m_ij² + 10/B sum_i ||z_i−z_teacher,i||²`。

其中1/4来自runner外层`.5*pair_loss`与pair_loss内部`.5`，不是额外提出的系数。对应

`dL/dz_i = alpha_i W(s_i−t_i) + (alpha_i/2) sum_j a_yi,j m_ij d_ij + (20/B)(z_i−z_teacher,i)`。

参数梯度再乘归一化双支特征对adapter的Jacobian。历史pair常数部分及其他head项不在此显式导数中，但W本身已由旧信息决定。不能因此说旧记忆不保护更新，或以标量pair大小判断其直接梯度大小。

## 停止梯度的判读边界

这是交替求解/条件更新的实现，不能仅由`no_grad`断言应加入隐式求导。若某个内部头精确最小化与外层一致的完整条件目标，在满足相应条件时，包络定理可使完整最优值的一阶导数不需要显式头导数；此处仅源码契约分析，没有逐batch验证目标完全一致、精确驻点或包络条件。PCG小残差亦不等于adapter收敛。

旧共同shift在epoch边界和任务末由当前类估计，历史矩据此更新；它没有跨batch可导历史路径。任务开始teacher依赖当时状态，但训练本任务时视为固定，不能据当前类FD大小推定旧类真实漂移或运输误差。

## 准入结论与下一证据问题

已确定的功能边界是：原A有当前图的识别/竞争/FD直接梯度，以及旧充分统计通过头与R的间接复用；它不是旧类经验表示风险的端到端优化。此结论不提供“去掉历史项/增加旧梯度/调FD或beta”的性能依据，不复活已关闭响应、PMT、竞争或冻结时点网格。A/F1仍是组合行为控制，不能将其开发集差异唯一归因某项。

当前仍未形成可启动的新假说。下一步应只审查已有训练记录能否区分两条可证伪解释：（1）当前类判别方向确实改变；（2）仅历史矩运输或解析读出变化造成收益。现有字段缺少逐类梯度、旧类新表示误差与共同shift，因此不能从现有batch均值完成该区分。先形成单一候选的必要观测与合法状态清单，明确哪些仅当前阶段可取得、哪些历史不可恢复；不为填空重放旧图或重新运行原Task1。若候选只重述既有响应/运输/线性重参数化，则直接reject，不启动。

源定位：run_multilabel.py:24–52；ct3p_core.py:35–45；module_additions.py:79–98；run_module_additions.py:171–235、285–337；core1_competition.py:116–132；prototype_coherent.py:90–97、133–169。均为41e6545，机器可读摘要见GRADIENT_CONTRACT.json。原全部实验成本与失败不变。
