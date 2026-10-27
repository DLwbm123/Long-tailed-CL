# R74 后证据与 R75 类别风险测度

R67–R74 全部单 HK74002 六 stage 完成。预定采样主候选均未稳定超共同 SFT；最终 old/new 两组均负，不只旧类遗忘。原型支持梯度合成反例不是现实失败原因证据，D50/R59 双边反馈历史负结果保留，不启动其复验。R68 先验、R69 支持坐标、R70/R71 guide 保护、R72 inverse、R73 reward、R74 文本伪样本均不叠加或调参救援。

当前直接 TRAIN 事实来自 R67 保存的全部 TAILNESS_TABLES，不访问新图像或 VAL。自然 query 下，固定 surrogate 权重给 class c 的先验质量为 n_c*w_c。六 stage TARS 尾类质量占比约18.24/24.66/28.15/30.59/32.50/37.21%，按类测度对应38.46/46.67/52.94/52.63/52.38/47.83%。最终各类质量最大/最小445.58；原误差为0的类仍因clamp_min1保留自然计数质量。这是实际保存权重的代数性质，不是已测逐类梯度或 old-policy p 分布，不证明唯一失败原因。

R71 只重加权 CE guide 来投影 AdamW proposal，原 GRPO surrogate 风险测度没变；其负结果不能视为对 surrogate 类别测度的直接检验。R72 按 old 正确概率逆系数补偿个体信号，保留原 TARS 类别测度；产生22402.95最大有效权重和更大KL后失败。本轮不补偿 h(p)，保留组标准化的全部期望偏置和采样方差；只检验类别风险质量。

单机制：原 TARS 与预先固定的均匀类别 surrogate 风险比较，匹配每stage自然query下总权重质量。M=sum(n*w)/N，候选 a_c=M*N/(C*n_c)。因此sum(n*a)=sum(n*w)，各类n*a均等。匹配M是控制总surrogate对KL/anchor尺度，不用把总体降权当类别风险收益。候选最大权重按六stage约121.26/114.10/102.98/115.06/127.26/173.54，来自现有counts和固定公式，不搜索cap/强度/gamma。

这是已知 importance/class reweighting 应用于现有 categorical GRPO，非新RL创新，也不是CE替代。按类等权的是归一化组 surrogate 的条件风险，不是宏 BA；h(p)和类内难度仍在。候选存在更大稀有行梯度方差和过拟合风险，尤其3个TRAIN样本与极小VAL类。KL和anchor仍按原自然query不加权，明确不是同时均衡整个正则化目标。若失败应完整保留，不调cap/LR/epoch/G救援。
