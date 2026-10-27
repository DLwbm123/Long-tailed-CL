# D76：端点 TRAIN 表示诊断的必要性与边界

R67–R75 九轮完整单seed开发实验没有稳定采样GRPO增量。R75尾类对SFT净+2.597403pp来自一个VAL7类别正确4→6，其余尾类变化抵消；平均/最终下降，exact也不支持。不能以此挑某个类别/分支或承诺下一项改动一定有效。

现有保存结果包含正确计数、训练KL、奖励/clip、固定权重和单位化最终原型，但没有每类适配前后完整 TRAIN 表示的类内离散程度，也没有按类 endpoint 正确概率。不能由单位化原型角度、聚合hard accuracy或KL推断这些未保存量。R74文本混合只改原型，D50/R59双边反馈及支持梯度合成反例也不补这些观测。因此本轮只补全测量，不再凭想象启动性能训练。

D76覆盖固定HK74002全部6stage、每stage同一组已封存SFT/originalTARS/R75macro端点，完整已到达TRAIN所有行；没有新候选优化、额外epoch、VAL/TEST/未来类输入。每个端点只前向一次得到同一组512维单位嵌入，在RAM计算class sums/squarednorm/fold sums与完整原型。单位化原型parity核对既有saved head。原参数及LoRA delta、grad=None保持，根本不实例化optimizer。

主测量是类内均值长度、无偏类内方差trace、不同样本平均paircos、最近其他class centroidcos；以相反metadata component fold原型计算每个真实query的raw/natural正确数、margin和正确概率分布，再只保存按class汇总。所有行曾参与端点训练，非独立holdout；component也非患者保证。全类均值几何是描述性统计，不能证明临床形态、因果遗忘或可用泛化增益。

附带h(p)只把端点当前概率代入原binaryG8快照解析式，是假设current=old且clip未激活的局部系数；不是历史训练的oldpolicy p或已发生梯度轨迹，不能与未保存的逐类TRAIN KL/梯度方差混同。h(1)有限趋于sqrt7，而CE方向此时为0；不错误声称h(1)=0。

不设性能采用门、梯度方向门、候选淘汰或“最佳阶段”。全部18端点/87651实际TRAIN嵌入行/36raw-natural TRAIN计数路线均做完后分析，保留全部class分母与正负变化。若看到变化，只能形成进一步可证伪问题，不能宣称原因唯一或新方法成功。不进行参数/类别/seed/epoch搜索；原R65停止、CLS+patch暂停、D13defer及其他旧闭合结果保持。
