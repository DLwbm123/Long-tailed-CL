# R83：固定唯一图像的类别损失份额动作

已知 class-loss weighting 的动作可控性诊断，不是新 RL 算法或控制器效益试验。固定 R81 全六阶段学习器与 AdamW 状态，所有候选同一图像及顺序，仅第一步 CE 份额变化；奖励/LR/anchor/查询划分保持。0 控制器更新、0 新 VAL/TEST。

## 事前门与完整结果

极少类执行门：**True**；跨查询识别门：**False**；两者合取：**False**。未更改任何门。

| stage | 宏 vs uniform pp | 尾 vs uniform pp | 宏 vs fixedmean pp | 尾 vs fixedmean pp |
|---|---:|---:|---:|---:|
| 0 | -0.172844 | -0.194126 | -0.355439 | -0.419086 |
| 1 | -0.120866 | -0.010629 | -0.214155 | +0.000000 |
| 2 | -0.028831 | +0.080081 | -0.015521 | +0.044092 |
| 3 | -0.098394 | -0.108470 | -0.030250 | +0.015983 |
| 4 | +0.149095 | +0.331262 | -0.019219 | +0.040650 |
| 5 | +0.058914 | +0.013702 | +0.106821 | +0.135630 |

| 查询方向 / 指标 | vs uniform pp | vs fixedmean pp | vs action0 pp |
|---|---:|---:|---:|
| forward BA | -0.035487 | -0.087960 | -0.086572 |
| forward tail_BA | +0.018636 | -0.030455 | +0.022455 |
| reverse BA | -0.129656 | -0.285583 | -0.310013 |
| reverse tail_BA | -0.457432 | -0.837229 | -0.854014 |

事前识别门要求：四个全局宏/尾对uniform与fixedmean增量均正，至少5/6stage四项同时正，且至少9/12stage-fold有2/3随机流tau>0。实际四项同时正stage数 1/6，排名strata 12/12。反向只作固定敏感性，不选更好方向。

## 极少类与全部类别响应

| 类 | 参数越数值门 / 比较数 | stage内至少4/6越门的stage数 | 优先时自身第一步CE质量均值 | 自身风险改善/损害/未越门次数 |
|---|---:|---:|---:|---:|
| 1 | 36/36 | 6/6 | 0.254395 | 7/28/1 |
| 2 | 36/36 | 6/6 | 0.258789 | 15/21/0 |
| 12 | 36/36 | 6/6 | 0.258789 | 1/35/0 |

全部非零类别动作比较 648；参数越数值门 648，风险越数值门 648。duplicate 参数L2最大 0，类 Brier 差最大 0。全部23类方向/份额计数见 PRIORITY_CLASS_RESPONSES 与 AFFECTED_CLASS_RESPONSE_COUNTS；不公开Q逐类风险幅值。

Qa/Qb原protected-Brier utility tau-b均值 0.548266、中位 0.568694；定义 36/36，正 36/36。这些是同端点TRAIN敏感性，不是临床效益。

所有动作的完整曝光向量差均为0；改变的是CE质量，不是图像数。rho=.25固定，无weight/奖励/LR搜索。极少类即使参数可控，也没有新增独立信息，仍可能强化对少量样本的拟合。loss质量等同不等于梯度或AdamW作用等同。

## 固定平均权重对照与归因

uniform仅含唯一action0+类别动作，duplicate与fixedmean排除。平均动作loss/gradient等于固定meanweight目标已用CPU证明；非线性AdamW端点不因此相等。完整 FIXED_MEAN_CONTRASTS 保留fullQ/Qa/Qb下固定mean与uniform相对action0的差异；若固定mean本身获益，不能归给策略选择。OLD_NEW_HEAD_TAIL保留QB相同eligible类集的旧头/旧尾/新头/新尾分解，空组null，不删小类改门。

## 工程核验、覆盖与费用

全部36blocks/756branches/36fixedmean controls/2268真实视觉optimizer/9072micro/0controller/0VAL/0TEST封存。saved verify证明固定唯一batch、CE类质量、源delta+完整AdamW、步数和有限端点；补充检查prototype与保存支持、完整Q/slice计数恒等式及保护效用。结案0新模型评分/优化。

第一次CPU软件自检发现fixedmean标量除整数计数隐式FP32再加到FP64权重，质量和断言失败；在native/视觉update/VAL之前修成FP64计数，公式/rho/容差/科学门不改。原失败源码、日志、退出/费用和零视觉更新审计保留，全部原PID退出后经新现场准入启动attempt2；没有partial optimizer恢复或时钟重置。两次各3个CPU toy optimizer，共6，不能当视觉更新。详见ENGINEERING_NOTES。

原钟至退出 5544.233974s（1.540065h）；两attempt workerCPU总计 7547.695717s、workerwall总计 5275.364188s。workerRSS峰 6645833728B/group独立采样峰 6668861440B，disk新增峰 8885428619B；前置设置/间隔仍计入原钟，完整结案CPU未独立采样不是0。

6821行FP32prefix RAM 4127960064B/parity 2.23517418e-07，root/remote spill0，只清再生prefix。全部学习support/features/delta/AdamW/起点和统计均保护，未复制大权重或碰他人进程。

class1在fold1仍缺双component查询覆盖，fullQ保留该类、Qa/Qb同eligible类集缺该类；不能把其他类结果外推为它的奖励可靠性。原SFT/其他context已见TRAIN，不是独立患者/holdout；36固定状态块/3共同流不是3新训练seed，756端点不是独立重复。固定状态单动作不覆盖原两动作交互或长程策略收益。

## 科学结论边界

本报告按两道预定门分别解释动作可控性和识别选择效益；不能仅因动作有效/奖励非零而启动更强策略优化。原R80最终SFT联合门失败、D>C、R81未胜uniform、R82识别门失败均保持。未新增VAL/TEST，不据本结果改主候选/删小类/搜索系数。若混合或不稳定，保留不足而不承诺GRPO成功。

公开仅自有source/protocol、安全宏统计/类别方向次数/曝光与质量/cost/report。图像身份、逐例数组、Q逐类Brier/margin、候选权重/支持features、原日志和私有配置路径不公开。

## 冻结选择器的奖励—识别分解（描述性，不新增 selector）

同一原protected-Brier选择器，在Qa选优→Qb时全部36块的utility相对uniform均为正，平均+0.0021416854；其中宏Brier平均变化-0.0008607529、保护损害变化-0.0012809326，恒等式U=-M-D可复算。但这36块中21块宏BA下降、18块尾BA下降。反向utility平均+0.0013670319，31块正，其中12块宏BA下降、14块尾BA下降。它表明风险代理的可转移优化价值不等于识别选择价值，并非所有奖励都纯噪声；不证明Brier数学错误或唯一失败因果。未试新selector、未扫系数、0新模型评分/优化。

class1优先动作自身风险改善/损害/未越数值门为7/28/1，class2为15/21/0，class12为1/35/0。提高训练损失份额和参数可控性没有保证查询风险改善；这与小样本信息不足/过拟合相容，但不是对单一原因的确认。不能再通过rho/LR/更新次数搜索把本轮救成正结果。
