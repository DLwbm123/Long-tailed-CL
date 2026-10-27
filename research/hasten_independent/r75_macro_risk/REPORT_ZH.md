# R75：匹配总质量的类别风险重加权，局部尾类正信号与整体负结果

本轮只改变原 categorical GRPO surrogate 的固定类别权重形状。原 TARS 自然 query 下最终尾类占47.826%类别，但仅37.2085%加权质量；这不是已证失败原因。用本stage TRAIN计数与已保存SFT错误表固定 w，令 M=sum(n*w)/N，候选 a_c=M*N/(C*n_c)。每类 n_c*a_c 相等，总 sum(n*a)=sum(n*w)，避免把整体surrogate降权误作类别风险改善。它是已知 importance/class reweighting，不是新 RL 创新、加权CE替代或宏BA保证。类内组标准化 h(p)和采样方差保持，KL及anchor仍自然平均、不乘类别权重。

四个分支全部HK74002六stage，stagewise从同R67 SFT独立开始、不carry后训练delta，无新SFT。原binary G8/popstd+1e-8/clip.2/KL.04/anchor.01、2epochAdamW.001wd.001cosine2/batch256micro64、natural20cos+logcount策略、raw20cos最终推理、dynamic oppositefold detach支持及old/ref冻结均保持。没有文本融合、近邻reward、inverse、guideprojection或全支持梯度，没有短pilot/VALscreen/参数救援。

|方法，百分比|阶段平均BA|最终BA|最终尾类BA|
|---|---:|---:|---:|
|H0|55.630100|56.454053|36.857416|
|共同SFT|66.840041|64.876782|44.924938|
|tars_grpo|66.115954|63.655574|44.737644|
|macro_grpo|65.411597|62.741784|47.522341|
|tars_exact|65.456019|63.954743|44.161723|
|macro_exact|64.695657|62.808625|42.366890|

预定主候选 macro_grpo 对原 tars_grpo 的平均/最终/尾类差值为 -0.704357/-0.913790/+2.784697 个百分点；对共同SFT为 -1.428444/-2.134998/+2.597403 个百分点。尾类局部上升，但阶段平均和最终下降，不是整体获益，不以H0差距代替SFT增量。

同目标 macro_exact 对 tars_exact 差值 -0.760362/-1.146117/-1.794833 个百分点，对SFT -2.144384/-2.068157/-2.558048，同样不支持整体改善。两个原TARS控制重现旧同seed指标，不是新增独立seed复现；不后改精确分支为主候选。

最终宏尾类对SFT净增益为 +2.597403pp。逐class正确计数分解显示：class_index15 的VAL分母7，从SFT的4个正确变为6个正确，贡献全部净增益 +2.597403pp；另两个分母21的尾类各少/多1个正确，贡献 -0.432900/+0.432900pp 相互抵消，其余尾类相对SFT不变。因此不能称广泛尾类改善、独立患者获益或稳定医学优势。完整全class分母及正确计数在 FINAL_CLASS_DECOMPOSITION.json，未删小分母类或修改门。

|分支|真实TRAIN平均KL|零优势组率|采样正确动作率|clip比例|查询平均固定权重|
|---|---:|---:|---:|---:|---:|
|tars_grpo|0.046914|74.7578%|89.5722%|5.3905%|1.426089177|
|macro_grpo|0.181018|72.5588%|88.2290%|7.7789%|1.426089177|
|tars_exact|0.019433|不适用：精确积分|不适用：精确积分|不适用：精确积分|1.426089177|
|macro_exact|0.031845|不适用：精确积分|不适用：精确积分|不适用：精确积分|1.426089177|

四分支查询平均固定权重均1.426089177，逐stage总质量及宏分支各类质量恒等式全部核对；不是新权重没有应用。宏采样KL为0.181018、原采样0.046914，clip也升高；匹配平均总权重并未匹配梯度尺度/方差或参数偏移。更少零优势组同样不等于更好效果。没有逐类old-policy概率或梯度方差测量，不能用这些聚合记录证明唯一因果；精确arm零采样计数不能解释为零优势率0%。

最终stage TRAIN跨fold宏/尾BA从共同SFT起点88.505737/83.220751降至macro_grpo的76.188190/67.011612；原tars_grpo为81.596113/72.194357。macro_exact为85.266472/79.614736，tars_exact为86.867615/80.691374。跨foldTRAIN曾用于优化，不是独立holdout，也不是CE风险，不能把这些曲线直接视为泛化或逐例因果证据。全部stage的0/1/2epoch曲线保留。

saved-only verify→analyze通过944optimizer/3696原目标micro backward/24stage-arm分支/72curve/30真实VAL读数(24新+6共同SFT)。全部24head封存后才任何VAL，额外saved_diagnostics仅做计数恒等式，0新模型评分/优化。原参数冻结49152LoRA、strictrestore/fold/oldref、原TARS表与候选权重公式/总质量/queryweightsum、未缩放KL/raw推理/seal/分母/cleanup均核。

首次结案verify因新增权重检查中的计数向量n覆盖总行数n，发生Tensor布尔歧义；改名nvec后完整核验通过。修复仅涉及保存结果核验，未改训练输出、重训、更新optimizer或重置时钟。VERIFY_REPAIR_AUDIT.json保留错误与修复边界。

控制器记录原钟总历时2650.704041秒（44.18分钟），worker及后代CPU3622.561488秒，workerRSS峰7450222592B，独立采样group+controller峰7418015744B，新增实际磁盘峰55081294B。两个RSS峰的采样时点不同，不伪造同刻大小顺序。此费用是原run记录，不冒称覆盖后续人工分析/发布所有CPU。

仅my-gpu A100物理0/1，30GB总RAM、每GPUreserved<12GiB及原20h时钟未重置。rootcap0/noOSswap；root和remote prefixspill实际0，FP32parity最大2.2352e-7、8519rows，声明再生prefix已清，所有学习产物保留。只读复用历史SFT1180/4620和原模型权重，不计新增优化；不恢复用户停止R65。

单HK单seed反复开发数据，少数VAL类仅1或2个样本，不能三域BOTH/3seed/TEST/独立患者或临床确认。保留尾类局部正信号，不将其替换原平均/最终结论，不继续扫权重cap/强度/LR/G/epoch救援。原TARS与类别测度重加权的完整对照为后续表示学习/风险诊断提供约束，不承诺必须让GRPO赢。

公开自有source/protocol/完整stage-class计数/全部曲线/权重质量表/诊断/费用/正负报告；不公开图像、身份、逐例数组、token/features、模型权重、原日志、私有配置路径或第三方全文。
