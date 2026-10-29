# R80：长尾持续学习训练分配 GRPO 完整结果

**结论：存在相对固定均衡分配的条件正增量，但未通过原 SFT 最终性能保留门，也没有建立正确奖励归属的 GRPO 独立贡献。** 主候选始终是 protected_controller，不将结果更好的打乱奖励对照改成主候选。

| 分支 | 六阶段平均BA | 最终BA | 最终尾类BA |
|---|---:|---:|---:|
| balanced_short | 64.443502 | 60.951858 | 42.779034 |
| macro_controller | 65.130314 | 61.740939 | 45.885454 |
| protected_controller | 65.071594 | 61.768916 | 46.107184 |
| shuffled_controller | 65.046855 | 62.185713 | 47.086129 |
| balanced_long | 60.457244 | 60.441780 | 36.049304 |
| frozen_initial | 62.141098 | 57.205089 | 37.661690 |
| historical_stagewise_sft | 66.840041 | 64.876782 | 44.924938 |

## 预定比较与结论

| 主候选减去对照 | 平均差pp | 最终差pp | 尾类差pp |
|---|---:|---:|---:|
| balanced_short | +0.628092 | +0.817059 | +3.328150 |
| macro_controller | -0.058721 | +0.027977 | +0.221729 |
| shuffled_controller | +0.024739 | -0.416797 | -0.978945 |
| balanced_long | +4.614349 | +1.327137 | +10.057880 |
| frozen_initial | +2.930496 | +4.563828 | +8.445494 |
| historical_stagewise_sft | -1.768447 | -3.107865 | +1.182245 |

- 对 balanced_short：平均/最终/尾类同向增加，但这是训练分配方案的整体比较，不能单凭它归功于控制器学到了正确的奖励与动作关系。
- 对 macro_controller：最终仅 +0.027977pp、尾类 +0.221729pp，阶段平均反而 -0.058721pp。损害惩罚的尾类净收益全部来自 class9 的 VAL41 中多正确1张，不能称广泛尾类保护。
- 对 shuffled_controller：最终 -0.416797pp、尾类 -0.978945pp，未显示正确奖励归属优于固定轮转错误归属。该对照不支持当前学习策略的独立收益主张。
- 对 balanced_long：最终 +1.327137pp、尾类 +10.057880pp；该控制在固定共享数据流上反复花费额外监督更新，也可能受重复曝光/优化轨迹影响。不能据此声称 GRPO 普遍优于所有同计算监督学习。
- 对历史原SFT：平均 -1.768447pp、最终 -3.107865pp、尾类 +1.182245pp，原最终/尾类联合保留门失败。原SFT各stage独立训练，与本轮stage0初始化后carry协议不同，所以它是强历史参照而不是本轮机制的因果控制。
- 对同初始化冻结编码器：三项正增量；该比较同时包含监督更新、训练分配和控制器，不单独识别GRPO。

## 小分母贡献与全类别分母

所有23类保留，尾类固定为 TRAIN<250 的11类。`results/ALL_CLASS_COUNTS.json` 保留七个方案全部stage的正确数/分母；`CLASS_CONTRIBUTIONS.json` 给出主候选减各对照的每类贡献，不删除 VAL1/2 的类别。

主候选对 balanced_short 的尾类 +3.328150pp，变化为：class9 VAL41 多2张 +0.443459pp；class12 VAL2 多1张 +4.545455pp；class14 VAL21 多1张 +0.432900pp；class16 VAL5 少1张 -1.818182pp；class18 VAL33 少1张 -0.275482pp，其余尾类不变。class12 的一张图贡献超过全部净增量。

作为敏感性说明，扣除 class12 的正贡献后，对 balanced_short 的尾类净差变为 -1.217304pp、最终宏BA净差 -1.356854pp；对原SFT的尾类净差变为 -3.363209pp。这不改变正式指标/分母/门，只说明收益集中，不能宣称广泛尾类泛化。

最终主候选旧头类/旧尾类/新头类 BA 为 74.892090/46.107184/82.292577；最终阶段没有新尾类，该项为null而不是0。原SFT对应 82.043453/44.924938/88.778569；头类的旧类和新类均下降，不能把损失唯一解释为旧类遗忘。全部stage四组结果在 ANALYSIS.json。

## 方法实际执行

监督视觉学习器：原 BiomedCLIP 最后两层Q/V LoRA r8 alpha8，49152参数，原图unit512与原型raw20cos推理。每个分支只从R67 stage0 SFT初始化一次，此后六stage承接视觉delta、AdamW、控制器及其AdamW状态；不加载后续stage SFT，不加入kernel、翻转、文本、TARS或新视觉正则。监督损失 CE(20cos+log完整已到达TRAIN计数)+.01原始Biomed表示anchor。视觉AdamW .001/wd.001，stage前4context用.001、后4用.0005，阶段之间moments保留。

控制器是共享8→64→64 tanh类别编码器加set-mean pooling，无类别ID embedding。输入含计数、可用支持量、类别年龄/新类、TRAIN Q Brier和margin及其同fold变化、近期保留曝光。动作0均衡填充，动作c优先类别c。每context从同视觉/AdamW起点运行8个候选+1均衡参考，每轨迹3次监督更新；前两次192自然共同图像+64控制名额，最后一次共同256图像。batch内图像不重复；小类不足64时使用均衡唯一图像补齐。

Q/S是原metadata component相反fold，Q不参与当前inner梯度或prototype，所有类别两侧非空；Q曾被初始SFT使用，也可能出现在别的context的S中。Q是反复使用的控制器TRAIN反馈，不是独立验证集，不保证患者独立。每条轨迹更新后用自身encoder重新编码S建原型，再在全Q上评价。

奖励是负类别平均Brier减旧类∪尾类正损害的均值，两个系数固定1；旧类anchor用前stage最终模型，新tail用本context起点。均衡参考的共同baseline在组内标准化时抵消。控制器使用G8/populationstd+1e-8、clip[.8,1.2]、对合法动作均匀参考的KL .04、AdamW .0003/wd0、每context4次更新。该KL属于控制器行为约束，不是视觉模型知识保持。

保留轨迹index=(stage*8+context)%8提前固定；没有按reward挑赢家。balanced_short同样做9条均衡分支但只保留固定一条；balanced_long用同数实际更新依次保留所有9条。B仅宏奖励，C宏奖励+损害，D只把8个reward轮转一位再分配优势。详细冻结规则见 PROTOCOL.md。

## 控制器、绝对回报与实际分配

三个GRPO控制器均实际更新192次，总576次；主控制器参数相对初值L2变化1.802900，平均均匀参考KL .014083。主候选48个context奖励population std均非0，均值 .00138438；四次小更新的importance clipping比例为0，不能声称clip实际介入。参数变化/非零奖励不等于效果。

主候选48次固定保留中有17次相对同起点均衡参考绝对回报为负，其中2次组内优势为正。没有一个context是8个候选全部负回报；“相对好”仍不等于绝对改善。各context安全聚合在 REWARD_AND_EXPOSURE_CURVES.json。

耦合随机数下，主候选与宏奖励控制的768个采样动作有740个相同(96.354167%)，与打乱奖励控制有621个相同(80.859375%)；这是动作一致性描述，不是独立重复或对原因的证明。

主候选实际保留曝光的尾类比例13.522678%，低于均衡控制15.087891%。主候选342次优先尾类动作平均只占22.766082个优先名额，401次优先头类动作每次都可占满64；受唯一图像可用量影响，动作的实际强度不对称。不能把尾类动作次数解释成相同强度的尾类训练，不能用 n*w 或曝光次数替代实测参数梯度。完整类别曝光和名额影响聚合均保留。

## 工程核验、时钟与费用

CPU唯一分配/合法动作/小轨迹exact policy-gradient/归一化baseline抵消/轮转检查通过。真实到达TRAIN2张图像native检查无optimizer/VAL，embedding最大误差2.12341547e-7≤2e-5；49152、冻结、有限梯度和严格恢复通过。

全训练：6480次实际视觉optimizer/25920micro；保留1872次(A/B/C/D各144，E1296)，240 contexts/2160奖励端点；30训练head+6冻结参考head全部封存后36次新VAL，0TEST。原SFT6端点仅复用封存计数。saved-only核验通过，包括真正跨stage carry模型/视觉优化器/控制器/控制器优化器；结案0新模型评分/0新优化。

独立原钟训练与清理总历时 23284.564257s（6.4679小时），worker进程组CPU 32860.212212s；训练worker RSS峰 7330750464B、含controller进程组采样峰 7353974784B。训练guard采样新增磁盘峰 857765523B，root/remote前缀spill均0。原钟20h未重置，物理A1000训练/1冻结前缀，GPU2/3和其他进程未使用/改动。

guard退出费用是训练进程组范围；saved-only结案核验与分析另有CPU费用，未独立采样全程峰值/完整CPU，不能记为0或将guard峰声称为全部结案峰。原始deadline继续约束结案；闭合回执记录含交付的原钟elapsed。ALLOCATION_INFLUENCE.json包含这项小型saved审查的单独wall/CPU。

同数inner backward和S/Q编码遍历不等于严格同GPU秒：首臂承担冷前缀建立成本，控制器、anchor和封存读出也有额外开销。按stage实际wall记录在COST.json，不能宣称与历史标签GRPO等FLOPs或精确等运行时间。全部学习支持、视觉/控制器/优化器head保留为私有科学产物，仅清声明再生prefix。

核验修复：原均衡分支Brier逐位相等断言在两个context失败，最大差2.38418579e-7；曝光、分配、三步损失和类别正确数完全相同。改为atol1e-6、rtol0，并同时核上述计数/损失；FP32差异来源未唯一定位。该修复仅saved-verifier，不重训/评分、不改科学门。详见VERIFICATION_NOTES.json。

## 科学边界与后续决策

本轮闭合为“有限条件正增量、主门失败、奖励归属贡献未成立”。不把随机/打乱对照的收益改写为GRPO成功，不以额外监督控制较差包装普遍计算优势。单HK74002、相同VAL反复开发；新seed也不自动消除开发适应性或创造独立患者。

当前不升级为已验证主线，不进行LR/G/epoch/温度/权重/读出救援，也不启动R79。下一步首先依据已保存TRAIN检查动作映射的可辨识性和正确奖励相对无学习随机分配的必要对照，避免再开只改奖励细节的一轮。没有据本轮VAL挑类别对、阶段或赢家。持续研究授权保留，任何后继必须另冻单一假设、匹配对照和预算。

训练分配/RL回放有已知先例，当前实现不是仅凭使用GRPO即可声称新算法：[Replay Scheduling](https://arxiv.org/abs/2209.08660)、[Adaptive Memory Replay](https://arxiv.org/abs/2404.12526)。设计来源为用户转交的[Pro讨论](https://chatgpt.com/share/6ae2da47-b768-83ea-93ad-8712e6af1fff)，本报告为本项目实际实现和封存证据的独立总结。

## 复现与公开范围

公开自有source/protocol、完整class-stage计数、TRAIN正确数曲线、类别曝光/宏reward聚合、费用、修复说明。没有图像、身份信息、逐例输出、概率向量、support features、权重、private配置路径或第三方全文。没有公开逐类Q Brier/margin，避免极小类均值退化为逐例模型输出。

运行需合法取得原始资产、原manifest split/metadata component、冻结BiomedCLIP依赖和R67初始SFT；不在此复制或重新下载。PUBLIC_CONFIG_SCHEMA.json是匿名路径示例与固定科学配置；把源码通过中性入口映射到 /run/experiment，RUN_CONFIG指向私有完整配置。先check_decision.py→preflight.py→train.py，在守护进程原钟下执行；结束后verify.py→analyze.py→closeout.py→audit_allocation.py。不要把示例路径当真实服务器地址，私有图像/权重/运行环境未随仓库发布。

## 结案后 TRAIN 元数据动作审查（0模型评分/0优化）

使用相同S/Q component划分、common stream、合法动作mask和采样uniform重建固定均匀控制器的动作，未训练该反事实模型。主控制器768次动作与固定均匀策略591次相同(76.953125%)，实际保留轨迹96次决策中78次相同(81.25%)；打乱奖励控制分别712/768(92.708333%)和93/96(96.875%)。这说明策略确实偏离初始均匀行为，但无法从动作重合率推断固定均匀控制器的训练性能。下一项必要机制控制是固定均匀类别动作，不应直接把C相对固定均衡A的收益当作已学会的策略收益。本项wall16.096643s/CPU23.160358s，以saved TRAIN元数据计算，无新图像/模型前向/VAL/优化。
