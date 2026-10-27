# D76：完整六阶段 TRAIN 端点表示诊断

本轮为描述性测量，不是新方法或性能筛选试验。固定 HK74002，六阶段各对既有SFT、原TARS、R75宏类别采样三个端点编码所有已到达TRAIN行。完整18端点、87651行模型TRAIN嵌入，复用同嵌入计算36条raw/natural TRAIN统计路线；0optimizer、0backward、0VAL/TEST。两张真实TRAIN图像预检只是原始embedding parity/strictrestore工程核对，不构成效果证据。

所有类别已参与端点训练，相反metadata component fold原型避免直接使用同组件支持，但不是独立holdout或患者独立保证。每个端点完整TRAIN原型与原saved head的parity<=2e-5，原参数及49152LoRA delta完全不变，grad=None。私有class sums/平方范数/fold sums支持saved-only恒等式复核；原图、逐例p/features和模型权重均不公开。

|最终stage端点|类等权均值长度|无偏类内variance trace|最近其他centroid cosine|raw TRAIN宏BA|raw TRAIN尾BA|natural TRAIN宏BA|natural TRAIN尾BA|
|---|---:|---:|---:|---:|---:|---:|---:|
|sft_reference|0.765674|0.424963|0.738393|88.505737|83.220751|77.369695|55.190105|
|tars_grpo|0.777997|0.405696|0.775308|81.596113|72.194357|70.275669|42.422365|
|macro_grpo|0.787226|0.386355|0.793313|76.188190|67.011612|67.406364|41.758410|

最终stage后训练并非简单地增加类内离散：原TARS和宏类别分支的类等权within trace均比SFT低，均值长度更高，同时最近其他类centroid更相似，raw TRAIN宏/尾正确率却下降。因而“类内方差变大”不是所有阶段失败的统一解释；更紧的类簇也不自动表示类别可分或泛化更好。不能从这些端点统计识别因果，尚未测实际参数梯度方差或连续更新轨迹。

阶段0–4的两个后训练端点，类等权within trace都比各自SFT高，最近其他centroid cosine都比SFT低；最终stage方向反转。不能挑一阶段或只报某几何指标。全部18端点、各class计数与安全标量见结果JSON。本轮没有读取新的VAL，既有VAL结论仍以R67–R75原报告为准，TRAIN差值不能代替它们。

natural策略加入logTRAINcount，在所有18端点其TRAIN宏/尾BA均低于相同端点raw20cos路线；这只是同一表示下已声明两种计分方式的TRAIN描述。R68移除先验的完整训练对照已未获支持，因此不能用此诊断复活先验搜索或声称验证收益。最终SFT/TARS/宏分支的raw正确类概率按类均值分别约0.782001/0.721219/0.645945；natural约0.706622/0.644528/0.606095，不是历史oldpolicy概率分布。

h(p)仅将端点概率代入current=old、clip未激活的binaryG8假设快照系数；它不是实际训练旧策略、参数梯度或方差测量。h(1)有限趋于sqrt7而CE方向本身趋于0，不能将h(1)误记0。没有根据这些诊断训练或筛除新候选。

软件检查通过variance/paircos恒等式、fold计数、排列不变性与概率/h边界；native两TRAIN图像parity2.1234e-7/strictrestore/finite通过。全部18端点saved-onlyverify→analyze通过，结案0新增模型评分/优化。私人moments向量仅留source科学产物；公开时移除固定分位数数组，避免极小类别分位数退化成逐例概率，保留class聚合均值、正确数、几何标量与完整分母。

控制器原钟412.158700秒（6.87分钟），worker及后代CPU543.208724秒，workerRSS峰6286983168B，独立采样group+controller峰6271127552B，新增实际磁盘峰40871292B。原20h时钟未重置；排队等待在实际运行前，不计入该run钟。原钟不冒称覆盖后续全部分析/发布CPU。

仅物理A1000端点推理/1冻结prefix，RAM总30GB/GPU各reserved<12GiB；rootcap0/noOSswap。声明再生prefix已清，prefix编码6821个TRAIN行、parity最大1.6578e-7，root/remote spill均0；全部科学产物及历史权重保留。正式87651行端点计数不含native复算或冻结prefix工程前向，相关开销计入控制器费用。共享R67SFT/R75savedheads只读，无复制大模型、修改旧结果或恢复R65。

本轮补齐表示层面的证据，不能称新方法成功、三seed/三域BOTH/TEST或临床证据。现阶段不据几何表直接叠加新的聚类/间隔/保护项，也不通过先验、权重或epoch搜索救援；后续方法仍需独立的TRAIN依据、可证伪假设、匹配对照和完整冻结实验。

访问范围补充：沿用的数据加载器会读取既有完整TRAIN manifest的标签、组件与固定顺序/计数元数据，再按已到达stage构造输入；因此不声称未来类manifest元数据完全不可见。编码器输入、支持与原型仅包含当前已到达类，未加载未来类图像/features/端点head，未用未来类元数据选择方法或更新参数。原冻结协议与代码保留，该披露界定其中“无未来类输入”的实际实现范围。
