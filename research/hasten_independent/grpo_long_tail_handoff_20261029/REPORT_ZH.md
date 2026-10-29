# 给 Pro 的长尾 GRPO 设计交接：R67–R77 与 D76/D78

截至 2026-10-29；本次仅汇总已保存并校验的结果，未新增图像前向、模型评分、优化或 VAL/TEST 评测。用户希望 Pro 围绕 long-tail 任务设计 GRPO；本文提供事实与约束，不预先指定新奖励或要求 GRPO 必须胜出。

## 1. 先读结论

- 原始 SFT 是主要参照。R67–R75 的预定采样主候选没有同时改善阶段平均、最终和尾类 BA；高于 H0 的差距不能全部归于 RL。
- **R77 有小幅条件正结果**：distribution-GRPO 相对相同分布读出的零后训练 SFT，阶段平均/最终/尾类为 **+0.167141/+0.225051/+0.519648pp**；同目标 exact 为 **+0.727905/+0.546529/+1.400567pp**。两者均应保留，不能将 exact 改成事后主候选。
- R77 相对原 SFT 为 **+2.618943/−2.498135/−3.188441pp**。读出本身贡献了大部分阶段平均提升，最终和尾类仍低于原 SFT，不能称整体成功。
- R75 macro-GRPO 尾类 +2.597403pp，但平均/最终下降；尾类净增量全部来自一个 VAL=7 的类别正确数 4→6，其余尾类变化抵消。不是广泛尾类提升。
- 已经试过先验对齐、固定支持、实际 AdamW 增量保护、逆信号补偿、错误类别差异奖励、文本伪原型、类别等权 surrogate。不要只换名称重新提出这些路线。
- 所有现代结果是 **同一个 HK seed74002 的重复开发**，不是多 seed 独立确认；极小 VAL 分母和多轮开发限制外推。

## 2. 数据、指标和评测边界

Hyper-Kvasir（HK），自然长尾；六阶段已见类别数 13/15/17/19/21/23。最终 TRAIN=6821、VAL=1698，23 类；尾类按原冻结规则 TRAIN count<250，共11类。类别索引为既定内部顺序，不在本文重新定义语义映射。

- BA：已见类别 recall 的等权平均；阶段平均是六阶段 BA 的算术平均。
- 最终 BA：stage5 全23类；最终尾类 BA：stage5 冻结尾类子集等权平均。
- 最终 TRAIN 类计数：[626, 3, 6, 85, 288, 403, 736, 488, 252, 167, 595, 261, 6, 34, 83, 25, 23, 635, 127, 662, 18, 654, 644]。
- 最终 VAL 类计数：[165, 1, 1, 21, 66, 113, 182, 123, 60, 41, 150, 61, 2, 8, 21, 7, 5, 164, 33, 160, 4, 153, 157]。
- 部分类只有1或2个 VAL 样本。一个 VAL=1 类的一次变化可改变23类宏BA约4.347826pp；若属于11个尾类，尾类BA约9.090909pp。不能据此删类或改门。
- 训练中元数据 component 两折不保证患者独立；TRAIN 描述性分析不是独立 holdout。模型只接触已到达类图像/支持/head；不使用未来类图像训练。
- 新分支全部 head 封存后才读取 VAL；R67–R75 通常30个真实 readout（24新+6SFT），R74/R77为36个（额外同读出SFT对照）。无中途VAL筛选、无TEST确认。

## 3. 共同算法：实际做了什么

这是 BiomedCLIP 上的**单步类别采样 GRPO**，不是生成式 VLM TARS 的完整复现。输入为确定性图像预处理；只训练视觉 Transformer 最后两层 Q/V LoRA，rank8、alpha8，共49152参数；前十层和基础权重冻结。

每个 stage 从对应 R67 SFT 独立开始后训练，不 carry 上一 stage 的 RL delta。SFT 为10epoch监督训练。后训练自然 query 顺序、2epoch、AdamW lr0.001/wd0.001、cosine2、batch256/micro64；每arm全六stage236updates/924micro。支持在每epoch/每queryfold开始刷新后detach固定，query使用相反折支持；old在同fold开始冻结，reference为固定SFT及其对应支持。支持分支不反向传播。

原型类分数为 `20*z·normalize(mean_i f_ci)`，z和f均为单位512维特征。TRAIN策略 logits加完整已见TRAIN类 `log N_c`，正式推理用 raw 分数，不加count prior。

从 old 类别策略有放回抽 G=8 个类别动作，奖励 `r=1[action==label]`；组内优势 `A=(r-mean(r))/(population_std(r)+1e-8)`。ratio为当前动作概率/old动作概率，clip范围[0.8,1.2]。最小化：

`L = -w_y * mean_j min(ratio_j*A_j, clip(ratio_j)*A_j) + 0.04*KL(pi_current || pi_SFT) + 0.01*||z_current-z_original_Biomed||²`。

TARS权重在后训练前由**原SFT原型TRAIN交叉折错误率**固定：`u_c=[(mean_count/N_c)*(1-Acc_c)]^0.75`，`w_c=max(1,u_c/max(median(u),1e-8))`。只乘surrogate，不乘KL/anchor。普通GRPO的w=1。Exact精确积分同一组标准化/clip的G8二元目标；**不是CE**，也不应为exact编造采样zero率。

原 `sft_continue` 直接返回普通CE，未加入reference KL；不能将它称为与GRPO正则完全匹配的CE对照。R72有weightedCE+KL/anchor，但仍缺 plainCE+KL 的同配置对照。

SFT曲线的lr在scheduler.step之后记录：epoch9记录 `2.4471741852423235e-5`，这是最后一个实际训练epoch所用lr；epoch10记录0是训练结束后的scheduler值。后训练重新建立AdamW并从0.001开始，约为SFT最后实际lr的 **40.863458倍**。该配置差异已核实，但未证明是负结果原因。低LR四臂尚未运行；不得把“发现差异”写成“修复后有效”。

## 4. 完整结果

单位%，越大越好。差值单位百分点，顺序为阶段平均/最终/尾类。下表固定展示每轮事前指定的采样主候选，**不是从所有arm挑赢家**。全部40个后训练arm（含重复匹配控制）原精度见 [ALL_ARMS.csv](ALL_ARMS.csv)、[EVIDENCE.json](EVIDENCE.json)。重复控制不算新seed。

|参照|阶段平均BA↑|最终BA↑|最终尾类BA↑|
|---|---:|---:|---:|
|H0|55.630100|56.454053|36.857416|
|原SFT|66.840041|64.876782|44.924938|
|R67继续CE|66.580250|62.497152|39.854724|
|R67普通GRPO|66.577956|63.597961|42.511518|

|轮次与预定主候选|阶段平均BA↑|最终BA↑|最终尾类BA↑|相对原SFT差值pp（平均/最终/尾）|
|---|---:|---:|---:|---|
|R67 tars_grpo|66.115954|63.655574|44.737644|-0.724087 / -1.221208 / -0.187294|
|R68 raw_tars_grpo|65.295557|63.727152|43.586882|-1.544483 / -1.149630 / -1.338056|
|R69 fixed_tars_grpo|66.133591|63.554042|44.737644|-0.706449 / -1.322740 / -0.187294|
|R70 protected_tars_grpo|65.828378|62.006397|42.032016|-1.011662 / -2.870385 / -2.892922|
|R71 balanced_guard_grpo|65.494932|62.495759|42.962128|-1.345109 / -2.381023 / -1.962810|
|R72 compensated_grpo|64.899613|63.053406|43.866204|-1.940427 / -1.823376 / -1.058734|
|R73 hardnegative_grpo|65.776356|63.472624|44.482559|-1.063685 / -1.404158 / -0.442379|
|R74 semantic_grpo|65.870120|63.759645|43.464787|-0.969921 / -1.117137 / -1.460151|
|R75 macro_grpo|65.411597|62.741784|47.522341|-1.428444 / -2.134998 / +2.597403|
|R77 distribution_grpo|69.458984|62.378647|41.736497|+2.618943 / -2.498135 / -3.188441|

### 逐轮机制与结果含义

- R67：TARS相对普通GRPO尾类+2.226126pp，但平均−0.462003pp；原SFT仍更强。sample与exact没有一致优劣。
- R68：将策略改为raw score以对齐推理，主候选失败。训练count prior差异不是已证bug，不自动复活先验救援。
- R69：固定SFT支持坐标，主候选失败；fixed-exact尾类45.249254，比SFT高0.324315pp，但平均65.640457/最终64.816552仍低。
- R70/R71：自然CE/均衡CE guide约束的是**实际AdamW增量**，不是仅投影原始梯度；保护实际生效，仍无整体收益。局部一阶约束不是有限步泛化保证。
- R72：补偿二元组归一化的概率依赖期望因子；最大有效权重22402.95、TRAIN KL增加后失败。compensated-exact平均66.903941略高SFT，但最终/尾类低。局部CE梯度方向关系不是全局算法等价或新RL创新。
- R73：区分近邻错误的奖励令415/58434个原全错组产生信息，sample/exact仍未整体改善。新增非零信号本身不保证有益方向。
- R74：单单位冻结文本伪原型。semantic-SFT（零后训练）67.182120/63.974439/42.831274，本身伤最终/尾类；不能只报告相对较弱同读出控制的恢复。
- R75：在自然query下使每类surrogate总质量相等，匹配原TARS总质量；不是全目标均衡，KL/anchor仍自然测度。主候选平均/最终负，尾净+2.597403pp全部由class15 VAL7正确4→6贡献；macro-exact64.695657/62.808625/42.366890。macro-sample TRAIN KL0.181018 vs原0.046914。

### R77：必须分开读出与后训练贡献

R77类分数为 `logsumexp_i(20*z·f_ci)-log n_support_c`，使用全部相反折真实支持，不做nearest-k/带宽搜索。控制仍原型读出；其余原TARS配置固定。权重仍由原SFT原型错误率计算。最终推理各自raw score；kernel raw不是单个cosine。已知角度核读出，不是新RL。

|R77端点|阶段平均BA↑|最终BA↑|最终尾类BA↑|
|---|---:|---:|---:|
|原SFT原型|66.840041|64.876782|44.924938|
|distribution-SFT，0后训练|69.291843|62.153597|41.216849|
|prototype-GRPO|66.115954|63.655574|44.737644|
|distribution-GRPO，预定主|69.458984|62.378647|41.736497|
|prototype-exact|65.456019|63.954743|44.161723|
|distribution-exact|70.019748|62.700126|42.617416|

分布SFT本身相对原SFT+2.451802/−2.723185/−3.708090pp。GRPO在相同读出上+0.167141/+0.225051/+0.519648pp，是小幅条件正增量；exact同样有小正增量。尚无跨seed/患者独立确认，不能抹去，也不能夸大。

主候选相对distribution-SFT的尾类净增量有大抵消：class12 VAL2少1正确−4.545455pp；class15 VAL7多3正确+3.896104pp；class3 VAL21多2 +0.865801；class13 VAL8多1 +1.136364；class18 VAL33多1 +0.275482；class9 VAL41少5 −1.108647；其余不变。均为对11类尾均值的贡献。

TRAIN mean KL distribution-sample0.018924616 vs prototype0.046913815，zero组65.9359% vs74.7578%，动作reward86.4719% vs89.5722%。reference评分映射也变了，KL不等于统一参数距离或泛化保证，不能唯一归因于方差。

R77需存最多6821×512 FP32支持向量，约13.97MB，对比23个centroid约47KB；最大head14.27MB。24head约246MB，加6份SFT支持约60MB。推理比较最多6821支持而非23原型；额外kernel运算和6次SFT VALforward需披露，不能称同FLOPs/同记忆。

## 5. 已有诊断对“长尾GRPO设计”的约束

### 类别风险与组信号

R75保存的权重审计显示，原TARS尾类名义质量占比最终37.2085%，类别等权对应47.8261%；最终各类 `N_c*w_c` 最大/最小445.58。此量是surrogate名义曝光，不是参数梯度质量，也不证明尾类梯度被头部支配。

二元G8奖励在old正确概率p下，混合组概率为 `1-p^8-(1-p)^8`。全对与全错组优势均为0，因此“zero多”不等于“尾类全错多”。R67 TARS保存58434组、43684零优势组、467472动作、418725正确动作；计数可推出至少39435个零组全对，即至少90.2733%的零组为全对下界，**不是精确逐组测量**。未保存逐类历史old-policy轨迹，不得从汇总值编造尾类p或梯度。

同query组内把全部奖励乘正常数会被均值/标准差归一化抵消（epsilon效应除外）；TARS是在归一化之后加权。仅把尾类奖励常数乘大未必改变更新。Exact已多次为负，不能把采样方差当唯一失败原因。R72的逆因子补偿也已失败。

已有R67–R74审查核96个old/new差值：最终八个采样主候选都同时损伤old/new类，不只旧类遗忘。原两折支持stop-gradient不等于实现bug：D50/R59双边反馈历史负，合成支持梯度反例只表明方向可能不同，不能证明医疗数据收益或自动复验全支持梯度。

### D76：表示几何不是因果诊断

18端点、87651 TRAIN嵌入、36 raw/natural统计；0优化/VAL。最终SFT/TARS/macro的mean norm约0.765674/0.777997/0.787226，within trace0.424963/0.405696/0.386355，nearest centroid cosine0.738393/0.775308/0.793313，而TRAIN宏88.505737/81.596113/76.188190、尾83.220751/72.194357/67.011612下降。stage0–4的几何方向与最终反向。不能概括为所有阶段“类内更紧导致失败”，端点p/几何也不是历史梯度方差。

### D78：输入方向敏感，尚无增强训练效益证明

18端点、87651翻转嵌入+87651保存原图特征、72配对/144单视图统计，0optimizer/backward/VAL。固定原图支持，仅query水平翻转（PIL RGB、resize/crop前）。最终TRAIN：

|端点及自身raw读出|原图宏BA|翻转宏BA|原图尾BA|翻转尾BA|
|---|---:|---:|---:|---:|
|SFT prototype|88.505737|74.264994|83.220751|61.863878|
|prototype-GRPO prototype|81.596113|73.335535|72.194357|61.817356|
|distribution-GRPO kernel|82.722033|74.988439|75.013580|65.790741|

prototype raw宏18端点17降1微升，kernel raw宏18降；raw尾类各16降。不是所有类/阶段都下降。更小原图/flip差距可来自原图性能也变低，不能直接称GRPO更鲁棒。该诊断不能证明23类翻转均保标签，不能证明增强能改善原图VAL。

R77核均值单位化候选只读审查还发现，其score修正可分解为count项与geometry项，最终与logcount相关约0.90–0.95。不是纯几何机制；未据此启动新训练，不换带宽/去对角救援。

## 6. 未运行方案与交接要求

- **R79**：已冻结并实现原图+固定水平flip query与duplicate原图匹配对照，四arm、原图支持/推理固定；仅编译通过，CPU软件与native尚未执行，未启动实验钟、0optimizer。保留实现，不把D78当增强已有效，也不自动启动。
- **Pro此前低扰动四臂**：plainCE+KL、weightedCE+KL、主GRPO、exact，统一用SFT末实际lr；仅建议，未实现为新运行、未验证效益，用户尚未明确放开“不救援LR”边界。
- 用户最新要求是针对**长尾识别**本身设计GRPO。本轮整理push后由用户交给Pro设计；等待用户带回设计，不自行并行启动另一候选。持续研究授权未整体撤销，已完成历史与资源保护继续有效。

建议Pro给出：一个明确的长尾失败机制及支持/反驳证据；奖励/优势/采样/正则的完整公式及梯度作用；是否退化为CE、类别加权或score prior；与R72/R73/R75的实质区别；同读出零后训练SFT、匹配采样控制与同目标exact如何归因；额外计算/存储预算；阴性结果如何解释。不要仅凭“更复杂”或新命名声称算法创新。

现有约束：单HK74002全六stage；不按VAL挑类别对/阶段/arm，不改旧门，不短pilot淘汰或网格；不自动调LR/G/epoch/weightcap/带宽/k/温度/融合。新方案若确需改变边界应明确指出并由用户决定；不覆盖已运行历史。

资源：仅既定两张A100角色（train/prefix），host总RAM≤30GB，实际执行新实验独立20h预算不重置，各服务器新增实际存储<50GB，原准入与硬guard继续。所有大文件留既定远程存储；不清历史权重/支持/图像。未来新模型诊断也应先冻结TRAIN-only用途和预算。

## 7. 成本、证据入口与公开范围

R67–R74原钟合计5.022746h、CPU26627.466641s、8732updates/34188objective micro，R70/71额外7392guide autograd、198stage分支/642curve/246VAL。R75原钟2650.704041s；D76 412.158700s；R77 2727.902920s；D78 462.827361s。每轮原始费用见EVIDENCE.json及对应报告；这些不是从首轮到交接的自然时间，不含全部轮间人工/本地分析发布费用。

本交接报告只读现有保存结果，0新模型评分/优化。公开本报告、全部arm汇总和聚合分母；已有各轮公开目录含完整class-stage计数/曲线/诊断/成本与自有源代码。没有发布图像/身份/逐例数组、支持向量、特征、权重、私有配置、原日志或第三方全文。

- [R67 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/363bf426c4e7c3792e6269fda1c3b53c17d7d1ec/research/hasten_independent/r67_clip_grpo/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/363bf426c4e7c3792e6269fda1c3b53c17d7d1ec/research/hasten_independent/r67_clip_grpo/PROTOCOL.md)。
- [R68 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/073d05a4d8d3997fc4581feb245424702c882663/research/hasten_independent/r68_aligned_grpo/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/073d05a4d8d3997fc4581feb245424702c882663/research/hasten_independent/r68_aligned_grpo/PROTOCOL.md)。
- [R69 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/4e70ed1c355818504512d4840fd0ba8e62b69073/research/hasten_independent/r69_stationary_support/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/4e70ed1c355818504512d4840fd0ba8e62b69073/research/hasten_independent/r69_stationary_support/PROTOCOL.md)。
- [R70 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/a31b624d62b12468dbca70091795cd2917bdea0c/research/hasten_independent/r70_risk_guard/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/a31b624d62b12468dbca70091795cd2917bdea0c/research/hasten_independent/r70_risk_guard/PROTOCOL.md)。
- [R71 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/945979b26f7aa6e7681de70157111074504aea85/research/hasten_independent/r71_macro_guard/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/945979b26f7aa6e7681de70157111074504aea85/research/hasten_independent/r71_macro_guard/PROTOCOL.md)。
- [R72 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/2ae6bf68a987f9d201d5d0a066c5bd2f6870b83e/research/hasten_independent/r72_signal_compensation/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/2ae6bf68a987f9d201d5d0a066c5bd2f6870b83e/research/hasten_independent/r72_signal_compensation/PROTOCOL.md)。
- [R73 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/a6c78c3b4c12f6939d3a7213eae2f26fe55376e4/research/hasten_independent/r73_contrastive_reward/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/a6c78c3b4c12f6939d3a7213eae2f26fe55376e4/research/hasten_independent/r73_contrastive_reward/PROTOCOL.md)。
- [R74 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/05fde7ab01e8518a1c7c6e5bb739bf562882cd1d/research/hasten_independent/r74_semantic_prototypes/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/05fde7ab01e8518a1c7c6e5bb739bf562882cd1d/research/hasten_independent/r74_semantic_prototypes/PROTOCOL.md)。
- [R75 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/9ca10114add58e312507abf3ce2bbd1de7474635/research/hasten_independent/r75_macro_risk/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/9ca10114add58e312507abf3ce2bbd1de7474635/research/hasten_independent/r75_macro_risk/PROTOCOL.md)。
- [R77 完整报告](https://github.com/DLwbm123/Long-tailed-CL/blob/52819958e066f16d0a7b4875ff3c0041895ae6f9/research/hasten_independent/r77_distributional_support/REPORT_ZH.md)；[协议](https://github.com/DLwbm123/Long-tailed-CL/blob/52819958e066f16d0a7b4875ff3c0041895ae6f9/research/hasten_independent/r77_distributional_support/PROTOCOL.md)。

D76/D78及证据综审入口：

- [D76报告](https://github.com/DLwbm123/Long-tailed-CL/blob/ed30a2c8b88d94b4451da090bec5a4605a2b73c9/research/hasten_independent/d76_train_representation_census/REPORT_ZH.md)。
- [D78报告](https://github.com/DLwbm123/Long-tailed-CL/blob/02dcb27743e63e289bea185b92b8bff9b74ac554/research/hasten_independent/d78_orientation_sensitivity/REPORT_ZH.md)。
- [R67–R74综审](R67_R74_REVIEW_ZH.md)；本文保留其负证据与小分母限制。

附：[核均值归一化未采用候选审查](KERNEL_NORMALIZATION_REVIEW_ZH.md)。
