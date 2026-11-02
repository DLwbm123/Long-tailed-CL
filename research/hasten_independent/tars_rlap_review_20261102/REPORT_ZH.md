# TARS-GRPO与RLAP研究交接：方法、已尝试路线和完整证据边界

整理日期：2026-11-02（北京时间）。[主指标CSV](KEY_METRICS.csv)。用途：交给Pro审阅研究方向与下一机制，不是论文成功声明。本报告读取既有协议和已保存聚合报告；未新增训练、模型评分、VAL/TEST评测或历史端点扫描。R90运行状态为本次只读查询快照，后续可能变化。

## 1. 请先读的结论

1. **原BiomedCLIP监督适配是强参照。** R67原监督起点六阶段平均宏准确率66.840041%、最终宏64.876782%、最终尾类44.924938%。TARS-GRPO为66.115954%/63.655574%/44.737644%，未超过它。高于较早H0不能归因于GRPO。
2. **真正GRPO并非完全没有局部正信号，但没有稳定的增量证据。** R77在相同分布式读出下有小幅后训练正增量；同目标精确积分更好，且两者最终/尾类仍低于原SFT。R80训练分配的条件收益未超过打乱奖励和固定均匀动作控制。R87判决策略未超过同容量监督和精确奖励优化。
3. **R88完整RLAP重实现胜过两个内部消融，但不是标准GRPO成功。** 最终总体准确率66.548882%、宏48.472684%、尾31.372347%。完整版本相对移除RLPO的最终宏增量4.312309pp，但尾类只有0.027116pp；必要总门失败。
4. **R89均衡+蒸馏不是稳定改进。** 预定JOINT相对匹配BASE六阶段平均宏/尾+1.623779/+1.412291pp，最终却−1.569555/−2.524789pp；三个极少样本类最终召回全0。不能换成表现较好的某个消融作主候选。
5. **TARS与RLAP的绝对数值不是公平机制对决。** 骨干、监督起点、旧数据可用量、适配结构和训练轮数不同。TARS的63.655574%是宏准确率，不能与RLAP的总体66.548882%直接比。
6. **当前R90是桥接诊断，尚无准确率结果。** 它检验“支持证据驱动的早期视觉提示构建”，同时对比真正GRPO、监督、原始utility精确期望和同组归一化clip目标的精确期望。不是完整RLAP，也尚未直接证明原BiomedLoRA主线改善。
7. 用户的两方向仍有效：①借鉴RLAP改善原真正GRPO；②以RLAP为基础形成长尾持续学习方法。原方法主线不应因复现外部方法而被替代。用户最近询问RLAP换BiomedCLIP，已列为R90后的优先匹配后继，**尚未启动、尚无结果**。

## 2. 数据、指标与比较口径

### 2.1 当前主要研究设置

- HyperKvasir既有23类TRAIN/VAL划分，固定到达顺序13+2+2+2+2+2，seed74002。
- TRAIN共6821张，最终VAL共1698张。按到达顺序的TRAIN类数：626,3,6,85,288,403,736,488,252,167,595,261,6,34,83,25,23,635,127,662,18,654,644。
- 尾类定义固定为完整TRAIN类别样本数<250；最终11类。极少样本类索引1/2/12，TRAIN为3/6/6，最终VAL为1/1/2。类12多或少正确一张，就可改变最终尾类均值约4.545455pp。
- metadata-component划分不等于患者独立。部分训练反馈来自反复使用的TRAIN子划分，不能称独立holdout；同一VAL已用于多轮开发。这里只是单数据集单seed探索，不是三seed、独立患者或临床确认。
- R67–R81及R88–R89相关正式结果按各自协议读取VAL；R82–R87大部分是TRAIN内部机制诊断，**不可把它们的查询指标与VAL绝对值混排**。本报告涉及的后续实验没有TEST结果。

### 2.2 三种指标不要混用

- 总体准确率micro = 全部图像正确数/全部图像数，头部样本影响大。
- 宏准确率macro/BA = 已到达类别recall的等权均值。
- 尾类宏召回率tail BA = 固定尾类中已到达类别recall的等权均值。
- 六阶段平均 = 每阶段指标等权平均，不是把跨阶段重复图像混成独立样本。
- 表中“pp”是百分点差，不是相对百分比。随机策略的精确期望正确数可以是小数，不是新增独立病例数。
- R67公开主表只列宏及尾类；本次不把它们冒充micro，也不为补齐表格重新评分。各报告中的完整class/stage计数仍保留。

## 3. 原TARS-GRPO：我们实际实现的是什么

依据：[R67协议](../r67_clip_grpo/PROTOCOL.md)、[完整结果](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r67_clip_grpo/REPORT_ZH.md)。这是受TARS原则启发的CLIP分类式改编，**不是原生成式VLM论文的原样复现**；不将外部论文报告的效果当作本项目证据。

### 3.1 表示、原型和监督起点

BiomedCLIP视觉骨干前10个block冻结，只在最后2个block的Q/V加入LoRA，rank8、alpha8，共49,152个可训练参数。图像输出单位化512维向量。

对当前类c，用相反metadata-component fold的已到达TRAIN支持图像构建原型：

`p_c = normalize(mean_{i:y_i=c} f_theta(x_i))`。

分类原始分数为`20*cos(f_theta(x),p_c)`。正式推理不加类别先验、不做额外校准。原型支持编码按epoch/fold刷新并detach；训练梯度通过query编码进入LoRA，不对支持编码双边求导。

先建立共同SFT：`CE(20cos + log n_c) + 0.01*原始Biomed图像表示漂移约束`，AdamW lr0.001/wd0.001、cosine10，10epochs，batch256/micro64。监督LoRA跨6stage携带，每stage重置监督优化器。

### 3.2 真正类别动作GRPO

每个SFT阶段分别分出4条2epoch后训练分支：继续监督、普通GRPO、TARS-GRPO、同目标精确积分。后训练delta不携带到下一stage；下一stage仍来自共同SFT。因此R67是**stagewise posttraining**，不是端到端在线RL持续学习。

- state：query图像及当前原型分类器；action：离散类别预测。
- 行为策略logit：`20cos + log n_c`；每query从冻结old policy有放回采样G=8个类别动作。
- reward：预测正确1，错误0。
- advantage：`A_i=(r_i-group_mean)/(group_population_std+1e-8)`；全同reward组保留在分母中，优势为0，不重采样或强制成功动作。
- ratio：`rho_i=pi_theta(a_i|x)/pi_old(a_i|x)`，使用真实old logprob；surrogate为`min(rho_i*A_i,clip(rho_i,0.8,1.2)*A_i)`。
- 固定参考：同stage SFT，categorical forward KL系数0.04；表示anchor系数0.01。KL/anchor不乘TARS权重。
- 后训练优化：AdamW lr0.001/wd0.001、cosine2、2epochs，batch256/micro64。

### 3.3 TARS权重与精确控制

在各stage后训练前，用共同SFT的TRAIN crossfold错误率固定：

`t_c=[(mean_train_count/n_c)*(1-SFT_TRAIN_crossfold_accuracy_c)]^0.75`；
`w_c=max(1,t_c/max(midpoint_median(t),1e-8))`。

它强调稀缺且监督模型表现较差的类别，只在组标准化之后乘surrogate，不乘KL。自然query采样不变。其TRAIN错误估计和原生成式TARS来源中的估计方式不同，必须披露。

`tars_exact`对Binomial(G8,p_correct)的组正确数、以及错误动作身份的old条件概率，精确积分**同一个clipped、组内标准化、TARS加权目标**。它不是普通CE，也不是直接原始reward期望。只有把两种exact控制分清，才能区分目标变化与采样效应。

### 3.4 R67完整结果与重要诊断

| 版本 | 阶段平均宏% | 最终宏% | 最终尾类% |
|---|---:|---:|---:|
| H0历史参照 |55.630100|56.454053|36.857416|
| 原SFT |66.840041|64.876782|44.924938|
| 继续监督2epoch |66.580250|62.497152|39.854724|
| 普通GRPO |66.577956|63.597961|42.511518|
| TARS-GRPO预定主候选 |66.115954|63.655574|44.737644|
| TARS同目标exact |65.456019|63.954743|44.161723|

TARS对原SFT为−0.724087/−1.221208/−0.187294pp；对普通GRPO尾类+2.226126pp，但平均−0.462003pp。相对继续监督的最终/尾类改善，不能替代相对原SFT的失败。

普通GRPO/TARS零优势组约77.06%/74.76%；TARS采样正确动作率约89.57%。大量零优势不等于大量全错，因为全对组也零优势。同目标exact也未胜SFT，故不能把全部问题归为采样方差。R67实际逐步TRAIN KL未保存，不声称该项已测；R68以后才记录相关KL。

## 4. TARS与原生LoRA路线已经尝试过什么

以下表格列每轮**预定采样候选**，不是逐轮挑最好arm。除另注明外，R68–R75/R77复用R67同一SFT、同seed完整6stage，各stage独立后训练，不增加SFT，不根据VAL救援调参。

| 轮次与干预 | 候选平均宏% | 最终宏% | 最终尾类% | 主要判定 |
|---|---:|---:|---:|---|
| R68 raw-policy对齐 |65.295557|63.727152|43.586882|去掉RL行为/ratio/KL中的logcount先验，未胜SFT；零优势率下降也未转成尾类收益|
| R69 固定SFT支持原型 |66.133591|63.554042|44.737644|把动态支持坐标固定，采样收益接近零；不能声称支持漂移原因已解决|
| R70 自然CE方向保护 |65.828378|62.006397|42.032016|投影实际AdamW增量，阻止当前CE一阶增加；最终/尾类更差|
| R71 均衡CE方向保护 |65.494932|62.495759|42.962128|guide用N/(C*n_c)类别权重；相对R70局部恢复，但仍低原TARS和SFT|
| R72 解析信号补偿 |64.899613|63.053406|43.866204|按二元组归一化梯度系数的逆数补偿；信号放大未获收益|
| R73 三档hard-negative奖励 |65.776356|63.472624|44.482559|正确+1/固定最相似错误−1/其余0；出现全错非零优势组仍未改善|
| R74 冻结文本伪样本原型 |65.870120|63.759645|43.464787|每类加入固定1个文本伪样本，不搜提示；最终尾类下降|
| R75 匹配总权重的宏类别风险 |65.411597|62.741784|47.522341|尾类局部正，但平均/最终下降，不能只报尾类胜出|
| R77 完整支持分布角核 |69.458984|62.378647|41.736497|平均提高，最终/尾类牺牲；相同读出SFT后有小增量，但exact更强|

### 4.1 已排查的具体假设与对照

- **R68**：2×2 natural/raw policy × sampled/exact，最终推理一律raw20cos。raw采样零优势率74.76%→66.55%，尾类反而低1.150762pp；移除先验不是充分修复。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r68_aligned_grpo/REPORT_ZH.md)
- **R69**：dynamic/fixed prototypes × sampled/exact。fixed_exact最终宏64.816552%、尾45.249254%，对SFT最终−0.060230pp、尾+0.324315pp；这是次要exact局部正信号，不能换成主候选。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r69_stationary_support/REPORT_ZH.md)
- **R70/R71**：原GRPO梯度不混入CE；CE只提供增量投影guide，AdamW内部moment不投影。一阶当前batch条件不保证有限步、变化原型或VAL宏风险。guide实际触发，但不能改善最终结果。[R70](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r70_risk_guard/REPORT_ZH.md)、[R71](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r71_macro_guard/REPORT_ZH.md)
- **R72**：current=old、未clip时，二元组奖励期望梯度是CE上升方向乘系数h_G(p)，低p约sqrt(G−1)*p。补偿只乘surrogate的detached逆系数。snapshot梯度关系不等于全训练轨迹等价。另有补偿exact和weightedCE对照；采样最大实际有效权重约22402.95，增大信号未带来收益。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r72_signal_compensation/REPORT_ZH.md)
- **R73**：hardnegative依据训练前固定SFT相反fold原型，非临床语义oracle。三档exact枚举45种reward-count组合。415/58434个组全错但有优势，仅0.7102%；不证明增加错误动作区分就足够。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r73_contrastive_reward/REPORT_ZH.md)
- **R74**：`normalize(n_support*unit_visual_mean+frozen_text)`。增加同readout、0优化的semantic-SFT以分离读出和GRPO。semantic-SFT自身最终/尾类低于原SFT；不能把后训练恢复说成整体超过原方法。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r74_semantic_prototypes/REPORT_ZH.md)
- **R75**：`a_c=M*N/(C*n_c)`，M为原TARS自然query平均权重，匹配总权重质量但不匹配梯度方差。尾类对SFT+2.597403pp的净增量全部可由一个VAL分母7的类别多对2张解释，其他变化相抵；并非广泛尾类改善。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r75_macro_risk/REPORT_ZH.md)
- **R77**：`score_c=logsumexp_i(20*z·f_ci)-log(n_c)`，原型对照仍20cos；不调带宽/k/温度。0后训练distribution-SFT平均69.291843/最终62.153597/尾41.216849。采样相同读出增量+0.167141/+0.225051/+0.519648pp；distribution_exact为70.019748/62.700126/42.617416，增量更大，但二者最终/尾仍低原SFT。保存完整支持最多6821×512，推理对全部支持而非23原型；不能声称等记忆/FLOPs。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r77_distributional_support/REPORT_ZH.md)

### 4.2 描述性诊断，不等于新方法胜出

- **D76**：18个已有端点、六阶段完整TRAIN表示统计。最终TARS类内variance trace比SFT低、最近异类centroid更相似，TRAIN识别反而下降；不能统一归因为“类内方差变大”。0优化/VAL/TEST。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/d76_train_representation_census/REPORT_ZH.md)
- **D78**：固定原图支持，测水平翻转query敏感性。SFT最终TRAIN宏88.505737%→74.264994%，但这是重复TRAIN描述，不是VAL增强收益；GRPO差距更小也可能是原图起点下降。0优化/VAL/TEST。[报告](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/d78_orientation_sensitivity/REPORT_ZH.md)
- **R79双视图训练覆盖**仅形成过协议/排队设计，仍暂停；不把它列成完成的性能实验，不恢复。低LR、R65 USERSTOP、D13等其他暂停分支同样不因本报告恢复。

## 5. 从“预测类别”改成“控制学习或判决”的路线

这些也是引入完整RLAP前为GRPO主线做的探索。不同实验回答不同问题，不是一个固定最终模型不断堆模块。

### 5.1 R80：GRPO控制实际监督训练分配

视觉学习器仍是BiomedCLIP末两层Q/V LoRA。仅从R67 stage0 SFT初始化一次，此后视觉参数、视觉AdamW、控制器及其AdamW全部跨6stage携带。不能与R67各stage独立后训练混称。

每类state8项：计数、可用S量、年龄、新类标志、当前TRAIN Q Brier/margin、同fold margin变化、近期保留曝光；共享8→64→64 tanh及set pooling，无class-ID embedding。动作0均衡分配、动作c优先类别c。每context8候选+1均衡参考，各3个真实监督更新；前2步192共同自然图像+64控制名额，最后一步共同256图像。每batch图像唯一，少类不足名额时均衡补齐，造成动作强度不对称。

Q不进入该context的inner梯度/原型，但历史SFT及其他context可能见过，不能称独立holdout。端点重新编码S建原型、全Q反馈。奖励是负macro Brier−old∪tail正损害均值；旧类anchor为前stage，新tail为context起点。G8、clip0.8–1.2、uniform合法动作KL0.04，每context4个控制器更新。预定保留轨迹index按stage/context循环，不按reward选最好轨迹。

| R80/R81版本 | 平均宏% | 最终宏% | 最终尾类% |
|---|---:|---:|---:|
| 固定均衡短训练 |64.443502|60.951858|42.779034|
| 宏奖励控制器 |65.130314|61.740939|45.885454|
| 保护奖励控制器（主候选） |65.071594|61.768916|46.107184|
| reward轮转错配控制 |65.046855|62.185713|47.086129|
| 固定均衡额外长训练 |60.457244|60.441780|36.049304|
| R81零学习均匀动作控制 |65.030549|62.244174|46.653229|

主候选对短均衡三项正，但最终/尾类不如错配控制；R81补充uniform后，主候选最终/尾类再低0.475258/0.546045pp。原SFT最终门也失败。不能把“随机训练分配有影响”写成“GRPO学会有效分配”。损害惩罚的局部尾收益还集中于小分母正确数变化。[R80](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r80_trajectory_allocation/REPORT_ZH.md)、[R81](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r81_uniform_allocation/REPORT_ZH.md)

### 5.2 R82–R83：动作确实起作用，是否就能泛化纠错？

固定R81六阶段已保存状态、完整AdamW、原S/Q角色，两fold和3共同随机流，在同起点枚举动作端点。Qa用于比较/选动作，Qb测迁移；反向预定敏感性，不择优方向。均是TRAIN内部诊断，非新VAL。

- **R82原曝光动作**：Qa→Qb相对唯一合法动作uniform的宏/尾增量+0.017395/+0.019310pp，仅3/6stage两正；utility排序tau均值0.432746。数值响应存在，但小类自身曝光常无额外名额；必要门失败。
- **R83固定唯一图像、改变CE份额**：同图同序，第一步为优先类分配固定rho0.25份额；均衡、uniform动作和固定meanweight均有控制。648/648动作参数/风险响应超过数值门；tiny类参数作用均成立，但Qa→Qb宏相对uniform−0.035487pp、尾+0.018636pp，相对fixedmean宏/尾−0.087960/−0.030455pp。排序tau升至0.548266，却只有1/6stage四项正，识别门仍失败。
- class1在fold1无法分出真实双查询component，相关可靠性不可识别；不复制/重分割补覆盖。

结论：非零梯度、强参数响应、稳定代理reward排名，均不足以证明宏/尾纠错。不能再把“没有actuation”当全部失败原因。[R82](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r82_action_identifiability/REPORT_ZH.md)、[R83](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r83_loss_share_action/REPORT_ZH.md)

### 5.3 R84–R86：冻结表示的边界/结构/图像信息诊断

| 轮次 | 新机制 | 对强参照的结果 | 限制 |
|---|---|---|---|
| R84 | S-only监督残差线性头，平衡与自然风险、同单位二次锚 | 平衡对NCM平均宏/尾+0.303752/+0.237636pp；最终−0.084318/−0.809157pp | 3/6stage四正、tiny失败；不是RL增益 |
| R85 | 等级邻接图共享边界，控制unstructured/degree-only/family-mean | 对独立头平均宏−0.049964/尾+0.016416pp | 仅1/6stage六正，目标tiny没有新增收益 |
| R86 | CLS＋全部196patch均值，与双CLS等维、同平衡监督头比较 | 对双CLS平均宏/尾+0.134297/+0.334994pp；对NCM最终+0.536976/+0.522838pp | 早期负，4/6stage四正、tiny仅一类改善，必要门失败 |

这些结果说明某些信息/边界确有条件收益，但没有建立GRPO必要性，也不能仅凭冻结表示诊断断言表示已经充分。[R84](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r84_frozen_boundary/REPORT_ZH.md)、[R85](https://github.com/DLwbm123/Long-tailed-CL/blob/3c6913987869c973c18a1fe95cefd0c9eebfbba7/research/hasten_independent/r85_ordered_shared_boundary/REPORT_ZH.md)、[R86](../r86_fixed_patch_information/REPORT_ZH.md)

### 5.4 R87：两步分类仲裁与真正GRPO增量

此轮已受RLAP“动作应直接作用部署分类器”的启发，但没有复现RLAP。复用R81/R84的固定NCM/CLS分数，0视觉编码/优化，不读取VAL/TEST。

每图top1/top2固定；动作(c,d)只对原来d第一、c第二的图交换两者分数。两步预算、stop合法，第一步使用某无序类对后第二步禁用该类对两方向。不重新计算top2；真类低于第二名无法纠正。state27维，由支持数量、年龄、原型cos、无标签受影响比例/gap/probability等形成；27→32→32→1共享策略。部署使用无标签评测slice分数，所以是episode-transductive，需披露。

U uniform、CE同网络精确监督目标、E同网络精确终点保护utility、G真正G8 clip、H reward循环错配。双fold×双方向Qa/Qb×3条件stream×6stage，全部288策略/36864控制器更新；3stream不是3独立模型seed。所有策略先封存后读对应未进该方向policy loss的component标签，但历史TRAIN暴露不消失。

G相对U平均heldout期望宏/尾+0.795298/+0.461903pp；相对CE−0.202767/+0.037048pp；相对E−0.233530/−0.162660pp；相对NCM−0.671624/−0.283160pp。只有3/6stage全8项正，GRPO门失败。各学习臂在全部72个heldout及72个full cell中的modal轨迹均STOP，等同NCM；同tree的uniform modal本来就偏短STOP，不能据此称学会abstain。一次闭合的效用转移审查发现utility排序正也不自动转为识别收益，不重复选择器扫描。[协议](../r87_pairwise_decision_policy/PROTOCOL.md)、[报告](../r87_pairwise_decision_policy/REPORT_ZH.md)

## 6. RLAP：R88论文重实现究竟是什么

依据：[冻结协议](../r88_rlap_paper_reimplementation/PROTOCOL.md)、[结果](../r88_rlap_paper_reimplementation/REPORT_ZH.md)。官方论文来源见该协议；匿名代码URL返回404且未确认官方仓库，**不代表代码根本不存在**。用户明确接受论文重实现。下述是本项目实际配置，不冒称作者官方实现。

### 6.1 编码器与持续学习结构

- LAION400M CLIP ViT-B/16，底座冻结，视觉prompt4token、文本prompt4token可训练；编码器权重冻结不等于0视觉优化。
- 每task独立prompt与线性projection，projection初始identity，新task从前task复制初始化；旧task分支冻结。类别使用其birth branch。
- query在推理时通过所有已到达task分支，候选类使用对应birth branch；不读真实taskID。共享policy、easy/hard experts、attention跨task携带。
- 每stage只训练新类完整TRAIN＋旧类每类min(20,n)个distinct原图memory；不训练全部旧TRAIN。tiny类3/6/6不补成20。
- memory在各类到达时按无prompt冻结CLIP到类centroid距离选，稳定tie；各臂共享ID。这与各臂动态特征选memory有明确偏离，目的是隔离权重/loss。

### 6.2 加权视觉原型与三源融合

每epoch刷新detach的exemplar特征bank。共享policy MLP512→384→192→96→1（LayerNorm/GELU）输入为exemplar特征减当前均匀类原型均值，类内softmax得到w_i；最终层zero初始化为uniform。原型由bank特征与w_i加权形成；policy可接CE梯度，prompt经query/text路径接梯度，bank路径停止。

候选类c的困难度按`d_c=1−[cos(v,p_visual_c)+1]/2`。这里文本仅一个类模板，文本样本与其类原型cos恒1，这是本实现的退化和假设。所有候选类在训练/推理均同样计算，避免训练用真标签路由、推理无法复用。

soft router在easy线性expert与hard三层expert之间混合；固定sigmoid斜率10，阈值参数经sigmoid、初始0.5。三源attention输入expert(v)、视觉原型、文本原型，1536→128ReLU→3softmax。最终类分数为三项cos的attention加权和：expert(v)对视觉原型、v对视觉原型、v对文本原型，再除固定分类temperature0.07。标量融合细节是事前重实现假设，不声称论文明确给出。

### 6.3 RLPO目标与“不是标准GRPO”的区别

exemplar reward为“与自身类原型cos−最相近错误类cos−0.1×平均原型cos”；在exemplar batch中减均值/除population std+1e-8，advantage停止梯度。

`L_RLPO = −mean[A_i*log(w_i)] + 0.01*mean_class KL(w || reference)`；reference初始uniform，在task边界做参数EMA0.9。总目标：

`L = difficulty-weighted CE + contrastive + 0.1*L_RLPO + 0.01*MoE balance`。

**没有从old policy随机采样离散动作，没有importance-ratio clipping，不能叫标准clipped GRPO。** CE_WEIGHT移除整个RLPO项但保留同权重网络/其他模块，UNIFORM改用固定均匀原型权重。FULL对CE_WEIGHT可以检验这套重实现中RLPO项的条件增量，不能证明随机GRPO必要性。

训练20epochs/task，AdamW1e-4/wd0.01，batch16/micro2；contrastive实际micro2，保留重复类文本的false-negative约定。contrastive温度可学习、初始0.07、约束0.01–1；这些不等于分类温度。完整18端点和优化器先seal后一次VAL。

### 6.4 R88完整结果

| 版本 | 平均micro% | 平均macro% | 平均tail% | 最终micro% | 最终macro% | 最终tail% |
|---|---:|---:|---:|---:|---:|---:|
| FULL |76.145552|54.233843|33.786034|66.548882|48.472684|31.372347|
| CE_WEIGHT |71.844894|52.148522|33.631689|58.833920|44.160376|31.345231|
| UNIFORM |73.409314|52.890491|33.461124|61.837460|45.836223|31.342999|
| bounded NCM |70.338483|53.117106|38.782777|66.607773|49.419505|33.765130|
| all-TRAIN NCM |76.541214|55.613984|37.313835|72.850410|52.036732|33.126338|

FULL对CE_WEIGHT最终micro/macro/tail+7.714962/+4.312309/+0.027116pp；对UNIFORM+4.711422/+2.636461/+0.029348pp。但仅2/6stage对两个消融的macro/tail四项全正，final对bounded NCM macro/tail−0.946821/−2.392783pp，必要总门失败。三学习臂tiny最终0/0/0，NCM0/0/50（分母1/1/2）。

一次已完成的保存TRAIN权重诊断：FULL在stage0/1/2中最大权重>0.99的类数6/13、7/15、5/17，CE_WEIGHT均0；归一化熵约0.361/0.363/0.610，对照约0.991/0.985/0.978。loss和参数finite并不排除权重极化风险。负总loss可来自负adv降低对应权重，不等于识别更好；没有证明极化是VAL损害的唯一因果，也没有按此追改训练。

## 7. R89：在RLAP基础上加入类别风险与旧类保持

这是用户方向2的2×2机制诊断，不是新GRPO试验或最终创新。四臂从同LAION预训练起步，完整6stage×10epoch：BASE/MACRO/RETENTION/JOINT，JOINT事前指定。R88为20epoch，因此跨轮不同不能归因于模块或事后选epoch。

### 7.1 精确改动

- **MACRO**：仅分类项乘`|T_s|/(C_s*N_{s,y})`，其中N是实际stage训练行数（旧memory＋新TRAIN），使分类风险在期望上class-mean。difficulty系数保留，contrastive/RLPO/balance不重新加权。
- **RETENTION**：本臂前stage模型作冻结teacher。对旧memory，teacher旧类softmax扩展新类概率为0，与student全部已到达类softmax做`KL(q_old_extended || p_student_all)`；temperature1、系数1。按旧类等权校正，在完整microbatch平均，新类行贡献0。它不仅保持旧类相对logit，还惩罚新类夺走旧行概率。旧birth branch特征可复用，无新增teacher图像编码。teacher会保留错误，不是oracle。
- **JOINT**同时加入两者；其他R88模块、路由、memory/优化配置一致。stage0没有旧类，所以BASE=RETENTION、MACRO=JOINT为配对预期，实际校验通过。

### 7.2 完整平均和最终结果

| 版本 | 平均micro% | 平均macro% | 平均tail% | 最终micro% | 最终macro% | 最终tail% |
|---|---:|---:|---:|---:|---:|---:|
| BASE |74.671780|53.935437|35.299962|67.373383|50.492061|36.220928|
| MACRO |72.527022|52.980334|35.133371|61.130745|44.499261|29.301024|
| RETENTION |77.308997|55.132704|34.176689|69.434631|49.613042|31.825021|
| JOINT |76.154373|55.559216|36.712253|66.195526|48.922506|33.696139|

bounded/all-TRAIN NCM与R88同一冻结骨干参考，最终分别66.607773/49.419505/33.765130和72.850410/52.036732/33.126338（micro/macro/tail）。

JOINT对BASE平均宏/尾+1.623779/+1.412291pp，但最终−1.569555/−2.524789pp。仅stage4即1/6stage对三个控制的六项macro/tail同时正；final对bounded NCM宏/尾−0.496999/−0.068991pp，tiny门失败。最终tiny四学习臂均0/0/0，NCM0/0/50。

RETENTION最终micro比BASE高2.061249pp，但尾类低4.395907pp；旧头部宏59.551664%对BASE56.675881%，旧尾31.825021%对36.220928%，说明保护总体不等于保护尾类。JOINT最终正交互macro+5.302264pp、tail+8.791022pp，不等于它超过BASE。

旧class0最终突降仍在：BASE stage4 90.909091%→stage5 7.878788%，RETENTION94.545455%→5.454545%，JOINT93.939394%→4.242424%。干预未解决现象，但不据此把所有原因归结共享漂移或断言teacher完全保住旧memory。报告与全部逐stage/class/oldnew结果见[R89](../r89_rlap_longtail_retention/REPORT_ZH.md)。

## 8. 两条路线能怎么比、不能怎么比

| 维度 | R67 TARS/后继原生LoRA | R88/R89 RLAP |
|---|---|---|
| 底座 |BiomedCLIP|LAION400M CLIP B16|
| 视觉适配 |末两block Q/V LoRA|早期视觉prompt＋文本prompt及共享模块|
| 起点 |共同10epoch监督适配后2epoch RL|从预训练起步，20或10epoch任务训练，无额外SFT|
| 旧数据 |阶段监督/原型可用全部已到达TRAIN|仅旧类min20 distinct记忆＋新TRAIN|
| 跨阶段状态 |SFT携带；R67–R77 RL逐stage独立；R80视觉/controller才真正carry|birth branch旧分支冻结，共享模块携带|
| RL对象 |R67类别动作；R80训练分配；R87判决动作|R88/R89 exemplar权重RLPO，非标准GRPO|
| 强归因控制 |same-target exact、同参数监督、uniform、reward错配等|同权重网络去RLPO、uniform权重；R89 2×2|

就已报告的宏/尾绝对值，TARS最终63.655574/44.737644比R88 FULL48.472684/31.372347高15.182890/13.365297pp，比R89 JOINT48.922506/33.696139高14.733068/11.041505pp。**这不识别骨干、数据预算或GRPO各自贡献。** 原SFT仍比TARS更强，更不能因为TARS绝对值较高就宣称其RL增量成功。

总体准确率方面，R88为66.548882%，但R67本摘要没有对应micro，故不填总体排名。下一优先问题是BiomedCLIP替换RLAP骨干的匹配实验：先保持RLAP内部memory/轮数/loss控制不变以隔离骨干，然后才能以统一骨干/数据预算比较原生LoRA/TARS与RLAP。BiomedCLIP文本编码器与CLIP文本塔不同，必须验prompt插入、mask、冻结/梯度和strict restore，不能只替换权重名称。此实验尚未启动、没有结果。

## 9. R90：当前运行中的桥接诊断

[冻结协议](../r90_evidence_prompt_program/PROTOCOL.md)、[计时审计](../r90_evidence_prompt_program/NATIVE_TIMING_AMENDMENT.md)、[计数修复边界](../r90_evidence_prompt_program/COUNTER_BOUNDARY_REPAIR.md)。本报告快照：字典10/10轮已封存、1230正式视觉更新；控制器已到stage1（第二阶段）、完成25/32contexts的状态快照，累计1140正式controller更新；另一个更细活动快照为1147。尚无VAL/TEST及准确率结果。只是进度，不是效益证据，最终以运行输出为准。

### 9.1 为什么做、真正控制什么

R83说明参数可控仍不等于纠错，R87的旧决策选择器也未胜exact。R90改为：先由有标签支持证据构建**早期视觉编码函数**，同程序编码原型支持和query；不只给已有专家投票。单S支持下，原型权重恒1，但提示程序仍可改变表示。

底座仍LAION CLIP B16。两个slot在视觉block0和6之前，各有4选择：真NOOP（不插token）＋3组可学习4token/768维提示，共18,432个字典参数、16条完整程序。前slot提示持续通过后续block，后slot追加，所有程序同样编码S和query。两slot始终存在，避免短STOP天然modal偏置。

初始13类S=1955行监督训练唯一共享字典10epochs，uniform平均16程序CE，batch16/micro2、AdamW1e-4/wd0.01。每epoch刷新detach的程序特定支持原型；梯度仅query→prompt，无W/F字典梯度。固定epoch10后永久冻结，不对后续类额外SFT。后续各类到达时编码16程序最终512D向量；不使用无效的早期prefix缓存。字典共享隔离控制器效应，**不等于联合GRPO学习视觉权重，也未直接回到原BiomedLoRA完整主线**。

### 9.2 S/W/F角色、state和utility

复用原保存fold0角色：S构建原型，W有标签且部署可用，F仅作controller训练credit、不进state。角色在identity component上分离，不称患者独立。每类总memory≤20，按8/6/6及固定补空规则；旧类只用memory，新类F可用当前全F。tiny1 S1/W1/F1，tiny2/12 S3/W2/F1；不是总共只见1张的true one-shot。

state53维：各类计数/实际S/W数、年龄/new标志、W recall/margin/CE按global/old/tail/new池化，外加presence/slot/前动作、相对NOOP的W响应。第一slot看NOOP，第二slot看(a,0)实际W响应，无F或VAL标签；53→32tanh→32tanh→4共享策略。部署先由S/W决定程序，再分类未见query，不依赖评测slice标签或query-transductive state。

32contexts/stage，S cap循环1/2/FULL；W固定，F为新类全F＋旧类保留F。奖励是macro accuracy−old∪tail正损害均值，系数1。旧类anchor为同episode S的restricted-old NOOP，新tail为full-class NOOP。全部分母保留，不删除难类，重复context不是独立样本。

### 9.3 对照与事前声明

- C：同策略网络、精确期望标准化F CE。
- E：同网络、原始保护utility的全table标准化精确期望；**方向2预定候选**。
- X：同网络、G8组归一化/per-slot clipped目标的精确期望；不等于E。
- G：真实随机G8轨迹、old logprob/clip/组优势；**方向1预定候选**。
- H：G结构但reward固定轮转错配。
- U：uniform；NOP：两slot NOOP；WG：只用部署W的预定贪心程序。
- 另S-only NOOP与S+W+F的all-memory NOOP NCM，明确后者信息预算更强。

五学习臂相同state/参数/初始化/反馈/步数，AdamW0.0003/wd0，joint uniform16 KL0.04，clip0.8–1.2；每stage32groups×4updates，共6stage×5arm=3840controllersteps、30策略端点。G/X/H组标准化与C/E全table标准化分开披露，equal steps不等FLOPs。

E需全局macro/tail胜C/U/WG且≥5/6stage六项正；G另需胜E/X/H且≥5/6stage十二项正。两者final expected及modal macro/tail均需胜S-only NOOP和all-memory NCM，tiny各非负且至少两类正，zero1e-8pp。全10字典/30策略/6memory部署状态封存后才统一VAL。E成功不能叫GRPO成功，G不胜X不能主张随机采样必要。正桥接还需要匹配完整原主线与SFT的研究，不自动达到投稿创新性。

### 9.4 工程故障与保留边界

一次native冷启动计时预测不满足原预算，正式fit前拒绝入场；随后审计固定同参数同数据的冷/稳态重复，改用事前指定第二pass、保留安全系数，并记录所有费用，未缩科学计划或重置24h。

正式字典及初始16程序特征完成后，程序因`program_feature_cache_rows`漏初始化在第一次controller更新前退出。审计全部10字典/optimizer状态、finite特征、memory/support一致，controller/VAL/TEST均0。修复仅补counter和从完整阶段边界继续，复用已封存字典及stage0特征，未重复字典训练、未恢复半途controller optimizer、未改loss/seed/门/原钟。失败日志、成本和所有science私存，安全边界说明已发布。本轮仍进行中，不能提前写通过结论。

## 10. 计算投入与结果完整性

这里列有统一字段且对主结论最相关的费用，不把不同采样范围相加成虚假的总GPU时间。CPU秒、wall秒不是GPU活跃秒，文档/发布/guard等未完整采样的CPU非0。完整逐轮费用见各报告。

| 实验 | 正式优化/端点 | 原run clock | worker CPU | 解释 |
|---|---|---:|---:|---|
| R67 | SFT1180＋四后训练臂944更新；30阶段模型 |3324.531s|4987.270s|包括共同SFT；之后复用SFT不能反复计成新监督训练|
| R80 |6480实际视觉更新，1872保留；576controller更新|23284.564s|32860.212s|候选rollout开销保留，不只报保留轨迹|
| R87 |288策略，36864controller更新，0视觉更新|见原报告|见原报告|CPU缓存分数研究，非GPU训练收益|
| R88 |18端点、360epochs、31260visual更新|72555.694s|83686.107s|另CPU准备及本机toy费用见原报告；native已含worker不重复加|
| R89 |24端点、240epochs、20840visual更新|43549.436s|56266.315s|主worker wall43406.869s，native/toy含worker|
| R90 |计划1230visual＋3840controller，尚未完成|固定24h原钟|尚未封账|工程失败及边界衔接费用均保留，不仅报最新attempt|

所有结果应按预定主候选与控制解释；不按VAL选最好epoch、阶段、类别、arm。极小VAL分母的正确数变化必须披露。保存参数、优化器、特征和完整私有结果以备审计，但公开交付不含它们。本报告不开展旧端点再扫描，也不复活暂停分支。

## 11. 请Pro重点审阅的问题

1. **问题定位**：当前失败是否更应归为“奖励与部署决策的关系不稳定”“表示/原型信息不足”“类别风险与类内组标准化不一致”，还是另有可检验原因？请区分证据支持与推测；不要只凭零优势率/聚类紧度/KL下降指定唯一原因。
2. **真正GRPO的必要性**：R67有同目标exact，R87有同参数CE/raw-exact，R90又有同目标X。什么动作/反馈结构能让组相对优化有明确价值，而不是把已知可微分类目标换成高方差估计？若没有理由，请明确建议移除或改变GRPO角色，而不是先保证它必须赢。
3. **原主线迁移**：若从RLAP借鉴表示/原型构建，怎样在BiomedCLIP原监督起点和统一旧数据预算下接入，并保住原SFT最终宏64.876782%/尾44.924938%的强参照？不要以击败较弱继续监督或不同骨干作为充分成功。
4. **长尾持续学习**：R89普通均衡/蒸馏失败，tiny中一张图就主导几个pp。新机制应增加什么可迁移信息或跨类/阶段结构？如何避免只是强化极少支持自拟合，或用singleton权重恒1伪装有动作自由度？
5. **RLAP重实现风险**：请审查单模板text-cos=1、candidate routing、标量融合、support-bank stopgradient、micro2 contrastive、shared frozen memory selection等偏离，哪些值得另立匹配检验，而不是把本结果当作者代码表现。
6. **下一实验优先级**：BiomedCLIP-RLAP匹配骨干替换与原BiomedLoRA机制迁移，怎样最小化混杂且服务方法主线？请给明确假设、控制、部署state/action/reward、梯度路径、样本信息预算、预定正负门及24h内完整计划。
7. **投稿贡献**：用户希望最终方法有与问题相称的机制完整性和实质创新，不以普通均衡/蒸馏/conditional prompt pool改名作为终点，也不堆无效模块。请指出真正区别于已有conditional prompt/meta-prompt/prompt-pool/训练调度方法的机制；若现有证据不够，明确缺失实验。

约束：不建议根据已经多轮暴露的VAL扫LR/epoch/阈值/k/温度、挑类别或择优方向；不恢复明确暂停路线；不改进行中的R90；新独立实验每次≤24h、每台新增<50GB，预先冻结对照和门。下一研究不逐轮等待审批，但也不能为同时推进两方向而无据制造实验。

## 12. 证据索引与交付范围

- 原TARS及R68–R77：本报告第3–4节链接对应已发布REPORT_ZH与同目录PROTOCOL；历史对照重现同seed不是新seed确认。
- R80–R87：第5节链接完整方法、class/stage计数、query覆盖与预定门。R82–R87的TRAIN查询不能替代独立VAL收益。
- R88/R89：第6–7节链接完整REPORT_ZH，其中包含全部stage/class/oldnew、两种NCM信息预算、费用与保存校验。
- R90：第9节链接事前协议和两份工程边界修订；效益结果待完成，不用训练loss/原生检查代替。
- 公开新增仅此交接报告与主指标CSV；不包含原图、身份、逐例模型输出、Q逐类风险/margin、feature、权重、优化器、原日志、私有路径或第三方全文。Pro如需更多细节，可按索引查看既有安全源码/协议/聚合结果；不要索取或公开私有患者数据来补推断。
