# CPC唯一候选规格V1（R91事前固定科学方案；软件与预算准入前不fit）

2026-11-02。本科学配置在R91首次实现和真实fit前固定，初始状态为未入场。只有软件、初始参数收据、原生检查和完整预算通过才进入正式fit；实际状态见工程收据及STATUS。本文是唯一配置，无配置网格或结果择优。以用户提供CPC提案服务原BiomedCLIP主线，已关闭R90保持关闭。

## 数据与部署口径

固定HK74002六阶段，原SFT全历史信息预算，绝不称严格20memory CL。沿用已核角色：S=原S、B=原W、F=原F，A=S∪B；三侧component disjoint，tiny1 1/1/1，tiny2/12 3/2/1。没有独立E。S/B不交换角色，全部构建器拟合查询只来自B，F仅前置诊断/actor TRAIN信用；底座SFT历史见过完整TRAIN必须披露。每阶段只访问已到达类的该阶段SFT特征版本。原特征顺序绑定已经PASS，不重做审计。

所有类均参与分类竞争。S稀缺档位固定循环1/2/4/8/FULL，实际取min(cap,池大小)，FULL取该类全部S；少类不补复制。每个支持集合内唯一；按manifest固定顺序、seed74002与(stage,update,episode,cap)确定起点循环选，查询B使用该阶段全部B、不对同类重复扩增。所有出现次数/查询分母记录。用于actor的64context也依序循环五档，以context//5确定支持起点。

最终S_dep使用该阶段全部已到达TRAIN，与原SFT正式支持预算一致；仅在所有构建器/actor/静态选择规则完全seal后再建最终state/候选。预先作为F的样本此时可进入S_dep，全部控制同一支持清单；这些行不是独立评价。训练episode与部署支持量不一致明确报告。最终p0直接用该stage原SFT head原型，防止FP32支持重算顺序改变NOOP；新原型修正均从相同p0出发，不只给NOOP特殊不同基准。正式评估仍用原SFT score scale20、stable class-index argmax tie。

## 主构建器及留类诊断

类state为1541维：p0单位均值512；S无偏对角variance512（m1数值0但mask0）；其他类关系向量512；5scalar为log1p(n_raw)/log1001、log1p(m)/log6822、variance可估计mask、age/5、新类flag。n_raw与m分别保存。关系对其他类j以softmax(5*p0_c·p0_j)固定attention，输出sum_j a_cj*(p0_j−p0_c)，不使用标签IDembedding，不使用F/query分布。该scale5是唯一未调优工程默认，不能依据结果修改。

MLP1541→128tanh→512，263424参数，float64 CPU。第一层默认配对随机初始化，最终层weight/bias为0。共享g输出经(I−p0p0T)切向投影，r=u*min(1,.1/max(norm(u),1e−12))，alpha0/.5/1，alpha0必须精确返回p0而非再normalize改变它。norm0安全。AdamW3e−4/wd1e−4，主构建器跨stagecarry参数与优化器；每stage256updates，每update平均4个预定episode loss再单次优化。loss为classmean B CE(20cos,p(alpha1))＋.1*classmean(1−p·unitmeanB)，无F梯度。本方法保存6个stage主构建器snapshot及6个诊断snapshot，禁止沿用R90字典步骤名/计数。

留类诊断另起完全独立g与AdamW，同结构/初始化/更新预算。固定held类别集合为ordered class index mod3=1，即1/4/7/10/13/16/19/22（只用已到达者）。这些类的任何特征、标签、支持和竞争原型都不进入该诊断g的拟合，任何先前stage也不得进入；避免carry泄漏。只在诊断计算时以其合法S作为test-time输入，与全部已到达类竞争。其他类可参与诊断支持/state。不是新的视觉模型seed，也不称患者独立。诊断目标仍F，且F将用于actor，故只能称构建器未拟合类的TRAIN机制检查，非独立最终验证。

训练/保存全部六阶段主与诊断g后，再按各stage不可变snapshot计算前置门。后stage g不回写前stage。主/诊断共2×6×256=3072正式builder optimizersteps，4episode/update；不漏算辅助模型。

## 前置候选迁移门：在actor正式fit前一次判断

每stage固定五个支持档位（每档offset0），共享主/diag对应snapshot，B、F保持原角色，不选档位。以下所有差值都由相同S/query与NOOP比较，单位pp。

1. 主Nfull对NOOP在F上的六stage×五档等权平均macro>0、tail>=0，至少4/6stage的档位平均macro/tail同时>=0。
2. 诊断g的固定held类F：全类竞争下，held macro和held tail的六stage×五档均值均>0，至少4/6stage held macro>=0；目标tiny1不得较NOOP新增净错误。支持量/类别重复不称独立重复。
3. 小块选择迁移：每stage、每支持档，固定最早四类0/1/2/3枚举3^4=81，其他类NOOP；B上最大U，lexicographic tie，固定在F评价。相对该块81动作均匀期望，六stage×五档全局macro/tail均>0，至少4/6stage两项>=0。块固定不能据结果换类或选方向。B曾用于g拟合，F是模块层面分离反馈，不能称全模型holdout。81oracle仅作诊断，不当可部署结果。

三门合取。任一失败则按预定构建器/迁移负结果结束该独立实验的actor与VAL部分，保存主/诊断全部状态、完整覆盖/分母和成本，不替换候选/档位/门或提前VAL。门可失败，不保证CPC会进入actor。

## 六actor与静态控制的唯一目标

actor类state为builder1541＋r向量512＋norm1=2054，MLP2054→64tanh→3，131715参数，final层zero→uniform。全部类参数共享、给定S/state后各类动作条件独立，非自回归。只有S/元数据/候选几何输入。AdamW3e−4/wd0，参数和optimizer跨stagecarry；64contexts/stage，每context4updates，六学习臂共9216updates。每臂配对初始化；G/H采样使用配对uniform但因policy不同不能保证相同动作。所有候选与分数银行停止梯度，不回传g/视觉/支持bank。

B_exact为macro+tail精确期望，风险D为old∪tail逐类positive part，旧类参考为原共同上一stageSFT在同F上的正确率，新tail为本episode NOOP。Z3，lambdaT=lambdaD1，KL为.04*mean_class KL(pi||uniform3)。没有独立E审查池，此处E是actor臂名称。

- M：原提案半解析B_exact＋风险条件counterfactual积分，G8、old4updates固定、类sum、oldpi(k)权重、clip.8–1.2，不按std缩放风险优势。
- S：同M，其他类上下文与q相同，局部k按oldpi采样，不对k积分。局部ratio/优势与M相同，保证估计的是同一冻结old clipped局部目标；另报保留同一精确baseline时实际需要的原生最小计算成本，不能假定免去三k。
- H：同M保留B_exact，仅每类q在k上循环一位，再重算old baseline；不沿组i假打乱。小空间验证其梯度真改变。
- E：最大化B_exact/3−相同KL，无风险，完整精确当前策略概率梯度。
- G：真正factorized-action sampled GRPO，终点reward U=(B−D)/3，G8联合动作，terminal group-popstd+1e−8 advantage，每类ratio clip并sum_c，old每4updates固定，同meanclass uniform KL。明确是factorized局部ratio配置，不冒称joint-ratio PPO或M同目标。
- C：优化离散期望classmean CE＋相同KL，不用软混合原型。每一个update按当前策略重新采8个其他类联合上下文，对各类局部k三值枚举，q_cost停止梯度，loss=sum_c sum_k pi_current_c(k)*q_cost_c(k)＋KL。这是无偏的当前离散期望CE梯度估计，不是大空间exact CE；在不同context分布下不声称目标值估计可直接相加。额外四次上下文/table开销单列，不以同更新步数宣称等call。

固定控制N0/Nhalf/Nfull和U精确expected/modal，U modal tie0；静态搜索每stage在唯一FULL S训练context用F与U，初始全0，按class0..C−1坐标扫两轮，各坐标枚举3候选、只严格改善才更新，tie保持较小动作；预算6*C次联合U评估（即2轮×C×3），最后动作保存，不按VAL挑轮数。在S_dep候选上使用同一alpha向量，不用并入F后的候选重新看F选动作。各stage搜索独立固定，不称可迁移策略。

## 部署/方法归因/计算报告

主指标逐类modal动作部署，tie0，保存最终原型；expected另报。方法M相对同协议原SFT：平均macro和finalmacro>=+.5pp，finaltail>=+1pp，平均tail>=0，>=4/6stage macro/tail同时非负，tiny1/2/12不能新增原SFT正确病例损害。另报单病例贡献敏感性，去除最大单病例正贡献只做说明不替换指标。

目标/策略必要性：M对C、E、Nfull、固定TRAIN搜索的平均及最终modal macro/tail四项全部严格正（zero1e−8pp）；同时报告对G/H/S所有正负。若只胜U不胜强控制，不归因策略学习。M对H的平均及最终modal macro/tail均>0才支持正确风险信用。M与S等步数结果不自动说明估计器效率。

效率口径：每臂预存64/128/256累计stage actor更新的部署动作（相当context16/32/64），都事前固定，全部seal后统一报告，不选best。用于同质量时间比较的预定阈值是相对NOOP在TRAIN F固定FULL支持的macro和tail均非负且macro至少+.5pp；到达该阈值的第一个预定检查点时间/真实rewardcall计数，仅为TRAIN优化效率，不证明VAL质量。若M达到而S未达到，报告censored；若二者都未达到，效率未成立。S原生baseline所需三k成本与固定同context配对审计费用分开，不伪造节省。另在固定小块81上下文计算同目标估计器均值/方差，是局部估计器诊断，不推出共享参数总梯度方差必降。

## 完整预算及尚需执行的入场条件

所有学习在已有冻结512D TRAIN特征上CPU4执行；6actor/2builder估计参数+AdamW端点<.3GB，候选/score/context/日志保守<1GB logical，NFS2.1x＋额外空间按8GB独立cap及512MiB早停，新增实际每server<50GB，保护历史。正式期望6主g+6诊断g+36最终actor+72中间actor（两个额外检查点）及所有optim/搜索/部署信息全部保留。middle checkpoint只报告固定曲线，不作为挑选最终版本。

单独24h从新实验首次代码/数据/软件实际工作开始，含准备/queue/native/前置诊断/分析/发布，不重置；留90s保存/退出。CPU4/400000s、RAM总30GB worker28/group29每2s/file1GiB、独占锁。GPU1 only用于必要Biomed原生两TRAIN图和seal后的VAL图像编码，64GiB RAM/14GiBfree准入/reserved12GiB；无GPU0/旧进程修改/下载底座。是否可复用已保存VAL编码尚未核查，不在seal前读取其数组；默认预留六stage1698图上界10188次视觉编码及原生开销。

准入要求：新编号无覆盖确认、最终config/protocol、所有新数学单测、初始小模型参数固定收据（按提案明确要求可对小init参数记录hash，不哈希大型数据/权重）、真实TRAIN无fit原生读出与梯度/restore、所有必要臂与诊断的保守实测全预算。计时超预算则正式fit前不准入，不缩builder/actor/对照救援。实际准入结果与费用见ENGINEERING_SUMMARY.json和ENGINEERING_NOTES.md；正式训练不等于效益门已通过。

R91 implementation starts under its original24h clock. Earlier read-only design audits are preparation provenance/costs, not new scientific optimization. New software/native attempts stay on this clock.

## Pre-fit implementation clarifications
S retains exactly M's old conditional baseline b=sum_k oldpi*q, so native S also needs all three risk evaluations in general. Its native minimum is counted honestly as3k, not an independently sampled baseline (which would change clipped target). Only its local gradient action is sampled; M integrates it. During divergent training each arm uses its own old distribution with paired uniform RNG; identical-context M/S estimator audit is separate. No computational savings are assumed.
Candidate scoring banks use original FP32 matrix products per candidate, then cast scores to float64 for probability/objective arithmetic. NOOP uses original head prototypes and original20*z@p.T operation, preserving the inference path; builder fitting retains float64 differentiable scores. Native comparison still must pass.

Pre-fit software audit found constant-q floating subtraction could emit tiny nonzero advantage; compute b after subtracting common q(k0), an algebraically identical baseline with exactly zero for identical local risks. Initial CPU software failed this strict-zero assertion; no optimizer/real data/VAL ran. Preserve this engineering attempt/cost on the original clock.
