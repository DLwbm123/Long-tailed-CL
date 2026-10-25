# R40 后主线证据整理：下一轮启动条件

本次仅整理已封存报告，不重新训练、编码、模型前向或评分。当前无训练进程，持续自主研究授权保持。

|候选解释|已具备的证据|已完成干预及边界|当前决策|
|---|---|---|---|
|教师不可靠导致辅助监督失败|D20确认一致率跨域差异；R21按标签过滤可实现|R19、R21、R22、R23均未通过；全覆盖、类中心逐步覆盖也不稳定|不再改变教师阈值、监督强度或类中心归一化|
|旧类功能漂移主导|D36发现新增Text模块改变旧提示欧氏输出|R31功能KL整体恶化；R37旧欧氏Text精确固定仍未通过|不将漂移可见当作主要性能原因；不扫描路由/蒸馏系数|
|Gaussian越界主导|D29确有跨类越界；D34真实TRAIN存在新增类别竞争|R30纠正NIH尾改善但三域门失败；R35固定真实特征回放大幅恶化|两项干预不能证明Gaussian无用或所有真实回放无效；不扫回拉和memory数量|
|线性容量不足|原Image/Text适配器为仿射；读出与骨干有交互|R24线性独立head失败；R28同精度GELU对照最终恶化|不扩大激活/宽度/读出搜索，GELU影响parent初始化的边界保留|
|几何与识别梯度冲突主导|D39固定TRAIN probe跨NIH种子冲突可见|R40全程过滤可达但最终总体与尾无稳定收益|停止原geometry priority，不扫方向、组件和容差|
|映射保护不足主导|D13发现保护一维，类均值差覆盖低|尾类从首次出现已弱；覆盖不具尾类特异性；漂移不指向映射主导|继续DEFER_DISCRIMINATIVE_PROJECTION_TRAINING，不扩大/学习空间|
|融合或记忆度量失配主导|封存两分支均错比例高，D15读出交互存在|R16、R17、D26均失败|不以融合、温度、方差幂或收缩扫描代替训练机制改进|

这些负结果只拒绝已固定候选，不等价于证明其广义机制都不可能有效。跨轮数值差异也不能作单因素因果分解：协议有新目标、设备、精度、预算等差别，分别以各轮事前参照判断。

下一轮必要条件：有一个尚未被上述干预直接检验的机制；明确它能改善哪一训练行为；现有TRAIN证据或必要最小诊断能够区分支持与不支持；在运行前写下单一配置、控制、完整三域门、存储/时钟/停止条件。仅有“可换一个loss/模块”不足以启动。若诊断直接复述既有曝光、漂移、竞争、均值覆盖或梯度冲突，停止重复诊断。

当前没有已满足这些条件的新训练候选，不能把空闲GPU或持续授权当作启动依据。继续主线假设筛选，不重开已拒绝路线，不擅自解除D13暂缓或Spatial/CLS+patch暂停。下一心跳以最新STATE为入口，仍无新证据时静默；有实质新路线再通知。

所有历史开发集均反复使用；无TEST/独立确认/临床结论。保留H0与全部正负结果。

## 外部原始来源筛选（2026-10-24，未据此启动实验）

- [HASTEN 原文 HTML v1](https://arxiv.org/html/2511.15633v1)：本次仅复核摘要和引言。其层级锚定与共享映射保护动机不能替代本项目跨轮负结果，也不能据此解除D13暂缓。
- [Hyperbolic Multimodal Continual Learning，PMLR官方摘要](https://proceedings.mlr.press/v306/liu26dq.html)：仅摘要证据，提出以共享双曲等距变换保留跨模态关系，并区分关系漂移与层级失真。本次没有阅读PDF或附录，未核验理论前提、完整损失、数据协议和计算成本。

“相对关系保持而非绝对旧logit锚定”可作为待审查的不同解释，但现有信息不足以直接立训练：R31已否定一个固定均值功能KL配置，D13显示初始尾识别弱；两者既不等价于否定所有关系保持，也不为它提供正支持。需先读原方法完整条件，并明确它能解决本地哪个尚未检验的TRAIN问题；不能把新论文结论移植为本项目因果证据。当前记为SOURCE_SCREEN_ONLY，不新增或优化论文对比方法，不重跑既有诊断。

## HMCL-CA 适用性核对：暂不立训练

来源：[作者原文 HTML，2609.29329v1，第4–6节与7.1节](https://arxiv.org/html/2609.29329)。本次阅读这些章节；未完整审阅证明附录、代码或全部实验。原2608.09572 HTML入口404，未将其记为全文已读；新题名版本单独列出，不假设两版本实现完全相同。

原文在冻结编码器后的线性Lorentz头上推导共享旋转约束，并将实际AdamW位移校正后向任务起点收缩。其结论需要固定的受保护表示系统及相应正则条件，低秩实现只保护保留子空间。

本地适用性判断（推断）：HASTEN新增Image/Text适配器改变头输入，不能直接套用冻结输入的保证；当前已有postAdam实际增量保护，所以仅采用校正放置位置不构成新的本地主线机制。照搬CA还要重新定义保护系统/空间，不能绕过D13暂缓。其保持目标也尚未解决本地首次尾类识别偏弱。状态改为NOT_SELECTED_ARCHITECTURE_AND_EVIDENCE_GAP，不开展论文比较复现、不据其摘要直接启动本地训练。该判定不否定原方法在其自身条件下的结果。

## D41/R42 新证据

D41未训练首任务TRAIN探针支持固定首Image/Text恒等fc初始化，零步改善不预言训练效果。R42完整45全新阶段/44070步/135计数检查通过但BOTH门false，NIH final/tail较H0−1.333333/−.952381pp，ISIC+.512340/0、HK+1.425636/+2.722089pp。保留H0，不选择HK采用。固定初始identity路线关闭，不扫描矩阵、幅度、单侧、alpha、阶段、LR或融合。数据集合/时点/分母不同，不能用D41与R42数值相减归因“训练破坏对齐”。完整报告在r42_initial_identity/RESULTS_ZH.md。

## D43：固定适配器数平均不支持

36增量边界990两模态仿射范数行，无优化/分类评分。原sum范数变化较小，除以模块数使跨阶段绝对log变化更大；NIH Image/Text预设联合条件均0/3seed，其他域也不支持，停止固定mean路线，不扫sqrt(K)/指数/单侧/stage/class缩放。不是训练性能证据，保留完整负结果。

## D44：原生对比风险归约的新诊断

新增21个条件TRAIN方向cell（NIH6、ISIC3、HK12），每域3seed，A/B严格不相交，0优化/VAL/TEST。原生对角双向contrast只改风险归约，区别R05/R14/R22/R38。NIH3/3seed宏/尾cos增益大，但ISIC尾下降.614159、HK尾下降.023281，预先跨域门false，不启动固定归约训练，不放宽门或扫强度/尾权重。完整数据见d44_native_class_risk/RESULTS_ZH.md。

后继待核查问题：有限样本下原每类mean/cov估计的不确定性是否足以支持不同的记忆统计机制。该问题不同于D26全局对角度量、R30修正抽样点、R35固定exemplar；当前仅列为待检验，不宣称已有支持，也不直接启动收缩参数搜索。无活动run时应做限定研究动作，不再仅重复状态读取。

## D45/R46：少样本协方差的任务驱动改进

D45 54类×fold TRAIN密度cell，固定双fold样本数/维度收缩改善NIH初始尾及三域宏NLL，0优化/神经前向/VAL/TEST。只支持另立R46回放统计估计实验，不是准确率支持。R46仅旧类Gaussian派生协方差，原mean/cov/projection/NCM/fusion不改，P限已到达旧类，lambda由n,d固定，无收缩/正则搜索。R46现已完整exit0，BOTH门false：NIH平均/最终/尾较H0+.091784/计数持平/+.634921pp；HK尾−1.899483pp，超伤害限。135checks/45eval=36new+9shared/12120newsteps通过，保留H0。拒绝固定协方差收缩，不扫lambda/P/正则/对角full/采样数。密度收益未转化为稳定识别收益。

## D47：固定语义先验均值规则不支持

原少样本均值不确定性尚未被协方差干预检验，故采用其他fit类TRAIN图文offset加原固定文本为先验，固定方差/先验差距权重，双fold54cell比较native NCM heldout TRAIN判别。NIH宏+.009260/−.449422pp、尾0/−3.333333pp；ISIC宏−.134000/+.375306、尾0/0；HK宏+.003566/+5.440309、尾0/+16.666667。事前NIH双fold宏尾各+1pp门false，不采用有利HK/fold，不启动，不扫先验/权重/prompt。统计拟合与TRAIN评分实际存在，无优化/神经前向/VAL/TEST。报告见d47_semantic_mean/RESULTS_ZH.md。此轮已实际推进独立有限机制诊断，后续需要不同且事前可区分的新机制，不能重复本诊断或救援固定均值规则。

## D48：图像局部关系机制的有限筛选

核对原生loss及R22/R23/R24/R35干预，直接视觉关系目标尚有结构区别。先检验固定同support类均值与cosine 1NN的双fold TRAIN判别（每类支持/查询各最多64，严格不相交），54cell；NIH尾−10.344828/−13.333333pp，ISIC尾−38.461538/−42.857143，HK尾−32.172619/−6.068314，预设可行性false。不启动依据该固定近邻结构的训练、不扫k/支持数/距离。支持预算与稀有类support不等限制保留，不据此否定全部视觉关系学习；本轮确有代码核对与新的TRAIN区分诊断，不是状态重复。完整见d48_visual_neighborhood/RESULTS_ZH.md。

## D49：最强竞争类二元任务风险筛选

原训练有pairedcontrast而class logits只用于训练正确率，故检验不同主损失softplus(max错误classlogit−真实classlogit)，原几何项保持；不是R08全类CE或D44归约。21个条件TRAIN梯度cell对共同disjoint全类CE风险，NIH宏/尾cos变化−.902356/−1.117401，ISIC+.907831/−.355873，HK+.288995/−.584980；NIH0/3支持，跨域尾受损，预设门false。不启动训练、不扫margin/softness/竞争类数/weight/seed/fusion。参数与层级RAM恢复、finite通过不等于效果。直接任务风险方案确已诊断，下一候选需不同机制，不重复D49。完整见d49_hardest_class_risk/RESULTS_ZH.md。

## D50：直接视觉类原型反馈

Image adapter通过实际TRAIN行与视觉类均值的cosine CE共同学习，区别R23对TEXT CE、R24 detached head、R22对TEXT CE、R32层级叶、D48原始1NN、D49替换主风险；不改读出/融合。新增固定系数1、原目标保留；raw means与A均排除B，21cell对共同B原hyper全类CE方向。NIH宏/尾cos变化−.068622/−.091538，ISIC+.138793/−.084550，HK+.072026/−.054517；NIH0/3支持，跨域false。新梯度仅活跃Image adapter且非零，参数/层级RAM恢复通过但不等效果。拒绝固定反馈、不训练、不扫coeff/temp/normalization/priority/seed/fusion。方向门是局部筛选而非性能因果，不否定所有视觉目标。完整见d50_visual_prototype_feedback/RESULTS_ZH.md。

## D51：真实fit-only类别文本参数化学习pilot

用每类zero-init Text_encode残差测试共享文本仿射耦合之外的结构，原paired/hierarchy/geometry目标保持；与R37/R24/R42不同。三域3seed双fold两arm共36个全新fit-only模型，严格TRAIN奇偶分折、每类每折fit/holdout最多64；2epoch，不加载learned checkpoint或VAL。成功552steps，加attempt1已完成native18steps共570实际步；候选firstforward设备错在0候选更新，原钟修复，失败保留。NIH三个种子两折平均最终宏BA变化−4.186063/−.361530/−1.632184pp、尾−19.367816/−27.198276/−8.160920pp；ISIC/HK平均保护门失败。预设NIH0/3支持，不启动完整训练、不扫幅度/正则/seed/epoch/融合。新参数初始零/核心同control/RNG/非零finite梯度/模型训练均有核验；两epoch小样本初始阶段不是完整增量或泛化结论。完整d51_class_text_residual/RESULTS_ZH.md。

## D52：删除可学习公共平移的结构pilot

根据D13强公共谱但非bias因果、D51额外类自由度无益，另立Image/Text均永久无bias的原点保持结构；原生损失/alpha/map/projection/fusion不改，剩余初始化与RNG exact。不是一次init/模块平均/读出中心化。36全新fit-only模型、552steps，2epoch严格TRAIN双fold，NIH74002/3/4宏/尾Δpp分别+.260955/+1.695402、+2.079741/+2.586207、−.916128/−4.971264；仅1/3支持，ISIC尾平均约−4.12pp超限，门false。HK局部改善不选择采用，不启动、不扫单侧bias/幅度/初始化/epoch/seed/fusion。谱仅提供有限动机不证明bias原因，不否定全部受约束结构。完整d52_origin_preserving/RESULTS_ZH.md。

## 用户要求综合分析后的D53整体结构试验

共同证据与新方向记录COMMON_FAILURES_AND_STRUCTURAL_DECISION.md。测试视觉粗专家/语义fine专家的联合后验，重构主训练目标和预测，含native/flat/factorized匹配主线消融，54新fit-only模型828steps，非融合调参。相对native宏/尾pp：NIH−.036877/−11.642720、ISIC+11.793155/+44.047619、HK−7.494427/−4.277178；相对flat NIH+1.987488/−1.130268、ISIC+19.276557/+68.772894、HK+1.647287/+1.835029。NIH0/3双参照支持、跨域false。ISIC尾类为singleton group，不将其收益泛化成层级普遍有效，不择域，不扫tree/routing/temp/prior/expertblend/coeff/seed/epoch。停止每轮强求新编号，保持综合机制研究和真正任务学习证据。完整d53_factorized_decision/RESULTS_ZH.md。

## 用户要求的训练充分性审计与结论纠正（R54）

TRAINING_SUFFICIENCY_AUDIT.md记录：D51–D53单模型18/8/20steps，对原完整初始9400/630/620步仅约.19%/1.27%/3.23%，并且fit头类截64改变类别分布。无平台/holdout轨迹，筛选未标定完整训练预测能力，不能否定整个结构，也不能笼统称所有负结果只是未收敛。保留原门false与全部数值，结论限定为screen配置；D41→R42和D45→R46的正筛选失败不能估算普遍预测率。用户明确要求复核，覆盖此前针对D53追加日程的执行停止；唯一主线R54验证同一因子决策，4h12GB，先uncapped TRAIN-fold10epoch全曲线，再事前承诺三arm完整原数据/日程/增量训练（不根据A选赢家），135全新stage132210full steps；完整BOTH采用门保持。当前后台运行，以STATE为准。


## R54完整训练充分性验证与R55/D56后继
R54完整18audit/135freshstage、PhaseA31980+PhaseB132210=164190实际优化步完成，factorized/flat原BOTH门false；fixed factorized拒绝不外推所有结构无效。R55选择类别均衡视觉适配后冻结+真实统计全局解析判别，完整9base20epoch63900steps/45analyticstages/90readouts，均衡候选两门false，ISIC/HK尾类相对H0下降约7.96/7.41pp；不选有利HK宏、不扫lambda/dim/map/epoch救援。D56全TRAIN重代入零优化核对显示balanced尾TRAIN83.55/100/99.66%却历史开发尾12.38/51.23/30.41%，泛化落差不能仅归训练不足，也不证明唯一过拟合原因。下一研究须围绕全局类均衡下的少样本泛化约束和旧新竞争，拒绝散点模块。全部数值/全seed与曲线见各RESULTS_ZH与results，public179a4b7。当前无训练进程；持续自主研究/唯一hasten不暂停。

## R57 完整共享低秩判别与真实历史风险
用户明确要求从公式落实完整主线，冻结R57：unit512真实视觉均值无类专属可训练参数，共享rank16的I+UV^T、F范数硬界.25；全部已到达真实TRAIN特征完整记忆，每stage10epoch全局等类CE，共三域三seed45stage52110步。不用短pilot筛选，合成0optimizer工程预检通过后真实后台启动，原4h6GB时钟保持。假设是限制自由度且直接训练旧尾/新头共同边界；不保证泛化。不是R55随机高维ridge参数救援或R35固定8行回放；额外真实记忆成本和不同训练曝光披露。当前运行，以STATE现场为准，source/protocol public51061ab；效果尚未知。

## R57完整结束：真实历史风险和强共享约束仍未带来判别收益
exit0总206.114048s/202420224bytes，45完整训练stage52110步/90保存响应/45x11曲线及约束、均值prefix、memory与restore全部通过。shared平均/最终/尾NIH28.635582/21.777778/10.476190 ISIC51.371887/45.768234/57.371795 HK52.783830/51.403691/30.026540；H0/R28两门false、全域joint0/3，保留H0，固定配置拒绝不扫rank/bound/scale/epoch。45/45终点boundactive、末三objective描述平台；无holdout不能断言泛化原因。保存权重谱显示NIH/HK主方向能量>99.9%，effective rank近1；共享与rawNCM全曝光预测改变仅9/4050、6/1935、31/22014。尾BA三域与rawNCM相同，不等于所有逐例预测完全相同。NIH最终TRAIN宏24.32尾29.10亦弱，不是所有失败都高拟合过拟合。后继须解耦共享判别与类分布证据的结构问题，不再强弱约束扫参；不能把rank塌缩当作放宽bound肯定有效。无活动训练，持续自主研究。

## D58归因边界与R59完整分布判别
D58三域三seed初始/最终18全TRAIN cell，0optimizer/新拟合/VAL/TEST，exit0总26.007378s/163840bytes。确认R57双因子尺度U约4→33–257/V约.001–.008且乘积固定.25；初始完整直接矩阵梯度已有主方向，但初阶Adam乘积方向有效秩约3.8–4，不能说首步强制rank1。最终因子方向近rank1，直接G仍有非径向分量，不是固定rank可行KKT残差。早期moments/参数未保存，不能定位塌缩时刻或因果分离Adam/投影。单均值不充分仅是假设非唯一根因。R59以512共享正权度量代替双因子，以每类全真实支持角核混合代替均值；TRAIN查询自身支持严格剔除、其类分母n-1，全部已见类全局等风险，梯度经过query/support两边。初始/每stage10完整epoch，全三域三seed45stage52110步，4h6GB，已真实后台运行，预设distribution_fixed/centroid_shared消融，不挑读出替代主候选。R59不是R57半径/秩救援、D48最近k或Gaussian密度选参，效果待完整终点。source+完整D58 publicdaf700b，原门/预算/历史保持。

## R59完整结束：分布证据的收益依赖域，原采用门失败
三域三seed45stage52110steps135读出/495曲线核验，exit0总1626.785019s/200187904bytes。候选平均/最终/尾NIH28.255438/19.333333/8.571429 ISIC48.092563/43.697462/49.358974 HK58.635087/55.124069/41.277032；H0/R28两门false全域joint0/3，保留H0。共享metric仍有226/465/453有效维数，45/45stage NLL下降和目标描述平台，不是R57近rank1因子失衡，也不证明全局收敛。LOO非独立holdout，不能和R55重代入差值宣称泛化解决。相同learned度量，分布vs单中心HK尾+11.03pp、ISIC约-8.01、NIH持平；学习度量相对固定分布HK反降，不择域/ablation采用。旧尾错到任一后到head描述计数各seedNIH13/82、ISIC10/25、HK8/70，不能只靠old/new校准解释全部错误。后继转向医学图文语义/真实全局等类判别接口，不重复Gaussian/均值/融合/短pilot或参数救援。无活动训练、持续研究。完整公开交付以STATE回执为准。

## D60联合接口审计与R61完整匹配训练
15全到达TRAINstage几何（另attempt1重复NIHstage0，共实际16cell），0optimizer/训练新分类器，存在类均值统计估计；原clock148.756273s/exit143360bytes。严格textbitwise失败maxabs5.96e-8数值舍入，修为记录差异且要求2e-6，输入/指标未改，失败保留。原始语义宏不强于视觉LOO，不能当oracle或冻结表示上限；原NIHauthor TRAIN68058行20列恰1positive及原manifest对应无明确错误。配对image→text CE重复文本引入log(batchmultiplicity)的梯度恒等式已核，但不解释native全部loss。联合综合见R54_R59_JOINT_SYNTHESIS.md。R61单主线共享视觉representation+固定语义cosine决策+全真实历史等类风险，matched paired目标和tasklocal语义置换两完整控制，初始/数据/batch/日程/预测公式一致。全三域三seed135stage156330步180读出，4h6GB原钟，已真实后台，不能从D60预言效果。用户随后授权离开12h自主继续、30min、不逐轮审批、无总GPU时间预算，未来run不沿用任意4h上限；当前R61冻结协议保持，新增累计<50GB/server，确认无依赖临时产物可清理，历史证据仍保护。

## R61完整结案与原视觉骨干适配准备
exit0总2811.676530s/247726080bytes(closeout249212928)，135stage156330steps/1485curves/180responses核验。候选avg/final/tail NIH30.664903/23.222222/11.111111 ISIC52.242356/50.981709/64.529915 HK55.485786/56.351127/36.751135，H0/R28两门false，保留H0。完整matchedglobalCE比paired三域宏更高但尾更低；正确语义非一致优于task内置换（NIH置换更好，HK原语义宏更好），不能认定loss或text独自主因。共享representation比frozen语义宏提高，原cache含可学习信息，不是内在不可分上限证据。完整结果b607d58已双端交付。
后继R62准备进入视觉backbone内部共享适配，非再换global512读出。已做有界原source目录stat、每域1TRAIN图像实际解码、旧Biomed权重存在，q107v已import并读取原Timmforward接口，sourceGPU有余量。尚未做新编码/大复制/GPU训练。规划冻结前10visionblocks缓存原生FP32tokens，训练末2attentionq/vLoRAr8，保持原pool/proj/semanticglobalrisk及真实历史；源单chunk暂存、pro逐domain可再生成cache生命周期严格<50GB实际窗口占用，不删结果/重置峰值账。计划20h独立原钟（非任意4h），正式协议/缓存数值等价/跨serverguards尚需工程实现，不冒称已启动。用户12h持续授权/唯一hasten保持。


## R62完整结案：原生视觉适配提高若干宏指标，尾类泛化仍失败
原20hclock全程保留，exit0总28302.872598s，45stage52110updates/208050microbackward/495curves/90保存响应核验全部通过。候选avg/final/tail NIH34.388521/23/6.984127 ISIC53.873505/52.887993/52.457265 HK57.944613/58.341124/33.689028，H0/R28两门false、三域两参照joint均0/3，保留H0。NIHavg对H0+2.033583但final−3.333333、tail−5.079365；ISIC/HK尾−6.730769/−4.133471。45/45末三objective未达1%描述平台，但最终TRAIN尾NIH约94.63%、ISIC100%、HK约81.22%，开发尾6.98/52.46/33.69，未平台与泛化落差同时存在，不以追加epochs救援或宣称已定位唯一过拟合因果。初始/后到尾的出生和终点全类计数保留。源NFS st_blocks约2倍导致storageguard在0optimizer时停；分段ACK保留36416已编码行、原钟续完，全部失败/费用保留。sourcepeak44.689GB/propeak41.894GB低于45GB，各域全部科学产物保存后仅删再生token/base，历史保护不变。当前无活动训练，不再监测旧PID。
下一选择支持条件化的共享视觉适配：其他真实历史支持定义类别、组不相交query驱动风险、同一决策覆盖训练和增量推理，替代固定文本槽直接拟合。只读TRAIN metadata审计各类>=2 identity_component可实施；不把component普遍称患者、无优化/VAL/TEST/效果筛选。R63已写decision接口和工程计划，尚无冻结完整协议/模型前向/新编码/GPU训练；下一必须完成fold与交替刷新训练实现及预算，预检后直接全三域实验，不停在公式或用短pilot否定。