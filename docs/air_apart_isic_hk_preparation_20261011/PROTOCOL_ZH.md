# AIR 与完整 APART：ISIC/HK 外部方法比较准备协议

用户批准先核对作者代码、预训练和预算、做CPU兼容检查，当前残余漂移及RFMiD差异先验两轮完整交付后再安排GPU实验。当前状态为CPU_PREPARATION_COMPLETE，0新增GPU进程、0正式更新、0真实GPU预检通过；没有启动后台GPU队列。

## 作者来源与移植边界

- AIR作者仓库 [Analytic-Continual-Learning](https://github.com/ZHUANGHP/Analytic-Continual-Learning/tree/0afd6bf87622e808ce7f730f8556e78229b9e758)，固定commit `0afd6bf87622e808ce7f730f8556e78229b9e758`。核对AIR.py、GeneralizedARM、RandomBuffer及ACILLearner.learn。[论文v2](https://arxiv.org/html/2408.10349v2)与通用仓库默认参数有差异，本文比较为明确配置的医学迁移，不称论文原数据集复现。
- APART作者仓库 [APART](https://github.com/vita-qzh/APART/tree/f3c5b8da5908b2f78266744a737f7f6bff45a43d)，固定commit `f3c5b8da5908b2f78266744a737f7f6bff45a43d`。核对Learner、AdapterVitNet、双池ViT及apart_cifar_shuffle配置。[论文](https://arxiv.org/html/2409.07446v1)。不使用本项目已添加ConCM等机制的第三方Learner冒充完整作者方法。两作者MIT许可随准备协议保存。

只改变数据接口、固定权重读取、医学频数embedding容量、封存评价接口和checkpoint必要状态。保留作者核心公式、损失、路由、优化器及双池推理；不加入项目解析竞争头、local_transport、残余漂移或校准。

## 数据、标签与评价权限

现有train/开发val只读复用，不下载或重新抽样，不打开test/reserved。ISIC18718train/295val，8类、order=[4,7,5,1,2,6,0,3]、tasks4+2+2。HK6821train/1698val，23类、order=[4,5,6,22,20,1,2,15,14,12,10,11,19,16,7,0,17,9,18,13,21,8,3]、tasks13+2×5。seed74002。原标签映射为order中的列索引，推理只允许已引入列；不能把原label当作增量列。train/val image及identity_component交叉0；不扩大对患者隔离/临床泛化的断言。

原数据供给和每类val分母详见DATA_BUDGET_AUDIT。ISIC class0的train10529、HK最大train736，HK极少类val5/6仅1、19仅2、21仅4，逐类变化不夸大。AIR/APART使用当前任务全部原train，不人为下采样尾类。原方法optimizer使用既有80/20 fit/meta、边界统计包含全部train；两个外部方法没有该meta机制，不强塞80/20，报告optimizer曝光差异。开发val不是meta或超参选择集。

## 共同预训练

都使用与现有ISIC/HK同一个缓存权重文件：timm `vit_base_patch16_224.augreg_in21k_ft_in1k`，ViT-B/16、224，ImageNet21k预训练再ImageNet1k微调。不调用在线pretrained=True自动下载，也不沿用AIR通用代码中torchvision SWAG/384默认。只依据当前缓存来源及实际核心tensor映射声称共同初始化，不把同架构误称同权重。

CPU已对冻结timm ViT strict加载、APART的qkv拆分/mlp重命名后198个核心tensor精确匹配；仅adapter池、router和原生分类head为新增初始化参数。没有重新做全量文件哈希。正式运行仍需锁定该文件路径和同一映射检查。

## AIR 固定日程

冻结768维backbone及作者随机ReLU投影。正式投影8192维、float64、无bias，原RandomBuffer初始化，seed74002，所有阶段同一投影。每任务完整train一次、batch32、固定推理transform，无基任务SGD、无adapter更新。每任务边界解一次头，不每batch读取val。仅保留聚合A/C/n、投影和head，不保存旧逐样本特征缓存或图像。

使用作者GeneralizedARM的原归一化：alpha_c=(1/n_c)×K/sum_j(1/n_j)，A=sum(alpha_c A_c)+gamma I，B=C diag(alpha)，W=A^-1 B。gamma=2000，来自v2论文ImageNet-R参数，作为两个医学数据集统一的事先迁移选择；不是医学验证的最优值，不进行gamma/投影维度网格或val选参。通用仓库默认gamma=.1不是本次配置，必须在表格中披露差异。

ISIC逐任务批数126/90/370，共586；HK123/4/2/24/22/41，共216。共802次解析fit batch、9次阶段solve，0optimizer/adapter更新，不用“0训练成本”表述解析计算。仅逐类8192² float64 A最终占ISIC4.0GiB/HK11.5GiB，额外head/聚合/逆矩阵及临时张量还需余量，实际GPU峰值待真实预检。

作者cnt是普通tensor、没有注册进state_dict；checkpoint必须显式保存/恢复cnt，不能仅存head或state_dict。CPU已验证恢复后重新求解精确一致；这只修复保存接口，不改变ARM。

## 完整 APART 固定日程

保留两个adapter池、每池5个adapter、bottleneck64、instance/class-frequency router、两个可学习linear head和pull约束。训练使用作者loss1、双head ensemble CE、router加权few CE及两段match loss，epoch0..4使用频数theta100目标，epoch5..9使用作者第二段约束。推理用两池logits相加、仅seen列，不替换成解析头，不添加headnorm校准/ConCM/FD/回放。

每任务10epoch、batch48、workers4、drop_last=false。作者AdamW原有效pool lr=.0003、其余trainable lr=.003、betas(.9,.999)、eps1e-8。作者配置weight_decay=0，但param group键写成`weight decay`，PyTorch实际weight_decay=.01；保留作者实际行为，不悄悄改为0。作者get_scheduler在init_cls!=increment时每任务cosine，Tmax10/eta_min1e-5；现有项目改编版的仅首任务cosine不能拿来替代。每任务重置optimizer，无梯度clip新增。

ISIC更新840/600/2470=3910；HK820/30/10/160/150/280=1450，两条共5360。所有train图每任务10次optimizer曝光；不是与原两epoch方法相同预算。当前只授权这4个外部轨迹，不偷偷追加2epoch APART或10epoch项目方法。报告为共同数据/预训练、方法原日程的迁移对比，不能声称等更新量条件下胜负。

作者频数embedding只有501行，医学真实频数会越界。仅容量扩为ISIC10530/HK737，保留原501行精确不变，追加行使用seed74002+1000003；不截断/压缩频数，不按val决定容量。loss仍只使用当前已seen类的真实train频数。

作者incremental_train直接取source=test，且每5epoch计算test accuracy，与封存协议冲突。后续医学task driver使用current-train loader，原_train核心保留；_init_train仅去掉周期评价与其日志分支，保留纯训练日志。CPU合成fixture中，封存版与原作者完整10epoch循环参数精确一致；真实运行不能用假0评价值代替原test指标。全训练结束前不得产生开发val分类结果。

## CPU证据与未完成项

CPU_COMPATIBILITY_CHECK PASS：AIR作者ARM逐批增量与直接联合加权ridge一致，max相对残差1.95e-16；固定投影/显式cnt恢复通过。APART作者完整12层ViT实际224输入CPU forward/backward，双池/router/两个head有限非零梯度，核心冻结；两数据集所有task的原损失/optimizer/scheduler及封存循环精确回归。Synthetic Tiny训练130更新只为fixture，不是正式科学轨迹。未读取医学影像、test或新分类指标。

首次CPU fixture进度wrapper返回range，缺set_description，原作者训练日志处失败；第二次修复wrapper后PASS。两次私有回执保留。驻留10.700723+21.807225=32.507947 CPU秒，成功pipeline13.429077已包含在第二次驻留中，不再相加。APART首次代理源码获取TLS35在解包前失败，同代理再次获取成功；0GPU/0科学重跑。不能声称全准备过程无工程故障。

未完成：生产医学worker完整train/checkpoint/eval接线验收、4个真实GPU零更新预检、8192维矩阵实际GPU分配/求解峰值与耗时、任何新效果。CPU小维ARM不是正式8192维GPU验证，不能仅据CPU PASS宣布GPU训练ready。完成生产接线须保留同一方法/参数/预算，冻结源码后再预检，不能先跑再补协议。

## GPU顺序、屏障与交付

本次不启动GPU进程或修改当前两轮。下一阶段先完成module_drift10和rfmid_contrast4的GitHub/NAS完整交付，核对各本机/NAS独立DELIVERY的commit一致、匿名报告HTTP200、NAS_saved，PROGRAM终态/active空且实际worker退出；不能只因GPU空闲或训练结束绕过交付。

届时完善并冻结生产worker/collector/launcher，真实NAS nfs/空间/小写读probe和GPU0/1每卡>=32768MiB准入；不借2/3、不杀其他进程，不回退根盘。中性入口，私有路径经环境传递，启动后完整ps/NVML核对。4项真实零更新预检全部PASS后，队列ISIC_AIR→HK_AIR→ISIC_APART_full→HK_APART_full；两卡只按既定队列并行slots，不承诺某方法独占某卡。AIR预检为冻结表征/投影及shadow ARM计数与solve，不要求不存在的adapter反传；APART预检为真实firstbatch backward、0optimizer，核心不变/双池router/head梯度有限。原生方法不混用预检规则。

全部4条按精确预算TRAINED后，统一开发val评价全部9个数据集阶段。正常4preflight+4fit/train+4eval=12费用记录，每次驻留单独计，即使job/operation相同不去重；失败全部额外计、停止依赖、不自动重跑或放宽检查。24h从未来实际launch计，CPU准备不冒充GPU时间，历史独立round成本只累加一次。当前没有GPU启动时间或deadline，不能制造新24h回执。

外部表完整列AIR/APART和原static_pc/local_transport，当前已授权漂移两臂交付后也完整列出，不按val择优隐藏。原结果只严格匹配复用、不重训/重新收费。报告final/average BA、Acc、tail、forget、old/new、macro F1，全部stage/31类n与recall、固定训练频数组正负、trainable参数/状态存储/真实训练与推理费用；外部方法完整超参和预算差异必须随表公开。单seed、重复开发val，仅开发证据，test继续封存，不称医学论文benchmark精确复现/临床泛化。

真实结束匿名收集只聚合指标/状态/费用/标量诊断，禁止影像/模型/逐样本/身份/原始日志或私有路径外传。GitHub显式代理推送，核对账号/分支/范围/remoteSHA/匿名精确报告HTTP200，再NAS匿名public_delivery和独立DELIVERY/HANDOFF；交付不完整不得称结束，不另开实验。
