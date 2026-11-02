# R93 配对支持分类间隔一致性：唯一冻结机制诊断

2026-11-02。独立新实验，不恢复R91/D92。目标仅检验配对支持margin一致性是否在保持识别收益时减少支持成员扰动；不称最终创新或GRPO效益。lambda=1唯一默认，不搜索；若失败结束此机制，不增actor救援。0encoder优化/actor/VALTEST，全历史BiomedSFT预算，原S/B/F及角色历史暴露限制沿用，不称独立患者或受限memoryCL。

## 模型、数据与配对

原R91 Builder1541→128tanh→512、切向残差cap.1、alpha0/.5/1、float64，AdamW3e−4/wd1e−4，原CE_B+.1Bmean辅助保持。全六stage当前已到达TRAIN特征，保存版本/roles直接复用，不读VAL。不用R91已训练参数，从相同seed74002初始化。control lambda0与candidate lambda1，分别main/held两个独立模型，各自carry6stage；held永久排除mod3=1所有拟合支持/查询/竞争原型。

每model每stage256updates，每update4个支持对，两个arm完全相同支持对/查询/init/预算；总6144buildersteps/24576pairbackward/49152支持episode。每pair两侧监督loss平均，新增classbalanced mean_{j!=y}((margin_a−margin_b)/20)^2。两侧均反传，原型和视觉base不训练。main全已到达B，held仅非held B。

cap1/2/4/8/FULL按pairindex mod5；每类m=min(cap,n)实际数一致，不复制；sorted原S池，a offset=(74002+100003stage+pairindex//5)%n；若m<n，b offset=(a+max(1,m))%n，否则b=a。循环抽取m个唯一成员，两侧可重叠但不称独立。FULL和singleton两侧相同是合法无扰动，不制造FULL重复；其他类变化仍可能影响该类分数。保存每pair每类实际数/成员重叠汇总，私存完整IDs可重构，不公开。

## 新工程准入

新软件：pair margin loss在两侧相同为0、共同逐query全类分数平移不变、class permutation等变、有限且双侧梯度、真NOOP和参数restore。实际TRAIN小pair上原生Builder梯度/有限检查，无optimizer/encoder。采用固定cap4新实际pair的第二次完整forward/backward计时，第一次也计成本；不是挑最快。预算按最坏C23/B1713相对初始C13/B行数的保守比例再乘4，另3600秒分析/诊断余量，包含全部6144updates×4pair。若原clock剩余不足则fit前不准入，不改步数/删held/control。计时CPU/workerwall全保留。

## 一次固定TRAIN机制读出

所有24builder+optim阶段状态封存后，才统一对F诊断。每stage×五cap×固定四个pairindex(5*10000+capindex+5*j,j0..3)，两侧相同计数、循环offset，始终披露FULL重复无新信息。共120情境/arm，各side分别NOOP/half/full，主结果固定full、half完整报告不选best。held只取未拟合类指标但全类共同修正，禁止称逐类自身修复。

评估query F，loss间隔一致性的F版本为新稳定性指标；classbalanced全竞争对差平方，实际可变化支持情境总体和FULL零扰动分别报。识别报告每侧full−NOOP宏/尾、old/new、head/tail，均值及六stage、tiny病例得失仅私有聚合。稳定下降不能代替准确率，残差范数与相对NOOP预测变化率另报，排除靠归零修正伪稳定。

唯一机制门：main在可变化支持情境F margin variance候选/control<=.90且control>1e−12；候选对control全局宏/尾变化均>=0，至少4/6stage宏尾同时>=0；候选相对自身NOOP全局宏>0/尾>=0；held对control全局宏/尾均>=0；tiny不新增相对control正确病例损害。zero1e−8pp。任何失败科学negative，仍完整报告。无pvalue/独立重复声明，F开发使用已披露。无新actor准入、无需根据结果加训练。

## 部署边界

本轮不读VAL，不以F作query同时把F并入support。FULL S非全部TRAIN部署；实际部署原型起点若将来进主线仍用原SFT精确head，支持全已到达TRAIN，必须在后继另立完整评估协议。本轮不从训练稳定性外推部署成功。

## 预算、恢复与交付

FIRST起24h含本轮实现、软件/准入/profile/fit/分析/发布，绝不重置。CPU-only CUDA空，无新encoder不需GPU，仍只允许未来GPU1；RAM64GiB准入、worker28/group29每2s/总30GB、CPU4/400000s/file1GiB、独占共享memorylock、disk2GB+512MiB早停，每server新增<50GB。科学remote-home，NFS小probe/free先准入，保护全部历史。不重训/盲resume，active只审计。

成功需exit0/6144steps/24sealed/120contexts每arm/分析verification/完整正负与费用交付；工程失败或机制negative分别标注。安全source/protocol/全stage-cap聚合报告按本机显式7897proxy、existing deliverygit发布，exactremote/匿名REPORT200/source+localDELIVERY。不得公开逐例/身份/图/features/weights/optim/原始日志/privatepaths。所提一致性是通用正则诊断，不因实现就宣称原创贡献。
