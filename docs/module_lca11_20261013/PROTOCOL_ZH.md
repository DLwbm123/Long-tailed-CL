# 新版LCA-inspired：ISIC/HK主四格冻结协议

用户明确授权主四格，只复用已交付base/local_transport，新增模块单独与组合；GPU0/1、test/reserved封存、全部正负和AIR差距及新类保护完整交付。不是原版LCA复现。机制消融AC/AO/AD未授权，本轮不加入，不调参/换seed/放宽门/自动失败重跑。

A0=原base、AL=原local_transport；AB=lca_supported（base+模块）、AP=local_transport_lca_supported。ISIC/HK各AB/AP，新4条轨迹；ISIC每臂100/72/296=468，HK98/4/2/20/18/34=176，共1288 adapter更新，0policy。base/local_transport/AIR只复用本地公开RESULTS，不读取旧NAS模型重新评价或收费。前DAT/Hao完整GitHub/NAS交付和实际退出已核对。

同seed/split74002、同readonly train/val和缓存预训练ViT双adapter、原order/task_sizes、原fit/meta身份80/20、2epoch/batch64/workers4、AdamW.0003/WD.01/clip5/FD10/ridge.001/proximal.01/beta.5/cosine eta_min1e-5。CONTROL_MATCH核对历史CONFIG科学字段/交付、train/val身份互斥和每task fit/meta/更新聚合。具体身份不公开，未哈希影像或模型，不声称未测量的逐字节资产身份。

## 模块

数学设计原封沿用docs/lca_supported_design_20261012/PROTOCOL_DESIGN_ZH.md和supported_old_margin_v1：原common_shift先减掉；current-fit配对teacher/current特征拟合支持空间内仿射残余map，ridge.001；利用task-start原旧类mu/Q外推旧类等质量风险，原充分统计/原型/共同运输不替换。组合删除span(R_base-R_local)重叠，两次去泄漏，无删除能量归一化；这防止额外惩罚抵消局部先验放宽，不声称区域运输真实校正了这些漂移。

H_ref在epoch为candidate Gram+ridge R+proximal I，boundary无proximal。S=.1 U_used/(1+max(raw/retained白化风险谱半径))，相对曲率上界.1。M仅old centered block，新类行列0。目标J_native+.5trace(W' S W M)，Hessian算子native_H(W)+S W M，监督与竞争RHS不变。S/M/map在epoch内detach/freeze；boundary全current train含原meta重估。首task无old，S/M应为0；真实预检不证明非零激活。

reference head/base_pairs保留原control构造；controller、batch competition/proximal解、显式loss、boundary解共享frozen S/M，PCG复用原Cholesky/Woodbury预条件。zero路径返回原解。推理仍一个原global线性头，无额外分数或学习参数、图像/逐样本特征回放、Gaussian采样、RL或模型合并。M不直接压缩新类列，但原竞争耦合、old均分数及old-vs-new margins仍可能损伤新类，不能称无损保护。旧类漂移只是外推代理；报告覆盖rank、old support min/mean/max/defined数、map/intercept norm、fit MSE、raw/retained trace、去重rank/leakage、relative bound、solver和direct-new-force。

## 运行与评价屏障

中性后台父子入口，GPU仅0/1；每卡>=32768MiB且NAS nfs/free/小写读探针通过才运行，不干预别人。24h从真实launch起，含资源等待/预检/训练/评价，不重置。4真实first-batch backward零optimizer预检全部PASS（模型state未变、有限非零梯度、solve<=1e-8、0policy/test）才TRAIN。独立preflight名字与日志，正式train不覆盖。全部4轨迹精确TRAINED/全部stage与步数一致后统一development val；训练入口不读取val manifest，独立evaluation barrier还核对四STATUS。

checkpoint保存模块version、dataset/order/seed/split/source/task/cumulative steps与frozen S/M、原bank/head；只当前新轮私有。评价在打开val之前检查屏障，每stage检查metadata、old/new M和解析头重建相对误差<=1e-8，float32 readout严格等于保存double头的cast。重建不替换已保存读出。任何失败停止依赖，保留全部驻留和私有诊断，不自动重跑。

## 贡献和重点

五格含AIR，每数据集完整final/avgBA、tail、forget、old/new、final/avgAcc；所有stage Acc/BA与31类n/recall。AB-A0、AP-AL、AP-AB、AP-A0、交互(AP-AL)-(AB-A0)全部pp，另AB-AIR/AP-AIR全六指标；共14组。AIR-gap=AIR最终BA−候选最终BA；原gap/新gap/缩小量完整报告，不按gap选stage、seed或参数。旧F1未提供留缺失，不重评旧模型。

分别AB/AP相对base冻结门：BA>=+1pp、tail>=−.5、forget<=+1、old/new>=−1；另对自身control要求BA>=+1和新类平均>=−1。全部正零负公开，不因平均门通过隐藏单类损伤。重点ISIC0/3、HK8/3，所有新类都分无下降/下降门内/明显损伤描述。只有双数据集均过相应门才称开发贡献；不自动追加确认seed。主四格不能分别归因旧估计/去重/M保护，组合与独立贡献也不能混同。

## 费用与交付

正常4preflight+4train+4eval共12GPU操作，每次驻留出现收费一次，失败额外全计；旧复用费用0。已闭合历史GPU315757.8385525835秒，各独立round+旧83019.61353103025仅一次，不相加cumulative。CPU准备/报告/等待与并行wall单列，早设计CPU21.392109549604356秒历史不重计；本轮CPU全部驻留含失败，pipeline已经在驻留内。

完成后一次匿名collector只取metrics/STATUS、epoch/boundary纯标量、4预检标量和PROGRAM费用。禁止影像、权重、IDs、逐样本分数/特征、mu/Q/B/U/S/M、中心、原型/边矩阵、private配置/rawlog/traceback外传。源码/协议/全部正负及失败/REPORT/贡献/AIR gap/逐stage逐类/AUDIT/费用显式本机代理推送GitHub feat/prototype-analytic，验证remote SHA/精确commit报告匿名HTTP200，再NAS独立public_delivery/DELIVERY/HANDOFF。未核实不称完成。

单seed/order、反复development val、极少val分母、缺独立患者证据；非原论文复现、非test/临床/统计显著或等计算预算胜负。CPU和真实预检只证明工程路径兼容，非模块效用。按既定后台运行约定，完成启动检查后不保持聊天等待，不默认新建持续监测。
