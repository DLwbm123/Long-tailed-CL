# 有限旧类风险可行更新：冻结协议

## 动机、完整机制与边界

已交付的 contrast_replay 主臂0/13门、frozen_expansion主臂4/10门、functional_margin主臂3/13门。后者辅助确实执行，局部分支冻结也确实成立；这些证据不能唯一归因为漂移、竞争或权重。共同未建立的是实际全seen判别风险的可靠保护。这里不续跑任何旧配方，不加蒸馏、teacher、对比项、refit、NCM或融合。

本轮改变可接受的参数状态转移。每个实际到达阶段，在每个旧类最多32患者、每患者一个不可变冻结输入feature上，计算当前模型全seen分类CE均值 R_c。到达前风险 r_c 固定，阶段内主臂必须满足 R_c(theta) <= r_c + .02 nat（FP32数值容差1e-6；独立FP64验收容差2e-5）。这是逐旧类约束，不是max-class目标、出生logit/间隔软惩罚或旧H线性响应空间投影。它约束保留样本均值，允许个体伤害，不约束未保存患者或验证集风险，不声称零遗忘。新类别加入本身改变softmax分母，因此必须提供可行到达规则。

三臂共同模型为1536→128 SiLU→1536零up残差、L2 embedding、scale25余弦head及每类可训练bias，410250→425620参数。固定encoder不更新。到达时新head来自当前真实到达影像的患者等权冻结输入均值；不读未来图或旧全feature。所有臂共用新增bias规则：令每个旧记忆行的旧logsumexp为A_i、新列未移位logsumexp为B_i，新增bias共同平移 min(0,min_i(A_i-B_i)+log(expm1(.01)))。从而每个旧行到达CE增量<=.01 nat，留下阶段预算余量。它是同臂共同的可行起点构造，不是主臂专属校准或事后救分。旧bias可继续训练。

每步先求共享抽样的balanced CE梯度g，clip norm5。PROJECT及FEASIBLE对所有旧类完整记忆的CE求梯度G_c，投影到 Gd>=0 的交集：min_d ||d-g||²/2。约束逐行归一不改变半空间，小型dual用固定坐标下降（最多2000 sweeps、KKT1e-8），方向残差<=2e-5，否则工程停止，不能静默采用失效方向。FEASIBLE另检查真正参数候选 theta-.01*2^-k*d，k=0..11；只接受所有旧类风险均在阶段固定上限内的第一个候选，无候选恢复原参数。恢复后再次检查。无动量/weight decay，不会在投影后暗加更新。所有臂SGD lr.01，种子74002固定；无辅助weight。

一阶投影不能保证有限非线性步可行，例如r(theta)=theta²在theta=0处梯度0却在非零更新后增大。这是数学动机，不是已确认旧实验病因。主臂阶段预算允许最多每次实际到达.02 nat累积，5次对同一旧记忆最多.1nat加数值容差；不声称总体损失或预测固定。可以拒绝全部后期步骤，这是要如实测量的可塑性风险，不自动调预算或增加步数。

## 强对照与公平性

- ER：普通balanced CE原梯度SGD。
- PROJECT：相同目标、逐旧类梯度的GEM式一阶可行投影。
- FEASIBLE（唯一主臂）：相同投影后加入实际有限步可行转移。

同模型结构/容量、输入feature记忆、metadata选择、到达顺序、当前源、抽样和dropoutmask、机会数、初始化规则；不称相同FLOPs或实际更新数。ER不使用额外旧风险梯度，PROJECT不缩步，主臂接受/拒绝另记。GEM是成熟对照，本轮是其feature级、单全seen头、逐类约束适配，不是忠实task-aware GEM复现，也不据此称新TMI贡献。

主文依据：Lopez-Paz & Ranzato, Gradient Episodic Memory for Continual Learning, arXiv:1706.08840v6，已读第3节公式6—11和算法1及周边设置，未声称读全附录。https://arxiv.org/pdf/1706.08840 。原GEM依据一阶局部约束；本轮完整到达可行性、逐类风险账本和非线性有限步接受是待证伪设计。旧ISIC/HK response_projection/constrained_response是固定线性读出/旧H子空间及响应惩罚，本轮不读取其数据、头或响应，不复活其配方。

## 数据、训练与门

NIH原68058 train，6实际阶段10+2*5，原20类顺序及患者分割不变。20个真实基类资格编码单列；不读test/Etest、不下载数据、不恢复任何旧实验cache/pack；当前类全feature仅RAM，阶段末释放，保留每类32患者一图的metadata seed记忆。encoder ImageNet CLS+patchmean1536/L2、FP16；FP32训练、FP64部署、TF32off、CUBLAS_WORKSPACE_CONFIG=:4096:8在CUDA前设置。

每步seen无放回8类，每类8次患者均匀/患者内影像均匀有放回抽样，64逻辑行x2 inputdropout.1=128views。基类2000、之后各500机会=4500/臂，共13500机会；每臂288000逻辑行/576000views。不是收敛标准。每臂实际accepted updates、old-risk backward图映射、candidate forward、head到达/账本/probe/资格/验收均独立计费，患者数不与重复映射相加。全部旧失败成本保留，新实验账独立清晰。

主门仍13项全部必须通过：FEASIBLE final>=91/300、六阶段mean>=36.005291005291%、tail7>=25/105、initial>=44/150、later>=41/150；分别对ER/PROJECT final+4、mean+1pp、tail4+2/60、later不降。并要求主臂运行中每步和最终阶段有限记忆风险可行、独立数学/预测/记忆/计数/资源验收通过。任何一项失败关闭，不改约束预算/solver/lr/步数/seed/容量/记忆/门或换主臂救分；不追加headrefit/融合。若只部分完成，缺失指标保持缺失。

18模型全部val前封存；300val只编码一次、每stage实际seen前缀评分，每臂1350全模型query行。300val为重复开发筛查，不是独立临床test或同协议SOTA。独立CPU NumPy18score<=2e-6且预测全同；独立metadata memory选择、旧feature逐项完全不变、患者交集0、阶段覆盖与counts，首阶段三臂state完全同；每次到达CE增量和主阶段末风险上限重算、每后期阶段保存的一次真实参数转移及投影半空间证据重核。中间所有步由在线检查，独立验收不冒称重放全部轨迹。验收0编码/优化，额外CPU模型映射单列。

## 硬预算、真实GPU准入与交付

仅my-gpu GPU0。新独立原钟从本轮远端准入/写读probe开始记录，首次到完整科学/验收组退出最多8h，所有准备、资格、失败和等待共享，不能延长；3,000,000,000B硬限、2.7GB预停，RSS16GiB/GPUreserved8GiB，5s全根+中性入口按max(st_size,st_blocks*512)核算，提前60s TERM/30s必要KILL。预计产物<1GB，不复制数据/权重。无科学重试；资格或工程故障先停止，不能绕过PASS bulk。更早本地准备未知不记0。

启动代码强制先取基类10类各2患者共20真实训练图，正式reader/preprocess→正式GPU FP16 encoder→正式GPU FP32 128views三臂完整forward/backward/clip/SGD→用基类内部8+2模拟到达、head扩展、旧记忆梯度/投影/有限步检查→保存加载→CPU FP64/独立NumPy。工程模型丢弃，真实20编码/每臂2步（6步）及额外映射/数学全部计费。GPU合成或CPU资格均不能替代该链。必须QUALIFICATION PASS后才首次完整基类编码。

完整父子argv与GPU PID为中性入口现场核对。健康不重启，唯一现有30min automation原位跟进。完成后全组退出、独立验收、三臂全指标/拒绝率/风险账本/全部成本/中文报告及匿名源码聚合经有效代理push，核远端SHA及匿名200后交付，无需用户审核。身份/影像/feature/权重/逐图分数/原始日志/transition witness私有，不扩大清理。
