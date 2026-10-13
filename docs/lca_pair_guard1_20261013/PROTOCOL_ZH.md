# 新版 LCA-inspired 主四格：冻结 checkpoint 评价协议

用户本次只授权 ISIC/HK 主四格，复用原 base/local_transport 的 18 个阶段 checkpoint，不重训、不调参，只用 GPU0/1 和已有 development val。test/reserved 封存。不重启旧轮、复用训练不重复收费；不自动重试或追加实验。

## 方法与不变量

采用已公开设计 `1ea8ba02940bb0374dc51c050d930e329df5302f` 的 `old_pair_residual_guard_v1`，数学和参数保持不变。A0 为原 base，AL 为原 local_transport（固定主控制）；B=A0+新版，P=AL+新版。主四格不包含 P_raw/P_plain；守卫前分数仅用于幅度和间隔诊断，不计算其分类效果表或选参数。

每个阶段独立重建原解析头 W0，保持 bank、原 competition 权重、R 和 RHS 不变。B 为居中的旧类均值正交基（机器精度定秩），V 为旧类 Helmert 对比基，新类行精确零，ΔW=BZVᵀ。全部旧类有向对的残差为 max(0,1−μcᵀ(W0c−W0j))，权重 n_c/(n_c+1)/[O(O−1)]。原完整 native Hessian 投影得到 H；在 H 白化坐标中将对残差曲率最大特征值规范为 1，求解二次目标。目标间隔 1、native 能量半径 0.1、64 次标量信赖域二分均固定。没有新优化器更新、旧样本读取或漂移真值。首阶段旧类数 0，严格恒等；18 个 sidecar 中正常 14 个非平凡解析求解。

保存头按原 float32 numpy readout 得到基准分数；校正也用相同特征的 float32 readout。新类分数保持基准值；旧类分数用 `原旧类最大值 − (校正后旧类最大值 − 校正后各旧类分数)`。这保持每输入旧新预测组别、每个新类 one-vs-rest 预测集合、真实新类正确性与 raw true-class margin。真实新类若仍错分为旧类，错误旧标签可以变化，须单列“错→另一错”；不能把它误报为新类正确性下降或遗漏。新类 softmax 概率不保证不变。成为后续旧类的类别可以变化。最终守卫是非线性 readout，不能称单一线性解析头。

新版只修正原头留下的旧类内部均值间隔残差，与 local_transport 改先验的作用位置不同；使用的旧均值仍是原统计，不能宣称新增独立证据、可靠测得旧类漂移或保证互补增益。

## 来源、生产验收与统一评价

原 base：`module_additions1_20261009`，科学源 `0cfe6b648f5de1b8e8783265e7a0b97b64939952`，交付 `ecd783ffd7b32eea4e2e2a1311b9f033231f1677`。原 local_transport：`module_local2_20261010`，科学源 `0f0d8a0717c6da09109cc9205cccf44c490d3e84`，交付 `4b6e0c87887b101ae2970738919760f00e19726e`。只读取各自 ISIC/HK runs 下已有 stage 文件。

原文件未嵌入完整 source/order/seed metadata；通过原 CONFIG、独立 DELIVERY、固定目录与文件低成本 size/mtime、checkpoint module/seen/steps/全部 class n、原头重建共同验收。不能声称旧 checkpoint 有不存在的嵌入元数据。原保存头为 float32，CPU double 重建相对范数差≤1e−5，并逐元素 atol=2e−6、rtol=2e−5；求解残差≤1e−8。这些容差在任何真实 checkpoint 准备前固定，不因失败放宽。正式 OFF 分支全部逐阶段类召回、分母和准确率还须精确复现公开结果。

先做合成 CPU 生产检查（0医学影像/0旧checkpoint/0GPU），包含真实模块求解与 sidecar、早开 val/非法范围拒绝、首阶段空旧类、完整报告及坏分母拒绝。既有设计 CPU 检查仅为历史，不重计。随后冻结 Git 科学源及本协议、代理公开验证，再部署到独立 NAS source。

正式协调顺序：4 条 CPU checkpoint 解析准备 → 全部 18 个 sidecar 封存 → 4 条真实 GPU 零更新预检全 PASS → 4 条统一阶段评价。CPU 准备只读原 checkpoint 统计，0影像；预检用各最终 stage 的现有 val 首 64 图，真实 encoder 前向、参数/buffer 精确不变、有限 readout、CPU/GPU guard 精确一致，0optimizer/0policy。预检不计算分类指标；无需制造推理模块梯度要求。失败停止依赖、保留所有驻留与私有错误，不能自动重跑。

ISIC order=[4,7,5,1,2,6,0,3]，tasks=[4,2,2]；HK order=[4,5,6,22,20,1,2,15,14,12,10,11,19,16,7,0,17,9,18,13,21,8,3]，tasks=[13,2,2,2,2,2]。seed/split74002。原更新累计分别 [100,172,468]、[98,102,104,124,142,176]，本轮新增更新0。全部 train/val class counts 和原公开阶段参考固定于 PROTOCOL_LOCK.json。

评价只直接读取 val.csv，不调用会打开 train/test 的 manifests()。ISIC val295、HK val1698；各 stage 只评价已见类别。bicubic Resize224/Normalize(.5)，batch64、workers0、shufflefalse，与原评价一致。单个 checkpoint 同一次 encoder 特征同时得 OFF/ON 分数，特征与逐样本结果不保存/外传。全套阶段必须完整，不能先挑最终或最好 checkpoint。sidecar 不反馈其他阶段。四轨迹顺序 ISIC_base、ISIC_local_transport、HK_base、HK_local_transport，两卡两个槽。实际启动时冻结2小时总期限，含准备/等待/预检/评价，不能重置。

## 全部正负交付及门

全四格×两数据集：最终/平均BA、tail、forget、old/new、最终/平均Acc、macroF1；AIR使用已交付指标。完整 B−A0、P−AL、P−B、P−A0、交互、B−AIR、P−AIR 六指标pp；分别列 AIR 差距前后及缩小值，不只展示改善。

所有 stage 的支持度/recall/Acc/BA/F1；全部31类最终结果、每stage逐类/all/old/new 的改变、纠正、破坏、错→另一错、净正确；旧类内部/旧→新/新→旧/新类内部的错误流；原始与守卫后间隔正零负及均值/极值、分数作用大小、信赖域是否限幅。小类 n<5 连续 margin 抑制为 null，计数与recall完整。新类 one-vs-rest 决策/正确性与 raw margin 及旧新组别验收精确0变化；wrong-old identity 变化单列。

主门固定 P−AL，B−A0 同样完整报告：两个数据集同时 BA≥+1pp、old≥+1pp、tail≥−0.5pp、forget≤+1pp，加精确新类保护。没有真实改善或缺失不得宣布模块有效；不按 val 调参。统计正则非零不是效用。

正常 GPU 费用8条（4 preflight+4 evaluate），CPU checkpoint 准备4条单列；所有失败逐次记录，不去重同名 operation。历史各独立 round 一次，禁止累计再累计；AIR与原训练0新费用，当前重评价有费用。CPU pipeline 包含在进程驻留内，墙钟另列。参数/峰值/真实 checkpoint 与 sidecar 大小/val曝光按实际报告，不虚构货币单价。

只有匿名汇总、源码、协议、报告、审计/费用公开；权重/sidecar/统计矩阵/中心/逐样本/影像/患者身份/private配置路径/raw日志保留 NAS 私有。完成后代理 push 同仓库分支，remote SHA 与精确提交报告匿名 HTTP200 验证，再 NAS 独立 public_delivery/DELIVERY/HANDOFF，不能只留 NAS。无新自动监测。单seed/order、重复development val，非独立患者/临床或封存test证据；不再追加实验。
