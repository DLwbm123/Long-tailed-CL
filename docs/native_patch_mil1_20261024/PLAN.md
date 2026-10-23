# 原生patch多实例主线：协议冻结，实现已补齐、真实GPU资格尚未执行

当前状态IMPLEMENTED_PENDING_REAL_GPU_QUALIFICATION，不是科学启动。已完成的native_token、structured、prompt、crossfit、full_covariance均关闭，本实验不能改变其参数救分。假说：在完整且永久冻结的原生768维patch空间中，直接学习类别条件取证并统一新旧类判别，可能利用被CLS+patch均值及其全协方差都丢掉的局部模式。保留原生global旁路，不重学ViT、不压缩到128维、不建局部Gaussian、多原型分布或旧分数融合。类注意力能表达均值相同但局部模式不同的两个bag，是表达能力事实，不是NIH有效性或失败根因证明。

一手依据为Ilse等Attention-based Deep Multiple Instance Learning（ICML2018），已定点读原文2.1–2.4，特别式7–9的归一化可学习集合聚合：https://arxiv.org/html/1802.04712 。原文使用图像bag的attention/gated attention；本方法是NIH增量类别条件原生patch/有限历史token适配，不是忠实复现或已证明创新。原文结果不作为NIH预期收益。两篇ICLR27队列条目wQqMfweGWo/9YOIbxEDQH当前正文访问403/验证页，仍只属元数据候选，没有读全文或确认当前接收状态。

## 四臂、表示与目标

NATIVE_RIDGE：同一次新编码的原生global1536/L2输入、患者/类别等质量完整矩、ridge1e−4全seen解析头，强原生控制；不以旧分数替代本轮控制。MEAN、SHARED_ATTN、主CLASS_ATTN：同冻结原生token、同患者有限记忆/更新/实际draw和原生global旁路，区别为局部mean、图像共享attention、类别条件attention。global旁路均为25*cos(global_z,w_c)，global_z=normalize([raw_CLS,mean(raw_patch)])。patch局部h_p=L2(raw_patch_p)。local分数=25/sqrt(768)*v_c^T sum_p a_c,p h_p，v初始0；MEAN的a均匀，SHARED的a=softmax_p(10*normalize(q_shared)^T h_p)，CLASS为各seen类q_c。不做instance标签、位置硬框或注意力就是病灶/临床解释的声称。SHARED→CLASS改变类条件容量，参数与FLOPs差异必须披露，不称等参数单因子。

预训练全部12层/最终norm永久eval且冻结，原入口FP16 patch+position、原attention FP32完整body，最终197×768 token封存FP16；该额外量化是本协议输入，不声称与旧未量化fullgraph端点逐位一致。global和局部均用同封存token，完整768局部保留，不用128维固定投影。当前到达类所有token仅RAM；旧输入只保留每类≤32患者、每患者1张由seed74002和原metadata hash规则固定选出的token，预计终点615患者，三学习臂共享只读输入不多给主臂历史信息。NATIVE保存完整解析统计、学习臂保存有限token，统计与记忆权限不同明确报告，不声称等内存。没有全train磁盘cache、旧缓存/pack恢复或旧原图重放。

仅类实际到达时，global w以本类患者等质量global均值初始化；CLASS新q以本类患者等质量patch-unit均值方向初始化，SHARED q只在base以基类等质量方向初始化，之后持续学习。v新行0，旧行及旧q/w不重置；未来类无参数或数据。新旧类全部在统一softmax竞争，允许旧q/w/v在有限合法token重放上继续学习，冻结的是图像坐标而非旧判别器。无taskid路由或多专家选择。三学习臂同balanced CE，无蒸馏/额外RC/协方差/温度扫描；AdamW lr3e−4/wd1e−4/clip5、每stage统一重建优化器，FP32/micro8，64逻辑行，seen中抽8类各8次患者等权/患者内影像等权（可有放回，不能称64独立患者）。base3000、后来各500，总5500/臂、16500科学更新。固定步数不称收敛。seed74002，无数据增强或dropout/TF32，全部实际draw三臂配对。

官方NIH同10+2×5/order、68058train/300val。只读当前到达类图像，后继源数据权限不扩大。新编码不借旧IN/CLIP静态cache；不得test/Etest、future提前初始化、val梯度/调参/早停/挑checkpoint。24模型（6×4）全部val前封存，val仅一次新编码，六stageseen前缀统一预测，非独立临床test/SOTA。

## 资格、预算、验收与失败关闭

主17门固定final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial≥44/150、later≥41/150；对Native/MEAN/SHARED各final+4、mean+1pp、tail4+2/60、later不降。CLASS>SHARED匹配机制四门另列，但容量差异保持披露。失败关闭，无科学重试，不扫query数/temperature/lr/步数/记忆/seed/门，不换主臂/削弱强控制。

首次实际数学检查起原钟8h，所有后续编码准备/资格/等待/失败/正式/独立验收共钟，不按远端新根重置。未启动的GPU实验不冒称正在训练。全新根+中性入口实际新增<2GB/1.8GB预停，预计physical<1.4GB含全部memory/query/proof/解析矩，RSS32GiB/GPUreserved8GiB；hostfree40GiB/GPUfree8GiB及实际NFS余量小写读准入。原窗口30GB/remote27GB预停/local100MB余量全部旧根加本新根/入口不重置、不删证据。正式运行必须独立于NFS扫描的整组timeout（deadline前60sTERM/30sKILL）及资源guard，中性argv和GPU PID核对。

先数学counterexample/梯度/置换/初始旁路/扩类保持；再20合法基类真实GPU链：reader→预训练token→量化→三臂2实际clip/AdamW更新、两真实类内部到达/固定记忆选择、global/query非零更新、预训练不变、GPU FP32/CPU FP64/独立NumPy分数与实际保存加载/梯度检查。全部资格模型丢弃。真实4维数学不能代替GPU768维/197token；CPU不能代替GPU。实际两步吞吐和内存、预算预测1.5*5500*三臂配对step+10800s须在原剩余钟内，否则不bulk、不放宽。资格新编码20及完整pipeline接口2张额外调用、丢弃优化6等逐项计费，禁止记0。

最终独立NumPy从封存前缀token计算一次原生完整12层FP64响应对量化token proof（至少query300及每类有限到达proof），不重编码原图；从各stage实际input token/封存w/q/v重算18学习组score和6原生ridge组，score≤2e−6/预测全同、normal residual≤1e−9；实际训练probe目标独立复算，患者/记忆选择/旧token保持/原生冻结/阶段权限/封存/全部计数/资源/完整组退检查。资格、额外body、训练/梯度、独立响应/score、失败与未存等待费用全部保留，嵌套时间不相加。仅源码/聚合结果公开；原图/token/feature/权重/矩/score/身份/实际路径私有，最终按7897代理push+远端SHA+匿名访问交付。当前运行器/守卫/验收器已补齐并通过语法编译，但真实GPU资格、吞吐/内存和科学启动尚未发生，不能宣称已具实际运行证据。

## 资格前工程范围澄清（不改科学配方或硬预算）

GPU FP32→独立FP64响应与实际训练probe目标用原同源5e−5工程阈值，CPU FP64→NumPy实际readout score仍≤2e−6；真实完整64 vs micro8三臂梯度≤3e−4、实际AdamW≤2e−5，另6丢弃参考更新，加原两步三臂6，共12，全部计费。最终FP16 token的逐分量舍入界为abs(actual16−NP64)≤2^−11*abs(NP64)+5e−5，实际head分数始终用实际封存FP16输入做2e−6独立验收；不声称GPU FP32→FP16与CPU FP64逐位相同。为可重建累计解析统计，六stage另存当前到达的Gram总量；全probe和query/prefix、memory等精确文件布局的保守physical预计修正为约1.65GB，原2GB硬限/1.8GB预停不变，不能删旧证据腾额。以上均在首次真实GPU资格与新影像编码前写定。
