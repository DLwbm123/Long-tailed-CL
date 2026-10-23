# 到达式患者定向固定类几何与原生低秩迁移：冻结主线协议

状态PROTOCOL_FROZEN_IMPLEMENTATION_PENDING；只有核心源码，语法检查不是数学或GPU资格。没有科学RUNNING，不继承任何旧配方的准入或性能。本文是独立主线构建，不是优化论文对比方法。

## 来源、证据和假说

本轮实读[NC-FSCIL原论文](https://arxiv.org/pdf/2302.03004)第3节定义1、第4.1—4.4节及5.3节（正文p4—9相关段），不是仅摘要。该文固定全标签ETF分类器，以dot-regression训练归一化特征；base训练backbone和projection，后来冻结backbone、训练projection，保留旧类中间特征均值。理论把样本特征当独立优化变量，不是共享深网络不遗忘保证。未读全附录或审计官方代码；OpenReview链接触发验证页，未重复访问。其自然图像FSCIL结果不支持NIH效果、临床结论或本适配创新。

旧NIH完整科学证据表明共享可塑upper的新类可以学、旧类判别及竞争明显退化；全协方差和患者crossfit并未满足门。两个最新工程失败没有方法负性能证据。新假说：**在原生内部以受限Q/V低秩更新适配医学表示，同时为旧新类维持同一个固定目标几何，以正类方向回归替代变化的全seen push梯度**。这同时改变医学表示可塑范围、目标几何及训练目标，不是旧score冻结/patch注意力/密度/缓存读出配方复活。固定target仍不固定输入表示或完整score，有限重放仍可能遗忘。

## 本轮明确适配，非忠实NC-FSCIL复现

原始ViT-B/16已有权重只读；第一8块、最后4块全部原始参数与norm永久冻结，只在最后4块attention Q/V加rank8 LoRA（alpha8、A Kaiming、B零，无adapter bias/dropout）。三学习臂同一份初始A/B拷贝，全部阶段持续更新这些98,304个参数，非上层dense微调/每批新expert/128维压缩。拼接原生CLS与patch均值1536并L2，头为25余弦。LEARNED_CE可塑head从本次合法患者等质量原生均值初始化；FRAME_CE固定几何+CE；主FRAME_DR同几何+0.5*(target_cos−1)^2，无CE混合/额外损失/权重搜索。NATIVE_RIDGE为本轮同输入的完整患者类矩强锚点，不能拿弱可塑臂替代。

固定几何仅在真实类别到达时建立行：总20类是已知协议元数据，rho=−1/19；base10行由当期原生患者均值M按G10^(1/2)(MMᵀ)^(-1/2)M生成，G_n=(1−rho)I+rho11ᵀ。仅这10行存在。后来每新类以旧行W的Gram系统构造与每旧行内积rho的平行分量，再取当前患者均值在旧span外的单位残差补足单位范数；最后第20行由旧行唯一确定。旧行逐元素保持，不建立未来类行/实例/标签。这里是本项目推导的因果几何构造，不是原文随机全标签ETF预初始化；不继承其定理。任何Gram/残差不合格即停，不换方向、重排类、回退随机轴救分。

四臂与三个学习臂实际患者draw、LoRA起点和优化机会匹配；固定/可塑head活跃参数不同，DR/CE梯度不同，诚实披露，不称等活跃参数/FLOPs。两固定臂相同初始target，FRAME_CE隔离DR目标；LEARNED_CE是强迁移训练对照。Native另有全矩记忆，格式不等于有限token记忆。

## 数据、数值路径和更新

沿用author20单标签train68058、val300、原order10+2×5与seed74004，患者train/val隔离。只有当前新类完整图和旧类每类≤32患者/每患者1图的第一8块197×768 FP16 token；旧token来自本轮实际编码，禁止借旧cache、旧权重/score/head。记忆按既有metadata患者优先固定键选择，实际draw三臂逐步相同：8seen类不放回，每类8患者有放回，再患者图内有放回，逻辑64/micro8；不是64独立患者。base3000+later500×5=5500/臂，共16500正式更新。AdamW lr3e-4/wd1e-4/eps1e-8/betas(.9,.999)/clip5，每stage同规则reset，无dropout/增强/TF32/搜索/seed追加。

本独立方法规范数值路径在运行前明确：transform仍原resize224/normalize(.5,.5,.5)；embedding在FP64计算后量化FP16，第一8块FP64，输出量化FP16作为固定记忆输入；最后4块/LoRA/head及优化器均FP64。原始FP32预训练权重精确转FP64，不下载替代权重。FP64用于确定此新方法的可审计数值目标，不追改或解释已关闭FP32整链失败，不借其未保存误差放宽门。CPU/独立NumPy应比较此**新定义接口**，另报告量化误差边界和reader/embedding proof。GPU8GiB和真实吞吐必须实际满足，不假定FP64可行。

完整当前token仅stage RAM，结束释放；只保存有限memory、40训练proof、300query embedding/lower接口、小LoRA状态、head、完整Native矩及draw/训练probe，不存完整当前token表或预训练大权重副本。每类Native以患者等质量、类等质量1537维仿射ridge1e-4，实际新类到达才更新，在线独立NumPy核对当期全矩。

## 资格、验收与门

真实NIH/GPU前先完成独立小维数学目标/导数、到达式Gram/旧行不变/禁止未来行、LoRA零B旁路与参数路由/保存加载的核心检查。它属于本方法的首次资格，不是重跑已完成归约/Adam合成诊断。所有数值在assert前写原子receipt，联合条件各项单独报告；异常时必须把已完成计数写到finally，禁止再次只留下缺失联合误差。

20合法基类图真实GPU原生768/197token，内部8+2到达，三臂各两次真实micro8更新；同初态/同保存梯度独立FP64 AdamW递推，实际LoRA更新、原始参数精确不变、head权限、固定Gram及旧行不变、reader/embedding和完整8+4层独立NumPy、GPU/CPU保存加载、实际时间/存储准入均PASS才bulk。FP64分数/目标≤2e-6，目标导数小维≤2e-6，同梯度AdamW误差≤1e-10+1e-10|预期参数|。不要求或冒称旧full64/micro8轨迹等价。实际最差两阶段三臂canonical配对step，以1.5×5500×pairedstep+10800s严格小于原剩余钟−60s；资格独立审计费用另记，仍消耗原钟，非ETA。

四臂×六stage24模型全部在val读取前seal，一次300val新编码。独立NumPy完整原生8+4层query与40训练proof、24组5400score行、prediction全同、LoRA/固定权重/目标Gram、患者隔离、实际draw/有限memory/到达权限/完整Native系统与所有计数、每stage三臂最后64行真实训练probe目标。不是全训练轨迹重放或重新读全部历史图。

主17门保留：绝对final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial10≥44/150、later10≥41/150；对三控制各final+4、mean+1pp、tail4+2/60、later10不降。机制FRAME_DR对FRAME_CE同四门。完整中文四臂表与原分母、各stage和所有失败/未知费用公开，不选主臂或用val调参。原300val是重复开发筛查，不是独立临床test/SOTA。

## 原钟、物理预算、停止

保守原first取本轮研究触发2026-10-23T18:30:48.618Z（BJT10-24 02:30:48），12h deadline=1792823448.618；研究/实现/准备/资格/等待/传输/正式/验收共享，从此不重置、不延长，不另起远端12h。当前代码未执行资格或科学。

新根+中性入口硬3e9B/2.7e9预停、RSS32GiB/GPU8GiB，预计logical<1GB但不能当physical；已观察NFS大文件blocks约2×logical，必须在本轮准入按实际block物理占用校准预计最坏峰及临时副本，再满足余量。原30GB/remote27GB预停/local100MB累计包含所有旧根/入口/论文源码，旧末采样12.311GB不重置不删。初始hostfree40GiB/GPUfree8GiB、实际NFS4余量与小写读探针，独立NFS无关整组timeout截止前60s TERM/30s KILL及资源guard、所有父子argv中性。

任一数学/权限/资源/时间/资格失败关闭固定配方，不自动重试或放宽；门全PASS才一次bulk。结束需独立验收或诚实失败记录、完整自有组退出、中文表/全部成本与未知、有效7897proxy push+远端SHA+匿名报告验证。无新增数据下载/test/Etest/future初始化/val调参/旧路线恢复；身份/图像/tokens/权重/矩/逐图score/原log/全文私有。当前尚无实际准入或活动科学，不据草案宣称可运行。
