# R124：真实类别到达下TARS×保留奖励的机制实验

## 动机与区别

本协议在实现/native/科学训练前冻结。原始准备钟和12小时硬截止见SCIENTIFIC_FREEZE，全部准备、失败、重试、子进程共钟，不重置。主线检验长尾机会分配和候选级保留反馈能否共同改善真实连续泛化。RaPO的连续保留奖励在组优势之前改变候选排序，CTAN跨任务保存尺度；不是R115当前TRAIN风险角度投影、R116旧锚点KL阈值或R123取证输入救援。它借鉴已发表方法的机制，不声称简单组合已有创新，也不将作者数值当本地效果。

R113已有真实前缀分类器更新和记忆编辑策略，因此本轮明确不学习记忆编辑、选择旧候选或更新预算：训练的是生成式视觉语言策略本身，零历史样本回放，五阶段各自模型/优化器连续，旧任务通过前一任务模型锚点而非旧FIT样本反馈约束。与R67的固定闭式特征分类器也不同。

## 数据与访问边界

固定原HK23 TRAIN6821/VAL1698，不使用TEST。类别按原canonical ID依次5+5+5+4+4到达，顺序见freeze，不依VAL或已完成分数挑顺序/seed。实验调度器可知道固定类别schema、到达顺序和manifest元数据；学习器不得接触未到达图像、feature、预测、反馈或训练目标。旧224冻结视觉cache只作只读物理后端，mmap不等于读取权限；所有切片均先由严格accessor核current-task TRAIN白名单。任务2起，旧TRAIN同样不可读取，历史样本预算为0。VAL只在独立审计路径读取已到达类，不能给优化器、校准表或策略状态。

## 初始化和连续状态

generic Qwen2.5-VL3B，224/64视觉tokens、冻结vision，语言q/v LoRA r8 alpha16；验证task B全0，绝不载入R114/R121等旧任务SFT。仅首任务类0..4做共同2自然epochSFT，shuffle74005、batch16/micro2、AdamW2e-4/wd0/clip1/cosine306（4868queries），作为所有臂相同的首次领域支持，不算RL贡献。共同首任务SFT封存并只评一次已到达VAL，然后复制参数给五臂。任务2..5无额外SFT，每臂独立延续自己参数、AdamW状态和（RaPO臂）sigma。post阶段首次新建AdamW2e-5/wd0，之后不在任务边界清零；每次任务开始冻结该臂上一任务结束的策略作为reference/anchor。模型、优化器、锚点和sigma是模型状态，不是历史样本。

五臂GRPO/TARS/RAPO/TARS_RAPO/TARS_CE（下文CE均指此匹配加权控制）。每到达任务各128次optimizer尝试，8queries/update，四RL G8；同一固定自然当前TRAIN有放回tape及同response seed，所有臂查询索引一致。CE用相同8query教师强制标签，但无G8生成/KL、前向更少，费用完整披露，不假称同计算或由比较证明RL必要。

## 权重、奖励与优化

各臂任务开始前仅对当前TRAIN做一次自己policy的greedy校准，不访问旧/未来。固定TARS原rarity×error^.75、median floor1e-8、minweight1，再除当前自然采样下的平均权重Z，使PG平均强度1；该归一化是本轮明示适配，不是原样复现TARS。GRPO/RAPO臂用1；TARS_CE也使用自身当前表的同一均值归一化类别权重，以免通过较弱的未加权监督控制夸大RL增量。表在任务内固定，不能从VAL或部分结果调整。

准确率奖励为exact canonical class-name match且正常EOS。RaPO臂任务2起：D=max(mean_mask(logp_behavior_TF−logp_anchor_TF),0)，Rret=exp(−20D)，Rtotal=Raccuracy+.5Rret，全部detach。current/anchor采用同micro2 teacherforcing数值路径，避免把采样缓存差异当漂移；未更新同policy行为ratio须精确1。保留奖励进入组内减均值之前；TARS权重在优势形成后，仅乘PG，绝不乘KL。RaPO分母是sigma的EMA：先sigma←.999sigma+.001std(当前8×8总奖励)，再除sigma+1e-6；sigma初始1.0并跨真实任务持久，组均值仍按每query计算。普通GRPO分母为各query组std+1e-6。原论文未明确sigma初值，本轮固定1.0不扫描，属于适配而非声称作者配置。

四RL其余均原clip.2、同KL beta.04、lr2e-5、gradclip1、max24/temp1原manual decoder。保留loss-level KL用于所有RL，RaPO部分是加到同底座的组合实验，不声称原RaPO复现。每阶段ref为自己上一任务policy，不是generic或其他臂。合法全对/全错组可为零步；不得过滤、重采。记录实际参数L2/非零步/Adam尝试、逐类组组成、正确响应、retention与任务奖励/adv/漂移、CTAN轨迹和所有成本；非零错误PG不算有效学习。

## 指标与九门

任务1共同评估只执行一次并在表中显式共享；任务2..5先完成并封存全部五臂，才统一评已到达VAL，所有阶段完整报告，不据中间门改参/停臂/择优checkpoint。25条arm×stage表含5条首阶段共享读出，实际21个评估调用序列；不是25独立重复。所有评分为重复开发VAL，不是独立患者/临床证据。

主TARS_RAPO对TARS：四个post阶段平均seen macro>=+1pp、平均seen tail>=+2pp、最终首任务head相对共同SFT下降<=1pp、任一post阶段old-tail相对TARS下降不超过1pp（空group跳过，分母报告）。主对RAPO和CE分别post平均macro>0/tail>0（4门）；平均classwise peak-to-final forgetting小于TARS（1门）。合9门。mainvsGRPO、所有逐类/old/new/head/tail/tiny/micro、漂移分布、实际计算完整报告，不额外设事后门。旧R117等失败门不变，不能靠换为CL指标宣称它们成功。

## 工程、预算和停止

CPU fixtures核TARS归一化、ret reward在ref常量、组均值/EMA跨阶段、正确与错误排序、零步合法、禁止旧/未来TRAIN切片、仅已到达VAL；native在首任务合法两图上做samehidden/samehead parity，另4RL各一次真实backward/optimizer、一次首任务batch16 CE，以及batch16greedy profile，全部丢弃并严格还原generic/ref。native使用80生成响应（4×2×G8+16greedy）/4TRAIN视觉编码/5optimizer尝试，实际可能零步如实记录；sampled/TF差异披露，行为路径ratio1。依据实测SFT/RL/校准/已到达VAL全矩阵1.5倍+1800秒余量，必须小于原钟余量才可进入科学SFT。此为运行准入，不是效果准入。

独立后台guard每2秒含全部R114–R123保留根、本机200MB余量及新root，在9GB预停，全部累计新物理严格<10,000,000,000B，不重置。新checkpoint/优化器/报告预计<600MB，无新模型/特征副本或下载，单文件256MiB/core0、CPU4/RSS48GB/GPUallocator56GiB/freeFS>=1GB、真实/data ext4和读写探针，GPU1/pro5000 only，中性argv，不干预他人。原始12h<=24h，deadline−120秒保存/停止。失败保留成本/源码/旧钟，根因修复，不科学调参或复活旧配方。

结束0后CPU-only saved-only验证所有访问日志/各臂独立连续状态/firsttask generic/306SFT+2560post曲线/旧任务零回放/九门/权重数学/奖励与sigma/optimizer尝试和真实位移/所有成本与封存顺序。实际计数由manifest固定分母推导，native/所有失败另加，checkpoint和参数文件不公开。完整负结果也交付。正结果只是新单order开发条件，需另立确认方案，不追加seed挑分或宣称创新/临床成功。只发布源码、协议、安全聚合和成本，禁止逐例预测、私有索引、身份、图像、特征、权重、私有路径、原日志/PDF/凭据；所有GitHub显式7897proxy、allowlist/no-force、remoteSHA/匿名200。

实施前修订2：在任何新模型调用、native或优化之前，将普通CE明确为匹配TARS权重的CE；只为公平归因，五臂数/更新/查询/原钟不变。v1原件保留，PRE_SCIENCE_SPEC_REVISION记录0查询/0更新；不以对照增强为支线搜索。
