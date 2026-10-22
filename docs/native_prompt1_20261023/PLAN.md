# NIH：原生响应约束的基类深层提示与固定空间全历史解析判别

## 可证伪假说与作用边界

既有kernel主臂65/300低于同轮线性强控制75/300，而同核类质量归一较自然质量控制提高49张；token可塑上层17/300、局部压缩结构48/300也未过门。本轮不改变这些关闭配方。检验：在完整原生1536维信息空间中，仅在基阶段学习小规模深层提示，并用原始无提示响应约束避免基阶段过度偏移；随后冻结表示，使用真实患者/类别等质量的充分矩联合解析判别，能否同时提高旧类、新类和尾类。

这是服务主线的BASIL Eq2/实现表8–9及固定空间充分统计的适配组合，不是忠实复现、新TMI贡献或已有效机制。实际阅读限BASIL3.2及附录实现表，并未声称读完全部证明/官方代码。来源：[BASIL](https://openreview.net/forum?id=Uyz13nNEDN)。提示拓扑和完整CLS+patchmean是本任务适配，不继承原论文效果。

## 完整机制

共享冻结原ViT全部预训练参数；图像正式reader后执行冻结patch/position前缀，197×768 FP16 token仅当前阶段共享RAM一份，无磁盘cache。三个模型将相同token转换为FP32执行全部12块，保留CLS与全部patch均值拼接L2，共1536维。缓存量化是各臂共有计算协议，不声称与全FP32原始图像路径逐位一致。

在blocks1–8每块插入6个768维提示；提示输出在该块后丢弃，保留原197个token进入下一块。深层提示共36,864参数。只有提示和临时10类余弦头训练，主目标为25倍余弦CE + .05×mean(1−f_prompt·f_original)。teacher是相同输入上无提示的完整响应，detach、不读取未来。初始化提示std .02，临时头为完整基类患者等权原生均值；CE和RC相同初始化、逐步实际抽样。固定9400步=ceil(60129/64)×10平衡曝光等价轮，非十次无放回遍历。AdamW lr .003/wd1e-4、clip5、64逻辑batch、micro8，无增强/dropout/调度。每步随机8基类，每类8患者再均匀影像；少患者类允许有放回。训练后临时头丢弃，提示/表示全部永久冻结。

每类实际到达时，在各自冻结空间取所有当期真实输入；phi=[L2(feature),1]/sqrt2。G_c=mean_patient(mean_image(phi phiT))、m_c同权均值。保留真实类充分矩与每类每臂两行有限映射审计，不保留旧完整图像/token/feature表。所有seen类等质量，W=(mean_c G_c+1e-4I)^−1[m_c/K]，6stage×3臂18次CPU FP64联合解，预测不使用task-id。全部当前类矩在完整到达表上独立NumPy重算后释放；以后表示不变，不做非线性旧矩运输。该权限比有限32患者记忆更强，不能混称原A/B或exemplar-free（有有限审计token）。

## 公平强控制

FROZEN_NATIVE：完整原生表示＋相同充分矩解析流程。PROMPT_CE：同提示/临时头/配对batch/9400更新，去掉响应约束。主PROMPT_RC：唯一增加.05响应约束。三臂下游1537维/数据/到达/患者类质量/正则/solver/评估相同。冻结臂没有提示训练，两训练臂多36,864提示及15,360临时头参数；不称同可训练容量/训练FLOPs。CE也计算配对teacher相关项用于一致账本但权重0。预训练参数0更新不等表示从未学习。

## 冻结门与预算

主13门全部必须通过：final≥91/300、mean≥36.005291005291%、tail7≥25/105、initial≥44/150、later≥41/150；对两控制分别final+4、mean+1pp、tail4+2/60、later不降。失败关闭，不扫lr/提示数/层/步数/正则/consistency/ridge/seed/门，不换主臂。300val仅重复开发筛查，不是临床test/SOTA；不test/Etest、不val调参、不未来类初始化。

首个远端准入起8h原钟，准备/数学/资格/失败/等待/科学/独立验收全部共享，不重置延长。新根+中性入口合计5e9B硬限/4.5e9B预停，按max(size,blocks×512)，NFS约双倍计费已计入预估；RSS40GiB/GPUreserved8GiB，实际hostfree48GiB/GPUfree8GiB及NFS余量write-read准入。5s独立组守卫，原截止前60sTERM/30s必要KILL。此前三个窗口根及所有入口保留在从23:15:19起累计30e9B账本/27e9预停，本地100MB余量，不分名重置、不删证据腾额。更早本地准备时长未知，不记0。

## bulk前真实GPU与独立验收

20合法基类不同患者图，正式reader→GPUFP16前缀→GPUFP32完整12层/正式64micro8，两提示臂各两个完整forward/backward/clip/AdamW更新（base8类及8+2模拟到达），实际提示delta>0且预训练参数delta0；三臂CPU FP64完整1537维患者矩/两次solve，所有class矩/有界token/probe保存加载，FP64与独立NumPy全部12层attention/GELU/提示路由及normal-system一致才PASS。资格模型全丢弃，4优化/6solve/20编码另计。数学合成仅附加资格，不代替真实GPU全链。

18模型val前seal，300val图一次前缀；每臂FP64完整响应300行，复用到6seen前缀。自动独立CPU NumPy同样每臂300行完整12层，再解18系统/18score同预测且max≤2e-6；额外每类每臂两行真实映射审计FP32差<5e-5，normal residual<1e-9。独立metadata重建全部配对抽样、患者交集0、seen/封存/计数/资源。是保存端点和到达表检验，不冒称全优化轨迹或旧原图重放。保存只复制有界紧凑tensor，原safetensors只读不复制340MB到新根。

科学账本至少包括20资格prefix+68058scienceprefix+300valprefix；基类teacher60129完整body；两提示臂各601600训练前反向行/9400更新；每臂68058完整body映射、300FP64query、18headquery共4050；类矩和线上独立统计各204174映射行，CPU NumPyquery900及classproof120完整body行另计，所有资格/数学/加载准备/失败费用保留。0后续SGD不等0计算。完成全组退、完整三臂中文结果/门/成本及独立验收，通过有效7897代理push、远端SHA、匿名标题可读后交付，无需审核。
