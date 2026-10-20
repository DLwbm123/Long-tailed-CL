# PRODUCER_GROUP_CROSSFIT：整个监督producer的组排除诊断

独立问题：先前P/C训练类2几乎完全正确而开发下降，是否存在超过“仅拟合读出”效应的适配响应复用差距？这是记忆/组外泛化诊断，不是旧P/C救援、logit remapping方法复现或新CL性能候选。旧门永久保留；method_success=false。

## 文献与Pro依据

用户Pro分析要求分开固定参考、保留训练头/refit与竞争力；既有结果已否定换头是唯一瓶颈。本轮不追加旧头变体，而检验整个producer是否见过query组。

Memorized Logits Reverse Post-Hoc Correction in Long-Tailed Recognition（https://openreview.net/forum?id=3Fq8NvlSTg）§3.3式3由排除整fold的训练算法生成OOF logits；附录C.1每个fold都保留所有类，C.2在OOF表内部选择remapper。本文实际阅读本地正文§3.1/3.3、C.1-C.3。本轮不拟合其remapper、不搬其GLMC/CIFAR超参，不称复现。论文按图分层，本项目更严格采用已有已知组分层，组数两半差≤1但图数可能不等。

Feature Support Erosion in Long-Tail Fine-Tuning（https://openreview.net/forum?id=LYEiAGvdJw）附录D.1选定中间表示上的margin AGOP与下游loss梯度有明确区别；具体层/归一化位置未唯一定位。本轮不称SADP复现、不估AGOP或据开发损害选方向。均为under-review快照，非录用。

## 排除链与固定设计

仅原initial训练2555（组数129/16/93/653），不读取开发681的标签/特征参与计算、评分、选参，不读未来/test/rawimages。容器inputs0与canonical像素chunk可能物理含开发行，加载后立即按原valid mask排除，开发行不进入任何统计/模型前向/评分。只用原预训练权重，原特征生产链审计无本任务encoder更新；不认证大规模预训练语料无重叠。

先使用标签进行一次按类分组划分（只此设计用途）：每类排序组、固定seed74290打乱，交替分配两半，所有图随组；每组必须单一类。16组的类1每半8组；若任类任何半少于8组，未决停止，不换seed。禁止通过图ID推断新身份；只沿既有已知groups。

每个producer k只在support半组训练。support-only统计包括组权、两级mapper、alpha=.01自由截距ridge初始头、所有训练batch/optimizer；无预训练外的query监督。alpha.01与lr.003/800步继承既定研究配方，在fold前固定，不用任何query/开发选参；这并不消除整个研究历史的适应性，内部query不是外部独立确认。

强简单控制F_k：同support-only mapper/ridge，encoder始终冻结。T_k：相同起点及mapper/ridge头初始化，encoder102912既有bias+6144double四类头参数CE联合训练并部署终点。两者均不用全2555拟合的旧teacher/mapper或任何旧learned encoder；读取旧实验配置仅定位只读像素/权重，不读其学习产物。原native双head冻结。

T0/T1各800步，每步每类8不同support组各1图，共32槽、各25600槽；同训练seed74250配方、各自optimizer重置，AdamW .003/wd0/betas(.9,.999)/eps1e-8，无搜索/早停/新模块/惩罚。训练只support半组；query绝不入CE、norm、方向/类别权或checkpoint选择。不是旧O每步交替query反馈。两个T端点先ENCODER_SEAL，之后每个encoder仅一次对2555初始训练图编码、FEATURE_SEAL；query只在此无梯度评分阶段前向。F用原冻结g，无额外F编码。两个完整T诊断artifact含absolute bias/mapper/PERM/三角support A、B/W(六列)/元数据各≤19058800，W来源CE非当前A/B的ridge解，回读核全分数/bias/A。无最终全2555重训、无新alpha/head refit。

## 比较与停止门

对每行同时构建seen响应（其所属support半producer）和out响应（另一producer），F/T同样构建。四个全2555表均保留原y/group绑定；每fold seen/query和合并全表的BA/逐类计数/CE/margin/救回损害全部报告。F控制同时受读出支持复用影响，不能把T seen-out全部归encoder效应；两个不同producer的比较仍非样本干预因果。

主估量Δ_BA=(BA_Tseen−BA_Tout)−(BA_Fseen−BA_Fout)，类2同理，单位pp。两项点门同时Δ_BA≥5pp、Δ_class2≥10pp，才做固定seed74292/10000次类内已知组配对bootstrap，全部四表共享draw；两项Bonf分位[.05/4,1−.05/4]下界均>0才CONDITIONAL_ADAPTATION_REUSE_GAP_SIGNAL，否则STOP_REUSE_GAP_EVIDENCE_INSUFFICIENT。未过点门STOP_REUSE_GAP_POINT_GATE，不补区间0、不挑类/半组/指标升主。bootstrap仅条件于这两个训练producer，不估训练随机性，不是外部泛化确认。所有类及两fold保留，即使类1少也不掩盖。

raw margin只做同源描述，不把T大head尺度当更好表示；不以平均margin代替BA风险。无新保护方向、无未来扩类/CL成功声明。

## 执行前校验与预算

源码封存后SYNTHETIC：组划分互斥/全覆盖/每类保留、poison query标签/g后support输入exact不变、支持权/mapper/自由截距正规方程残差、10k推断成功/失败/缺失、JSON有限。预定故障注入同包装器四退出/失败/资源/协调回执，必须按预期失败。PREP只拟合两个support教师并封存划分。实际PREFLIGHT仅半0支持32图：G batch32原cache≤1e-10、micro舍入披露、fresh32 vs checkpoint8 bias/head梯度atol1e-5/rtol1e-3、一步finite/native双头不变/bias恢复exact，probe弃用，失败停止不放宽容差自动重试。

独立首次synthetic起共享3h硬截止（含预检/失败/子阶段/等待），独立driver进程组deadline guard提前10秒终止、5秒kill，协调每2s核时间/空间/RSS。远端+本地/Git/helper16MiB新增严格<1GiB，单文件512MiB，RSS12GiB；预计新增<384MiB。旧窗17GB软停/20GB硬限，按max(logical,allocated)+默认cache1040384+localGit1GB计。GPU0启动前显存及实际NFS挂载/容量/小写读probe；复用原像素/模型/环境，不新增像素复制；所有cache/tmp NFS，PYTHONDONTWRITEBYTECODE，中性主子argv/GPU名。不删历史或干预他人。无并行训练或第二监测。

driver严格SYNTHETIC→预定INJECT→PREP→PREFLIGHT→TRAIN→EVALUATE，必要phase exit0且正式预检pass才继续。验收两个800/51200监督槽、两2555单次FEATURE_SEAL、两个完整artifact、两support ridge解、全部4表封存后聚合、finite/组排除/parity/容量/全部退出退休；wait4含后代只计一次，guard重叠wall不加。完成公开源码协议全聚合成本失败摘要并核远端SHA/匿名HTTP200；DELIVERY本地NFS双端，所有模型/feature/身份/逐例/原日志/PDF私有。唯一scale-med30min继续，健康静默。
