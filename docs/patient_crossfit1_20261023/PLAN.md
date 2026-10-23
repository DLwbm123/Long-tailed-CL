# NIH患者交叉拟合解析读出一致性：完整协议冻结

## 假说及进入条件

前轮native_prompt已完整验收/诊断/交付ff05052，Native/CE/RC final75/78/67、base53/53/59、later41/35/22，主2/13门失败。原配方关闭。已有基阶段收益但后来类无收益，因此只进行一次患者外解析读出风险的匹配检验，不声称临时CE与ridge失配已是唯一根因，不扫原参数或复活旧competition压缩比较器/最坏风险。

完整预训练ViT全部冻结，原生CLS+patchmean1536维保留；8块×6深层提示及原生响应锚定.05，基阶段学习后永久冻结。每实际类到达时所有真实患者/类别等质量充分矩、全seen CPU FP64 ridge1e-4联合判别，24模型在val前封存。无taskid/生成/非线性旧矩运输、无可塑后期表示。三训练臂相同提示初始化/实际episode/更新数/冻结时点/下游历史信息，固定Native为强原生控制。成熟深提示/可微ridge的NIH组合适配，不声称忠实复现或新TMI；证据来源和阅读边界延续BASIL方法3.2及实现表8–9/前轮报告。

## 四臂

FROZEN_NATIVE不训练提示。CE_RC为临时全10基类余弦CE头＋.05原生完整响应差，之后丢弃临时头。RIDGE_IN与主RIDGE_OUT都是两次患者/类别等质量可微ridge，平均双向监督平方风险＋同.05RC；IN在S拟合评价S、Q拟合评价Q，OUT在S拟合评价Q、Q拟合评价S。IN/OUT唯一变化为拟合/评价侧配对，因此OUT−IN才是患者交叉拟合机制比较。CEvsOUT改变读出/损失，不能解释为单因子。

内头在全10基类输出，episode中未支持类别target及ridge系数0；CE保留10类竞争。单侧d_i=1/(8*n_c)，phi=[L2(f),1]/sqrt2；Z=sqrt(D)phi,T=sqrt(D)Y；W=Z^T solve(ZZ^T+1e-4I,T)。最多32×32系统，不用显式inverse，不detach内头。Outer每侧按同患者/类别质量求平方误差的类别维和，双向平均；RC同权平均。不同CE/平方风险尺度及CE临时头额外15360可训练参数明确披露，不称全臂同FLOPs/活跃容量；IN/OUT提示参数36864完全同。

全局训练患者先分固定S/Q池：seed74002，SHA256(seed:patient_crossfit:globalpatient)首byte<128为S，否则Q。跨类同患者必须同侧。只读元数据准入S11107/Q10989、交集0，未搜索seed。每步8基类，每类每侧n=min(4,S人数,Q人数)无放回患者，再患者内均匀影像；S后接Q。class15仅7独立训练患者（S3/Q4），所以含class15的episode62行，否则64；不复制凑64独立患者。不同类可有同一侧共享患者，患者/类别权重定义逐类但两侧绝不重叠。训练query是监督训练数据，不是独立验证或未见患者证据。

9400固定配对episode/每训练臂，来自相同10基类nominal曝光量的预定训练预算；三臂总28200正式更新，实际影像行根据62/64实际计数而非预填601600。AdamW lr.003/wd1e-4/std.02/clip5、micro8、无dropout/调度/增强。CE联合裁剪提示与临时头，IN/OUT仅提示。基类共享一份197×768 FP16前缀RAM；全12块GPU FP32。为避免跨样本求解被micro独立求导替换，先无梯度算全部episode特征，叶特征上构建完整两个解析系统反向，再逐micro重算body并VJP，参数不在两遍之间更新。额外缓存前向逐行计费；各训练臂同重算方法。正式无完整图token表磁盘cache，不借旧特征/提示/头。

## 冻结门/预算/停止

主17门全部要求：final≥91/300、六stage mean≥36.005291005291%、tail7≥25/105、initial≥44/150、later≥41/150；对Native、CE_RC、RIDGE_IN分别final+4/mean+1pp/tail4+2/60/later不降。机制OUT对IN同四相对门另列，机制门通过不等方法成功。失败关闭，不换主臂/削弱对照/扫ridge、RC、lr、提示、步数、seed、门。300val仅重复开发筛查，非临床test/SOTA。

本实验首次远端准入至全组退出**16h**不重置，全部准备/资格/故障/等待/训练/验收共享，仍<全局24h；比前轮更长仅因本轮三训练臂和额外缓存重算，**不延长前轮原钟**。全根+中性入口6e9B硬限/5.4e9预停（NFS max(size,blocks512)，约2倍逻辑）；RSS40GiB/GPUreserved8GiB，准入hostfree48GiB/GPUfree8GiB/NFS余量写读。所有旧窗口根及中性入口纳入原23:15基线累计30e9B/远端27e9预停/本地100MB余量，不分名重置、不删证据腾额。四families类Gram只保存一次，原大safetensors只读、不复制。

独立5s硬guard，原截止前60sTERM/30s必要KILL。bulk前额外可行性门：从真实62/64资格三训练臂两步耗时求配对step，预测1.5×9400×step+10800s保守余量须小于原剩余钟，否则停止不bulk、不改步数续命。预测不是保证ETA或成绩推断。科学单次不自动重试。

## bulk前真实GPU全链与完整验收

78合法真实基类图（每类每侧最多4，class15每侧3），reader→GPUFP16 patch/position→GPUFP32完整12层，三训练臂每臂64/62各一次缓存前向/叶特征双系统/梯度/VJP/micro8余6/clip/AdamW，实际提示改变、预训练参数delta0；加8真实行完整图与micro重算的梯度和实际AdamW更新对照（2丢弃opt/4内solve），不得只比分数。三训练臂qual共6丢弃opt，合计GPU资格8opt；四臂完整类矩/8+2内部到达CPU FP64 solve各两次共8、全部class/有限token保存加载storage检查、FP64与独立NumPy完整12attention/真实解析训练目标及normal-system。CPU数学另做NumPy独立目标/有限差分、完整小图vs缓存梯度和AdamW一致（6CPU丢弃opt）；合成不代替真实GPU。全部PASS及可行性门通过才bulk，丢弃资格模型恢复同初始提示。真实目标差2e-6、完整图/缓存实际GPU梯度3e-4和更新2e-5、CPU数学FD2e-6/目标1e-9/缓存梯度3e-5/更新3e-6；阈值不后改。

每实际到达当前完整patient矩独立NumPy复算后释放，历史类矩只读；四臂每类2个有界token/feature审计，不能称exemplar-free。24heads全val前seal，300valprefix一次，每臂CPU FP64完整响应300复用六头；独立NumPy完整12层每臂300＋每类2proof行、24系统/24score≤2e-6预测全同，FP32proof≤5e-5/normal≤1e-9；metadata独立重建全局患者分池及全部配对episode（padding−1不算影像）、62/64约束/双内系统计数、患者交集0/封存计数资源。端点不是全历史原图或全优化轨迹重放；0新增编码优化验收。

完整成本必须记录三臂每步两个body前向/一个反向、IN与OUT各18800内层dual solve，CE0；正式后期24解析系统另计，不能只报24。teacher/当前映射/资格78/实际62或64逻辑行/CPU数学opt6与GPU资格opt8/原钟等待未知准备/全部验收/资源均保留。未知分段不记0，不重复加驻留内部或总钟。完成全组退、中文四臂表/17门/机制门/独立验收/全部成本及有效7897代理push/SHA/匿名访问再交付。test/Etest不读、不未来初始化、不val改参，不恢复HyperKvasir/HASTEN/旧CLIPcache或修改两tools；隐私产物不公开。
