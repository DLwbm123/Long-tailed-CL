# HASTEN CIFAR100-LT 复现正确性审计

结论：74.42% 是这次运行真实、完整的最终准确率，但尚不能将它解释成“论文原设置已正确复现，因此 HASTEN 方法不行”。核心方法代码与官方提交一致、当前任务抽样核对通过；预训练、预处理、构树和环境没有形成严格匹配控制，且官方投影代码存在可复现的正确性问题。原始结果不撤销、不改分数；改变的是归因与复现验收结论。

## 1. 实现与实际配置

对实际运行端的9个关键文件做了本次用户要求的定向源码比较：models/hasten.py、models/base.py、utils/inc_net.py、utils/data_manager.py、utils/data.py、utils/ort.py、utils/lorentz.py、trainer.py、convs/linears.py，均与冻结官方提交063f295d08991d4da5b1f46a61bfcd279a94dbc6相同。没有全仓扫描/哈希。

包装器保留官方的图文线性适配器、共享映射、层级/蕴含/映射正则损失、梯度投影、Gaussian旧类特征回放及stat=.85的最终融合。sample_num=0表示每步覆盖全部旧类、各抽一个Gaussian特征，不是禁用回放。optimizer字符串adam在官方实现中实际构造AdamW；lr=.001、wd=.05、batch64、每stage10epoch、cosine、其余CLI默认值核对一致。完整6stage更新数920/120/320/80/170/110，总1720，与各阶段真实行数和batch ceiling相符。固定末epoch，不选最优epoch。

终点评价直接调用与官方BaseLearner.eval_task相同的_eval_cnn，不是取最后一次训练batch或纯双曲诊断准确率。平衡测试集下总体top1等于逐类宏召回；全部已见类别共同竞争。标签名称与原CIFAR meta顺序一致，仅下划线转空格；未发现标签错位、漏类或使用不同测试分母。

执行细节需纠正：train/test DataLoader改为2个worker，但官方用于类统计的sample_loader仍硬编码8个worker。此前“全部loader2”的资源说明不准确。所有子进程继承同一4核affinity，实际预算未超；未将其当作性能归因。HASTEN官方_set_random把Torch种子固定为1，class-order/Python种子1993；APART Torch种子1993。这是保留各自标准配方，而不是共同随机流配对。

## 2. 数据协议：实际APART数据管理器独立核对通过

本次执行了APART原DataManager（仅绑定已存在的CIFAR文件、不建立APART模型），与HASTEN保存的SPLIT_RECEIPT逐项核对：100类顺序、每类长尾计数、10847个原始训练行索引全部相同；增量50+10×5；测试每类100张，共10000张。APART data.py/data_manager.py与其冻结Git提交无改动。核对的是可复现的标准代码抽样；历史APART运行当时未保存逐行快照，不能冒称已恢复其历史逐例身份或给出配对显著性。

|阶段|当期真实训练行|HASTEN实际更新|APART标准配置推算更新|
|---|---:|---:|---:|
|0|5859|920|1230|
|1|754|120|160|
|2|2031|320|430|
|3|485|80|110|
|4|1040|170|220|
|5|678|110|150|

APART末列为由10epoch/batch48推算，不伪装成新测训练日志。二者样本相同，不意味着图像曝光变换、更新数、旧类统计回放或算力相同。

## 3. 骨干与预处理：确有重大未匹配变量

|项目|本次HASTEN|已有APART标准结果|
|---|---|---|
|骨干|CLIP ViT-B/16，512维跨模态投影|ImageNet AugReg ViT-B/16，768维视觉骨干及适配池|
|预训练|LAION400M e32图文对比|标准timm0.6.12绑定，旧运行记录为ImageNet21k预训练/1k微调AugReg|
|训练图像变换|确定性Resize224×224、CenterCrop224、CLIP Normalize|RandomResizedCrop224(scale .05–1)、随机水平翻转、ToTensor|
|测试变换|Resize224×224、CenterCrop224、CLIP Normalize|Resize256、CenterCrop224、ToTensor|
|训练|AdamW .001、wd .05、batch64|AdamW .003、wd0、batch48；适配池较低学习率|

这些预处理来自各自官方数据代码，不是此次漏加CLIP normalization或擅改APART。已追踪APART的AdapterVitNet.forward到forward_features/patch_embed，未见额外输入Normalize；这确为该标准代码路径，不能为追分擅自替换成另一预处理。HASTEN本地权重文件绑定到官方laion400m_e32标签对应的同名release资产（598533543bytes），日志确认加载；当前open_clip工厂对标签路径和本地文件都调用同一load_checkpoint。未发现加载BiomedCLIP、B/32或随机权重的证据。未做全权重逐字节比对或跨版本数值等价试验；APART旧历史权重本次未重新取回，不能声称重新验证了历史权重字节。

运行环境为Torch2.1.2/torchvision0.16.2/open_clip2.26.1。官方requirements写2.6.0/0.21.0/2.32.0，README另写1.13.0/0.15.0/2.30.0，原仓环境说明本身不一致。本次没有在任一完全匹配环境完成论文正控，环境影响未定量排除。

## 4. 语义树：结构通过，论文监督未被复现

官方附带50_10树与seed1993实际类别顺序不匹配：初始阶段缺26个已到达类、包含26个未来真实类；用户明确允许由当前模型按官方提示重新生成。本次六棵树通过官方check_only：已见类覆盖、唯一根、无环、无多父、无未来真实类节点、旧边不删除/改接均通过。虚节点数31/33/35/35/35/36，真实类平均深度约3.76–3.83。

但该检查只证明输出结构，不证明语义监督质量与论文树一致。生成者不同，树的拓扑/深度/父类提示也不同，未做独立语义质量验证或匹配构树消融。另需收紧此前“无未来信息”的表述：当前会话在构树前已经看过完整类别和官方树，没有隔离且可审计的逐阶段构树调用记录；只能确认各阶段输出不包含未来真实类节点，不能证明构树上下文从未接触未来类名。不能把输出校验当作生成过程因果隔离证明。

## 5. 官方投影代码存在可复现问题，不能等同于差距原因

utils/inc_net.py第81–92行用torch.svd(C)的第三返回值命名Vh后再转置。PyTorch明确规定torch.svd返回U,S,V，torch.linalg.svd才返回Vh；这里转置后再取列，一般不再得到所需输入特征的低能子空间。第99行再以P@gradient左乘Linear的权重梯度；对输入特征保护、Y=X@W.T而言，输入方向应对应右侧，左乘一般不能保持原输入输出。方法训练在hasten.py第466–476行实际调用该路径。本次没有改动这些上游实现。

用官方LearnableLorentzExp在16维旋转正定协方差上做CPU矩阵fixture（0图像/预训练模型前向、0backward、0optimizer）：

|检查量|官方实现|参考代数|
|---|---:|---:|
|所取基与受保护子空间重叠范数|1.57567|1.36e-15|
|固定受保护输入下，投影梯度对应的输出变化范数|5.67039|正确基且右乘约5.22e-15|

即使只修基、仍左乘，变化范数仍5.83519。此fixture证明投影一般不实现预期的固定输入保护；不测CIFAR准确率、不证明修复能补回10.48pp。AdamW自适应预条件/weight decay还使“梯度在子空间内”不能直接推出“实际参数位移也在子空间内”。真实适配器输入随stage变化也进一步限制保护保证。没有在本次审计偷偷修代码重训或用新结果替换旧结果。

PyTorch主来源：https://docs.pytorch.org/docs/2.14/generated/torch.svd.html 。冻结官方源码位置：https://github.com/LAMDA-CL/NeurIPS2026-HASTEN/blob/063f295d08991d4da5b1f46a61bfcd279a94dbc6/utils/inc_net.py 。

## 6. 目前究竟能解释多少

HASTEN初始50类82.18%，APART90.42%，尚未发生跨阶段遗忘时已经差8.24pp；最终差10.48pp。可以排除“全部差距只由后续遗忘造成”的说法，不能把8.24或2.24pp当成骨干/遗忘的因果占比，因为类别集合等也随stage变化。

最终按TRAIN频率的头/中/尾：HASTEN81.31/74.63/66.13，APART旧记录89.80/84.26/79.93。少样本类别差距更大是描述证据，不区分骨干、增强、层级监督、Gaussian统计估计或优化目标的原因。

准确结论是“当前标准配方的HASTEN-LT移植运行低于当前标准APART历史实测”，不是“同骨干、同增强、同树、同环境的HASTEN理论上限低于APART”。没有普通平衡CIFAR原协议正控；没有同骨干冻结读出/预处理匹配控制；没有投影正确性修复的事前匹配完整控制。仅凭本轮不能完成这几个因素的性能因果分解。

后续如继续确认，应先独立冻结正确性正控与必要匹配比较，保留原结果，不做按test挑树/调stat/lr/epoch救援；不擅改正在进行的R64。用户本次要求为先审计，审计未新增训练。

## 审计执行边界

本次是定向源码/配置/已存输出检查、APART数据管理器核对和CPU矩阵反例，无新增图像模型前向、backward或optimizer。首次审计的torchvision根路径绑定漏掉datasets.cifar别名，意外触发原CIFAR包下载；按该审计进程PID及其打开文件双重确认后停止，仅将本次创建的部分下载移入本run的audit保留。未运行模型、未改原数据或实验结果；相关I/O收据单独保留。R64进程/配置/时钟没有改动。
