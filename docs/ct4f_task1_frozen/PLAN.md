# CT4-F：正常Task1末冻结编码器的解析参照

状态：预先固定；独立于已关闭CT3-P。用户已授权基于结果自行设计/执行有限后续实验，并取消GPU小时限制。

依据：CT3-P交付6f6e15fb63bb0ac4b4034d14f4ef0c5a16d3fb34。ISIC C3−C1 +5.584pp但current −10.883pp；HK F的终端预测与P完全相同。FD是否有超出“保留Task1编码器”的适配净价值尚未检验。

假说：保持正常Task1训练后编码器不变，精确追加新类统计，可能解释大部分FD前缀收益；若FD显著优于此固定参照，则持续适配有额外证据。允许负结果，不保证方向。

固定矩阵：仅新增F1，一个正常CT1-U Task1末冻结编码器+联合1536维特征CBRidge。HK/ISIC×原1993/1994/1995顺序×Task1/2/3（2→4→6），共18阶段/72逐类行。六个既有Task1父状态保持原哈希/恢复校验；不新增独立S0、不重训Task1。Task1统计直接继承；Task2/3仅提取当期fit，原统计精确累积，无A/T输运。全encoder/adapter/keys/head固定，无optimizer step，无FD训练。此冻结是新控制的定义，不改写CT3-P继续训练的定义。

数据、图像预处理、原始权重、类别映射、pointwise FP32提取、batch48、float64 joint-L2及CBRidge均复用原代码。ridge为原1/C类均衡目标、lambda0.001、无bias，保留cross-Gram。不更换模型，不搜索参数，不依据val修改方法。

访问：合法继承Task1统计，后续每阶段只读当期fit；旧fit重读0、未来fit0、test/reserved所有访问0、不读取任何oracle资产。全部18个W和父网络谱系锁定后才生成val预测；每阶段sample_id排序，只评已见类。对照使用CT3-P已保存的同布局val预测，不读取test。原父状态、CT3-P数据/预测/配置只读。

工程：严格恢复六父；Task1 W与原CT3-P C0相等，Task1新score/分类与原布局一致；每阶段模型hash不变、零训练；每类样本计数匹配锁；统计有限、ridge残差达原阈值。第一单元测吞吐，检查GPU/磁盘。若不一致立即保留证据停止，不选择较好实现。

主比较C3−F1；固定次比较C2−F1、F1−P、F1−C0。同阶段BA/old/current/HM/tail、逐类正确数/分母、最差seed和零召回。prefix_terminal_BA限6类，不能称全8/23类Final BA。2000次seed48001按类component配对bootstrap，固定模型不重采样seed，跨seed共享规范sample/component基础抽样。singleton与反复val开发局限保留。

资源：单GPUworker，GPU不设小时截止但矩阵仅18单元；预计约20–40分钟，真实耗时另报。CPU解析/报告上限30分钟，新增磁盘上限2GiB、空闲至少1GiB；六个只读父副本总663855666字节，私有W/score预计<10MiB，不保存特征。无新神经checkpoint、epoch或step。启动后保留小时监测；禁止重跑取最好、参数搜索、Task4+或holdout。完成后STOP并分析，不自动触发更大训练。
