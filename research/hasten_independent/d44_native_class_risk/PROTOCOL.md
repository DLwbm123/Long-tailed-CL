# D44 原生对比风险归约诊断（事前冻结）
假设：不改变自然batch、负例矩阵或对角正例，仅把原双向paired contrast的行风险从按行平均改为“先本类平均、再出现类平均”，可能使更新方向更符合尾类/宏类风险。R05改采样同时改负池和RNG；R14改图像→文本分母频次；R38改文本正集合；R22只在保留原目标之外加全类CE辅助梯度，均不是本机制。

最小诊断：R28linear64各域3seed只读stage0封存模型、当前stage0真实TRAIN缓存和标签，不使用VAL/TEST/图像/新编码。每初始类manifest前min(4,floor(n/2))行作为固定B探针；这些行从A候选池剔除。A为固定seed20261025+100*domainindex的自然shuffle按64分块，对每个初始尾类TRAIN<250选第一个含该类的完整batch；不重复同一batch。选择只用TRAIN标签，属于条件探针而非原实际训练轨迹。全部初始尾类分母/含尾batch保留，不能报告为独立患者或泛化。

同一A和模型计算原paired对角双向CE的逐行风险l_i=.5(CE_i(image→text)+CE_i(text→image))，原mean与固定mean_class(mean_rows(l_i))；其余原加权anchor/hierarchy/mapreg完全相同。取两种总梯度，不向.grad累积或optimizer使用。在互不相交的B上，原pairedCE按类均衡宏风险及尾类风险的梯度作为固定参考。比较cos(reference_gradient,total_gradient)：正值表示同单位长度负梯度步的一阶风险下降方向。无实际参数步长/优化，不能保证AdamW更新或accuracy。原None梯度仅在诊断向量拼接时置0，不写模型，全部网络和hierarchy状态恢复exact。

事前支持条件：NIH至少2/3seed，在所有条件A探针平均后，宏与尾参考的candidate cosine均>0，且各自相对原方向提高>=.01；ISIC/HK跨seed宏/尾cosine变化平均各>=-.01。完整所有cell及失败保存，不在看结果后改阈值、batch或权重。支持才启动固定“原生对比行风险类均衡归约”的完整三域新训练，仍用原BOTH采用门；不支持停止该归约，不扫强度/尾权重/组件选择/seed/融合。

0优化/VAL/TEST；与D39 geometry-v-contrast冲突不同，本次检验同一原contrast的风险归约相对不相交TRAIN探针风险的一阶方向，不重跑D39或旧科学评分。CPU4/GPU0reserved<8GiB/RSS8GiB，独立600s/300MB含16MiB余量/CPU2400s/file64MiB；原preflight起含全部修复重试，独立guard、中性主子argv、规定pro/data ext4容量写读probe/GPU余量。共享TRAIN资源只读，不复制旧模型、不删除历史。所有正负结果/完整source/protocol安全类级汇总公开交付。
