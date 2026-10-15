# 单位球面节点域：前瞻 CPU 机制检查

上轮9e5bc20已完整交付：固定节点凸权重ISIC近最优但MSE35.3421，未见误差/矩误差更差，部分权重塌缩。不能靠更多求解步解决该固定节点问题。下一假说是节点构造域和实际单位样本域不一致：原ball允许内点，而loss内部归一化会改变这些点的方向作用。高维均值/二阶矩Frobenius可忽视散布trace，零均值各向同性Q=I/d时零节点Fro误差仅1/d却丢失全部trace。它不证明当前失败原因；此前只存max norm不足以判断。

唯一结构改动：原矩Adam节点更新后由unit-ball投影改为unit-sphere归一化，norm<=1e-12拒绝而不发明方向，记录失败。没有新惩罚系数或超参数扫描。候选仍24矩节点步+24概率权重FW，查询96步针对不可变sphere24记忆，随后冻结。16节点、8查询、rank8、beta8、.02/.01、原seed263/264/265、数值门均不变。

五方法random、原ball moment48、sphere24、sphere48、sphere_weight。sphere24为sphere48相同优化前24步共享快照，不重复计算。原ball moment48为未降低基线；sphere48仅机制控制，不得事后切为主方法来宣称判别记忆成功。候选48记忆步，配对case实际两个48节点构造+24权重步；实际资源逐次报告，不称所有方法计算量相同。

四CPU case：球面范数/加权trace恒等、组合梯度FD、零向量拒绝、高维trace反例；32维N512K4、1536维N9434K8、1536维N1038K23，原合成非高斯单位支持/公开比例，不是医学fit。0医学images/features/checkpoint/val/test/GPU。部署只留合成节点权重；未来任务结束后不可回访旧医学支持。

全部五方法新增逐类min/max/加权radius、二阶trace、mean和Q误差诊断。weight阶段节点固定、FW训练单调和gap照旧；sphere能保持trace但不能保证高阶方向覆盖，权重仍可能塌缩，16节点低秩也未消除。48个未见随机/共同stress查询，不回传更新记忆；共同stress不是独立population检验，有限见证非sup证书。

主门仍为两个高维sphere_weight max<=.9各原moment48/random且mean不增。sphere48/radius正负独立解释，不换主门，不增加节点/querybank/步骤或调强度。共享stress随方法集合改变，不跨轮最大值作配对。

2CPU线程、CUDA隐藏、case硬1800秒/整体5500秒资源门。全部失败/每次wall/user/system/peakRSS保留，mode/operation包含驻留，不相加。此前四轮16次CPU1751.5632895277813秒另列，GPU历史400036.74851310626不变。只导出匿名类标量，不导出节点/矩阵/support/rawlog/traceback。

先源码协议公开冻结，再执行；终态一次收全，完整GitHub/NAS交付后机制分析。无完整序列、重压缩、医学分类或生产预算证据；过有限门也不直接医学GPU，失败不原实现GPU。连续失败必须如实披露，不用更多技巧替代贡献证据。
