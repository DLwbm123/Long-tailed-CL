# D45 少样本协方差的不确定性诊断（事前冻结）
目的：原Gaussian回放按类使用512维样本协方差+1e-5I。少样本类rank<=n-1，其未观测方向仅由小对角项支撑。检验固定数据驱动协方差收缩能否改善TRAIN内预测密度，而不是改分类读出或融合。区别D26全局对角度量、R30回拉采样点、R35真实exemplar；不重复已训模型评分。

只读三域stage0真实TRAIN unit特征/标签（NIH遵守原H-unit单位化）。各类按原manifest次序奇偶分为两个固定fold，互换fit/holdout，所有类和样本都保留；无随机划分或新RNG。仅TRAIN内统计估计，0优化/图像编码/神经网络前向/VAL清单特征响应/TEST；不能称0统计拟合或独立泛化。

每fold先以fit行算每类均值m_c和无偏协方差S_c（nfit=1时定义S=0），P=各到达初始类S_c等权均值。原C_base=S_c+1e-5I；唯一候选C_shrink=(1-lambda_c)*S_c+lambda_c*P+1e-5I，lambda_c=d/(d+max(nfit-1,0)),d=512。该权重由维度/样本数固定，无常数/强度/收缩目标搜索；两者均值相同，不归一化Gaussian输入。记录每类fit/holdout数、rank上界、lambda、每维Gaussian NLL、Mahalanobis/d与logdet/d；使用Cholesky解，不显式逆，全部finite和正定。

事前支持条件：NIH完整TRAIN<250的初始尾类，两fold各自按类等权候选NLL每维至少比原低.1 nat；NIH/ISIC/HK两fold各自全类宏NLL每维至多升.01 nat。通过只支持另立不确定性回放训练，不能当准确率收益。失败停止该固定收缩，不扫lambda/pooled权重/正则/对角或full选项/seed/融合。原模型统计受保护，只写新诊断结果。不得将TRAIN内leaveout当患者独立；均值不确定性未被修复。

独立600s/300MB含16MiB余量，CPU2400s/RSS8GiB/file64MiB/CPU4，预计新增<5MB，CPU-only无GPU分配。原preflight起含修复retry、独立guard、中性主子argv、pro/data实际ext4容量写读probe，旧cache/checkpoint只读不复制删除。所有类/正负结果保留，完成公开source/protocol/安全类级统计与中文报告。
