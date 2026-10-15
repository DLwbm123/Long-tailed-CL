# 固定子空间半径谱记忆 CPU：启动报告

4独立CPU检查已启动，无最终结果，不判PASS或GPU-ready。科学源：652153fbe30b4df1ec879578a1f421934e6be5ad

执行前remoteSHA/精确DESIGN匿名HTTP200、NAS nfs/free/probe、目标Python compile及启动后父子中性完整命令已核实。CUDA不可见，2线程；医学images/features/checkpoints/val/test不访问。启动及不重置deadline见STARTUP_AUDIT。

主候选与Gaussian控制共享同一均值/谱方向/特征值/尾方差，将Gaussian的独立主子空间和尾子空间半径固定为sqrt(k)、sqrt(d-k)，保持raw均值与协方差期望，改变高阶分布。16点8对噪声、同每类16d+16存储、同原moment48与查询/逐批门均不变。零方差块跳过，正方差投影退化拒绝不重抽。

固定块半径不保证总半径恒定、归一化后矩精确或方差下降；仅检验机制，不称修复低样本医学问题。候选和Gaussian对照共享一次谱拟合，baseline48单独保留，所有成本实际计费。

上一轮[完整结果](https://github.com/DLwbm123/Long-tailed-CL/blob/5225e121689ddf1884279a2d2bd962e451e1a689/docs/spectral_distribution_memory_design1_20261016/REPORT_ZH.md)：ISIC8/8、HK4/8、小规模6/8，整体未过。CPU239.64248802978545秒，十二轮47次累计4081.840160647873秒，新GPU0。HK低样本类raw协方差近乎精确仍有损失误差，有限8批方差高于isotropic，不能只凭此确认失败原因。

终态完整公开全部正负/失败/实际wall user system RSS；不修改健康源，不启动GPU。
