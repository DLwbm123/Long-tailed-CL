# 谱残差分布记忆 CPU：启动报告

4独立CPU检查已启动，尚无最终结果，不判PASS或GPU-ready。科学源：7bb5f4e19f4718e2ab33d04e058026f5b75440db

执行前remoteSHA/精确DESIGN匿名HTTP200、NAS nfs/free/probe、目标Python compile及启动后父子中性完整命令已核实。CUDA不可见、2线程；医学images/features/checkpoints/val/test不访问。启动及不重置deadline见STARTUP_AUDIT。

主候选将每类16d+16个float64的原存储预算重分配为均值、15个主方向与特征值、一个各向同性尾方差；实际秩min(15,d-1,nclass-1)，不足处固定补槽，不按查询选秩。Gaussian生成后单位归一化，每次仍16点、8对正负噪声。原moment48不降低，另含random和各向同性生成对照；各向同性对照存储更少，明确不是等容量控制。原48查询、8批逐批门不变。

候选无需记忆optimizer，但协方差构造和eigh有实际成本，不能声称计算等价或零开销。归一化可能改变均值和协方差，Gaussian不恢复高阶非高斯方向。当前仅合成支持，低样本统计可能包含原支持信息，不将此原型视为医学原始特征回放授权。

上一轮[完整结果](https://github.com/DLwbm123/Long-tailed-CL/blob/2e247f289447a90836551247befc5a56785bf185/docs/generated_moment_memory_design1_20261016/REPORT_ZH.md)：small8/8、ISIC0/8、HK8/8，整体失败；候选在ISIC/HK的同生成规律解析矩总误差也略差于posthoc。CPU518.763515999075秒，十一轮43次累计3842.1976726180874秒，新GPU0。旧结果非本轮完成证据。

本轮全部正负/失败/每次wall user system RSS将在终态统一公开；运行中源码不修改，未启动GPU。
