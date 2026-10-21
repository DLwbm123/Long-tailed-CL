# 人工指定Fibrosis复现取证：仅核对既有运行设置

用户要求先查清原因，不重训、调参、放宽门，再决定后继。读取原launch、q145入口、run_module_cycle/evaluate及原INPUT后，确认原评价未覆盖TF32，原q107v环境torch2.1.2/cu121/cuDNN8902默认matmul TF32 false、cuDNN TF32 true；F1 run/resume/boundary明确将两者设false。原/恢复解释器的库版本、NumPy/Pillow/torchvision/timm/scipy一致，不能把中性venv路径不同直接当版本不同。

唯一取证：同一原Task1 checkpoint、原seen10/150val、原batch64/num_workers0、原canonical变换与CPU NumPy float32读出，首先原设置cuDNN TF32=true，其次F1覆盖false；matmul TF32始终false，其余条件固定。各150诊断编码，共最多300，0训练、0优化、0新拟合；原Task1 head不改，saved F1 query只用于比较，不用于拟合或调参。记录实际运行前后flags、全部类计数、成对变化、Fibrosis真实margin与特征差；具体样本/分数只存私有，公开聚合。此二值对照由源码差异预先指定，不搜索其他精度/seed/批次/阈值。

判定：原设置严格复现旧每类计数，而false严格复现F1首阶段计数，且成对差只在Fibrosis一图，才可将该差异归到此运行设置。否则保留未决，不宣称查清、不继续搜索设置。即便原因确认，也不追认或覆盖原F1科学验收，更不在本任务重提取7929图或重跑F1。仅完成因果定位后向用户交付。

共原F1首次1792552424.3168447/截止1792581224.3168447、3GB实际预算、2.7GB预停、RSS8GiB和GPU0准入16GiB；独立guard进程组/提前保存退出。追加诊断300val须与原正式300、前次工程128分列，共验证编码728（若全部执行）；失败也计账。read-only版本探针耗时单列，不重置历史成本。只读原图/共享权重，不新数据、不缓存训练特征、不改原文件、不新建监测或并行研究分支。
