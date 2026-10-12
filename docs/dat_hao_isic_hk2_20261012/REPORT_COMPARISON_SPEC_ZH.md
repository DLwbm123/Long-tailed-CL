# 完成后的统一医学对比报告规格

这是用户新增交付要求，不改活跃科学源、预算、预训练、训练与评价时序。

主表一张：ISIC/HK×八行配置（原base、local_transport、residual_drift、组合，以及AIR/完整APART/DAT_paper/Hao_paper），各列最终/平均BA、tail、forget、old/new、Acc和已提供macroF1。主方法固定local_transport，同时保留全部我方配置，不按val挑最好一行。

AIR/APART/DAT/Hao各对四个我方配置、两数据集共32组六指标完整pp；DAT/Hao对AIR/APART另8组，共40组。正/零/负一律保留。另输出31类最终n/召回与逐stage全部n/召回、全部最终类级差异，不把净正确数变化当逐样本rescue/harm。

单独资源表：预训练与实现身份、epoch/batch/optimizer实际日程、精确updates与公开的图像曝光、解析批数、是否回放/不同图片bank及重复曝光、存储类型、已有参数/checkpoint/GPU峰值记录、可匹配的train/eval驻留。历史未提供的值留null/未提供；AIR0optimizer不是0费用，旧共用预检不乱摊到某方法。旧控制不重训/重评/重收费。

竞争力以local_transport对全部四外部的六指标和Acc逐项分析，高最终BA不能掩盖tail/forget/old/new损伤，优势与退步并列，并结合架构/预训练/更新和回放差异。仅冻结配置、单seed/order反复开发集的描述性证据，不称等资源、公平预算胜负、统计显著、封存test或临床泛化。

失败缺失不补0，也不能以未完整外部比较宣布竞争力。完整GitHub/NAS交付及聊天主表/全部正负/资源区别后，暂停同一监测。
