# 第七轮：当前任务类均衡的局部辅助目标归一化

第六轮局部类别监督执行与历史控制等价审计通过：ISIC通过原base开发门，但相对detached控制BA−0.0992pp、tail−1.2854pp；HK相对base仅BA+0.1300pp，未达+1门。候选局部特征梯度HK首任务约0.105、末任务约0.0031。这不是机制未执行，也不能证明梯度大小是唯一原因。

原辅助alpha=N_fit*w/(K_seen*batch_n)，当前fit每类w总和1，随机batch的期望总质量为K_current/K_seen，HK末任务为2/23。第七轮只把辅助CE分母改为K_current，保留原全局alpha、解析头、FD10、类中心、temperature1、系数1、优化器、clip和数据顺序。即local_alpha=global_alpha*(K_seen/K_current)，所有数据集/任务统一公式，不增加选类开关、幅度超参或val搜索。归一化使当前每类辅助质量相等、整体期望为1；单个随机mini-batch总质量可波动，不能声称每batch恒1。首任务K_seen=K_current，两目标和梯度精确相同。

ISIC归一化因子1/3/4；HK1/7.5/8.5/9.5/10.5/11.5，均由冻结任务划分决定。局部特征梯度变化还受图像/原型/类别/训练状态影响；更大辅助梯度可能损伤旧类或ISIC新类，FD10不是安全证书。这是归一化机制试验，不是保证HK过门。

保留ViT/adapter、原型统计、解析竞争头、第二轮区域运输局部先验以及第六轮同次前向可微四象限CE。中心每epoch冻结，来自当前fit和旧类聚合；只当前fit图像/标签进入CE。meta仍仅原controller来源及任务末拟合。旧影像/旧逐样本特征不回放。没有新数据源、新投影、可学习局部头或推理附加分数，RL和医学test/reserved/CIFAR封存。

## 固定矩阵与检查

两条新候选ISIC_local_normalized、HK_local_normalized，原base2条和第六轮local_supervised2条严格复用，总6比较轨迹。先两数据集真实0更新预检通过，再用GPU0/1/2固定派发ISIC→HK；ISIC468、HK176，共644新adapter更新，0策略。种子及split_seed74002，原shuffled顺序/task_sizes、2epoch/batch64/workers4/lr.0003/FD10/原竞争头均不变，不调参/挑seed/隐藏重跑。预检workers0且第一batch backward后返回、无optimizer更新。每正式batch局部特征梯度有限且>0，原全局/局部loss有限、clip error_if_nonfinite。

CPU小检查：不等类样本数时原总辅助质量K_current/K_seen，改后期望总质量1且每类同质量；辅助logit梯度按规定比例缩放；首任务权重精确一致；无效类数被拒绝。沿用第六轮双支局部梯度与detach全局梯度检查。

两条均TRAINED且计数一致后才EVALUATE。新候选首任务每类recall必须精确匹配第六轮local_supervised首任务，字段initial_task_control_equal=true，否则fail closed；后续任务故意更改辅助目标，不要求global/类召回与旧控制一致。不套旧历史transport全阶段/paired_global控制检查。失败保留输出和实际成本，原轮不得改协议/放宽门/自动正式重跑。

每轮墙钟≤24小时，GPU显存和NAS挂载/空间/小写读探针及中性完整进程规则照旧。正常2preflight+2train+2evaluate共6成功子进程；实际失败全部额外计费，以PROGRAM_STATE.costs计一次，墙钟/CPU单列，不重复复用基线成本。历史关闭GPU驻留126318.926678秒单列不重置。

## 目标与交付

相对原static_pc，ISIC/HK各自ΔBA≥+1pp、Δtail≥−0.5pp、Δforgetting≤+1pp、Δold/new≥−1pp。单数据集过门不算完成；双数据集通过才进入一次预注册两个新训练seed×两数据集×base/同冻结候选8条配对确认。开发集反复查看，新seed仍非独立患者/临床验证。

结果全完成后公开base/原seen归一化supervised控制/current归一化候选完整六指标和相对控制/base全部pp增量，每stage每类n/recall及全部正负；直接判断HK是否过门、ISIC新类是否无下降/仍下降但在保护门内/仍违规。诊断仅标量local_seen_classes/local_current_classes/local_normalization_factor、局部CE/梯度、全局loss、模块先验激活、solve残差≤1e−8和实际更新/成本。梯度非零、CE下降或质量归一化不等于净效用。无附加推理读出，不生成虚构的改变预测数量。

代码/协议/匿名聚合/审计/成本通过显式代理公开GitHub，并保存NAS匿名报告和私有交付回执，核对账号/分支/范围/远端SHA/匿名HTTP，不仅存NAS。身份、影像、权重、逐样本分数/特征、均值/统计或中心/原型/边矩阵、私有路径配置及原始diagnostics/logs不下载或公开。
