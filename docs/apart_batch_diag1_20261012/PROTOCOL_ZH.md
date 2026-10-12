# APART现有checkpoint：评价批次及w冻结诊断协议

用户批准只复用现有ISIC/HK APART checkpoint和已有val，诊断评价批次与w；0重训/0optimizer/0train影像访问，test/reserved始终封存。不换seed、不调阈值/学习率/模型，不按结果选择更高评价方式覆盖旧结果。旧训练源1080e4a099d684237f466d57615bd5fdca9d2a1e、旧结果295a0f0ff4977e3ebe8174c94185386357c40e42，作者f3c5b8da5908b2f78266744a737f7f6bff45a43d。原模型、统计、协议和交付包只读。

三模式固定batch48/不shuffle/相同旧eval变换：mixed_batch保留val manifest顺序、双池batchwise_prompt=True；class_concat按增量标签稳定拼接、双池batchwise_prompt=True；individual_routing保留混合顺序，仅将两个池batchwise_prompt=False，逐样本选adapter。逐样本仍batch48，不改模型权重或用标签决定逐样本路由。class_concat按真标签组织batch只是作者代码诊断，会引入标签相关的批次上下文，不能作为临床部署流程。

使用全部3个ISIC及6个HK阶段，不挑checkpoint；原order/labels/seen mask/val分母不变。共27个mode-stage评价、23949次val图像曝光，新增正式训练更新0。9份checkpoint metadata严格核对源/version/方法/dataset/order/seed/task/既有预算，strict载入完整双池/head/冻结原骨干；每模式前后参数版本不变，梯度关闭。首批真实forward检查有限/维度/配对身份，但不产生或利用选参指标。

w由同阶段checkpoint的assigner与冻结原骨干CLS计算，使用val类别对应的原train频数（元数据复用，不开train.csv）；仅在mixed模式预测完成后计算，不馈入任何三模式预测。每stage全部/旧/新/原theta100 rare-common及每类，公开n、mean/std/min/max、5/25/50/75/95分位数、w≤.01/≤.05/≥.95计数。近零screen固定为至少90%样本w≤.05且mean≤.05，仅描述checkpoint在val上的权重，不证明历史epoch何时坍缩，也不以小n夸大。

三模式按同一内部样本位置配对，公开每stage/每类预测改变、救回/损伤、双方正确/错误、任一池路由改变计数；不公开任何样本位置、ID、预测分数、w向量、特征或模型。完整最终/平均BA、tail/forget/old/new、Acc/macroF1、27stage和31类及全差值均报告。原频数head/tail规则不改。遗忘跨stage也受batch上下文影响，不能全部当作纯参数遗忘。

mixed必须与已公开原结果全部stage类recall精确复核；若不一致保留全部结果、费用和差异，标明原结果复核失败，不自动重试或挑更好模式，停止干净归因。若任意worker失败，保留部分匿名聚合、具体去标识错误和全部费用，missing不填0。

CPU检查、协议/源代理GitHub冻结发布并验证精确HTTP200后才启动。NAS实际nfs/free/读写probe，GPU仅0/1每卡free≥32768MiB，不碰2/3或他人进程。固定两个evaluation-only worker费用逐次记录，24h从实际launch起；CPU/协调器wall另列，旧训练/评价/数据0重复收费。没有新训练或自动重试，不恢复此前已暂停的小时监测。结束后匿名结果/完整正负报告/AUDIT/费用代理推送，remote SHA和精确匿名报告HTTP200核实，再独立NAS public_delivery/DELIVERY，禁止重复创建旧轮交付。

CPU检查PASS：原作者多数票反例/逐样本独立、稳定按类拼接与canonical配对、救回损伤统计、w固定screen和无效数拒绝、checkpoint权限/父轮屏障、完整三模式报告及错误分母拒绝。0医学影像/0GPU/0optimizer。CPU驻留9.426304秒包含pipeline0.388502秒，不相加。首次客户端CPU部署字符串换行被AST在SSH前拒绝，0NAS写入/0GPU；修正后CPU一次PASS，谱系保留。
