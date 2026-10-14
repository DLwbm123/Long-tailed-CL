# CRISK v4 冻结贡献对照协议（设计，未授权执行）

固定ISIC/HK、seed74002、原class order与task、两控制18阶段checkpoint和gain_joint1的72个数值候选。主方法预先固定local_transport+CRISK；base是贡献控制。源/data/order和支持数在PROTOCOL_LOCK.json中锁定。现有科学进程/源码/回执不改，无重训/新方向/步长扫描/val调参/test访问。

主四格为base、local_transport、base+CRISK、local_transport+CRISK。四项单因素贡献：

|对照|唯一替换|其余保持|
|---|---|---|
|NO_RESERVOIR|q用原模型OFF错误概率，不用目标事件计数锚定|条件r/h、漂移、供体和边界保护|
|NO_DRIFT|效用里的s_q/s_r/s_h为0|供体rho中的漂移可信度仍保留，避免同时改变借用|
|NO_DONOR_TRUST|供体rho固定1|供体身份/提案权重、目标n、伪计数上限、漂移和边界保护|
|NO_EDGE_GUARD|旁路相对间隔侵蚀准入|候选分数、新类守卫、净风险效用|

全部五模式使用同一18状态、四候选和同一目标计数；贡献只改变选择或准入，不能改变方向后继续称风险估计单因素。首stage严格identity；每个模式同时报告四候选、identity/NOT_READY及所有拒绝原因。拟议360条stage×candidate×selector审计、90个stage选择记录，不产生20条新训练轨迹。

未来若另获执行授权，先完成凭据生产接线与来源/权限检查、CPU全维检查及四条真实零更新预检，全部通过后只GPU0/1统一读取已有val评价；训练侧统计和全部选择在val访问前封存。旧训练图像如需补相对侵蚀或供体凭据，须明确获得纯推理权限；不能把本次设计授权当成医学图像访问授权。checkpoint与候选只读，test/reserved继续封存，无自动正式retry。

贡献报告固定各自控制和全部外部AIR参照，不挑val最好配置。完整六指标、Acc、可得macroF1、stage和31类n/recall、救回/破坏/旧旧及旧新流、RAW与新类守卫、间隔侵蚀分布、q/E/C及条件支持、预测符号/排名/偏差、供体评分、漂移敏感范围与凭据年龄全部正零负保留。无条件救回必须≤q，破坏≤1−q；训练内预测误差和val误差分列，不将前者称独立校准。

NOT_READY不是已完成医学方法，missing=null。GPU费用按真实每个进程驻留一次；复用控制/方向/AIR结果0新增收费，解析/CPU统计费用和并行wall分开。训练及val的重复stage不增加独立n。身份/样本/影像/权重/逐样本分数/特征/矩阵/私有路径不公开。

本次只做合成CPU设计验收与源/协议公开冻结。没有医学preflight，没有GPU实验，也没有效果结论。后续启动必须新授权，不能自动接续。
