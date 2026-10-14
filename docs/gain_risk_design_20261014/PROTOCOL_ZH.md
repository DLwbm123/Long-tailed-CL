# RISK v3 准备及贡献对照协议

唯一当前授权：重新设计风险估计/准入、解析头兼容性核对和合成CPU检查；不开医学实验、不重训、不调参、不读任何医学影像/checkpoint/val/test。不延续或重启已完成轮，不创建监测。

科学冻结内容见 PROTOCOL_LOCK.json；机制见 DESIGN_ZH.md；完整未来报告规范见 CONTRIBUTION_SPEC.json。准备状态是 CPU_DESIGN_ONLY、production_integrated=false，不能将设计源码提交称为医学部署源或GPU-ready。

## 主四格与三个贡献对照

主四格：base、local_transport、base+RISK、local_transport+RISK。主配置固定local_transport+RISK，不选val上更好的配置。

三个贡献对照在两个控制、两个数据集全部阶段都报告：

1. RISK − RISK_NO_DRIFT：保留校准、边界借用、逐旧类破坏约束；移除分数漂移包络及 uncovered 罚项。无漂移对照是明确不安全的机制诊断，不用于替换主方法。
2. RISK − RISK_NO_BOUNDARY：保留新漂移、事件校准和逐旧类约束；借用恢复原 v2 covariance family，保留未知借用误差罚项。比较的是边界匹配与分布混合这一组合机制，不能称单一 donor-distance 因果。校准凭据必须分别匹配各估计器。
3. RISK − RISK_NO_DAMAGE：只移除逐旧类破坏准入条件；宏平均下界、校准、漂移和边界借用仍保留。记录哪些稀少旧类风险使主配置返回identity及放开后实际得失。

严格保留四固定候选的全部估计和正负变化；旧JOINT_PRED_SELECTED、base/local_transport、AIR只复用已公开汇总。若未来读取18阶段checkpoint，必须单独获得生产与评价授权；不借此协议启动。

## 未来诊断/报告与准入屏障（当前不执行）

核对两个数据集原order、seed74002、val每类分母、control source/task/budget；全部18阶段准备/元数据封存后才允许四条真实零更新预检。四条全PASS才统一评价，必要推理仅GPU0/1，GPU2/3禁止。test/reserved始终封存。仅复用旧checkpoint，不重训、不改变任何旧轨迹或方向，不读取val来确定校准、供体信任或上限。

未来生产适配器必须绑定校准凭据的来源、独立性、目标域转移假设及候选模型版本。缺失项要有逐类identity原因。当前尚未实现凭据适配器、医学collector、生产worker或真实预检；CPU设计脚本不能冒充生产入口。

报告主四格全部六指标、Acc及已有F1、AIR差距、三个贡献对照、所有31最终类和全部stage n/recall、实际救回/破坏/净收益、旧旧/旧新混淆、新类守卫前后变化及归一化间隔。逐候选记录符号/排序错误、救回高估与破坏低估、数值误差、校准误差、漂移范围外质量、年龄、供体及借用罚项、逐旧类cap、拒绝原因。全部identity和所有正零负都保留，missing记null，不通过观察最优候选改主方法。

模型条件范围、独立同域校准频率界、医学实际val差异分开报告；不称临床置信、统计显著、独立患者或封存test泛化。原医学方向启发式不变，不能保证仅修准入就产生净增益。

## 费用与公开范围

此次只计合成CPU各次子进程驻留；pipeline包含其中，不相加。无医学GPU预检、正式GPU费用、训练更新或数据准备费用。旧轮费用只保留历史引用，不重复收费。

源码、协议、合成检查及匿名准备审计可公开；private运行目录、原始日志、私有路径、模型、矩阵、影像和样本信息不公开。准备交付和医学结果交付严格区分。任何未来运行失败保留所有费用和去标识原因，不自动正式重试。
